#!/usr/bin/env python3
"""
build_field.py
==============
Standalone script: build a REALISTIC simulated opponent FIELD for a DraftKings
NFL Classic contest and write it to CSV.

This is the "who am I playing against?" half of the pipeline. It does NOT pick
your lineups — it manufactures the thousands of lineups the rest of the field is
expected to submit, so the rest of the engine can score your candidates against
a believable crowd. (Your own lineups come from ``build_candidates.py``.)

WHAT IT DOES, STEP BY STEP
--------------------------
1. INGEST the slate (``nfl_ingest.build_slate``): the DK player pool + projected
   ownership + per-stat range-of-outcomes projections + the game schedule.

2. SIMULATE the slate (``sim_engine.simulate``): a correlated, game-consistent
   Monte-Carlo of DK points for every player across ``--n-sims`` sims. This is
   needed for the realism features below — a lineup's "projection" (to rank the
   overbuild) and each player's "ceiling" (p90, to weight the sharp builds).

3. STACK-OWNERSHIP CEILING SIGNAL (``stack_signal``): give popular offenses a
   small bump to their high-end sims, tied to projected stack ownership. The
   same boosted scores are used to rank the field here and to grade candidates,
   so it is a coherent re-weighting of reality (see ``--stack-boost``).

4. BUILD THE FIELD (``field_simulator.build_field``) for each ``--sizes`` entry.
   Three realism layers turn a naive ownership-imitation field into a believable
   one (a pure-chalk field is too soft — it would hand a no-skill lineup a
   positive edge, which is impossible):
     * SIZE MODEL — chalk "temperature" (beta) concentrates chalk in small
       fields / flattens it in big ones, and the QB-stack-size distribution is
       tilted toward bigger stacks for larger fields.
     * SHARP / CHALK MIX — ``--sharp-frac`` of the field is built "sharp"
       (bigger QB stacks, ceiling-weighted skill players, forced bring-backs) on
       the ownership base so it still stacks popular teams; the remainder is
       plain chalk (a realistic soft tail).
     * SUBMITTED, NOT RANDOM — the field is overbuilt ``--overbuild`` x and
       trimmed to the highest-projection lineups, because the field you actually
       face is the better lineups thousands of people submit, not raw draws.
   ``--own-uncertainty`` additionally rebuilds the pool from fresh ownership
   draws so the field mixes over ownership scenarios.

5. WRITE ``<outdir>/field_<N>.csv`` per size (one row per opponent lineup, slot
   columns as ``NAME (TEAM)``, plus Salary, QBstack and the lineup Proj), and
   print a summary of the realized stack-size distribution and chalk exposure.

USAGE
-----
  python3 build_field.py --sizes 1000 6000 20000 --n-sims 8000
  python3 build_field.py --sizes 6000 --sharp-frac 0.75 --overbuild 2.0
  python3 build_field.py --sizes 6000 --fast        # naive ownership field, no sim

The learned field grammar (QB-stack sizes, FLEX mix, ...) comes from
``field_params_nfl.json`` (see ``learn_field.py``); ``--params`` overrides it.
"""
import argparse
import os
from collections import Counter

import numpy as np

import nfl_ingest
import sim_engine
import field_builder as fb
import field_simulator as fs
import contest_sim
import stack_signal


def _prep_scores(slate, n_sims, seed, stack_boost):
    """Run the correlated sim, apply the stack-ownership ceiling boost, and
    attach each entity's ceiling (p90) as ``up`` for the sharp builds. Returns
    (boosted_scores, dk_mean) — dk_mean {key: mean points} ranks the overbuild."""
    sim = sim_engine.simulate(slate, n_sims=n_sims, seed=seed)
    names_by_team = stack_signal.offense_names_by_team(slate.entities)
    own_by_key = {e["key"]: float(e.get("own", 0.0)) for e in slate.entities}
    stack_own = stack_signal.team_stack_ownership(names_by_team, own_by_key)
    dkp = stack_signal.apply_stack_ownership_boost(
        sim.dk, names_by_team, stack_own, n_sims, strength=stack_boost)
    for e in slate.entities:
        arr = dkp.get(e["key"])
        e["up"] = (float(np.percentile(arr, 90)) if arr is not None
                   else max(float(e.get("own", 0.0)), 1e-3))
    dk_mean = {k: float(v.mean()) for k, v in dkp.items()}
    return dkp, dk_mean


def _summary(field, dk_mean):
    """Human-readable read on one built field."""
    n = len(field)
    stacks = Counter(lu["stack"] for lu in field)
    sal = [lu["salary"] for lu in field]
    proj = [fs._lineup_proj(lu, dk_mean) for lu in field] if dk_mean else []
    # chalk concentration: share of roster spots taken by the 20 most-used players
    use = Counter()
    for lu in field:
        for pl in lu["players"]:
            use[pl["key"]] += 1
    top20 = sum(c for _, c in use.most_common(20)) / max(1, sum(use.values())) * 100
    line = (f"  QB-stack dist " +
            " ".join(f"{k}:{100*stacks[k]/n:.0f}%" for k in sorted(stacks)) +
            f" | salary mean {np.mean(sal):.0f}"
            f" | unique players {len(use)} | top-20 roster share {top20:.0f}%")
    if proj:
        line += f" | lineup Proj mean {np.mean(proj):.1f}"
    return line


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 6000, 20000],
                    help="contest sizes (entries) to build a field for")
    ap.add_argument("--n-sims", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--seed-field", type=int, default=101)
    ap.add_argument("--medium", type=int, default=6000,
                    help="baseline field size the projected ownership describes")
    ap.add_argument("--chalk-sensitivity", type=float, default=0.30)
    ap.add_argument("--stack-tilt", type=float, default=0.12)
    ap.add_argument("--sharp-frac", type=float, default=fs.FIELD_SHARP_FRAC,
                    help="share of the field built sharp vs chalk (0 = naive)")
    ap.add_argument("--overbuild", type=float, default=fs.FIELD_OVERBUILD,
                    help="build this x field size and keep the top by projection")
    ap.add_argument("--stack-boost", type=float, default=0.05,
                    help="stack-ownership ceiling bump on the sim scores")
    ap.add_argument("--own-uncertainty", action="store_true",
                    help="mix the field over fresh ownership draws")
    ap.add_argument("--fast", action="store_true",
                    help="skip the sim: build a naive ownership-imitation field "
                         "(no overbuild ranking, no ceiling-weighted sharp part)")
    ap.add_argument("--params", default="field_params_nfl.json")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    slate = nfl_ingest.build_slate()
    params = fb.load_params(a.params)
    print(f"slate: {len(slate.players)} offense, {len(slate.dst)} DST, "
          f"{len(slate.teams)} teams, {len(slate.games)} games")

    dk_mean = None
    if not a.fast:
        _, dk_mean = _prep_scores(slate, a.n_sims, a.seed, a.stack_boost)
        print(f"sim: {a.n_sims} sims  (stack-boost {a.stack_boost})")
    else:
        print("fast mode: no sim — naive ownership field (soft; for comparison only)")

    for N in a.sizes:
        field = fs.build_field(
            slate.entities, params, N, dk_mean, n_med=a.medium,
            chalk_sensitivity=a.chalk_sensitivity, stack_tilt=a.stack_tilt,
            sharp_frac=(0.0 if a.fast else a.sharp_frac),
            overbuild=(1.0 if a.fast else a.overbuild),
            seed=a.seed_field + N, own_uncertainty=a.own_uncertainty)
        beta = fs.beta_for_size(N, a.medium, a.chalk_sensitivity)
        df = fb.lineups_to_df(field)
        if dk_mean is not None:
            df["Proj"] = [round(fs._lineup_proj(lu, dk_mean), 1) for lu in field]
        path = os.path.join(a.outdir, f"field_{N}.csv")
        df.to_csv(path, index=False)
        print(f"[field {N:>6}] beta {beta:.2f}  built {len(field)}  -> {path}")
        print(_summary(field, dk_mean))

    print(f"done -> {a.outdir}/")


if __name__ == "__main__":
    main()
