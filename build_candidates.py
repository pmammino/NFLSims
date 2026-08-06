#!/usr/bin/env python3
"""
build_candidates.py
===================
Standalone script: build a pool of CANDIDATE lineups — the lineups WE are
considering entering — for a DraftKings NFL Classic contest, and write them to
CSV with their simulated projection and ceiling.

This is the "what should I enter?" half of the pipeline. Where ``build_field.py``
imitates the crowd, this script deliberately builds SHARP, ceiling-seeking
lineups: the same QB-stack construction grammar, but tilted toward the shapes
that win tournaments rather than the shapes the crowd plays. Scoring these
candidates against the field and selecting an export set is done downstream
(``run_sim.py`` / the app); this script just manufactures and characterizes the
candidate pool.

FIELD vs CANDIDATE — the key difference
---------------------------------------
Both use the same builder, but with opposite postures:
  * FIELD picks stack TEAMS and players by projected OWNERSHIP (it models what
    the crowd does), on the field-shaped stack-size distribution.
  * CANDIDATES pick stack teams OWNERSHIP-BLIND (``uniform``) so we are not
    anchored to chalk, weight the individual skill/FLEX picks by simulated
    CEILING (p90) instead of ownership, lean toward BIGGER QB stacks, and force
    a bring-back from the QB's opponent more often. These are exactly the
    construction edges the win region shows over the field.

WHAT IT DOES, STEP BY STEP
--------------------------
1. INGEST the slate (``nfl_ingest.build_slate``).

2. SIMULATE (``sim_engine.simulate``): correlated DK points per player. Used to
   derive each player's ceiling (p90) — the candidate selection weight — and to
   report each finished lineup's projection and ceiling.

3. STACK-OWNERSHIP CEILING SIGNAL (``stack_signal``): the same small high-end
   bump to popular offenses used everywhere else, so the ceilings the candidate
   builder chases are consistent with how the field is graded.

4. BUILD CANDIDATES (``field_builder.Builder``) with the sharp posture:
     * ``uniform=True``            — ownership-blind stack-TEAM / QB choice.
     * ``use_upside=True``         — weight skill/FLEX picks by ceiling (p90).
     * bigger stacks               — ``candidate_stack_sizes(strength)`` tilts
                                     the QB-stack-size distribution upward.
     * ``bringback_prob``          — force a primary-opponent bring-back.
     * ``jitter``                  — optional lognormal shock to diversify the
                                     pool (spread near-equivalent players).
   Every one of these defaults toward "off" reproduces field-imitation, so the
   knobs below are what make a candidate a candidate.

5. WRITE ``<outdir>/candidates.csv`` (one row per candidate: slot columns as
   ``NAME (TEAM)``, Salary, QBstack, and the simulated Proj / Ceiling / Std),
   and print a summary of the pool's stack shapes and projection spread.

USAGE
-----
  python3 build_candidates.py --num-candidates 10000 --n-sims 8000
  python3 build_candidates.py --num-candidates 5000 --cand-stack-strength 0.8 \
          --cand-bringback 0.4 --jitter 0.2
  python3 build_candidates.py --num-candidates 5000 --no-cand-upside   # ownership-blind only
"""
import argparse
import os
from collections import Counter

import numpy as np

import nfl_ingest
import sim_engine
import field_builder as fb
import contest_sim
import stack_signal


def _prep_scores(slate, n_sims, seed, stack_boost):
    """Run the sim, apply the stack-ownership ceiling boost, attach each entity's
    ceiling (p90) as ``up`` (the candidate selection weight). Returns the boosted
    per-player score dict (used to characterize the finished lineups)."""
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
    return dkp


def _build_set(builder, n, cap_mult=40):
    lus, fails, limit = [], 0, n * cap_mult + 1000
    while len(lus) < n and fails < limit:
        lu = builder.build_one()
        if lu is None:
            fails += 1
            continue
        lus.append(lu)
    return lus, fails


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-candidates", type=int, default=10000)
    ap.add_argument("--n-sims", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--seed-candidates", type=int, default=2025)
    ap.add_argument("--stack-boost", type=float, default=0.05,
                    help="stack-ownership ceiling bump on the sim scores")
    ap.add_argument("--cand-stack-strength", type=float, default=0.6,
                    help="tilt candidate QB-stack sizes bigger (0 = field-shaped)")
    ap.add_argument("--cand-bringback", type=float, default=0.35,
                    help="candidate forced primary-opponent bring-back rate")
    ap.add_argument("--no-cand-upside", dest="cand_upside", action="store_false",
                    default=True,
                    help="weight candidate skill picks by ownership, not ceiling")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="lognormal shock on candidate selection (diversify pool)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the sim: ownership-blind build only (no ceiling "
                         "weighting, no Proj/Ceiling columns)")
    ap.add_argument("--params", default="field_params_nfl.json")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    slate = nfl_ingest.build_slate()
    params = fb.load_params(a.params)
    print(f"slate: {len(slate.players)} offense, {len(slate.dst)} DST, "
          f"{len(slate.teams)} teams, {len(slate.games)} games")

    dkp = None
    if not a.fast:
        dkp = _prep_scores(slate, a.n_sims, a.seed, a.stack_boost)
        print(f"sim: {a.n_sims} sims  (stack-boost {a.stack_boost})")
    else:
        print("fast mode: no sim — ownership-blind candidates, ceiling weighting off")

    # sharp posture: bigger stacks, ceiling-weighted picks, forced bring-backs
    cparams = dict(params)
    cparams["stack_sizes"] = fb.candidate_stack_sizes(
        params["stack_sizes"], a.cand_stack_strength)
    cb = fb.Builder(fb.Pool(slate.entities), cparams, seed=a.seed_candidates,
                    uniform=True, jitter=a.jitter,
                    use_upside=(a.cand_upside and not a.fast),
                    bringback_prob=a.cand_bringback)
    cands, fails = _build_set(cb, a.num_candidates)
    df = fb.lineups_to_df(cands)

    if dkp is not None:
        mat = contest_sim.score_matrix(cands, dkp, a.n_sims)   # (n_sim, n_cand)
        df["Proj"] = np.round(mat.mean(axis=0), 1)
        df["Ceiling"] = np.round(np.percentile(mat, 90, axis=0), 1)
        df["Std"] = np.round(mat.std(axis=0), 1)
        df = df.sort_values("Ceiling", ascending=False).reset_index(drop=True)

    path = os.path.join(a.outdir, "candidates.csv")
    df.to_csv(path, index=False)
    print(f"candidates: {len(cands)} built ({fails} fails) -> {path}")

    stacks = Counter(lu["stack"] for lu in cands)
    n = len(cands)
    print("  QB-stack dist " +
          " ".join(f"{k}:{100*stacks[k]/n:.0f}%" for k in sorted(stacks)))
    if dkp is not None:
        print(f"  lineup Proj mean {df['Proj'].mean():.1f}  "
              f"Ceiling mean {df['Ceiling'].mean():.1f}  "
              f"best Ceiling {df['Ceiling'].max():.1f}")
    print(f"done -> {a.outdir}/")


if __name__ == "__main__":
    main()
