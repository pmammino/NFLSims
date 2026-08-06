#!/usr/bin/env python3
"""
STAGE 1 / 4 — CORRELATED SIMULATIONS
====================================
Turn the raw slate into a correlated Monte-Carlo of DraftKings points for every
player. This is the foundation the other three stages read: the field, the
candidates, and the contest scoring all reason about the SAME per-sim score
arrays, so a QB and his pass-catchers boom together in the same simulated games.

WHAT IT DOES
------------
1. INGEST the slate (``nfl_ingest.build_slate``) from the four raw files.
2. SIMULATE (``sim_engine.simulate``): a hierarchical, game-consistent draw of
   each player's stat line -> DK points, ``--n-sims`` times. Teammates share
   game / team-offense / pass / rush latents, so the correlations are real.
3. STACK-OWNERSHIP CEILING SIGNAL (``stack_signal``, ``--stack-boost``): nudge
   popular offenses' high-end sims up a touch, tied to projected stack ownership.
   This is baked into the SAVED arrays so every downstream stage sees the same
   reality (set ``--stack-boost 0`` to disable).

INPUTS  (raw slate files in the working directory)
--------------------------------------------------
  projections.csv   per-PlayerID per-split (F/M/C) stat projections
  ownership.csv     the DK playable pool (Salary, Position, Ownership, ids)
  schedule.csv      Team,Opp[,Total,Implied] game pairings (+ Vegas)
  player_names.csv  id -> name crosswalk        (optional but recommended)
  dst_teams.csv     DST id -> team crosswalk    (optional)

OUTPUTS  (written to ``--outdir``, default ``out/``)
----------------------------------------------------
  player_dk_sims.npy      dict {entity_key: float32 array[n_sims]} of DK points,
                          INCLUDING the stack-ownership boost. THE stage-boundary
                          artifact — every later stage loads this.
  player_pool.csv         one row per playable entity with everything the build
                          and scoring stages need WITHOUT re-simulating:
                            key,name,pos,team,opp,salary,own,contest_id,proj,ceiling
                          (proj = mean sim points, ceiling = p90 sim points).
  player_projections.csv  the full human-readable player table (Proj, Floor p25,
                          Median, Ceiling p75, p10/p90/p99, Std, value, boom/bust).

USAGE
-----
  python3 simulate.py --n-sims 10000 --seed 20260709
  python3 simulate.py --n-sims 8000 --stack-boost 0.05 --outdir out
"""
import argparse
import os

import numpy as np
import pandas as pd

import nfl_ingest
import sim_engine
import exports
import stack_signal


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--stack-boost", type=float, default=0.05,
                    help="stack-ownership ceiling bump baked into the saved sims "
                         "(0 = off)")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    # ---- 1. ingest ----
    slate = nfl_ingest.build_slate()
    print(f"slate: {len(slate.players)} offense "
          f"({sum(p['matched'] for p in slate.players)} matched), "
          f"{len(slate.dst)} DST, {len(slate.teams)} teams, {len(slate.games)} games")

    # ---- 2. correlated sim ----
    sim = sim_engine.simulate(slate, n_sims=a.n_sims, seed=a.seed)
    rc = sim_engine.realized_correlations(sim, slate)
    print(f"sim: {a.n_sims} sims  realized QB-WR corr {rc['qb_wr_same']:.2f}  "
          f"WR-WR corr {rc['wr_wr_same']:.2f}")

    # ---- 3. stack-ownership ceiling signal (baked into the saved arrays) ----
    names_by_team = stack_signal.offense_names_by_team(slate.entities)
    own_by_key = {e["key"]: float(e.get("own", 0.0)) for e in slate.entities}
    stack_own = stack_signal.team_stack_ownership(names_by_team, own_by_key)
    dkp = stack_signal.apply_stack_ownership_boost(
        sim.dk, names_by_team, stack_own, a.n_sims, strength=a.stack_boost)
    if a.stack_boost > 0:
        print(f"stack-ownership boost: strength {a.stack_boost} on "
              f"{sum(1 for t in stack_own if stack_own[t] > 0)} offenses")

    # ---- outputs ----
    sims_path = os.path.join(a.outdir, "player_dk_sims.npy")
    np.save(sims_path, {k: v for k, v in dkp.items()}, allow_pickle=True)

    # player_pool.csv — the compact contract the build/scoring stages consume
    rows = []
    for e in slate.entities:
        arr = dkp.get(e["key"])
        proj = float(arr.mean()) if arr is not None else 0.0
        ceil = float(np.percentile(arr, 90)) if arr is not None else 0.0
        rows.append({
            "key": e["key"], "name": e.get("name", ""), "pos": e["pos"],
            "team": e.get("team", ""), "opp": e.get("opp", ""),
            "salary": int(e.get("salary", 0)),
            "own": round(float(e.get("own", 0.0)), 4),
            "contest_id": e.get("contest_id", ""),
            "proj": round(proj, 3), "ceiling": round(ceil, 3),
        })
    pool_path = os.path.join(a.outdir, "player_pool.csv")
    pd.DataFrame(rows).to_csv(pool_path, index=False)

    proj_path = os.path.join(a.outdir, "player_projections.csv")
    exports.player_table(sim, slate).to_csv(proj_path, index=False)

    print(f"wrote {sims_path}\n      {pool_path}\n      {proj_path}")
    print("next: build_field.py, build_candidates.py, then score_contest.py")


if __name__ == "__main__":
    main()
