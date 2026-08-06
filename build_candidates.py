#!/usr/bin/env python3
"""
STAGE 3 / 4 — CANDIDATE
=======================
Build the pool of CANDIDATE lineups — the lineups WE are considering entering.
This is the "what should I enter?" stage. Where ``build_field.py`` imitates the
crowd, this stage deliberately builds SHARP, ceiling-seeking lineups.

Reads the STAGE 1 (``simulate.py``) output — it does NOT re-simulate. Each
player's ceiling (p90) is the candidate selection weight, and the per-sim arrays
are used to report every finished lineup's projection and ceiling.

FIELD vs CANDIDATE — same builder, opposite posture
---------------------------------------------------
  * FIELD    picks stack teams / players by projected OWNERSHIP, field-shaped
             stack sizes.
  * CANDIDATE picks stack teams OWNERSHIP-BLIND (``uniform``), weights skill/FLEX
             picks by simulated CEILING (p90), leans toward BIGGER QB stacks, and
             forces a bring-back from the QB's opponent more often — the
             construction edges the win region shows over the field.
Every knob defaulting toward "off" reproduces field imitation, so the flags
below are what make a candidate a candidate.

INPUTS  (from ``--indir``, produced by ``simulate.py``)
-------------------------------------------------------
  player_pool.csv     key,name,pos,team,opp,salary,own,contest_id,proj,ceiling
                      (ceiling weights candidate skill picks)
  player_dk_sims.npy  {key: array[n_sims]} — to compute each lineup's Proj /
                      Ceiling / Std (a lineup ceiling needs the summed arrays,
                      not just per-player numbers)
  field_params_nfl.json (via ``--params``) — the learned QB-stack / FLEX grammar

OUTPUTS  (to ``--outdir``, default same as indir)
-------------------------------------------------
  candidates.csv      one row per candidate, sorted by ceiling:
                        Lineup, Salary, QBstack, QB, RB1, RB2, WR1, WR2, WR3, TE,
                        FLEX, DST, Proj, Ceiling, Std
                      Slot cells are ``KEY (TEAM)`` (entity key) so the scoring
                      stage can map each slot back to its sim array.

USAGE
-----
  python3 build_candidates.py --num-candidates 10000
  python3 build_candidates.py --num-candidates 5000 --cand-stack-strength 0.8 \
          --cand-bringback 0.4 --jitter 0.2
  python3 build_candidates.py --num-candidates 5000 --no-cand-upside   # ownership-blind only
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

import field_builder as fb
import contest_sim


def load_pool(indir):
    """Rebuild the builder's entity list from player_pool.csv; each entity's
    ``up`` (=ceiling) is the candidate skill-pick weight."""
    path = os.path.join(indir, "player_pool.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run simulate.py first")
    df = pd.read_csv(path)
    entities = []
    for r in df.to_dict("records"):
        entities.append({
            "key": str(r["key"]), "name": str(r.get("name", "")),
            "pos": str(r["pos"]),
            "team": "" if pd.isna(r["team"]) else str(r["team"]),
            "opp": "" if pd.isna(r["opp"]) else str(r["opp"]),
            "salary": int(r["salary"]), "own": float(r["own"]),
            "up": float(r["ceiling"]),
        })
    return entities


def load_sims(indir):
    """Load the {key: array[n_sims]} sim dict; returns (dk, n_sims) or (None, 0)."""
    path = os.path.join(indir, "player_dk_sims.npy")
    if not os.path.exists(path):
        return None, 0
    dk = np.load(path, allow_pickle=True).item()
    n = len(next(iter(dk.values()))) if dk else 0
    return dk, int(n)


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
    ap.add_argument("--seed-candidates", type=int, default=2025)
    ap.add_argument("--cand-stack-strength", type=float, default=0.6,
                    help="tilt candidate QB-stack sizes bigger (0 = field-shaped)")
    ap.add_argument("--cand-bringback", type=float, default=0.35,
                    help="candidate forced primary-opponent bring-back rate")
    ap.add_argument("--no-cand-upside", dest="cand_upside", action="store_false",
                    default=True,
                    help="weight candidate skill picks by ownership, not ceiling")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="lognormal shock on candidate selection (diversify pool)")
    ap.add_argument("--params", default="field_params_nfl.json")
    ap.add_argument("--indir", default="out", help="where simulate.py wrote its output")
    ap.add_argument("--outdir", default=None, help="defaults to --indir")
    a = ap.parse_args()
    outdir = a.outdir or a.indir
    os.makedirs(outdir, exist_ok=True)

    entities = load_pool(a.indir)
    dk, n_sims = load_sims(a.indir)
    params = fb.load_params(a.params)
    print(f"pool: {len(entities)} entities loaded from {a.indir}/player_pool.csv"
          + (f"  ({n_sims} sims)" if dk else "  (no sims found — Proj/Ceiling skipped)"))

    # sharp posture: bigger stacks, ceiling-weighted picks, forced bring-backs
    cparams = dict(params)
    cparams["stack_sizes"] = fb.candidate_stack_sizes(
        params["stack_sizes"], a.cand_stack_strength)
    cb = fb.Builder(fb.Pool(entities), cparams, seed=a.seed_candidates,
                    uniform=True, jitter=a.jitter,
                    use_upside=a.cand_upside, bringback_prob=a.cand_bringback)
    cands, fails = _build_set(cb, a.num_candidates)
    df = fb.lineups_to_df(cands)

    if dk is not None:
        mat = contest_sim.score_matrix(cands, dk, n_sims)   # (n_sims, n_cand)
        df["Proj"] = np.round(mat.mean(axis=0), 1)
        df["Ceiling"] = np.round(np.percentile(mat, 90, axis=0), 1)
        df["Std"] = np.round(mat.std(axis=0), 1)
        df = df.sort_values("Ceiling", ascending=False).reset_index(drop=True)

    path = os.path.join(outdir, "candidates.csv")
    df.to_csv(path, index=False)
    print(f"candidates: {len(cands)} built ({fails} fails) -> {path}")
    stacks = Counter(lu["stack"] for lu in cands)
    n = len(cands)
    print("  QB-stack " + " ".join(f"{k}:{100*stacks[k]/n:.0f}%" for k in sorted(stacks)))
    if dk is not None:
        print(f"  lineup Proj mean {df['Proj'].mean():.1f}  "
              f"Ceiling mean {df['Ceiling'].mean():.1f}  best {df['Ceiling'].max():.1f}")


if __name__ == "__main__":
    main()
