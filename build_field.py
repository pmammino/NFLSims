#!/usr/bin/env python3
"""
STAGE 2 / 4 — FIELD
===================
Build a REALISTIC simulated opponent field for a DK NFL Classic contest — the
lineups the rest of the field is expected to submit. This is the "who am I
playing against?" stage. It does not pick your lineups (that's
``build_candidates.py``); it manufactures the crowd so the scoring stage can
place your candidates against a believable set of opponents.

Reads the STAGE 1 (``simulate.py``) output — it does NOT re-simulate. A player's
projection (to rank the overbuild) and ceiling (to weight the sharp builds) come
straight from ``player_pool.csv``.

WHY THE REALISM MATTERS
-----------------------
A naive ownership-imitation field is too soft — it would hand a no-skill lineup
a positive edge, which is impossible. Three layers fix it:
  * SIZE MODEL     — chalk concentrates in small fields / flattens in big ones
                     (``--chalk-sensitivity``), stacks tilt bigger for big fields
                     (``--stack-tilt``).
  * SHARP/CHALK MIX— ``--sharp-frac`` of the field is built sharp (bigger stacks,
                     ceiling-weighted players, forced bring-backs) on the
                     ownership base; the rest is a chalk soft tail.
  * SUBMITTED      — overbuild ``--overbuild`` x and keep the highest-projection
                     lineups (the field is the better lineups people submit).
  * ``--own-uncertainty`` additionally mixes the field over ownership draws.

INPUTS  (from ``--indir``, produced by ``simulate.py``)
-------------------------------------------------------
  player_pool.csv   key,name,pos,team,opp,salary,own,contest_id,proj,ceiling
                    (own drives field selection; proj ranks the overbuild;
                     ceiling weights the sharp portion's skill picks)
  field_params_nfl.json (via ``--params``) — the learned QB-stack / FLEX grammar

OUTPUTS  (to ``--outdir``, default same as indir)
-------------------------------------------------
  field_<N>.csv     one row per opponent lineup for each ``--sizes`` entry:
                      Lineup, Salary, QBstack, QB, RB1, RB2, WR1, WR2, WR3, TE,
                      FLEX, DST, Proj
                    Slot cells are ``KEY (TEAM)`` (the entity key, so the scoring
                    stage can map each slot straight back to its sim array).

USAGE
-----
  python3 build_field.py --sizes 1000 6000 20000
  python3 build_field.py --sizes 6000 --sharp-frac 0.75 --overbuild 2.0
  python3 build_field.py --sizes 6000 --sharp-frac 0 --overbuild 1   # naive/soft field
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

import field_builder as fb
import field_simulator as fs


def load_pool(indir):
    """Rebuild the builder's entity list from player_pool.csv (STAGE 1 output).
    Returns (entities, dk_mean) — dk_mean {key: proj} ranks the overbuild; each
    entity carries ``up`` (=ceiling) for the sharp ceiling-weighted picks."""
    path = os.path.join(indir, "player_pool.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run simulate.py first")
    df = pd.read_csv(path)
    entities, dk_mean = [], {}
    for r in df.to_dict("records"):
        key = str(r["key"])
        entities.append({
            "key": key, "name": str(r.get("name", "")), "pos": str(r["pos"]),
            "team": "" if pd.isna(r["team"]) else str(r["team"]),
            "opp": "" if pd.isna(r["opp"]) else str(r["opp"]),
            "salary": int(r["salary"]), "own": float(r["own"]),
            "up": float(r["ceiling"]),
        })
        dk_mean[key] = float(r["proj"])
    return entities, dk_mean


def _summary(field, dk_mean):
    n = len(field)
    stacks = Counter(lu["stack"] for lu in field)
    sal = [lu["salary"] for lu in field]
    proj = [fs._lineup_proj(lu, dk_mean) for lu in field]
    use = Counter()
    for lu in field:
        for pl in lu["players"]:
            use[pl["key"]] += 1
    top20 = sum(c for _, c in use.most_common(20)) / max(1, sum(use.values())) * 100
    return ("  QB-stack " + " ".join(f"{k}:{100*stacks[k]/n:.0f}%" for k in sorted(stacks)) +
            f" | salary mean {np.mean(sal):.0f} | unique {len(use)} | "
            f"top-20 share {top20:.0f}% | lineup Proj mean {np.mean(proj):.1f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 6000, 20000])
    ap.add_argument("--medium", type=int, default=6000,
                    help="baseline field size the projected ownership describes")
    ap.add_argument("--chalk-sensitivity", type=float, default=0.30)
    ap.add_argument("--stack-tilt", type=float, default=0.12)
    ap.add_argument("--sharp-frac", type=float, default=fs.FIELD_SHARP_FRAC)
    ap.add_argument("--overbuild", type=float, default=fs.FIELD_OVERBUILD)
    ap.add_argument("--own-uncertainty", action="store_true")
    ap.add_argument("--seed-field", type=int, default=101)
    ap.add_argument("--params", default="field_params_nfl.json")
    ap.add_argument("--indir", default="out", help="where simulate.py wrote its output")
    ap.add_argument("--outdir", default=None, help="defaults to --indir")
    a = ap.parse_args()
    outdir = a.outdir or a.indir
    os.makedirs(outdir, exist_ok=True)

    entities, dk_mean = load_pool(a.indir)
    params = fb.load_params(a.params)
    print(f"pool: {len(entities)} entities loaded from {a.indir}/player_pool.csv")

    for N in a.sizes:
        field = fs.build_field(
            entities, params, N, dk_mean, n_med=a.medium,
            chalk_sensitivity=a.chalk_sensitivity, stack_tilt=a.stack_tilt,
            sharp_frac=a.sharp_frac, overbuild=a.overbuild,
            seed=a.seed_field + N, own_uncertainty=a.own_uncertainty)
        beta = fs.beta_for_size(N, a.medium, a.chalk_sensitivity)
        df = fb.lineups_to_df(field)
        df["Proj"] = [round(fs._lineup_proj(lu, dk_mean), 1) for lu in field]
        path = os.path.join(outdir, f"field_{N}.csv")
        df.to_csv(path, index=False)
        print(f"[field {N:>6}] beta {beta:.2f}  built {len(field)}  -> {path}")
        print(_summary(field, dk_mean))


if __name__ == "__main__":
    main()
