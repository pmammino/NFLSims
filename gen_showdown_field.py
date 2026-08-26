#!/usr/bin/env python3
"""
gen_showdown_field.py
=====================
Generate the **ownership-aware** DraftKings NFL **Showdown** opponent field and
show the flow end to end, in isolation from the rest of the pipeline.

Flow
----
  ingest single-game slate (projections + ownership + schedule)
    -> correlated sim  (sim_engine)
    -> split the ONE overall-ownership feed into Captain-slot vs FLEX-slot
       propensity  (showdown.split_ownership)
    -> draw the field: Captain ~ Captain-ownership, FLEX ~ FLEX-ownership, with
       the realism guards (chalk temperature, a sharp fraction from the smart
       candidate builder, overbuild-and-trim by projection)  (showdown.build_field)

The ownership feed carries one row per player = overall rostered ownership. This
is the crowd you actually face, so it is ownership-DRIVEN (contrast
``gen_showdown_candidates.py``, which ignores ownership).

Point it at a single-game slate's files. Example:

  python3 gen_showdown_field.py --ownership ownership.csv --n 20000 \
          --n-sims 10000 --out out/showdown_field.csv
"""
import argparse
import os
from collections import Counter

import numpy as np

import showdown as sd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projections", default="projections.csv")
    ap.add_argument("--ownership", default="ownership.csv")
    ap.add_argument("--schedule", default="schedule.csv")
    ap.add_argument("--names", default="player_names.csv")
    ap.add_argument("--n", type=int, default=10000, help="field lineups to build")
    ap.add_argument("--n-sims", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--n-med", type=int, default=6000,
                    help="baseline (medium) contest size for the chalk temperature")
    ap.add_argument("--chalk-sensitivity", type=float, default=0.30)
    ap.add_argument("--sharp-frac", type=float, default=0.35,
                    help="share of the field drawn from the smart candidate "
                         "builder (correlated/ceiling) vs pure ownership")
    ap.add_argument("--overbuild", type=float, default=1.5,
                    help="build this x --n and keep the top by projection")
    ap.add_argument("--ceiling-tilt", type=float, default=0.75,
                    help="how strongly the Captain-slot share skews to ceiling")
    ap.add_argument("--out", default="out/showdown_field.csv")
    ap.add_argument("--show", type=int, default=15, help="preview rows to print")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # ---- ingest + sim ----
    slate, sim, dk_mean = sd.scored_pool(
        projections=a.projections, ownership=a.ownership, schedule=a.schedule,
        names=a.names, n_sims=a.n_sims, seed=a.seed)
    teams = slate.teams
    print(f"slate: single game {' vs '.join(teams) if teams else '?'} · "
          f"{len(slate.entities)} players ({len(slate.players)} offense/K, "
          f"{len(slate.dst)} DST)")
    if len(teams) > 2:
        print(f"  ! {len(teams)} teams in this pool — Showdown expects ONE game; "
              f"point --ownership at a single-game file.")

    # ---- ownership split (the ownership-aware core) ----
    split = sd.split_ownership(slate.entities, ceiling_tilt=a.ceiling_tilt)
    meta = {e["key"]: e for e in slate.entities}
    cmass = sum(c for c, _ in split.values())
    fmass = sum(f for _, f in split.values())
    print(f"\nownership split  (Captain mass {cmass:.2f} of 1.0, "
          f"FLEX mass {fmass:.2f} of 5.0):")
    top = sorted(slate.entities, key=lambda e: -(split[e["key"]][0]
                                                  + split[e["key"]][1]))[:a.show]
    print(f"  {'player':26} {'pos':>3} {'own%':>6} {'CPT%':>6} {'FLEX%':>6}")
    for e in top:
        c, f = split[e["key"]]
        print(f"  {meta[e['key']].get('name', e['key'])[:26]:26} "
              f"{e['pos']:>3} {100*(c+f)/6:>5.1f}% {100*c:>5.1f}% {100*f/5:>5.1f}%")

    # ---- build the field ----
    field = sd.build_field(
        slate.entities, a.n, dk_mean, n_med=a.n_med,
        chalk_sensitivity=a.chalk_sensitivity, sharp_frac=a.sharp_frac,
        overbuild=a.overbuild, ceiling_tilt=a.ceiling_tilt, seed=a.seed)

    df = sd.named_lineups_df(field, slate, dk_mean)
    df.to_csv(a.out, index=False)

    _summary(field, meta, len(slate.entities))
    print(f"\npreview (top {a.show} by projection):")
    prev = df.sort_values("Proj", ascending=False).head(a.show)
    print(prev.to_string(index=False))
    print(f"\nwrote {len(df)} field lineups -> {a.out}")


def _summary(lineups, meta, pool_n):
    caps = Counter(meta.get(lu["captain_key"], {}).get("name", lu["captain_key"])
                   for lu in lineups)
    cap_pos = Counter(lu["captain"]["pos"] for lu in lineups)
    splits = Counter(lu["split"] for lu in lineups)
    sal = [lu["salary"] for lu in lineups]
    print(f"\nfield of {len(lineups)}:")
    print(f"  salary  min {min(sal):,}  mean {int(np.mean(sal)):,}  max {max(sal):,}")
    print(f"  team splits: " + "  ".join(f"{k} {100*v/len(lineups):.0f}%"
                                         for k, v in splits.most_common()))
    print(f"  captain position: " + "  ".join(f"{k} {100*v/len(lineups):.0f}%"
                                              for k, v in cap_pos.most_common()))
    print(f"  top captains: " + "  ".join(f"{k} {100*v/len(lineups):.0f}%"
                                          for k, v in caps.most_common(5)))


if __name__ == "__main__":
    main()
