#!/usr/bin/env python3
"""
gen_showdown_candidates.py
==========================
Generate the **ownership-blind** DraftKings NFL **Showdown** candidate lineups —
the lineups you would actually enter — and show the smart single-game
construction rules at work, in isolation from the rest of the pipeline.

Flow
----
  ingest single-game slate (projections + ownership + schedule)
    -> correlated sim  (sim_engine)  -> ceiling proxy (sim p90) per player
    -> build candidates by RULES, not ownership  (showdown.build_candidates):
         * pick a team to build around
         * Captain a ceiling player from it
         * keep the QB with >=1 pass-catcher (the intra-team passing stack)
         * force a bring-back from the other team (team split sampled from
           3-3 / 4-2 / 2-4 / rare 5-1 -- never one-sided 6-0)
         * guard against rostering a DST behind a big opposing stack
         * weight every remaining pick by simulated ceiling

Ownership plays no part here: the sim already correlates a single game, so the
job is to land on the structurally live builds. Example:

  python3 gen_showdown_candidates.py --ownership ownership.csv --n 10000 \
          --n-sims 10000 --out out/showdown_candidates.csv
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
    ap.add_argument("--n", type=int, default=10000, help="candidate lineups to build")
    ap.add_argument("--n-sims", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="lognormal shock on ceiling weights to diversify")
    ap.add_argument("--qb-stack-prob", type=float, default=0.85,
                    help="chance the focus QB is stacked with a pass-catcher")
    ap.add_argument("--opp-qb-prob", type=float, default=0.45,
                    help="chance the bring-back includes the opponent QB")
    ap.add_argument("--captain-skill-prob", type=float, default=0.90,
                    help="chance the Captain is a skill player (QB/RB/WR/TE)")
    ap.add_argument("--out", default="out/showdown_candidates.csv")
    ap.add_argument("--show", type=int, default=15, help="preview rows to print")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # ---- ingest + sim (ceiling proxy attached as `up`) ----
    slate, sim, dk_mean = sd.scored_pool(
        projections=a.projections, ownership=a.ownership, schedule=a.schedule,
        names=a.names, n_sims=a.n_sims, seed=a.seed)
    teams = slate.teams
    meta = {e["key"]: e for e in slate.entities}
    print(f"slate: single game {' vs '.join(teams) if teams else '?'} · "
          f"{len(slate.entities)} players ({len(slate.players)} offense/K, "
          f"{len(slate.dst)} DST)")
    if len(teams) > 2:
        print(f"  ! {len(teams)} teams in this pool — Showdown expects ONE game; "
              f"point --ownership at a single-game file.")

    # ---- build candidates (rules, ownership-blind) ----
    pool = sd.Pool(slate.entities)
    builder = sd.CandidateBuilder(
        pool, seed=a.seed, jitter=a.jitter, qb_stack_prob=a.qb_stack_prob,
        opp_qb_prob=a.opp_qb_prob, captain_skill_prob=a.captain_skill_prob)
    cands = sd._draw(builder, a.n)

    df = sd.named_lineups_df(cands, slate, dk_mean)
    df.to_csv(a.out, index=False)

    _summary(cands, meta)
    print(f"\npreview (top {a.show} by projection):")
    prev = df.sort_values("Proj", ascending=False).head(a.show)
    print(prev.to_string(index=False))
    print(f"\nwrote {len(df)} candidate lineups -> {a.out}")


def _qb_stack(lu):
    """A QB rides with >=1 same-team pass-catcher."""
    for p in lu["players"]:
        if p["pos"] == "QB" and any(
                q["team"] == p["team"] and q["pos"] in sd.PASS_CATCHER
                for q in lu["players"] if q["key"] != p["key"]):
            return True
    return False


def _summary(lineups, meta):
    if not lineups:
        print("no candidates built"); return
    caps = Counter(meta.get(lu["captain_key"], {}).get("name", lu["captain_key"])
                   for lu in lineups)
    cap_pos = Counter(lu["captain"]["pos"] for lu in lineups)
    splits = Counter(lu["split"] for lu in lineups)
    sal = [lu["salary"] for lu in lineups]
    n = len(lineups)
    both = sum(1 for lu in lineups
               if len(set(p["team"] for p in lu["players"])) >= 2)
    stack = sum(_qb_stack(lu) for lu in lineups)
    print(f"\ncandidates ({n}):")
    print(f"  salary  min {min(sal):,}  mean {int(np.mean(sal)):,}  max {max(sal):,}")
    print(f"  QB passing stack: {100*stack/n:.0f}%   both teams (bring-back): {100*both/n:.0f}%")
    print(f"  team splits: " + "  ".join(f"{k} {100*v/n:.0f}%"
                                         for k, v in splits.most_common()))
    print(f"  captain position: " + "  ".join(f"{k} {100*v/n:.0f}%"
                                              for k, v in cap_pos.most_common()))
    print(f"  top captains: " + "  ".join(f"{k} {100*v/n:.0f}%"
                                          for k, v in caps.most_common(5)))


if __name__ == "__main__":
    main()
