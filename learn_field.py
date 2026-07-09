#!/usr/bin/env python3
"""
learn_field.py
==============
Derive ``field_params_nfl.json`` (the field-construction grammar used by
``field_builder``) from real DK NFL contest standings — the NFL analog of the
MLB engine's field-params derivation, run in **aggregate-learning** mode.

What the standings CAN and CANNOT teach (they are keyed by player NAME only,
from other slates, with no team column):

  * LEARNABLE
      - FLEX position mix (RB/WR/TE). A player's natural position is inferred
        from the explicit slots he fills in other entries, then applied to his
        FLEX appearances. This is a genuine, data-derived distribution.
      - realized-ownership summary (chalk level / concentration) and lineup
        duplication — reported for calibration.
  * NOT LEARNABLE from these files (need player->team, which isn't present)
      - QB primary-stack sizes and bring-back rates. These use documented NFL
        GPP priors (``field_builder.DEFAULT_PARAMS``). Supply a name->team
        crosswalk to re-derive them later.

The output merges the priors with the learned FLEX mix and records a ``learned``
provenance block so it is always clear what came from data vs. priors.
"""
import glob
import json
from collections import Counter, defaultdict

import contest_ingest as ci
from field_builder import DEFAULT_PARAMS

MIN_FLEX_OBS = 50          # need this many FLEX appearances to trust the mix
SKILL_SLOTS = ("RB", "WR", "TE")


def natural_positions(contests):
    """name -> majority natural position, from non-FLEX slots across all entries."""
    votes = defaultdict(Counter)
    for cd in contests:
        for e in cd.entries:
            for slot, player in e.lineup:
                if slot in SKILL_SLOTS or slot == "QB" or slot == "DST":
                    votes[player][slot] += 1
    return {nm: c.most_common(1)[0][0] for nm, c in votes.items()}


def flex_distribution(contests, natpos):
    """Learned FLEX position mix (RB/WR/TE) from FLEX appearances."""
    flex = Counter()
    for cd in contests:
        for e in cd.entries:
            for slot, player in e.lineup:
                if slot == "FLEX":
                    p = natpos.get(player)
                    if p in SKILL_SLOTS:
                        flex[p] += 1
    return flex


def ownership_summary(contests):
    """Chalk-level / concentration stats from realized %Drafted."""
    out = {}
    for cd in contests:
        pcts = sorted((p.pct_drafted for p in cd.players), reverse=True)
        if not pcts:
            continue
        n = len(pcts)
        top = pcts[:max(1, n // 10)]
        out[cd.path] = {
            "n_players": n,
            "n_entries": len(cd.entries),
            "max_owned": round(pcts[0], 2),
            "mean_owned": round(sum(pcts) / n, 2),
            "top10pct_mean_owned": round(sum(top) / len(top), 2),
            "n_over_25pct": sum(1 for x in pcts if x >= 25.0),
        }
    return out


def duplication_summary(contests):
    """How many entries share an identical lineup (a large-field GPP signal)."""
    out = {}
    for cd in contests:
        keys = Counter(tuple(sorted(p for _, p in e.lineup)) for e in cd.entries)
        dupes = sum(c for c in keys.values() if c > 1)
        out[cd.path] = {
            "n_entries": len(cd.entries),
            "distinct_lineups": len(keys),
            "pct_duplicated": round(100 * dupes / max(1, len(cd.entries)), 1),
            "max_dupes_one_lineup": max(keys.values()) if keys else 0,
        }
    return out


def learn(paths, out_path="field_params_nfl.json"):
    contests = [ci.parse_contest_csv(p) for p in paths]
    natpos = natural_positions(contests)
    flex = flex_distribution(contests, natpos)
    total_flex = sum(flex.values())

    params = json.loads(json.dumps(DEFAULT_PARAMS))   # deep copy of priors
    learned_flex = None
    if total_flex >= MIN_FLEX_OBS:
        learned_flex = [[p, round(flex.get(p, 0) / total_flex, 4)]
                        for p in SKILL_SLOTS]
        params["flex_pos"] = learned_flex

    params["learned"] = {
        "source_files": [c.path for c in contests],
        "n_contests": len(contests),
        "n_entries_total": sum(len(c.entries) for c in contests),
        "flex_observations": total_flex,
        "flex_pos_learned": learned_flex,
        "flex_pos_source": "standings" if learned_flex else "prior",
        "stack_source": "prior (name-only standings lack team; supply a "
                        "name->team crosswalk to learn stacks)",
        "ownership": ownership_summary(contests),
        "duplication": duplication_summary(contests),
    }

    with open(out_path, "w") as fh:
        json.dump(params, fh, indent=2)
    return params


if __name__ == "__main__":
    paths = sorted(glob.glob("contest-standings-*.csv"))
    p = learn(paths)
    print(f"wrote field_params_nfl.json from {len(paths)} contest file(s)")
    print("  stack_sizes (prior):", p["stack_sizes"])
    print("  flex_pos:", p["flex_pos"], f"(source: {p['learned']['flex_pos_source']})")
    print("  flex observations:", p["learned"]["flex_observations"])
    for path, o in p["learned"]["ownership"].items():
        print(f"  {path}: max {o['max_owned']}%  mean {o['mean_owned']}%  "
              f">=25%: {o['n_over_25pct']} players")
    for path, d in p["learned"]["duplication"].items():
        print(f"  {path}: {d['distinct_lineups']} distinct / {d['n_entries']} "
              f"({d['pct_duplicated']}% dupes, max {d['max_dupes_one_lineup']})")
