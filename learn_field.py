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
      - QB primary-stack sizes, when a name->team crosswalk is available. We
        build one by joining ``player_names.csv`` (name<->id) to
        ``projections.csv`` (id->team); see ``build_name_team_crosswalk``. It
        only resolves players whose team is in the projection dump (a single
        week), so most lineups are only partially resolved. We therefore learn
        the QB-stack-size distribution from FULLY-RESOLVED lineups only (QB and
        all 7 skill players mapped), which is unbiased per lineup, and report
        the sample size so the confidence is explicit.
      - realized-ownership summary (chalk level / concentration) and lineup
        duplication — reported for calibration.
  * NOT LEARNABLE from these files (would need each standings slate's schedule)
      - bring-back rates and the DST-vs-own-stack rate. Distinguishing an
        opponent bring-back from an unrelated secondary/game stack needs the
        team->opponent pairing for that specific slate, which we don't have.
        These keep the documented NFL GPP priors; the learned SECONDARY-cluster
        distribution is recorded as a diagnostic that corroborates them.

The output merges the priors with whatever was learned and records a ``learned``
provenance block so it is always clear what came from data vs. priors.
"""
import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import contest_ingest as ci
from field_builder import DEFAULT_PARAMS

MIN_FLEX_OBS = 50          # need this many FLEX appearances to trust the mix
MIN_STACK_OBS = 200        # need this many fully-resolved lineups to trust stacks
SKILL_SLOTS = ("RB", "WR", "TE")


# --------------------------------------------------------------------------- #
# name -> team crosswalk (the piece the standings themselves don't carry)
# --------------------------------------------------------------------------- #
def _norm_name(nm):
    """Normalize a player name for matching across sources: strip accents, case,
    generational suffixes (Jr/Sr/II..), punctuation and hyphens."""
    s = unicodedata.normalize("NFKD", str(nm)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def build_name_team_crosswalk(projections="projections.csv",
                              names="player_names.csv"):
    """normalized-name -> team, by joining player_names (name<->id) to the
    projections dump (id->team). Only ids present in projections resolve, and
    only names that map to a single team are kept. Returns {} if files absent."""
    if not (projections and os.path.exists(projections) and
            names and os.path.exists(names)):
        return {}
    id2name = {}
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(names, newline="", encoding=enc) as fh:
                for r in csv.DictReader(fh):
                    pid = (r.get("ID") or "").strip()
                    nm = f"{(r.get('firstname') or '').strip()} " \
                         f"{(r.get('lastname') or '').strip()}".strip()
                    if pid and nm:
                        id2name[pid] = nm
            break
        except UnicodeDecodeError:
            id2name.clear()
            continue
    id2team = defaultdict(set)
    with open(projections, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            id2team[r["PlayerID"].strip()].add(r["Team"].strip())
    name2team = {}
    for pid, teams in id2team.items():
        nm = id2name.get(pid)
        if nm and len(teams) == 1:
            name2team[_norm_name(nm)] = next(iter(teams))
    return name2team


def stack_learning(contests, name2team):
    """QB primary-stack-size distribution from FULLY-RESOLVED lineups (QB + all
    7 skill players mapped to a team), plus a secondary-cluster diagnostic.

    Returns a dict with the learned ``stack_sizes`` (k=0..4, or None if too few
    resolved lineups), ``secondary`` cluster distribution, and coverage stats.
    Fully-resolved lineups are unbiased per lineup, so the distribution does not
    suffer the downward bias that partial resolution (an unresolved same-team
    mate counted as off-team) would introduce."""
    if not name2team:
        return {"stack_sizes": None, "n_resolved": 0, "n_entries": 0}
    prim = Counter()
    sec = Counter()
    n_entries = n_resolved = 0
    for cd in contests:
        for e in cd.entries:
            n_entries += 1
            qb = [pl for slot, pl in e.lineup if slot == "QB"]
            skill = [pl for slot, pl in e.lineup if slot not in ("QB", "DST")]
            if not qb or len(skill) != 7:
                continue
            qbt = name2team.get(_norm_name(qb[0]))
            steams = [name2team.get(_norm_name(pl)) for pl in skill]
            if not qbt or any(t is None for t in steams):
                continue                              # fully-resolved only
            n_resolved += 1
            prim[sum(1 for t in steams if t == qbt)] += 1
            c = Counter(t for t in steams if t != qbt)
            sec[max(c.values()) if c else 0] += 1
    learned = None
    if n_resolved >= MIN_STACK_OBS:
        w = {k: prim.get(k, 0) for k in range(5)}
        w[4] += sum(prim.get(k, 0) for k in prim if k > 4)   # fold k>4 into 4
        tot = sum(w.values())
        learned = [[k, round(w[k] / tot, 4)] for k in range(5)]
    stot = sum(sec.values()) or 1
    return {
        "stack_sizes": learned,
        "n_resolved": n_resolved,
        "n_entries": n_entries,
        "primary_counts": {int(k): int(v) for k, v in sorted(prim.items())},
        "secondary_dist": {int(k): round(v / stot, 4) for k, v in sorted(sec.items())},
        "crosswalk_size": len(name2team),
    }


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


def learn(paths, out_path="field_params_nfl.json",
          projections="projections.csv", names="player_names.csv"):
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

    # ---- QB primary-stack sizes: learn from a name->team crosswalk if we can
    name2team = build_name_team_crosswalk(projections, names)
    stacks = stack_learning(contests, name2team)
    if stacks["stack_sizes"]:
        params["stack_sizes"] = stacks["stack_sizes"]
        stack_source = (f"standings ({stacks['n_resolved']} fully-resolved "
                        f"lineups via projections+player_names crosswalk)")
    else:
        stack_source = ("prior (crosswalk resolved "
                        f"{stacks.get('n_resolved', 0)} lineups < {MIN_STACK_OBS} "
                        "needed; bring more overlapping projections to learn)")

    params["learned"] = {
        "source_files": [c.path for c in contests],
        "n_contests": len(contests),
        "n_entries_total": sum(len(c.entries) for c in contests),
        "flex_observations": total_flex,
        "flex_pos_learned": learned_flex,
        "flex_pos_source": "standings" if learned_flex else "prior",
        "stack_source": stack_source,
        "stack_sizes_learned": stacks["stack_sizes"],
        "stack_resolution": {
            "crosswalk_size": stacks.get("crosswalk_size", 0),
            "n_entries": stacks.get("n_entries", 0),
            "n_fully_resolved": stacks.get("n_resolved", 0),
            "primary_counts": stacks.get("primary_counts", {}),
        },
        "secondary_cluster_dist": stacks.get("secondary_dist", {}),
        "bringback_source": "prior (needs each slate's schedule to label the "
                            "QB's opponent; secondary_cluster_dist corroborates)",
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
    print("  stack_sizes:", p["stack_sizes"])
    print("    source:", p["learned"]["stack_source"])
    sr = p["learned"]["stack_resolution"]
    print(f"    crosswalk {sr['crosswalk_size']} names, "
          f"{sr['n_fully_resolved']}/{sr['n_entries']} lineups fully resolved")
    print("  secondary-cluster dist (diagnostic):", p["learned"]["secondary_cluster_dist"])
    print("  flex_pos:", p["flex_pos"], f"(source: {p['learned']['flex_pos_source']})")
    print("  flex observations:", p["learned"]["flex_observations"])
    for path, o in p["learned"]["ownership"].items():
        print(f"  {path}: max {o['max_owned']}%  mean {o['mean_owned']}%  "
              f">=25%: {o['n_over_25pct']} players")
    for path, d in p["learned"]["duplication"].items():
        print(f"  {path}: {d['distinct_lineups']} distinct / {d['n_entries']} "
              f"({d['pct_duplicated']}% dupes, max {d['max_dupes_one_lineup']})")
