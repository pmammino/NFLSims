#!/usr/bin/env python3
"""
nfl_ingest.py
=============
Build a simulate-ready **Slate** from the four raw NFL files:

  projections.csv  - per-PlayerID per-Split (C/F/M) stat-by-stat projections
  ownership.csv    - the DK playable pool (Salary, Position, Ownership, ids)
  schedule.csv     - optional Team,Opp[,Total,Implied] game pairings + Vegas
  dst_teams.csv    - optional crosswalk mapping pool DST ids -> Team

Key facts about the data (see README_nfl_sim.md):
  * join key is  ownership.RotoPlayerID == projections.PlayerID
  * projections are INDIVIDUAL players (incl. IDP defenders); there are no
    team-DST rows, so a team defense is aggregated from its defenders here
  * the contest-standings files are handled elsewhere (aggregate learning only)

The Slate exposes, for every playable entity, the DK fantasy-point triple
(floor p25 / median p50 / ceiling p75) plus - for matched offensive players -
the underlying per-stat triples, so the sim can work at the stat level and fire
yardage bonuses correctly.
"""
import csv
import os
from collections import defaultdict

import numpy as np

import dk_scoring as dk

SPLITS = ("F", "M", "C")           # floor(p25), median(p50), ceiling(p75)

# projections.csv column -> offensive scoring stat key
OFF_STAT_COLS = {
    "pass_yards": ["PassYards"],
    "pass_tds": ["PassTDs"],
    "pass_ints": ["PassInts"],
    "rush_yards": ["RushYards"],
    "rush_tds": ["RushTDs"],
    "receptions": ["RecCompletions"],
    "rec_yards": ["RecYards"],
    "rec_tds": ["RecTDs"],
    "fumbles_lost": ["offFumblesLost"],
    "two_pts": ["offPass2Pt", "offRush2Pt", "offRec2Pt"],
    "return_tds": ["specKORetTD", "specPuntRetTD"],
}

# aggregated team-DST counting stats -> projections columns (summed over defenders)
DST_STAT_COLS = {
    "sacks": ["defSacks"],
    "ints": ["defPassInt"],
    "fumble_rec": ["defFumblesRec"],
    "safeties": ["defSafeties"],
    "blocks": ["defBlockedKick"],
    "def_tds": ["defIntReturnTDs", "defFumbReturnTDs"],
    "st_ret_tds": ["specKORetTD", "specPuntRetTD", "specBlockedFGTD",
                   "specBlockedPuntTD"],
}

POS_OFFENSE = {"QB", "RB", "WR", "TE"}


def _f(v):
    """Parse a projections cell to float; NULL/blank -> 0.0."""
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == "" or s.upper() == "NULL":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _load_projections(path):
    """proj[pid][split] = raw row dict; also proj rows grouped by team."""
    proj = defaultdict(dict)
    by_team = defaultdict(lambda: defaultdict(list))   # team -> split -> [rows]
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            pid = r["PlayerID"].strip()
            sp = r["Split"].strip()
            proj[pid][sp] = r
            by_team[r["Team"].strip()][sp].append(r)
    return proj, by_team


def _load_ownership(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _load_names(path):
    """Optional crosswalk: player id -> 'First Last'. Tolerant of latin-1."""
    names = {}
    if not path or not os.path.exists(path):
        return names
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                for r in csv.DictReader(fh):
                    pid = (r.get("ID") or "").strip()
                    nm = f"{(r.get('firstname') or '').strip()} " \
                         f"{(r.get('lastname') or '').strip()}".strip()
                    if pid and nm:
                        names[pid] = nm
            break
        except UnicodeDecodeError:
            names.clear()
            continue
    return names


def _load_schedule(path):
    """team -> {opp, total, implied}. total/implied optional."""
    sched = {}
    if not path or not os.path.exists(path):
        return sched
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = r["Team"].strip()
            sched[t] = {
                "opp": (r.get("Opp") or "").strip(),
                "total": _f(r.get("Total")) or None,
                "implied": _f(r.get("Implied")) or None,
            }
    return sched


def _stat_triple(rowsplits, cols):
    """(F, M, C) sums of `cols` for one player; missing split -> 0."""
    out = []
    for sp in SPLITS:
        row = rowsplits.get(sp)
        out.append(sum(_f(row[c]) for c in cols) if row else 0.0)
    return tuple(out)


def _team_stat_triple(team_rowsplits, cols):
    """(F, M, C) sums of `cols` across ALL of a team's rows, per split."""
    out = []
    for sp in SPLITS:
        rows = team_rowsplits.get(sp, [])
        out.append(sum(_f(row[c]) for row in rows for c in cols))
    return tuple(out)


def _offense_fp(splits):
    """DK fantasy points at (F, M, C) from a matched player's stat splits."""
    fp = []
    for i, sp in enumerate(SPLITS):
        stat = {k: v[i] for k, v in splits.items()}
        fp.append(float(dk.score_offense(stat)))
    return tuple(fp)


def _replacement_fp(salary):
    """A near-zero, salary-scaled marginal for pool players with no projection.

    They are low-salary/low-ownership and rarely rostered, but must not vanish
    from the field pool. Median ~ salary/1000 pts, right-skewed."""
    m = max(0.5, (salary - 2000) / 1000.0 * 1.2)
    return (round(0.3 * m, 3), round(m, 3), round(1.9 * m, 3))


class Slate:
    """Container for the simulate-ready slate.

    Attributes
    ----------
    players : list of offensive-entity dicts, each with keys
        key, rid, pid, contest_id, pos, team, opp, game, salary, own,
        matched (bool), stats {stat:(F,M,C)} (only if matched), fp (F,M,C)
    dst : list of team-defense dicts, each with
        key, team, opp, game, salary, own, contest_id,
        count_stats {stat:(F,M,C)}, count_fp (F,M,C)
    schedule : {team: {opp,total,implied}}
    teams : sorted list of offensive teams on the slate
    games : {game_id: (teamA, teamB)}
    """

    def __init__(self, players, dst, schedule, teams, games):
        self.players = players
        self.dst = dst
        self.schedule = schedule
        self.teams = teams
        self.games = games

    @property
    def entities(self):
        """All simulatable entities (offense + DST) in a single list."""
        return self.players + self.dst


def _game_id(a, b):
    return "@".join(sorted([a, b]))


def build_slate(projections="projections.csv", ownership="ownership.csv",
                schedule="schedule.csv", dst_teams="dst_teams.csv",
                names="player_names.csv"):
    proj, by_team = _load_projections(projections)
    pool = _load_ownership(ownership)
    sched = _load_schedule(schedule)
    name_map = _load_names(names)

    # ---- offensive players ----
    players = []
    slate_teams = set()
    for r in pool:
        pos = r["Position"].strip()
        if pos not in POS_OFFENSE:
            continue
        rid = r["RotoPlayerID"].strip()
        salary = int(_f(r["Salary"]))
        own = _f(r["Ownership"])
        splits = proj.get(rid)
        matched = splits is not None
        if matched:
            stats = {k: _stat_triple(splits, cols) for k, cols in OFF_STAT_COLS.items()}
            team = splits.get("M", splits.get("C", next(iter(splits.values()))))["Team"].strip()
            fp = _offense_fp(stats)
        else:
            stats = None
            team = ""            # unknown team for unmatched scrubs
            fp = _replacement_fp(salary)
        rec = {
            "key": f"O{rid}", "rid": rid, "pid": r["PlayerID"].strip(),
            "contest_id": r["PlayerContestID"].strip(),
            "name": name_map.get(rid, f"#{rid}"),
            "pos": pos, "team": team, "salary": salary, "own": own,
            "matched": matched, "stats": stats, "fp": fp,
        }
        players.append(rec)
        if team:
            slate_teams.add(team)

    teams = sorted(slate_teams)

    # ---- schedule / games ----
    # opp + game id per team (from schedule.csv when present)
    def opp_of(t):
        return sched.get(t, {}).get("opp", "")

    games = {}
    for t in teams:
        o = opp_of(t)
        if o:
            games[_game_id(t, o)] = tuple(sorted([t, o]))
    for p in players:
        p["opp"] = opp_of(p["team"]) if p["team"] else ""
        p["game"] = _game_id(p["team"], p["opp"]) if p["opp"] else (p["team"] or "")

    # ---- team DST projections (aggregate defenders) ----
    dst_pool = [r for r in pool if r["Position"].strip() == "DST"]
    dst_map = _load_dst_map(dst_teams)
    dst_team_proj = {}     # team -> {stat:(F,M,C)}, count_fp
    for t in teams:
        rowsplits = by_team.get(t)
        if not rowsplits:
            continue
        cs = {k: _team_stat_triple(rowsplits, cols) for k, cols in DST_STAT_COLS.items()}
        cfp = []
        for i in range(3):
            stat = {k: v[i] for k, v in cs.items()}
            cfp.append(float(dk.score_dst(stat)))     # counting component only
        dst_team_proj[t] = {"count_stats": cs, "count_fp": tuple(cfp)}

    dst = _assign_dst(dst_pool, teams, dst_team_proj, dst_map, sched, opp_of)

    return Slate(players, dst, sched, teams, games)


def _load_dst_map(path):
    """Optional crosswalk: RotoPlayerID or PlayerContestID -> Team."""
    m = {}
    if not path or not os.path.exists(path):
        return m
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            team = (r.get("Team") or "").strip()
            for k in ("RotoPlayerID", "PlayerID", "PlayerContestID"):
                v = (r.get(k) or "").strip()
                if v:
                    m[v] = team
    return m


def _assign_dst(dst_pool, teams, dst_team_proj, dst_map, sched, opp_of):
    """Map the pool's DST rows to teams and attach aggregated projections.

    Uses `dst_teams.csv` when it identifies a row; otherwise falls back to the
    documented salary-rank heuristic: DK DST salary tracks Vegas expectation, so
    rank pool DSTs by salary and teams by projected DST counting points and zip.
    """
    teams_with_proj = [t for t in teams if t in dst_team_proj]
    # split pool rows into explicitly-mapped and to-be-inferred
    mapped, unmapped = {}, []
    for r in dst_pool:
        team = (dst_map.get(r["RotoPlayerID"].strip())
                or dst_map.get(r["PlayerContestID"].strip())
                or dst_map.get(r["PlayerID"].strip()))
        if team:
            mapped[team] = r
        else:
            unmapped.append(r)

    used = set(mapped)
    free_teams = [t for t in teams_with_proj if t not in used]
    # heuristic pairing on the leftovers
    free_teams.sort(key=lambda t: dst_team_proj[t]["count_fp"][1], reverse=True)
    unmapped.sort(key=lambda r: int(_f(r["Salary"])), reverse=True)

    out = []
    for team, r in mapped.items():
        out.append(_dst_rec(r, team, dst_team_proj, sched, opp_of))
    for r, team in zip(unmapped, free_teams):
        out.append(_dst_rec(r, team, dst_team_proj, sched, opp_of))
    return out


def _dst_rec(r, team, dst_team_proj, sched, opp_of):
    tp = dst_team_proj[team]
    opp = opp_of(team)
    return {
        "key": f"DST_{team}", "team": team, "name": f"{team} DST",
        "contest_id": r["PlayerContestID"].strip(),
        "rid": r["RotoPlayerID"].strip(), "pid": r["PlayerID"].strip(),
        "pos": "DST", "salary": int(_f(r["Salary"])), "own": _f(r["Ownership"]),
        "opp": opp, "game": _game_id(team, opp) if opp else team,
        "count_stats": tp["count_stats"], "count_fp": tp["count_fp"],
        # opponent implied total for the points-allowed model (neutral default)
        "opp_total": (sched.get(opp, {}).get("implied")
                      or sched.get(opp, {}).get("total") or 22.0),
    }


if __name__ == "__main__":
    s = build_slate()
    matched = [p for p in s.players if p["matched"]]
    print(f"offense: {len(s.players)} ({len(matched)} matched to projections)")
    print(f"DST: {len(s.dst)}  teams: {len(s.teams)}  games: {len(s.games)}")
    # show a marquee QB and its fp triple
    qbs = sorted([p for p in matched if p["pos"] == "QB"],
                 key=lambda p: p["fp"][1], reverse=True)[:3]
    for q in qbs:
        print(f"  QB {q['rid']:>6} {q['team']:>3} vs {q['opp']:<3} "
              f"sal {q['salary']} own {q['own']:.1f}%  F/M/C fp = "
              f"{q['fp'][0]:.1f}/{q['fp'][1]:.1f}/{q['fp'][2]:.1f}")
    ds = sorted(s.dst, key=lambda d: d["count_fp"][1], reverse=True)[:3]
    for d in ds:
        print(f"  DST {d['team']:>3} vs {d['opp']:<3} sal {d['salary']} "
              f"count F/M/C = {d['count_fp'][0]:.1f}/{d['count_fp'][1]:.1f}/"
              f"{d['count_fp'][2]:.1f}  opp_total {d['opp_total']}")
