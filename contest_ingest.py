#!/usr/bin/env python3
"""
contest_ingest.py
=================
Parse DraftKings **NFL** contest-standings CSVs (the dual-column export) — the
NFL analog of ``DFSSimsFull/contest_review.parse_contest_csv``.

The file interleaves two independent blocks per row:
  * left : Rank, EntryId, EntryName, TimeRemaining, Points, Lineup
  * right: Player, Roster Position, %Drafted, FPTS   (the realized-ownership /
           actual-score table; its rows are NOT aligned with the entries)

A lineup string looks like
    "DST Browns  FLEX Trey McBride QB Jameis Winston RB Jahmyr Gibbs ..."
tokenized here into (slot, player) pairs.
"""
import csv
import re
from collections import namedtuple

SLOT_RE = re.compile(
    r"(QB|RB|WR|TE|DST|FLEX)\s+(.+?)(?=\s+(?:QB|RB|WR|TE|DST|FLEX)\s+|$)")

Entry = namedtuple("Entry", "rank entry_id name points lineup")
PlayerActual = namedtuple("PlayerActual", "player roster_pos pct_drafted fpts")


def parse_lineup(s):
    """'DST Browns FLEX Trey McBride QB ...' -> [(slot, player), ...]."""
    out = []
    for m in SLOT_RE.finditer(str(s).strip()):
        out.append((m.group(1), m.group(2).strip()))
    return out


def _pct(v):
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return 0.0


def _f(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


class ContestData:
    def __init__(self, entries, players, path):
        self.entries = entries          # list[Entry]
        self.players = players          # list[PlayerActual]
        self.path = path

    def __repr__(self):
        return (f"ContestData({self.path!r}: {len(self.entries)} entries, "
                f"{len(self.players)} players)")


def parse_contest_csv(path):
    entries, players = [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # locate columns by name (the right block starts at the empty separator)
        idx = {name.strip(): i for i, name in enumerate(header)}
        c_rank = idx.get("Rank", 0)
        c_entry = idx.get("EntryId", 1)
        c_name = idx.get("EntryName", 2)
        c_points = idx.get("Points", 4)
        c_lineup = idx.get("Lineup", 5)
        c_player = idx.get("Player")
        c_rpos = idx.get("Roster Position")
        c_draft = idx.get("%Drafted")
        c_fpts = idx.get("FPTS")
        for row in reader:
            if len(row) <= c_lineup:
                continue
            eid = row[c_entry].strip() if len(row) > c_entry else ""
            if eid.isdigit():
                entries.append(Entry(
                    rank=int(_f(row[c_rank])), entry_id=eid,
                    name=row[c_name].strip(), points=_f(row[c_points]),
                    lineup=parse_lineup(row[c_lineup])))
            if c_player is not None and len(row) > c_player and row[c_player].strip():
                players.append(PlayerActual(
                    player=row[c_player].strip(),
                    roster_pos=row[c_rpos].strip() if c_rpos is not None else "",
                    pct_drafted=_pct(row[c_draft]) if c_draft is not None else 0.0,
                    fpts=_f(row[c_fpts]) if c_fpts is not None else 0.0))
    return ContestData(entries, players, path)


if __name__ == "__main__":
    import glob
    for p in sorted(glob.glob("contest-standings-*.csv")):
        cd = parse_contest_csv(p)
        print(cd)
        e = cd.entries[0]
        print(f"  top entry: rank {e.rank} {e.points} pts  slots={len(e.lineup)}")
        print(f"    {e.lineup}")
        top = sorted(cd.players, key=lambda x: x.pct_drafted, reverse=True)[:3]
        for pa in top:
            print(f"  chalk: {pa.player:24} {pa.roster_pos:5} "
                  f"{pa.pct_drafted:5.1f}%  {pa.fpts} fpts")
