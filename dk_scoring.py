#!/usr/bin/env python3
"""
dk_scoring.py
=============
DraftKings **NFL Classic** fantasy scoring, vectorized over numpy arrays so the
same functions score a single stat line or a full ``(N_SIMS,)`` simulation.

Rules (DK NFL Classic, full PPR)
--------------------------------
Offense
  passing    : 0.04 / yard (i.e. +1 per 25 yds), +4 / TD, -1 / INT, +3 @ 300 yds
  rushing    : 0.10 / yard (+1 per 10 yds),       +6 / TD,          +3 @ 100 yds
  receiving  : +1 / reception (PPR), 0.10 / yard, +6 / TD,          +3 @ 100 yds
  misc       : -1 fumble lost, +2 two-point conversion (pass/run/rec),
               +6 return TD (KO/punt), +6 own-fumble-recovery TD
DST
  +1 sack, +2 INT, +2 fumble recovery, +2 safety, +2 blocked kick,
  +6 any defensive/special-teams TD, and the points-allowed tier:
     0:+10  1-6:+7  7-13:+4  14-20:+1  21-27:0  28-34:-1  35+:-4

Every function accepts scalars or numpy arrays and returns the same shape, so a
per-simulation stat dict of arrays scores in one call.
"""
import numpy as np

# --------------------------------------------------------------------------- #
# Per-unit point values
# --------------------------------------------------------------------------- #
PASS_YD = 0.04
PASS_TD = 4.0
PASS_INT = -1.0
PASS_300_BONUS = 3.0

RUSH_YD = 0.1
RUSH_TD = 6.0
RUSH_100_BONUS = 3.0

REC_PPR = 1.0
REC_YD = 0.1
REC_TD = 6.0
REC_100_BONUS = 3.0

FUMBLE_LOST = -1.0
TWO_POINT = 2.0
RETURN_TD = 6.0

DST_SACK = 1.0
DST_INT = 2.0
DST_FUMREC = 2.0
DST_SAFETY = 2.0
DST_BLOCK = 2.0
DST_TD = 6.0

# points-allowed tiers as (upper_bound_inclusive, points); last is the 35+ floor
PA_TIERS = [(0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0)]
PA_FLOOR = -4.0


def _bonus(yards, threshold, points):
    """`points` when yards >= threshold, else 0 — vectorized."""
    return np.where(np.asarray(yards, dtype=float) >= threshold, points, 0.0)


def score_offense(s):
    """DK points for an offensive player from a stat dict.

    `s` maps stat name -> scalar or numpy array. Recognized keys (all optional,
    default 0): pass_yards, pass_tds, pass_ints, rush_yards, rush_tds,
    receptions, rec_yards, rec_tds, fumbles_lost, two_pts, return_tds.
    """
    g = lambda k: np.asarray(s.get(k, 0.0), dtype=float)
    py, ry, cy = g("pass_yards"), g("rush_yards"), g("rec_yards")
    pts = (
        PASS_YD * py + PASS_TD * g("pass_tds") + PASS_INT * g("pass_ints")
        + _bonus(py, 300, PASS_300_BONUS)
        + RUSH_YD * ry + RUSH_TD * g("rush_tds")
        + _bonus(ry, 100, RUSH_100_BONUS)
        + REC_PPR * g("receptions") + REC_YD * cy + REC_TD * g("rec_tds")
        + _bonus(cy, 100, REC_100_BONUS)
        + FUMBLE_LOST * g("fumbles_lost") + TWO_POINT * g("two_pts")
        + RETURN_TD * g("return_tds")
    )
    return pts


def points_allowed_score(pa):
    """DK DST points for the points-allowed tier of `pa` (scalar or array)."""
    pa = np.asarray(pa, dtype=float)
    out = np.full_like(pa, PA_FLOOR)
    # apply tiers from most-points (fewest allowed) down; later (looser) tiers
    # only fill cells not already assigned a tighter tier
    assigned = np.zeros(pa.shape, dtype=bool)
    for ub, val in PA_TIERS:
        hit = (~assigned) & (pa <= ub)
        out[hit] = val
        assigned |= hit
    return out


def score_dst(s, points_allowed=None):
    """DK points for a team defense from an aggregated stat dict.

    Recognized keys (all optional): sacks, ints, fumble_rec, safeties, blocks,
    def_tds (interception/fumble return TDs), st_ret_tds (kick/punt return TDs).
    `points_allowed` (scalar/array) adds the points-allowed tier; omit to score
    only the counting stats.
    """
    g = lambda k: np.asarray(s.get(k, 0.0), dtype=float)
    pts = (
        DST_SACK * g("sacks") + DST_INT * g("ints") + DST_FUMREC * g("fumble_rec")
        + DST_SAFETY * g("safeties") + DST_BLOCK * g("blocks")
        + DST_TD * (g("def_tds") + g("st_ret_tds"))
    )
    if points_allowed is not None:
        pts = pts + points_allowed_score(points_allowed)
    return pts


# projections.csv column -> offensive stat key (summing the granular TD buckets)
OFF_TD_PASS_COLS = ["offPassTD9", "offPassTD19", "offPassTD29", "offPassTD39",
                    "offPassTD49", "offPassTD50"]
OFF_RETURN_TD_COLS = ["specKORetTD", "specPuntRetTD"]


if __name__ == "__main__":
    # A 300-yd, 3-TD, 1-INT passing line: 12 + 12 - 1 + 3 = 26
    qb = score_offense({"pass_yards": 300, "pass_tds": 3, "pass_ints": 1})
    assert abs(float(qb) - 26.0) < 1e-9, qb
    # 8 rec, 100 yds, 1 TD: 8 + 10 + 6 + 3 = 27
    wr = score_offense({"receptions": 8, "rec_yards": 100, "rec_tds": 1})
    assert abs(float(wr) - 27.0) < 1e-9, wr
    # 99 yds rushing: no bonus -> 9.9
    rb = score_offense({"rush_yards": 99})
    assert abs(float(rb) - 9.9) < 1e-9, rb
    # points-allowed tiers
    pa = points_allowed_score(np.array([0, 3, 10, 17, 24, 30, 40]))
    assert list(pa) == [10, 7, 4, 1, 0, -1, -4], pa
    # DST: 3 sacks, 1 INT, 1 fum rec, 1 def TD, allow 10 -> 3+2+2+6+4 = 17
    dst = score_dst({"sacks": 3, "ints": 1, "fumble_rec": 1, "def_tds": 1},
                    points_allowed=10)
    assert abs(float(dst) - 17.0) < 1e-9, dst
    print("dk_scoring.py self-test passed")
