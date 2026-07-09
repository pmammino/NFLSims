#!/usr/bin/env python3
"""
sim_engine.py
=============
Correlated player simulation for NFL DFS, driven by the range-of-outcomes
projections (floor p25 / median p50 / ceiling p75) and scored by DraftKings
rules. This is the NFL analog of ``DFSSimsFull/sim_proj.py``.

Approach
--------
1. **Marginals.** Each quantity (a matched player's per-stat triple, an
   unmatched player's fantasy-point triple, a DST's counting triple) is turned
   into a *quantile function* that is piecewise-linear in standard-normal space
   through the three knots (z=-0.6745->floor, 0->median, +0.6745->ceiling), with
   linearly-extrapolated tails clamped at 0. This reproduces the provided
   percentiles exactly and inherits their skew, so ceilings stay fat.

2. **Correlation.** One standard-normal latent per entity is drawn from a
   Gaussian copula whose target correlation matrix encodes NFL structure
   (QB<->pass-catchers, WR<->WR, RB competition, game-stack, DST<->own offense).
   A matched player's own stats share that latent (plus a small per-stat
   idiosyncratic component) so a boom game lifts yards and TDs together.

3. **Scoring.** Sampled stats are scored per-sim by ``dk_scoring`` so yardage
   bonuses fire on the realized yardage. DST adds a points-allowed component
   derived from the *opponent's* simulated offense (so it anti-correlates with
   the offense it faces).

Output: ``SimResult`` with ``dk`` = {entity_key: np.ndarray[n_sims]} of DK
points (the stage-boundary artifact) plus per-stat means for the player table.
"""
from collections import defaultdict

import numpy as np
from scipy.stats import norm

import dk_scoring as dk

Z25, Z50, Z75 = -0.6744897501960817, 0.0, 0.6744897501960817

# ------------------------- correlation targets ---------------------------- #
# Pairwise target correlations by relationship (see README table).
SAME_TEAM = {
    ("QB", "WR"): 0.55, ("QB", "TE"): 0.50, ("QB", "RB"): 0.10,
    ("WR", "WR"): 0.30, ("WR", "TE"): 0.30, ("TE", "TE"): 0.25,
    ("RB", "WR"): -0.05, ("RB", "TE"): -0.05, ("RB", "RB"): -0.20,
    ("QB", "QB"): 1.0,
}
OPP_GAME = {                         # same game, opposing teams (bring-back)
    ("QB", "WR"): 0.18, ("QB", "TE"): 0.16, ("QB", "QB"): 0.15,
    ("QB", "RB"): 0.05, ("WR", "WR"): 0.10, ("WR", "TE"): 0.10,
    ("RB", "WR"): 0.05, ("TE", "TE"): 0.10, ("RB", "RB"): 0.02,
    ("RB", "TE"): 0.05,
}
DST_OWN_OFFENSE = 0.10               # DST with its own offense (field position)
DST_OPP_OFFENSE = -0.05              # DST counting stats vs a strong opponent
STAT_SELF_W = 0.85                   # within-player stat co-movement weight

# The floor/ceiling knots are p25/p75 (the inter-QUARTILE range), so the full
# distribution extends past them. Extrapolating the steep IQR slope linearly
# overstates the deep tails (a RB "p99" would exceed any real game), so the tail
# EXCESS beyond a knot is compressed and the latent is clipped.
TAIL_HI = 0.55                       # upper-tail slope damping
TAIL_LO = 0.80                       # lower-tail slope damping
Z_CLIP = 4.0


def _pair(a, b, table):
    return table.get((a, b)) or table.get((b, a))


def _target_corr(e1, e2):
    """Target correlation between two entities' latents."""
    p1, p2 = e1["pos"], e2["pos"]
    t1, t2, g1, g2 = e1["team"], e2["team"], e1["game"], e2["game"]
    same_game = bool(g1) and g1 == g2
    same_team = bool(t1) and t1 == t2

    if p1 == "DST" or p2 == "DST":
        dst, oth = (e1, e2) if p1 == "DST" else (e2, e1)
        if oth["pos"] == "DST":
            return 0.0
        if oth["team"] and oth["team"] == dst["team"]:
            return DST_OWN_OFFENSE
        if same_game:
            return DST_OPP_OFFENSE
        return 0.0

    if same_team:
        v = _pair(p1, p2, SAME_TEAM)
        return v if v is not None else 0.0
    if same_game:
        v = _pair(p1, p2, OPP_GAME)
        return v if v is not None else 0.05
    return 0.0


def build_corr(entities):
    """Symmetric target correlation matrix, projected to the nearest valid
    correlation matrix (PSD, unit diagonal)."""
    n = len(entities)
    R = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            c = _target_corr(entities[i], entities[j])
            R[i, j] = R[j, i] = c
    # project to PSD then rescale to unit diagonal
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    return R


def q_from_triple(z, floor, med, ceil):
    """Quantile map: standard-normal `z` -> value, piecewise-linear through
    (Z25->floor, Z50->med, Z75->ceil). Interior is exact; the tail EXCESS beyond
    a knot is damped (TAIL_HI/TAIL_LO) so p25/p75 are honored but the deep tails
    stay realistic. Clamped at 0."""
    f = min(floor, med)
    c = max(ceil, med)
    slope_lo = (med - f) / (Z50 - Z25) if med != f else 0.0
    slope_hi = (c - med) / (Z75 - Z50) if c != med else 0.0
    z = np.clip(np.asarray(z, dtype=np.float64), -Z_CLIP, Z_CLIP)
    interior = np.interp(z, [Z25, Z50, Z75], [f, med, c])
    v = np.where(z <= Z25, f + (z - Z25) * TAIL_LO * slope_lo,
                 np.where(z >= Z75, c + (z - Z75) * TAIL_HI * slope_hi, interior))
    return np.clip(v, 0.0, None)


class SimResult:
    def __init__(self, keys, dk_points, stat_means, meta, n_sims):
        self.keys = keys
        self.dk = dk_points               # {key: array[n_sims]}
        self.stat_means = stat_means      # {key: {stat: mean}}
        self.meta = meta                  # {key: entity dict}
        self.n_sims = n_sims


def simulate(slate, n_sims=10000, seed=20260709, total_scale=None):
    """Run the correlated sim over a Slate. Returns a SimResult.

    `total_scale` {team: multiplier} optionally reshapes each team's offensive
    output (Vegas overlay); defaults to 1.0 for every team (no-op)."""
    rng = np.random.default_rng(seed)
    entities = slate.entities
    n = len(entities)
    scale = total_scale or {}

    # ---- correlated latents ----
    R = build_corr(entities)
    L = np.linalg.cholesky(R)
    Z = (L @ rng.standard_normal((n, n_sims))).astype(np.float64)   # (n, n_sims)

    dk_points = {}
    stat_means = {}
    meta = {e["key"]: e for e in entities}
    idx = {e["key"]: i for i, e in enumerate(entities)}

    # ---- offense first (needed for DST points-allowed) ----
    team_off_sum = defaultdict(lambda: np.zeros(n_sims, dtype=np.float64))
    for e in slate.players:
        zrow = Z[idx[e["key"]]]
        sm = scale.get(e["team"], 1.0)
        if e["matched"]:
            stats = {}
            for stat, (F, M, C) in e["stats"].items():
                # yardage/TD stats scale with the team total; counting negatives
                # (INT, fumbles) do not
                mult = sm if stat not in ("pass_ints", "fumbles_lost") else 1.0
                zc = np.sqrt(STAT_SELF_W) * zrow + \
                    np.sqrt(1 - STAT_SELF_W) * rng.standard_normal(n_sims)
                stats[stat] = q_from_triple(zc, F * mult, M * mult, C * mult)
            pts = dk.score_offense(stats).astype(np.float32)
            stat_means[e["key"]] = {k: float(v.mean()) for k, v in stats.items()}
        else:
            F, M, C = e["fp"]
            pts = q_from_triple(zrow, F, M, C).astype(np.float32)
            stat_means[e["key"]] = {}
        dk_points[e["key"]] = pts
        if e["team"]:
            team_off_sum[e["team"]] += pts

    # opponent offense mean per team (for the points-allowed calibration)
    team_off_mean = {t: max(float(v.mean()), 1e-6) for t, v in team_off_sum.items()}

    # ---- DST: counting component (own latent) + points-allowed (opp offense) ---
    for e in slate.dst:
        zrow = Z[idx[e["key"]]]
        zc = np.sqrt(STAT_SELF_W) * zrow + \
            np.sqrt(1 - STAT_SELF_W) * rng.standard_normal(n_sims)
        F, M, C = e["count_fp"]
        counting = q_from_triple(zc, F, M, C)
        opp = e["opp"]
        if opp and opp in team_off_sum:
            # points allowed tracks how the opponent's offense did this sim,
            # anchored to the opponent's implied total
            ratio = team_off_sum[opp] / team_off_mean[opp]
            pa = np.clip(e["opp_total"] * ratio, 0.0, 70.0)
        else:
            pa = np.full(n_sims, e["opp_total"], dtype=np.float64)
        pts = (counting + dk.points_allowed_score(pa)).astype(np.float32)
        dk_points[e["key"]] = pts
        stat_means[e["key"]] = {"points_allowed": float(pa.mean())}

    return SimResult(list(dk_points.keys()), dk_points, stat_means, meta, n_sims)


# --------------------------------------------------------------------------- #
def realized_correlations(sim, slate):
    """Diagnostic: realized correlations of key NFL relationships vs targets."""
    dkp = sim.dk
    def corr(a, b):
        return float(np.corrcoef(dkp[a], dkp[b])[0, 1])
    out = {"qb_wr_same": [], "wr_wr_same": [], "qb_oppskill": [], "dst_oppoff": []}
    by_team = defaultdict(list)
    for e in slate.players:
        if e["matched"]:
            by_team[e["team"]].append(e)
    for t, players in by_team.items():
        qbs = [p for p in players if p["pos"] == "QB"]
        wrs = [p for p in players if p["pos"] == "WR"]
        for q in qbs:
            for w in wrs:
                out["qb_wr_same"].append(corr(q["key"], w["key"]))
        for i in range(len(wrs)):
            for j in range(i + 1, len(wrs)):
                out["wr_wr_same"].append(corr(wrs[i]["key"], wrs[j]["key"]))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in out.items()}


if __name__ == "__main__":
    import nfl_ingest
    slate = nfl_ingest.build_slate()
    sim = simulate(slate, n_sims=5000, seed=1)
    # top offensive players by mean sim DK points
    means = sorted(((k, v.mean()) for k, v in sim.dk.items()),
                   key=lambda kv: kv[1], reverse=True)[:8]
    for k, m in means:
        e = sim.meta[k]
        arr = sim.dk[k]
        print(f"{k:>9} {e['pos']:>3} {e['team']:>3}  mean {m:5.1f}  "
              f"p10 {np.percentile(arr,10):5.1f}  p90 {np.percentile(arr,90):5.1f}  "
              f"max {arr.max():5.1f}")
    rc = realized_correlations(sim, slate)
    print("realized corr:", {k: round(v, 3) for k, v in rc.items()})
