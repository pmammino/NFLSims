#!/usr/bin/env python3
"""
sim_engine.py
=============
Correlated player simulation for NFL DFS, driven by the range-of-outcomes
projections (floor p25 / median p50 / ceiling p75) and scored by DraftKings
rules. NFL analog of ``DFSSimsFull/sim_proj.py``.

Design — hierarchical, game-consistent allocation
-------------------------------------------------
The earlier version sampled every player's stats independently from a copula and
never reconciled them, so a QB and his receivers could each boom in the *same*
sim — double-counting the same passing yards and fattening the high tail. This
version enforces game logic:

1. **Latents.** Per sim: a game latent (shootout), a team-offense latent tied to
   it (bring-back / game-stack correlation), and team pass / rush latents under
   that.
2. **Marginals.** Each quantity is drawn from its *own* (floor, median, ceiling)
   triple via a quantile map that is piecewise-linear in standard-normal space
   with damped tails (`q_from_triple`), so p25/p75 are honored but deep tails
   stay realistic.
3. **Allocation (the key step).** The starting QB's passing line is sampled from
   his ranges; each pass-catcher's receiving line is sampled from *his* ranges;
   then the receivers are **rescaled so their receptions / rec yards / rec TDs
   sum exactly to the QB's completions / pass yards / pass TDs in every sim.**
   Team receptions == QB completions, team rec yards == QB pass yards, team rec
   TDs == QB pass TDs — game-consistent, and the receiver ceilings are bounded by
   the team's realized passing total (fixes the overshoot).
4. **Scoring.** Final stats are scored per-sim by ``dk_scoring`` (yardage bonuses
   fire on realized yardage). DST adds a points-allowed component derived from
   the opponent's simulated offense.

Output: ``SimResult`` with ``dk`` = {entity_key: np.ndarray[n_sims]} of DK points
(the stage-boundary artifact) plus per-stat means for the player table.
"""
from collections import defaultdict

import numpy as np

import dk_scoring as dk

Z25, Z50, Z75 = -0.6744897501960817, 0.0, 0.6744897501960817

# ---- latent loadings (see module docstring) ---- #
GAME_LOAD = 0.50          # game env -> team offense (shootout / bring-back)
PASS_FROM_OFF = 0.80      # team offense -> team pass latent
RUSH_FROM_OFF = 0.80      # team offense -> team rush latent
QB_FROM_PASS = 0.92       # team pass latent -> QB passing base
REC_FROM_PASS = 0.60      # team pass latent -> a receiver's involvement base
RUSH_FROM_TEAMRUSH = 0.70  # team rush latent -> a rusher's base
DST_FROM_OFF = 0.20       # own offense -> DST counting base (field position)
STAT_SELF_W = 0.85        # within-base per-stat co-movement

# tail damping — floor/ceiling are p25/p75 (IQR), so the true tails extend past
# them; extrapolating the steep IQR slope linearly overstates the deep tail. The
# upper tail is damped hard (and the latent clipped) to keep ceilings realistic
# across every position (elite-RB p99 ~70 / max ~88, QB/WR/TE max ~60-70).
TAIL_HI = 0.35
TAIL_LO = 0.80
Z_CLIP = 3.25

VOLUME_STATS = {"pass_yards", "pass_completions", "pass_tds", "rush_yards",
                "rush_tds", "receptions", "rec_yards", "rec_tds", "return_tds"}


def q_from_triple(z, floor, med, ceil):
    """Quantile map: standard-normal `z` -> value, piecewise-linear through
    (Z25->floor, Z50->med, Z75->ceil). Interior is exact; the tail excess beyond
    a knot is damped so p25/p75 are honored but deep tails stay realistic.
    Clamped at 0."""
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
    """Run the hierarchical, game-consistent sim over a Slate. Returns SimResult.

    `total_scale` {team: multiplier} optionally reshapes each team's offensive
    output (Vegas overlay); defaults to 1.0 for every team (no-op)."""
    rng = np.random.default_rng(seed)
    N = n_sims
    scale = total_scale or {}

    def unit(load):
        return np.sqrt(max(1.0 - load * load, 0.0))

    def combine(base, load):
        """base latent -> a child latent of unit variance."""
        return load * base + unit(load) * rng.standard_normal(N)

    def stat(base, triple, mult=1.0, w=STAT_SELF_W):
        """Sample one stat from its (F,M,C) triple around a base latent."""
        F, M, C = triple
        z = np.sqrt(w) * base + np.sqrt(1 - w) * rng.standard_normal(N)
        return q_from_triple(z, F * mult, M * mult, C * mult)

    # ---- game / team latents ----
    team_game = {}
    for e in slate.entities:
        if e.get("team") and e.get("game"):
            team_game[e["team"]] = e["game"]
    game_z = {g: rng.standard_normal(N) for g in set(team_game.values())}
    z_off, z_pass, z_rush = {}, {}, {}
    for t, g in team_game.items():
        zo = combine(game_z[g], GAME_LOAD)
        z_off[t] = zo
        z_pass[t] = combine(zo, PASS_FROM_OFF)
        z_rush[t] = combine(zo, RUSH_FROM_OFF)

    dk_points, stat_means = {}, {}
    meta = {e["key"]: e for e in slate.entities}

    by_team = defaultdict(list)
    loose = []                       # matched players with no team, or unmatched
    for p in slate.players:
        (by_team[p["team"]].append(p) if p["team"] else loose.append(p))

    team_off_sum = defaultdict(lambda: np.zeros(N))

    def zeros():
        return np.zeros(N)

    # ---- offense, team by team ----
    for team, players in by_team.items():
        sm = scale.get(team, 1.0)
        zp = z_pass.get(team)
        zr = z_rush.get(team)
        if zp is None:                        # team not in schedule/games
            zp = rng.standard_normal(N)
            zr = rng.standard_normal(N)

        matched = [p for p in players if p["matched"]]
        qbs = sorted([p for p in matched if p["pos"] == "QB"],
                     key=lambda p: p["fp"][1], reverse=True)
        thrower = qbs[0] if qbs else None
        skill = [p for p in matched if p["pos"] in ("RB", "WR", "TE")]
        catchers = [p for p in skill if p["stats"]["receptions"][1] > 0]

        # ---- QB passing line (the team's passing envelope) ----
        pass_tot = {"receptions": zeros(), "rec_yards": zeros(), "rec_tds": zeros()}
        if thrower is not None:
            zqb = combine(zp, QB_FROM_PASS)
            s = thrower["stats"]
            cmp_t = stat(zqb, s["pass_completions"], mult=sm)
            yds_t = stat(zqb, s["pass_yards"], mult=sm)
            ptd_t = stat(zqb, s["pass_tds"], mult=sm)
            pass_tot = {"receptions": cmp_t, "rec_yards": yds_t, "rec_tds": ptd_t}
            qb_stats = {
                "pass_yards": yds_t, "pass_tds": ptd_t,
                "pass_ints": stat(zqb, s["pass_ints"]),
                "rush_yards": stat(combine(zr, RUSH_FROM_TEAMRUSH), s["rush_yards"], mult=sm),
                "rush_tds": stat(combine(zr, RUSH_FROM_TEAMRUSH), s["rush_tds"], mult=sm),
                "fumbles_lost": stat(zqb, s["fumbles_lost"]),
                "two_pts": stat(zqb, s["two_pts"]),
            }
            _emit(dk_points, stat_means, thrower["key"], qb_stats)
            team_off_sum[team] += dk_points[thrower["key"]]

        # ---- receiver raws, then rescale to the QB passing totals ----
        raw = {}
        for p in catchers:
            zi = combine(zp, REC_FROM_PASS)
            s = p["stats"]
            raw[p["key"]] = {
                "receptions": stat(zi, s["receptions"], mult=sm),
                "rec_yards": stat(zi, s["rec_yards"], mult=sm),
                "rec_tds": stat(zi, s["rec_tds"], mult=sm),
            }
        alloc = _allocate(raw, pass_tot, N) if (catchers and thrower) else \
            {k: raw[k] for k in raw}

        # ---- each skill player: allocated receiving + own rushing + misc ----
        for p in skill:
            s = p["stats"]
            rec = alloc.get(p["key"], {"receptions": zeros(), "rec_yards": zeros(),
                                       "rec_tds": zeros()})
            zpr = combine(zr, RUSH_FROM_TEAMRUSH)
            st = {
                "receptions": rec["receptions"], "rec_yards": rec["rec_yards"],
                "rec_tds": rec["rec_tds"],
                "rush_yards": stat(zpr, s["rush_yards"], mult=sm),
                "rush_tds": stat(zpr, s["rush_tds"], mult=sm),
                "fumbles_lost": stat(zpr, s["fumbles_lost"]),
                "two_pts": stat(zpr, s["two_pts"]),
                "return_tds": stat(zpr, s["return_tds"]),
            }
            _emit(dk_points, stat_means, p["key"], st)
            team_off_sum[team] += dk_points[p["key"]]

        # ---- backup QBs: standalone from their own marginals ----
        for p in qbs[1:]:
            s = p["stats"]
            zb = rng.standard_normal(N)
            st = {"pass_yards": stat(zb, s["pass_yards"]),
                  "pass_tds": stat(zb, s["pass_tds"]),
                  "pass_ints": stat(zb, s["pass_ints"]),
                  "rush_yards": stat(zb, s["rush_yards"]),
                  "rush_tds": stat(zb, s["rush_tds"])}
            _emit(dk_points, stat_means, p["key"], st)
            team_off_sum[team] += dk_points[p["key"]]

    # ---- players with no team / no projection: replacement-level fp marginal ----
    for p in loose:
        zb = z_off.get(p["team"], rng.standard_normal(N)) if p["team"] else rng.standard_normal(N)
        F, M, C = p["fp"]
        pts = q_from_triple(combine(zb, 0.5), F, M, C).astype(np.float32)
        dk_points[p["key"]] = pts
        stat_means[p["key"]] = {}
        if p["team"]:
            team_off_sum[p["team"]] += pts

    team_off_mean = {t: max(float(v.mean()), 1e-6) for t, v in team_off_sum.items()}

    # ---- DST: counting (own latent) + points-allowed (opp offense) ----
    for e in slate.dst:
        zo = z_off.get(e["team"], rng.standard_normal(N))
        zc = combine(zo, DST_FROM_OFF)
        F, M, C = e["count_fp"]
        counting = q_from_triple(np.sqrt(STAT_SELF_W) * zc +
                                 np.sqrt(1 - STAT_SELF_W) * rng.standard_normal(N),
                                 F, M, C)
        opp = e["opp"]
        if opp and opp in team_off_sum:
            ratio = team_off_sum[opp] / team_off_mean[opp]
            pa = np.clip(e["opp_total"] * ratio, 0.0, 70.0)
        else:
            pa = np.full(N, e["opp_total"])
        pts = (counting + dk.points_allowed_score(pa)).astype(np.float32)
        dk_points[e["key"]] = pts
        stat_means[e["key"]] = {"points_allowed": float(pa.mean())}

    return SimResult(list(dk_points.keys()), dk_points, stat_means, meta, N)


def _allocate(raw, pass_tot, N):
    """Rescale each receiver's raw receptions/yards/TDs so the team totals equal
    the QB's completions/pass yards/pass TDs in every sim. If every receiver
    sampled 0 for a stat in a sim, split the QB total equally that sim."""
    keys = list(raw)
    out = {k: {} for k in keys}
    for stat_key, total_key in (("receptions", "receptions"),
                                ("rec_yards", "rec_yards"),
                                ("rec_tds", "rec_tds")):
        stack = np.vstack([raw[k][stat_key] for k in keys])   # (n_recv, N)
        col = stack.sum(axis=0)                                # (N,)
        total = pass_tot[total_key]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(col > 1e-9, total / col, 0.0)
        scaled = stack * ratio                                 # broadcast over rows
        # sims where every receiver sampled 0: split the total equally
        zero = col <= 1e-9
        if zero.any():
            scaled[:, zero] = total[zero] / len(keys)
        for i, k in enumerate(keys):
            out[k][stat_key] = scaled[i]
    return out


def _emit(dk_points, stat_means, key, stats):
    dk_points[key] = dk.score_offense(stats).astype(np.float32)
    stat_means[key] = {k: float(np.asarray(v).mean()) for k, v in stats.items()}


# --------------------------------------------------------------------------- #
def realized_correlations(sim, slate):
    """Diagnostic: realized correlations of key NFL relationships."""
    dkp = sim.dk

    def corr(a, b):
        return float(np.corrcoef(dkp[a], dkp[b])[0, 1])

    out = {"qb_wr_same": [], "wr_wr_same": []}
    by_team = defaultdict(list)
    for e in slate.players:
        if e["matched"]:
            by_team[e["team"]].append(e)
    for t, players in by_team.items():
        qbs = [p for p in players if p["pos"] == "QB"]
        wrs = [p for p in players if p["pos"] == "WR"]
        for q in qbs[:1]:
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
    means = sorted(((k, v.mean()) for k, v in sim.dk.items()),
                   key=lambda kv: kv[1], reverse=True)[:8]
    for k, m in means:
        e = sim.meta[k]
        arr = sim.dk[k]
        print(f"{k:>9} {e['pos']:>3} {e['team']:>3}  mean {m:5.1f}  "
              f"p50 {np.percentile(arr,50):5.1f}  p90 {np.percentile(arr,90):5.1f}  "
              f"p99 {np.percentile(arr,99):5.1f}  max {arr.max():5.1f}")
    rc = realized_correlations(sim, slate)
    print("realized corr:", {k: round(v, 3) for k, v in rc.items()})
