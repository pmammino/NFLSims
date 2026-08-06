#!/usr/bin/env python3
"""
stack_signal.py
===============
NFL port of ``DFSSimsFull/stack_signal.py``. Use projected STACK OWNERSHIP as a
small upside signal on the correlated DK score arrays, so the field's (and our
candidates') popular offenses hit their high-end outcomes a little more often.

WHY
---
The contest is scored from per-player DK sim arrays that are aligned by sim
index, so a team's QB + pass-catchers boom together in the same simulated
"games" (see ``sim_engine`` — the team-offense / pass latents). Today ownership
only drives WHO the field rosters — it has no bearing on how those players
actually SCORE. So a heavily-stacked chalk team and an ignored team with the
same projection have identical ceiling odds, and the optimizer keeps surfacing
low-owned noise stacks at the very top of the candidate results.

In real slates the crowd is not random: the most-stacked teams are
disproportionately the offenses the market (Vegas, weather, pace) expects to
erupt. Stack ownership therefore carries a little signal about which offenses
are most likely to post a tournament-winning game. This module folds that signal
in as a deliberately SMALL bump to a popular offense's high-end outcomes — a
nudge, never a driver.

HOW
---
For each team we look at the sims where that team's offensive players
COLLECTIVELY post a high-end score (the top ``1 - quantile`` of the team's
aggregate) and scale those players' DK points up by a small factor whose size
grows with the team's projected stack ownership (min-max normalized across
teams, so the lowest-owned offense gets no bump and the chalkiest gets the full
``strength``).

Because the bump is applied to the SAME sim indices for every player on the
team, the stack still booms together — it just booms a touch bigger. Players on
un-boosted teams (and every DST) are left exactly as they were, and the same
boosted arrays are used to score BOTH the field and our candidates, so this is a
coherent re-weighting of the simulated reality rather than a candidate-only
thumb on the scale.
"""
from collections import defaultdict

import numpy as np

# offensive positions whose points move with the team's passing/rushing game
STACK_POS = ("QB", "RB", "WR", "TE")


def offense_names_by_team(entities):
    """team -> [entity_key] for its offensive players (QB/RB/WR/TE). DST is
    excluded: its scoring is anti-correlated with the opponent's offense, not
    part of its own team's boom."""
    out = defaultdict(list)
    for e in entities:
        if e.get("pos") in STACK_POS and e.get("team"):
            out[e["team"]].append(e["key"])
    return dict(out)


def team_stack_ownership(names_by_team, own_by_key):
    """Projected stack ownership per team = sum of its offensive players'
    ownership. Mirrors the stack-team weight the field builder already uses."""
    return {t: float(sum(own_by_key.get(k, 0.0) for k in keys))
            for t, keys in names_by_team.items()}


def _team_signal(stack_own):
    """Min-max normalize stack ownership to [0, 1] across teams. The lowest-owned
    offense maps to 0 (no bump) and the highest to 1 (full strength); everything
    in between scales linearly. Returns {} when there's nothing to separate."""
    if not stack_own:
        return {}
    vals = list(stack_own.values())
    lo, hi = min(vals), max(vals)
    if hi - lo <= 0:
        return {t: 0.0 for t in stack_own}
    return {t: (v - lo) / (hi - lo) for t, v in stack_own.items()}


def apply_stack_ownership_boost(score, names_by_team, stack_own, K=None, *,
                                strength=0.05, quantile=0.80):
    """Return a NEW score dict with popular offenses' high-end outcomes nudged up.

    Parameters
    ----------
    score : {key: np.ndarray}
        Per-player DK sim arrays, aligned by sim index (length >= K).
    names_by_team : {team: iterable[key]}
        Offensive membership per team (keys must key into ``score``).
    stack_own : {team: float}
        Projected stack ownership per team (e.g. summed player ownership).
    K : int, optional
        Number of leading sims to operate on (defaults to each array's length).
    strength : float
        Maximum multiplicative bump applied to the chalkiest offense's high-end
        games. 0 (or no spread in ownership) -> no-op, original arrays returned.
    quantile : float in [0, 1)
        Defines a team's "high-end" sims: those at/above this quantile of the
        team's aggregate player score (0.80 -> the team's top ~20% of games).

    Only boosted players get a fresh (copied) array; everyone else — every DST
    and un-boosted offense — shares the original reference, so this is cheap.
    """
    if strength <= 0 or not names_by_team:
        return dict(score)

    signal = _team_signal(stack_own)
    out = dict(score)
    for team, keys in names_by_team.items():
        s = signal.get(team, 0.0)
        if s <= 0:
            continue
        members = [k for k in keys if k in score]
        if not members:
            continue
        n_sim = K if K is not None else len(score[members[0]])
        agg = np.zeros(n_sim, np.float64)
        for k in members:
            agg += np.asarray(score[k][:n_sim], np.float64)
        thr = np.quantile(agg, quantile)
        mask = agg >= thr
        if not mask.any():
            continue
        factor = 1.0 + strength * s
        for k in members:
            arr = np.array(score[k][:n_sim], np.float32, copy=True)
            arr[mask] *= np.float32(factor)
            out[k] = arr
    return out


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 20000

    names_by_team = {
        "CHALK": ["c1", "c2", "c3", "c4"],
        "MID":   ["m1", "m2", "m3", "m4"],
        "COLD":  ["d1", "d2", "d3", "d4"],
    }
    own = {"c1": 40, "c2": 30, "c3": 25, "c4": 20,
           "m1": 20, "m2": 15, "m3": 12, "m4": 10,
           "d1": 5,  "d2": 4,  "d3": 3,  "d4": 2}
    stack_own = team_stack_ownership(names_by_team, own)
    assert stack_own["CHALK"] > stack_own["MID"] > stack_own["COLD"]

    score = {}
    for team, members in names_by_team.items():
        shock = rng.normal(0, 1, K)               # team "game" latent
        for n in members:
            score[n] = (10 + 4 * shock + rng.normal(0, 3, K)).astype(np.float32)

    boosted = apply_stack_ownership_boost(score, names_by_team, stack_own, K,
                                          strength=0.08, quantile=0.80)

    same = apply_stack_ownership_boost(score, names_by_team, stack_own, K, strength=0.0)
    for n in score:
        assert np.array_equal(same[n], score[n])
    flat = apply_stack_ownership_boost(
        score, names_by_team, {t: 1.0 for t in names_by_team}, K, strength=0.5)
    for n in score:
        assert np.array_equal(flat[n], score[n]), "equal ownership -> no boost"

    def team_ceiling(d, members):
        agg = np.sum([d[n] for n in members], axis=0)
        return np.percentile(agg, 99)

    ch = team_ceiling(boosted, names_by_team["CHALK"]) / team_ceiling(score, names_by_team["CHALK"])
    md = team_ceiling(boosted, names_by_team["MID"]) / team_ceiling(score, names_by_team["MID"])
    cd = team_ceiling(boosted, names_by_team["COLD"]) / team_ceiling(score, names_by_team["COLD"])
    assert ch > md > 1.0, (ch, md)
    assert abs(cd - 1.0) < 1e-6, cd                 # least-owned: no bump
    assert ch < 1.10, ch                            # small nudge, not a takeover
    ch_med = (np.median(np.sum([boosted[n] for n in names_by_team["CHALK"]], axis=0)) /
              np.median(np.sum([score[n] for n in names_by_team["CHALK"]], axis=0)))
    assert ch_med < ch, (ch_med, ch)
    print("stack_signal.py self-test passed:",
          f"CHALK ceiling x{ch:.4f}, MID x{md:.4f}, COLD x{cd:.4f}, median x{ch_med:.4f}")
