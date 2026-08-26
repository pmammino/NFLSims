#!/usr/bin/env python3
"""
showdown.py
===========
DraftKings **NFL Showdown** (single-game) roster support — the Captain-centric
analog of the Classic ``field_builder`` / ``field_simulator`` stack.

Showdown roster
---------------
* Slots: ``CPT`` (Captain) + five ``FLEX``. Six players, all from the **one**
  game on the slate; any position (QB/RB/WR/TE/K/DST) fills any slot.
* Salary cap: ``$50,000``.
* Scoring: the Captain earns **1.5x** DK points and costs **1.5x** salary. The
  1.5x points multiplier is applied in ``contest_sim.score_matrix`` (via each
  lineup's ``captain_key``); the 1.5x salary comes from each entity's
  ``cpt_salary`` (see ``nfl_ingest._attach_captain``).

Ownership feed (the key modelling decision)
--------------------------------------------
The ownership feed carries **one entry per player** — the player's **overall**
rostered ownership across BOTH slots. A Showdown lineup has six roster spots
(1 CPT + 5 FLEX), so summed across the field:

    sum(overall ownership)  = 6      (six spots per entry)
    sum(captain ownership)  = 1      (one CPT spot)
    sum(flex ownership)     = 5      (five FLEX spots)

``split_ownership`` decomposes each player's overall ownership ``O_i`` into a
Captain-slot rate ``cpt_own`` and a FLEX-slot rate ``flex_own`` (``O_i =
cpt_own + flex_own``). The Captain slot is scarce (one per lineup) and skews to
high-ceiling players, so the split routes a ceiling-tilted share of each
player's exposure to Captain and leaves the rest at FLEX.

Two builders
------------
* ``FieldBuilder``  — **ownership-aware**. Draws the Captain in proportion to
  ``cpt_own`` and the five FLEX in proportion to ``flex_own`` (the crowd you
  actually face). Correlation is whatever ownership implies.
* ``CandidateBuilder`` — **ownership-blind**. Ignores ownership entirely and
  builds off smart single-game construction rules: pick a team to build around,
  Captain a ceiling player, keep the QB + a pass-catcher together (the intra-team
  passing stack), force a bring-back from the other team (the game rarely goes
  off one-sided), and weight every remaining pick by simulated ceiling. This is
  where the edge lives — the sim already correlates a single game, so the job is
  to land on the structurally live builds, not to imitate the field.
"""
import math
from collections import Counter, defaultdict

import numpy as np

import field_builder as fb   # reuse wchoice

SALARY_CAP = 50000
MIN_SALARY = 0               # Showdown pools are tiny; no hard floor by default
CAPTAIN_MULT = 1.5

SLOT_COLS = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
UPLOAD_HEADER = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]

PASS_CATCHER = ("WR", "TE", "RB")     # RB catches too (screen/checkdown game)
SKILL = ("QB", "RB", "WR", "TE")

# candidate team-split distribution: (focus_count, opp_count, weight). The focus
# team is the one you Captain from. Winning single-game lineups almost always
# carry both teams (the game has to go off), so there is no 6-0 build and 5-1 is
# rare; balanced 3-3 and onslaught 4-2 dominate.
CAND_SPLITS = [(4, 2, 0.34), (3, 3, 0.40), (2, 4, 0.16), (5, 1, 0.10)]


def cpt_salary(entity):
    """Captain-priced salary of an entity (explicit ``cpt_salary`` or 1.5x)."""
    cs = entity.get("cpt_salary")
    if cs:
        return int(cs)
    return int(round(CAPTAIN_MULT * int(entity["salary"])))


# --------------------------------------------------------------------------- #
# Ownership split: overall ownership -> (captain-slot, flex-slot) propensities
# --------------------------------------------------------------------------- #
def split_ownership(entities, ceiling_tilt=0.75):
    """Return {key: (cpt_own, flex_own)} splitting each player's OVERALL ownership.

    ``entities`` carry ``own`` (overall rostered %, any scale) and optionally
    ``up`` (a ceiling proxy, e.g. sim p90). Overall exposure is first rescaled so
    it sums to 6 (six roster spots per entry). The Captain slot (one per lineup,
    total mass 1) is assigned in proportion to ``own * (up/mean_up)**ceiling_tilt``
    — chalk that also has ceiling gets Captained most — then clipped so a player
    is never Captained more than their overall exposure. FLEX is the remainder
    (total mass 5). ``ceiling_tilt=0`` makes Captain share purely ownership-based.
    """
    keys = [e["key"] for e in entities]
    if not keys:
        return {}
    own = np.array([max(float(e.get("own", 0.0)), 1e-6) for e in entities],
                   dtype=np.float64)
    O = own * (6.0 / own.sum())                      # overall exposure, sums to 6

    up = np.array([max(float(e.get("up", e.get("own", 1.0))), 1e-9)
                   for e in entities], dtype=np.float64)
    upn = up / up.mean() if up.mean() > 0 else np.ones_like(up)
    g = np.power(upn, float(ceiling_tilt))           # Captain attractiveness

    # Split each player's overall exposure O_i into Captain and FLEX by a
    # per-player Captain SHARE s_i in [0, 1] (share of that player's rostered
    # lineups in which they are the Captain): cpt_i = s_i*O_i, flex_i = (1-s_i)*O_i.
    # This keeps cpt_i <= O_i by construction. Set s_i = k*g_i and solve k so the
    # Captain mass sums to exactly 1 (one CPT per lineup); when a heavily-tilted
    # player would exceed s_i=1, pin them there and re-solve for the rest. FLEX
    # mass then sums to exactly 5.
    s = np.zeros_like(O)
    active = np.ones(len(O), dtype=bool)
    target = 1.0
    for _ in range(50):
        denom = float((g[active] * O[active]).sum())
        if denom <= 0:
            break
        k = target / denom
        cand = k * g
        over = active & (cand >= 1.0)
        if not over.any():
            s[active] = cand[active]
            break
        s[over] = 1.0
        target -= float(O[over].sum())
        active &= ~over
        if target <= 0 or not active.any():
            break
    cpt = s * O
    flex = O - cpt
    return {k: (float(c), float(f)) for k, c, f in zip(keys, cpt, flex)}


# --------------------------------------------------------------------------- #
class Pool:
    """Indexes a single game's entities for weighted Showdown construction.

    Computes the Captain/FLEX ownership split and per-team position groups the
    candidate builder needs (QB, pass-catchers, DST) for stacking rules."""

    def __init__(self, entities, ceiling_tilt=0.75):
        split = split_ownership(entities, ceiling_tilt=ceiling_tilt)
        self.rows = []
        for e in entities:
            own = max(float(e.get("own", 0.0)), 1e-6)
            c, f = split.get(e["key"], (own / 6.0, own * 5.0 / 6.0))
            self.rows.append({
                "key": e["key"], "pos": e["pos"], "team": e.get("team", ""),
                "opp": e.get("opp", ""), "salary": int(e["salary"]),
                "cpt_salary": cpt_salary(e), "own": own,
                "cpt_own": max(c, 1e-9), "flex_own": max(f, 1e-9),
                "up": max(float(e.get("up", own)), 1e-9),
                "contest_id": e.get("contest_id", ""),
                "cpt_contest_id": e.get("cpt_contest_id", e.get("contest_id", "")),
            })
        self.teams = sorted({r["team"] for r in self.rows if r["team"]})
        self.by_team = defaultdict(list)
        self.qb_of = {}
        self.catchers_of = defaultdict(list)
        self.skill_of = defaultdict(list)
        self.dst_of = {}
        for r in self.rows:
            t = r["team"]
            if not t:
                continue
            self.by_team[t].append(r)
            if r["pos"] == "QB":
                # keep the highest-ceiling QB as the team's passer
                if t not in self.qb_of or r["up"] > self.qb_of[t]["up"]:
                    self.qb_of[t] = r
            elif r["pos"] == "DST":
                self.dst_of[t] = r
            if r["pos"] in SKILL:
                self.skill_of[t].append(r)
            if r["pos"] in PASS_CATCHER:
                self.catchers_of[t].append(r)

    def other_team(self, t):
        for x in self.teams:
            if x != t:
                return x
        return ""


# --------------------------------------------------------------------------- #
def _format(cap, flex):
    players = [cap] + list(flex)
    tc = Counter(p["team"] for p in players)
    split = "-".join(str(c) for c in sorted(tc.values(), reverse=True))
    total = cap["cpt_salary"] + sum(p["salary"] for p in flex)
    return {
        "players": players,             # index 0 is the Captain (order matters)
        "captain_key": cap["key"],
        "captain": cap,
        "salary": total,
        "cap_team": cap["team"],
        "teams": dict(tc),
        "split": split,
        # a Classic-shaped "stack" field so shared code (Counter tallies) is safe:
        # the count of Captain-team-mates (the team you leaned on), Captain aside.
        "stack": tc.get(cap["team"], 1) - 1,
        "slate_type": "showdown",
        "cells": [f"{p['key']} ({p['team']})" for p in players],
    }


# --------------------------------------------------------------------------- #
class FieldBuilder:
    """Ownership-aware Showdown field: Captain ~ cpt_own, FLEX ~ flex_own."""

    def __init__(self, pool, seed=None, jitter=0.0, salary_cap=SALARY_CAP,
                 min_salary=MIN_SALARY):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.jitter = float(jitter)
        self.cap = int(salary_cap)
        self.min_salary = int(min_salary)

    def build_one(self, max_tries=200):
        for _ in range(max_tries):
            lu = self._attempt()
            if lu is not None:
                return lu
        return None

    def _attempt(self):
        rows = self.pool.rows
        if len(rows) < 6:
            return None
        cap = fb.wchoice(self.rng, rows, [r["cpt_own"] for r in rows], self.jitter)
        used = {cap["key"]}
        flex = []
        for _ in range(5):
            avail = [r for r in rows if r["key"] not in used]
            if not avail:
                return None
            pick = fb.wchoice(self.rng, avail, [r["flex_own"] for r in avail],
                              self.jitter)
            flex.append(pick)
            used.add(pick["key"])
        lu = _format(cap, flex)
        if lu["salary"] > self.cap or lu["salary"] < self.min_salary:
            return None
        return lu


# --------------------------------------------------------------------------- #
class CandidateBuilder:
    """Ownership-blind Showdown candidates from smart correlation rules.

    Per lineup: choose a focus team, Captain a ceiling player from it, keep the
    focus QB together with >=1 pass-catcher (the passing stack), force a bring-back
    from the other team, and fill the rest by simulated ceiling. Team split is
    sampled from ``CAND_SPLITS`` so builds range over balanced (3-3) and onslaught
    (4-2/2-4) shapes. All picks weight the pool's ``up`` (ceiling), never ownership.
    """

    def __init__(self, pool, seed=None, jitter=0.0, salary_cap=SALARY_CAP,
                 min_salary=MIN_SALARY, qb_stack_prob=0.85, opp_qb_prob=0.45,
                 captain_skill_prob=0.90, dst_guard_prob=0.85, splits=CAND_SPLITS):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.jitter = float(jitter)
        self.cap = int(salary_cap)
        self.min_salary = int(min_salary)
        self.qb_stack_prob = float(qb_stack_prob)
        self.opp_qb_prob = float(opp_qb_prob)
        self.captain_skill_prob = float(captain_skill_prob)
        self.dst_guard_prob = float(dst_guard_prob)
        vals = [(a, b) for a, b, _ in splits]
        w = np.array([wt for _, _, wt in splits], dtype=float)
        self._split_vals = vals
        self._split_w = w / w.sum()

    # --- weighted ceiling pick over a row list ---------------------------- #
    def _pick(self, rows, used):
        avail = [r for r in rows if r["key"] not in used]
        if not avail:
            return None
        return fb.wchoice(self.rng, avail, [r["up"] for r in avail], self.jitter)

    def _sample_split(self):
        i = self.rng.choice(len(self._split_vals), p=self._split_w)
        return self._split_vals[i]

    def build_one(self, max_tries=300):
        for _ in range(max_tries):
            lu = self._attempt()
            if lu is not None:
                return lu
        return None

    def _attempt(self):
        P = self.pool
        rng = self.rng
        if len(P.rows) < 6 or len(P.teams) < 2:
            # degenerate single-team pool: fall back to a ceiling-weighted build
            return self._attempt_flat()

        focus = P.teams[rng.integers(len(P.teams))]
        opp = P.other_team(focus)
        n_focus, n_opp = self._sample_split()
        # clamp the split to what each team can actually supply (keep total = 6)
        n_focus = min(n_focus, len(P.by_team[focus]))
        n_opp = 6 - n_focus
        if n_opp > len(P.by_team[opp]):
            n_opp = len(P.by_team[opp])
            n_focus = 6 - n_opp
        if n_focus < 1 or n_opp < 1 or n_focus > len(P.by_team[focus]):
            return None

        used = set()
        roster = []

        def take(r):
            roster.append(r)
            used.add(r["key"])

        def count(team):
            return sum(1 for r in roster if r["team"] == team)

        # ---- Captain: a ceiling player from the focus team (skill, usually) ----
        cap_pool = P.skill_of[focus] if (P.skill_of[focus] and
                                         rng.random() < self.captain_skill_prob) \
            else P.by_team[focus]
        cap = self._pick(cap_pool, used)
        if cap is None:
            return None
        take(cap)

        # ---- focus passing stack: QB + >=1 pass-catcher together ----
        qb = P.qb_of.get(focus)
        if qb and qb["key"] not in used and count(focus) < n_focus \
                and rng.random() < self.qb_stack_prob:
            take(qb)
        # ensure at least one focus pass-catcher rides with the QB
        if any(r["pos"] == "QB" for r in roster if r["team"] == focus) \
                and count(focus) < n_focus:
            if not any(r["team"] == focus and r["pos"] in PASS_CATCHER
                       for r in roster):
                c = self._pick(P.catchers_of[focus], used)
                if c is not None:
                    take(c)
        # fill remaining focus slots by ceiling
        while count(focus) < n_focus:
            r = self._pick(self._focus_fill_pool(P, focus, roster, used), used)
            if r is None:
                r = self._pick(P.by_team[focus], used)
            if r is None:
                return None
            take(r)

        # ---- bring-back: opponent players (>=1 guaranteed by n_opp>=1) ----
        opp_qb = P.qb_of.get(opp)
        if opp_qb and opp_qb["key"] not in used and count(opp) < n_opp \
                and rng.random() < self.opp_qb_prob:
            take(opp_qb)
            # a double-QB game stack wants a catcher with the opp QB too
            if count(opp) < n_opp and not any(
                    r["team"] == opp and r["pos"] in PASS_CATCHER for r in roster):
                c = self._pick(P.catchers_of[opp], used)
                if c is not None and count(opp) < n_opp:
                    take(c)
        while count(opp) < n_opp:
            r = self._pick(self._opp_fill_pool(P, opp, focus, roster, used), used)
            if r is None:
                r = self._pick(P.by_team[opp], used)
            if r is None:
                return None
            take(r)

        if len(roster) != 6:
            return None
        lu = _format(roster[0], roster[1:])
        if lu["salary"] > self.cap or lu["salary"] < self.min_salary:
            return None
        return lu

    def _focus_fill_pool(self, P, focus, roster, used):
        """Focus-team fill candidates, applying the DST anti-correlation guard."""
        rows = P.by_team[focus]
        opp = P.other_team(focus)
        opp_off = sum(1 for r in roster if r["team"] == opp and r["pos"] in SKILL)
        # rostering your DST behind a big opposing offensive stack is a negative
        # correlation; drop the focus DST from the pool most of the time then.
        if opp_off >= 3 and self.rng.random() < self.dst_guard_prob:
            return [r for r in rows if r["pos"] != "DST"]
        return rows

    def _opp_fill_pool(self, P, opp, focus, roster, used):
        rows = P.by_team[opp]
        focus_off = sum(1 for r in roster if r["team"] == focus and r["pos"] in SKILL)
        if focus_off >= 3 and self.rng.random() < self.dst_guard_prob:
            return [r for r in rows if r["pos"] != "DST"]
        return rows

    def _attempt_flat(self):
        """No opponent in the pool: ceiling-weighted 1 CPT + 5 FLEX, no stacking."""
        P = self.pool
        used = set()
        cap = self._pick(P.rows, used)
        if cap is None:
            return None
        used.add(cap["key"])
        flex = []
        for _ in range(5):
            r = self._pick(P.rows, used)
            if r is None:
                return None
            flex.append(r)
            used.add(r["key"])
        lu = _format(cap, flex)
        if lu["salary"] > self.cap or lu["salary"] < self.min_salary:
            return None
        return lu


# --------------------------------------------------------------------------- #
def lineup_proj(lu, dk_mean):
    """A Showdown lineup's projection = FLEX means + 1.5x the Captain's mean."""
    tot = 0.0
    cap = lu.get("captain_key")
    for pl in lu["players"]:
        m = dk_mean.get(pl["key"], 0.0)
        tot += m
        if cap is not None and pl["key"] == cap:
            tot += 0.5 * m
    return float(tot)


def _draw(builder, count, cap_mult=40):
    lus, fails, limit = [], 0, count * cap_mult + 1000
    while len(lus) < count and fails < limit:
        lu = builder.build_one()
        if lu is None:
            fails += 1
            continue
        lus.append(lu)
    return lus


def build_field(entities, n, dk_mean=None, *, n_med=6000, chalk_sensitivity=0.30,
                sharp_frac=0.35, overbuild=1.5, overbuild_cap=12000,
                ceiling_tilt=0.75, seed=101):
    """Build a realistic opponent Showdown field of `n` lineups.

    Ownership-aware core (``FieldBuilder``) plus the Classic realism guards:
      * chalk temperature -- overall ownership is reshaped ``own^beta`` for the
        field size before the split, so small fields concentrate on chalk;
      * sharp mix -- ``sharp_frac`` of the build is drawn from the smart
        ``CandidateBuilder`` (correlated, ceiling-seeking) so the field is not a
        too-soft pure-ownership draw;
      * submitted-not-random -- overbuild ``overbuild x n`` and keep the top `n`
        by projection (Captain at 1.5x). Needs ``dk_mean``.
    """
    n = int(n)
    extra = 0
    if dk_mean is not None and overbuild > 1.0:
        extra = min(int(round(n * (overbuild - 1.0))), int(overbuild_cap))
    total = n + extra
    n_sharp = int(round(sharp_frac * total))

    beta = 1.0 - chalk_sensitivity * math.log10(max(n, 1) / max(n_med, 1))
    chalk_entities = _reshape_own(entities, beta)
    chalk_pool = Pool(chalk_entities, ceiling_tilt=ceiling_tilt)
    sharp_pool = Pool(entities, ceiling_tilt=ceiling_tilt)

    lineups = []
    lineups += _draw(CandidateBuilder(sharp_pool, seed=seed + 1), n_sharp)
    lineups += _draw(FieldBuilder(chalk_pool, seed=seed + 2), total - n_sharp)

    if dk_mean is not None and len(lineups) > n:
        lineups.sort(key=lambda lu: lineup_proj(lu, dk_mean), reverse=True)
        lineups = lineups[:n]
    return lineups


def build_candidates(entities, n, *, seed=2025, jitter=0.0, ceiling_tilt=0.75,
                     **kw):
    """Build `n` ownership-blind, correlation-aware candidate lineups."""
    pool = Pool(entities, ceiling_tilt=ceiling_tilt)
    return _draw(CandidateBuilder(pool, seed=seed, jitter=jitter, **kw), n)


def _reshape_own(entities, beta):
    """Chalk temperature over the single Showdown pool (own^beta renormalized)."""
    out = [dict(e) for e in entities]
    owns = [max(float(e.get("own", 0.0)), 1e-3) for e in out]
    tot = sum(owns)
    reshaped = [o ** beta for o in owns]
    rs = sum(reshaped)
    if rs > 0 and tot > 0:
        for e, r in zip(out, reshaped):
            e["own"] = r / rs * tot
    return out


# --------------------------------------------------------------------------- #
def lineups_to_df(lineups):
    import pandas as pd
    rows = []
    for i, lu in enumerate(lineups, 1):
        row = {"Lineup": i, "Salary": lu["salary"],
               "CaptainTeam": lu.get("cap_team", ""), "Split": lu.get("split", "")}
        for c, cell in zip(SLOT_COLS, lu["cells"]):
            row[c] = cell
        rows.append(row)
    return pd.DataFrame(rows)


def dk_upload(chosen_rows, slate, cols=None):
    """Build a DK-importable Showdown upload (CPT + 5 FLEX) from selected rows.

    The CPT cell maps to the player's Captain contest id; FLEX cells map to the
    base contest id. Each row's cells are ``"KEY (TEAM)"``."""
    import pandas as pd
    from portfolio import _split
    cols = cols or SLOT_COLS
    base_id = {e["key"]: e.get("contest_id", "") for e in slate.entities}
    cpt_id = {e["key"]: e.get("cpt_contest_id", e.get("contest_id", ""))
              for e in slate.entities}
    out = []
    for row in chosen_rows:
        ids = []
        for c in cols:
            key, _ = _split(row[c])
            idmap = cpt_id if c == "CPT" else base_id
            ids.append(idmap.get(key, key))
        out.append(ids)
    return pd.DataFrame(out, columns=UPLOAD_HEADER)


if __name__ == "__main__":
    # ---- self-contained self-test (no data files needed) ----
    # a 12-player single game: TeamA A0..A5, TeamB B0..B5 (A0/B0 = QB).
    rng = np.random.default_rng(0)
    ents = []
    for team, opp in (("A", "B"), ("B", "A")):
        for i in range(6):
            pos = "QB" if i == 0 else ("WR" if i in (1, 2, 3) else
                                       "RB" if i == 4 else "DST")
            ents.append({
                "key": f"{team}{i}", "pos": pos, "team": team, "opp": opp,
                "salary": 5000 + 800 * i, "own": float(2 + 3 * (5 - i)),
                "up": float(30 - 3 * i), "contest_id": f"c{team}{i}",
                "cpt_contest_id": f"C{team}{i}", "cpt_salary": int(1.5 * (5000 + 800 * i)),
            })

    # ownership split: Captain mass exactly 1, FLEX exactly 5, cpt_i <= O_i
    split = split_ownership(ents)
    cpt_mass = sum(c for c, _ in split.values())
    flex_mass = sum(f for _, f in split.values())
    assert abs(cpt_mass - 1.0) < 1e-6, cpt_mass
    assert abs(flex_mass - 5.0) < 1e-6, flex_mass
    own = np.array([e["own"] for e in ents]); O = own * 6 / own.sum()
    assert all(split[e["key"]][0] <= O[i] + 1e-9 for i, e in enumerate(ents))

    dk_mean = {e["key"]: e["up"] for e in ents}

    # ownership-aware field: every lineup 1 CPT + 5 FLEX, under cap, both slots ok
    field = build_field(ents, 500, dk_mean, seed=1)
    assert field and all(len(lu["players"]) == 6 for lu in field)
    assert all(lu["salary"] <= SALARY_CAP for lu in field)
    assert all(lu["captain_key"] == lu["players"][0]["key"] for lu in field)

    # ownership-blind candidates: both teams present (no 6-0), QB stacked often
    cands = build_candidates(ents, 500, seed=2)
    assert cands and all(len(set(p["team"] for p in lu["players"])) == 2
                         for lu in cands), "candidate had a one-sided lineup"

    def has_qb_stack(lu):
        for p in lu["players"]:
            if p["pos"] == "QB" and any(
                    q["team"] == p["team"] and q["pos"] in PASS_CATCHER
                    for q in lu["players"] if q["key"] != p["key"]):
                return True
        return False
    stack_rate = np.mean([has_qb_stack(lu) for lu in cands])
    assert stack_rate > 0.6, f"QB-stack rate too low: {stack_rate:.2f}"

    print(f"showdown.py self-test passed: cpt_mass={cpt_mass:.3f} "
          f"flex_mass={flex_mass:.3f}  field={len(field)} cands={len(cands)}  "
          f"cand QB-stack {100*stack_rate:.0f}%  "
          f"cand splits {dict(Counter(lu['split'] for lu in cands))}")
