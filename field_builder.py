#!/usr/bin/env python3
"""
field_builder.py
================
QB-centric DraftKings **NFL Classic** lineup builder — the NFL analog of
``DFSSimsFull/mlb_lineup_builder.py``. Builds realistic, ownership- and
stacking-aware lineups following the construction grammar the real GPP field
uses:

    QB  +  k pass-catchers from the QB's team  (the "primary stack")
        +  b players brought back from the QB's opponent (the "game stack")
        +  fills from other games
        +  DST (usually NOT the defense facing your own stack)

Roster (fixed): QB, RB, RB, WR, WR, WR, TE, FLEX(RB/WR/TE), DST — $50,000 cap.

The stack-shape / bring-back / FLEX distributions live in ``field_params_nfl.json``
(see ``learn_field.py``); ``DEFAULT_PARAMS`` below is used if the file is absent.
Selection weights are projected ownership, exactly as in the MLB builder; set
``uniform=True`` (candidates) to explore ownership-blind, or pass explicit
``team_weights``. ``jitter`` adds a per-draw lognormal shock to diversify a
portfolio.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}   # + 1 FLEX
FLEX_POS = ("RB", "WR", "TE")
SALARY_CAP = 50000
MIN_SALARY = 46000            # real lineups spend most of the cap
PASS_CATCHER = ("WR", "TE")

DEFAULT_PARAMS = {
    # primary stack = QB + k pass-catchers; probabilities over k
    "stack_sizes": [[0, 0.15], [1, 0.33], [2, 0.37], [3, 0.13], [4, 0.02]],
    # bring-back size (opponent players) when there is a stack (k>=1)
    "bringback": [[0, 0.45], [1, 0.42], [2, 0.13]],
    # position filling the FLEX slot
    "flex_pos": [["RB", 0.45], ["WR", 0.42], ["TE", 0.13]],
    "rules": {
        # probability the rostered DST is the one facing your primary stack
        "dst_vs_own_stack_prob": 0.08,
    },
}


def load_params(path="field_params_nfl.json"):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return DEFAULT_PARAMS


# --------------------------------------------------------------------------- #
class Pool:
    """Indexes the slate's playable entities for weighted lineup construction."""

    def __init__(self, entities):
        # entities: list of dicts with key, pos, team, opp, salary, own
        self.rows = []
        for e in entities:
            self.rows.append({
                "key": e["key"], "pos": e["pos"], "team": e.get("team", ""),
                "opp": e.get("opp", ""), "salary": int(e["salary"]),
                "own": max(float(e.get("own", 0.0)), 0.001),
            })
        self.by_pos = defaultdict(list)
        self.qbs = []
        self.dsts = []
        self.team_pass = defaultdict(list)     # team -> [WR/TE]
        self.team_skill = defaultdict(list)    # team -> [RB/WR/TE]
        for r in self.rows:
            self.by_pos[r["pos"]].append(r)
            if r["pos"] == "QB":
                self.qbs.append(r)
            elif r["pos"] == "DST":
                self.dsts.append(r)
            else:
                if r["team"]:
                    self.team_skill[r["team"]].append(r)
                    if r["pos"] in PASS_CATCHER:
                        self.team_pass[r["team"]].append(r)


def wchoice(rng, items, weights, jitter=0.0):
    w = np.asarray(weights, dtype=float)
    if jitter:
        w = w * np.exp(jitter * rng.standard_normal(len(w)))
    if w.sum() <= 0:
        w = np.ones_like(w)
    return items[rng.choice(len(items), p=w / w.sum())]


class Builder:
    def __init__(self, pool, params, seed=None, uniform=False,
                 team_weights=None, jitter=0.0):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.uniform = uniform
        self.team_weights = team_weights
        self.jitter = float(jitter)
        self.stack_k = _dist(params["stack_sizes"], self.rng)
        self.bringback = _dist(params["bringback"], self.rng)
        self.flex_pos = _dist(params["flex_pos"], self.rng)
        self.rules = params.get("rules", DEFAULT_PARAMS["rules"])

    # ---- weighted draws honoring uniform / team_weights / jitter ---- #
    def _w(self, rows, team_weight=False):
        if self.uniform:
            base = [1.0 for _ in rows]
        elif team_weight and self.team_weights is not None:
            base = [max(self.team_weights.get(r["team"], 1e-6), 1e-6) for r in rows]
        else:
            base = [r["own"] for r in rows]
        return base

    def _pick(self, rows, team_weight=False):
        return wchoice(self.rng, rows, self._w(rows, team_weight), self.jitter)

    def build_one(self, max_tries=300):
        for _ in range(max_tries):
            lu = self._attempt()
            if lu is not None:
                return lu
        return None

    def _attempt(self):
        rng = self.rng
        pool = self.pool
        need = dict(ROSTER)                       # remaining per-position counts
        flex_open = 1
        flex_target = self.flex_pos()             # desired FLEX position
        used = set()                              # entity keys
        chosen = []                               # list of (slot, row)

        def add(row, slot):
            nonlocal flex_open
            chosen.append((slot, row))
            used.add(row["key"])
            if slot == "FLEX":
                flex_open -= 1
            else:
                need[slot] -= 1

        def place(row):
            """Assign a skill player (RB/WR/TE) to its position slot or FLEX."""
            p = row["pos"]
            if need.get(p, 0) > 0:
                add(row, p)
                return True
            if flex_open and p in FLEX_POS:
                add(row, "FLEX")
                return True
            return False

        # ---- QB + primary stack ----
        if not pool.qbs:
            return None
        qb = self._pick(pool.qbs, team_weight=True)
        add(qb, "QB")
        qb_team, qb_opp = qb["team"], qb["opp"]

        k = self.stack_k()
        mates = [r for r in pool.team_pass.get(qb_team, []) if r["key"] not in used]
        k = min(k, len(mates))
        for _ in range(k):
            avail = [r for r in mates if r["key"] not in used]
            avail = [r for r in avail if need.get(r["pos"], 0) > 0 or
                     (flex_open and r["pos"] in FLEX_POS)]
            if not avail:
                break
            r = self._pick(avail)
            place(r)

        # ---- bring-back from the QB's opponent ----
        b = self.bringback() if k >= 1 else 0
        if b and qb_opp:
            opp_skill = [r for r in pool.team_skill.get(qb_opp, [])
                         if r["key"] not in used]
            for _ in range(min(b, len(opp_skill))):
                avail = [r for r in opp_skill if r["key"] not in used and
                         (need.get(r["pos"], 0) > 0 or
                          (flex_open and r["pos"] in FLEX_POS))]
                if not avail:
                    break
                place(self._pick(avail))

        # ---- fill the remaining position slots ----
        # Exclude the QB's team so the primary stack stays exactly `k` (its
        # high-owned pass-catchers would otherwise get re-drawn into WR fills).
        def fillable(pos):
            return [r for r in pool.by_pos.get(pos, [])
                    if r["key"] not in used and r["team"] != qb_team]

        for pos in ("RB", "WR", "TE"):
            while need.get(pos, 0) > 0:
                avail = fillable(pos)
                if not avail:
                    return None
                add(self._pick(avail), pos)

        # ---- FLEX (respect the sampled target position when possible) ----
        if flex_open:
            pref = fillable(flex_target)
            avail = pref or [r for p in FLEX_POS for r in fillable(p)]
            if not avail:
                return None
            add(self._pick(avail), "FLEX")

        # ---- DST (avoid the defense facing your own stack, most of the time) ----
        dsts = [r for r in pool.dsts if r["key"] not in used]
        if not dsts:
            return None
        avoid_vs = rng.random() >= self.rules.get("dst_vs_own_stack_prob", 0.08)
        stack_teams = {qb_team}
        if avoid_vs:
            filt = [r for r in dsts if r["opp"] not in stack_teams]
            if filt:
                dsts = filt
        add(self._pick(dsts, team_weight=True), "DST")

        # ---- feasibility: full roster + salary window ----
        if any(v != 0 for v in need.values()) or flex_open != 0:
            return None
        total = sum(r["salary"] for _, r in chosen)
        if total > SALARY_CAP or total < MIN_SALARY:
            return None
        return self._format(chosen, total, qb_team)

    def _format(self, chosen, total, qb_team):
        order = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
        byslot = defaultdict(list)
        for slot, r in chosen:
            byslot[slot].append(r)
        players, seen = [], Counter()
        for s in order:
            players.append(byslot[s][seen[s]])
            seen[s] += 1
        # primary stack = QB's pass-catchers/skill teammates (excludes the QB)
        stack = sum(1 for _, r in chosen if r["team"] == qb_team
                    and r["pos"] in FLEX_POS)
        return {
            "players": players, "salary": total, "qb_team": qb_team,
            "stack": stack,
            "cells": [f"{r['key']} ({r['team']})" for r in players],
        }


class _Sampler:
    """Weighted categorical sampler bound to a specific numpy Generator."""

    def __init__(self, vals, w, rng):
        self.vals = vals
        self.w = w
        self._rng = rng

    def __call__(self):
        return self.vals[self._rng.choice(len(self.vals), p=self.w)]


def _dist(pairs, rng):
    """Sampler over the first element of each `pairs` entry, by its weight."""
    vals = [p[0] for p in pairs]
    w = np.array([p[1] for p in pairs], dtype=float)
    return _Sampler(vals, w / w.sum(), rng)


def lineups_to_df(lineups):
    import pandas as pd
    cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
    rows = []
    for i, lu in enumerate(lineups, 1):
        row = {"Lineup": i, "Salary": lu["salary"], "QBstack": lu["stack"]}
        for c, cell in zip(cols, lu["cells"]):
            row[c] = cell
        rows.append(row)
    return pd.DataFrame(rows)


SLOT_COLS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]


if __name__ == "__main__":
    import nfl_ingest
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--uniform", action="store_true")
    a = ap.parse_args()

    slate = nfl_ingest.build_slate()
    params = load_params()
    pool = Pool(slate.entities)
    b = Builder(pool, params, seed=a.seed, uniform=a.uniform)
    lus, fails = [], 0
    while len(lus) < a.n and fails < a.n * 30 + 500:
        lu = b.build_one()
        if lu is None:
            fails += 1
            continue
        lus.append(lu)
    print(f"built {len(lus)} lineups, {fails} failed attempts")
    stacks = Counter(lu["stack"] for lu in lus)
    sal = [lu["salary"] for lu in lus]
    print("QB primary-stack size dist:",
          {k: f"{100*v/len(lus):.1f}%" for k, v in sorted(stacks.items())})
    print(f"salary mean {np.mean(sal):.0f}  min {min(sal)}  max {max(sal)}")
