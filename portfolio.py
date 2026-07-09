#!/usr/bin/env python3
"""
portfolio.py
============
Diversity-aware selection of a lineup PORTFOLIO for export, adapted for NFL from
``DFSSimsFull/portfolio.py``. The selection/EV algorithms are unchanged (rank or
concave-utility greedy, submodular EV coverage, per-entity caps/mins, overlap
ceiling, value groups); only ``lineup_features`` is NFL-aware:

  * primary stack  = the QB's team
  * secondary      = another team contributing >=2 skill players (a game stack /
                     secondary stack)
  * core           = the exact QB + pass-catcher stack (players on the QB's team)
  * the DST slot is the specially-capped slot (``dst_cap``), the other 8 are
    ``skill_cap``.

Cells are ``"KEY (TEAM)"``. Every control is OFF by default (caps=1.0,
overlap=1.0) so the defaults reproduce plain top-N-by-rank.
"""
import math
from collections import Counter, defaultdict

import numpy as np

SLOT_COLS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
DST_SLOT = "DST"


def _split(cell):
    """'KEY (TEAM)' -> ('KEY', 'TEAM'); tolerates a missing/odd team tag."""
    s = str(cell)
    if " (" in s and s.endswith(")"):
        name, team = s.rsplit(" (", 1)
        return name, team[:-1]
    return s, ""


def lineup_features(row, cols=SLOT_COLS):
    """Pull the bits the selector reasons about out of one result row."""
    names, teams = [], []
    for c in cols:
        nm, tm = _split(row[c])
        names.append(nm)
        teams.append(tm)
    # skill slots (everything but DST); QB is cols[0]
    skill = [(c, nm, tm) for c, nm, tm in zip(cols, names, teams) if c != DST_SLOT]
    primary = teams[cols.index("QB")] if "QB" in cols else ""
    # secondary: a non-primary team with >=2 skill players
    tc = Counter(tm for _, _, tm in skill if tm and tm != primary)
    secondary = next((t for t, n in tc.most_common() if n >= 2), "")
    core = frozenset(nm for _, nm, tm in skill if tm == primary and primary)
    return {
        "names": names,
        "playerset": frozenset(names),
        "primary": primary,
        "secondary": secondary,
        "pair": (primary, secondary),
        "core": (primary, core),
    }


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _unmet_mins(player_minn, team_minn, expo, teamc):
    out = []
    for nm, need in player_minn.items():
        if expo[nm] < need:
            out.append({"kind": "player", "name": nm,
                        "have": int(expo[nm]), "need": int(need)})
    for tm, need in team_minn.items():
        if teamc[tm] < need:
            out.append({"kind": "team", "name": tm,
                        "have": int(teamc[tm]), "need": int(need)})
    return out


def _cap_fns(N):
    def cap_n(frac):
        f = float(frac)
        return 0 if f <= 0 else max(1, int(round(f * N)))

    def min_n(frac):
        f = float(frac)
        return 0 if f <= 0 else min(N, int(math.ceil(f * N)))
    return cap_n, min_n


def _dst_index(cols):
    return cols.index(DST_SLOT) if DST_SLOT in cols else -1


# --------------------------------------------------------------------------- #
def select_portfolio(res_df, n_select, sort_cols, *, cols=SLOT_COLS,
                     eligible=None, skill_cap=1.0, dst_cap=1.0,
                     team_cap=1.0, pair_cap=1.0, core_cap=1.0,
                     max_overlap=1.0, group_of=None, group_cap=1.0,
                     player_caps=None, team_caps=None,
                     player_mins=None, team_mins=None):
    """Rank `res_df` by `sort_cols` (desc) then greedily accept lineups that keep
    every exposure cap and the pairwise-overlap ceiling satisfied. Caps are
    fractions of `n_select` (1.0 = off). Returns (chosen_rows, info)."""
    N = int(n_select)
    rdf = res_df.sort_values(list(sort_cols), ascending=False).reset_index(drop=True)
    cap_n, min_n = _cap_fns(N)
    dst_i = _dst_index(cols)

    scap, dcap, tcap = cap_n(skill_cap), cap_n(dst_cap), cap_n(team_cap)
    paircap, ccap, gcap = cap_n(pair_cap), cap_n(core_cap), cap_n(group_cap)
    group_of = group_of or {}
    player_capn = {nm: cap_n(fr) for nm, fr in (player_caps or {}).items()}
    team_capn = {tm: cap_n(fr) for tm, fr in (team_caps or {}).items()}

    def player_cap_for(name, i):
        if name in player_capn:
            return player_capn[name]
        return dcap if i == dst_i else scap

    def team_cap_for(team):
        return team_capn.get(team, tcap)

    player_minn = {nm: min_n(fr) for nm, fr in (player_mins or {}).items()}
    team_minn = {tm: min_n(fr) for tm, fr in (team_mins or {}).items()}
    player_minn = {nm: min(need, player_capn.get(nm, N)) for nm, need in player_minn.items()
                   if need > 0}
    team_minn = {tm: min(need, team_capn.get(tm, N)) for tm, need in team_minn.items()
                 if need > 0}

    expo = Counter(); teamc = Counter(); pairc = Counter()
    corec = Counter(); groupc = Counter(); dsts = set()
    chosen, chosen_sets, chosen_idx, skipped = [], [], set(), 0
    feats = [lineup_features(rdf.iloc[i], cols) for i in range(len(rdf))]

    def fits_maxes(f, names):
        if any(expo[n] >= player_cap_for(n, i) for i, n in enumerate(names)):
            return False
        if teamc[f["primary"]] >= team_cap_for(f["primary"]):
            return False
        if f["secondary"] and pairc[f["pair"]] >= paircap:
            return False
        if corec[f["core"]] >= ccap:
            return False
        if any(groupc[g] >= gcap
               for g in ({group_of[n] for n in names if n in group_of}
                         if group_of else set())):
            return False
        if max_overlap < 1.0 and chosen_sets:
            if max(_jaccard(f["playerset"], s) for s in chosen_sets) > max_overlap:
                return False
        return True

    def accept(pos, row, f, names):
        chosen.append(row); chosen_sets.append(f["playerset"]); chosen_idx.add(pos)
        for i, n in enumerate(names):
            expo[n] += 1
            if i == dst_i:
                dsts.add(n)
        teamc[f["primary"]] += 1
        if f["secondary"]:
            pairc[f["pair"]] += 1
        corec[f["core"]] += 1
        for g in ({group_of[n] for n in names if n in group_of} if group_of else set()):
            groupc[g] += 1

    def deficits_remain():
        return (any(expo[nm] < need for nm, need in player_minn.items()) or
                any(teamc[tm] < need for tm, need in team_minn.items()))

    def helps_deficit(f, names):
        if any(nm in player_minn and expo[nm] < player_minn[nm] for nm in names):
            return True
        return (f["primary"] in team_minn and
                teamc[f["primary"]] < team_minn[f["primary"]])

    # Phase 1: seed minimum-exposure targets in rank order
    if player_minn or team_minn:
        for pos, row in rdf.iterrows():
            if len(chosen) >= N or not deficits_remain():
                break
            f = feats[pos]; names = f["names"]
            if eligible is not None and not eligible(names):
                continue
            if not helps_deficit(f, names) or not fits_maxes(f, names):
                continue
            accept(pos, row, f, names)

    # Phase 2: fill remaining by rank
    for pos, row in rdf.iterrows():
        if len(chosen) >= N:
            break
        if pos in chosen_idx:
            continue
        f = feats[pos]; names = f["names"]
        if eligible is not None and not eligible(names):
            skipped += 1
            continue
        if not fits_maxes(f, names):
            continue
        accept(pos, row, f, names)

    info = _info(chosen, N, skipped, expo, teamc, pairc, corec, dsts,
                 player_minn, team_minn)
    return chosen, info


def select_portfolio_ev(res_df, n_select, pay, util, *, cols=SLOT_COLS,
                        eligible=None, skill_cap=1.0, dst_cap=1.0,
                        team_cap=1.0, pair_cap=1.0, core_cap=1.0,
                        max_overlap=1.0, group_of=None, group_cap=1.0,
                        player_caps=None, team_caps=None,
                        player_mins=None, team_mins=None, eval_sims=None):
    """Greedily build the export set maximizing E[util(portfolio $ return)] under
    the same caps. `pay` is (n_sim, n_row) dollars per candidate per sim; row i
    aligns with pay[:, i]. Returns (chosen_rows, info, W)."""
    N = int(n_select)
    rdf = res_df.reset_index(drop=True)
    n_row = len(rdf)
    pay = np.asarray(pay, dtype=np.float32)
    if pay.shape[1] != n_row:
        raise ValueError(f"pay has {pay.shape[1]} cols but res_df has {n_row} rows")
    n_sim = pay.shape[0]
    cap_n, min_n = _cap_fns(N)
    dst_i = _dst_index(cols)

    if eval_sims and int(eval_sims) < n_sim:
        step = max(1, n_sim // int(eval_sims))
        sel_idx = np.arange(0, n_sim, step)[:int(eval_sims)]
    else:
        sel_idx = np.arange(n_sim)
    pay_sel = pay[sel_idx]

    scap, dcap, tcap = cap_n(skill_cap), cap_n(dst_cap), cap_n(team_cap)
    paircap, ccap, gcap = cap_n(pair_cap), cap_n(core_cap), cap_n(group_cap)
    group_of = group_of or {}
    player_capn = {nm: cap_n(fr) for nm, fr in (player_caps or {}).items()}
    team_capn = {tm: cap_n(fr) for tm, fr in (team_caps or {}).items()}

    def player_cap_for(name, i):
        if name in player_capn:
            return player_capn[name]
        return dcap if i == dst_i else scap

    def team_cap_for(team):
        return team_capn.get(team, tcap)

    player_minn = {nm: min_n(fr) for nm, fr in (player_mins or {}).items()}
    team_minn = {tm: min_n(fr) for tm, fr in (team_mins or {}).items()}
    player_minn = {nm: min(need, player_capn.get(nm, N)) for nm, need in player_minn.items()
                   if need > 0}
    team_minn = {tm: min(need, team_capn.get(tm, N)) for tm, need in team_minn.items()
                 if need > 0}

    feats = [lineup_features(rdf.iloc[i], cols) for i in range(n_row)]
    elig = np.ones(n_row, dtype=bool)
    if eligible is not None:
        for i in range(n_row):
            if not eligible(feats[i]["names"]):
                elig[i] = False
    skipped = int((~elig).sum())

    expo = Counter(); teamc = Counter(); pairc = Counter()
    corec = Counter(); groupc = Counter(); dsts = set()
    chosen_pos, chosen_sets = [], []
    taken = np.zeros(n_row, dtype=bool)

    def gids_of(names):
        return {group_of[n] for n in names if n in group_of} if group_of else set()

    def deficits_remain():
        return (any(expo[nm] < need for nm, need in player_minn.items()) or
                any(teamc[tm] < need for tm, need in team_minn.items()))

    def helps_deficit(f):
        if any(nm in player_minn and expo[nm] < player_minn[nm] for nm in f["names"]):
            return True
        return (f["primary"] in team_minn and
                teamc[f["primary"]] < team_minn[f["primary"]])

    def fits(i):
        f = feats[i]; names = f["names"]
        if any(expo[n] >= player_cap_for(n, j) for j, n in enumerate(names)):
            return False
        if teamc[f["primary"]] >= team_cap_for(f["primary"]):
            return False
        if f["secondary"] and pairc[f["pair"]] >= paircap:
            return False
        if corec[f["core"]] >= ccap:
            return False
        if any(groupc[g] >= gcap for g in gids_of(names)):
            return False
        if max_overlap < 1.0 and chosen_sets:
            if max(_jaccard(f["playerset"], s) for s in chosen_sets) > max_overlap:
                return False
        return True

    W_sel = np.zeros(len(sel_idx), dtype=np.float64)
    cur_u = float(np.mean(util(W_sel)))
    for _ in range(N):
        avail = elig & ~taken
        if not avail.any():
            break
        avail_idx = np.where(avail)[0]
        u_new = util(W_sel[:, None] + pay_sel[:, avail_idx])
        gains = u_new.mean(axis=0) - cur_u
        order = np.argsort(-gains)
        picked = -1
        if deficits_remain():
            for li in order:
                i = int(avail_idx[li])
                if fits(i) and helps_deficit(feats[i]):
                    picked = i
                    break
        if picked < 0:
            for li in order:
                i = int(avail_idx[li])
                if fits(i):
                    picked = i
                    break
        if picked < 0:
            break
        i = picked; f = feats[i]
        chosen_pos.append(i); chosen_sets.append(f["playerset"]); taken[i] = True
        W_sel += pay_sel[:, i]
        cur_u = float(np.mean(util(W_sel)))
        for j, n in enumerate(f["names"]):
            expo[n] += 1
            if j == dst_i:
                dsts.add(n)
        teamc[f["primary"]] += 1
        if f["secondary"]:
            pairc[f["pair"]] += 1
        corec[f["core"]] += 1
        for g in gids_of(f["names"]):
            groupc[g] += 1

    chosen = [rdf.iloc[i] for i in chosen_pos]
    W = (pay[:, chosen_pos].sum(axis=1) if chosen_pos
         else np.zeros(n_sim, dtype=np.float64))
    info = _info(chosen, N, skipped, expo, teamc, pairc, corec, dsts,
                 player_minn, team_minn)
    info.update({
        "exp_return": float(W.mean()),
        "floor_p10": float(np.percentile(W, 10)),
        "median": float(np.percentile(W, 50)),
        "ceiling_p90": float(np.percentile(W, 90)),
        "cash_rate": float(np.mean(W > 0)),
    })
    return chosen, info, W


def _info(chosen, N, skipped, expo, teamc, pairc, corec, dsts,
          player_minn, team_minn):
    return {
        "chosen": len(chosen), "requested": N, "skipped_unmapped": skipped,
        "max_dst": max((expo[n] for n in dsts), default=0),
        "max_skill": max((expo[n] for n in expo if n not in dsts), default=0),
        "max_team": max(teamc.values()) if teamc else 0,
        "max_pair": max(pairc.values()) if pairc else 0,
        "max_core": max(corec.values()) if corec else 0,
        "distinct_pairs": len(pairc),
        "distinct_cores": len(corec),
        "distinct_primaries": len(teamc),
        "player_expo": dict(expo),
        "team_expo": dict(teamc),
        "dsts": sorted(dsts),
        "unmet_mins": _unmet_mins(player_minn, team_minn, expo, teamc),
    }


def detect_value_groups(meta, *, salary_tol=300, proj_tol=1.5, min_size=2):
    """Cluster near-equivalent players (same pos, close salary+projection) so the
    portfolio can spread exposure across them. `meta` is
    {name: {"pos","salary","proj","team"}}. Returns (group_of, groups)."""
    by_pos = defaultdict(list)
    for nm, m in meta.items():
        if m.get("proj") is None:
            continue
        by_pos[m.get("pos", "?")].append(nm)
    group_of, groups, gid = {}, [], 0
    for pos, names in by_pos.items():
        names.sort(key=lambda n: (meta[n]["salary"], meta[n]["proj"]))
        used = set()
        for i, anchor in enumerate(names):
            if anchor in used:
                continue
            cluster = [anchor]
            for other in names[i + 1:]:
                if other in used:
                    continue
                if (abs(meta[other]["salary"] - meta[anchor]["salary"]) <= salary_tol
                        and abs(meta[other]["proj"] - meta[anchor]["proj"]) <= proj_tol):
                    cluster.append(other)
            if len(cluster) >= min_size:
                for c in cluster:
                    used.add(c); group_of[c] = gid
                sals = [meta[c]["salary"] for c in cluster]
                prjs = [meta[c]["proj"] for c in cluster]
                groups.append({"id": gid, "players": cluster, "pos": pos,
                               "salary_lo": min(sals), "salary_hi": max(sals),
                               "proj_lo": min(prjs), "proj_hi": max(prjs)})
                gid += 1
    groups.sort(key=lambda g: len(g["players"]), reverse=True)
    return group_of, groups


if __name__ == "__main__":
    import pandas as pd

    def mk(qb, qbt, mates, rb, wr, te, flex, dst, ranks):
        row = {"QB": f"{qb} ({qbt})"}
        allocs = {"RB1": rb[0], "RB2": rb[1], "WR1": wr[0], "WR2": wr[1],
                  "WR3": wr[2], "TE": te, "FLEX": flex, "DST": dst}
        # place mates onto the QB team by overwriting WR1/WR2/TE as given
        row.update(allocs)
        row.update(ranks)
        return row

    # QB + 2 WR stack on DAL, DST NYG; several near-identical
    rows = []
    for i in range(6):
        rows.append({
            "QB": "q1 (DAL)", "RB1": "r1 (KC)", "RB2": "r2 (MIA)",
            "WR1": "w1 (DAL)", "WR2": "w2 (DAL)", "WR3": "w3 (BUF)",
            "TE": "t1 (SF)", "FLEX": "f1 (KC)", "DST": "d1 (NYG)",
            "Wins": 100 - i, "Top10": 0, "Top100": 0})
    # an alternate build: different QB stack + secondary KC stack
    rows.append({
        "QB": "q2 (PHI)", "RB1": "r3 (KC)", "RB2": "r4 (MIA)",
        "WR1": "w4 (PHI)", "WR2": "w5 (KC)", "WR3": "w6 (KC)",
        "TE": "t2 (PHI)", "FLEX": "f2 (SF)", "DST": "d2 (DEN)",
        "Wins": 40, "Top10": 0, "Top100": 0})
    df = pd.DataFrame(rows)

    f = lineup_features(df.iloc[0])
    assert f["primary"] == "DAL", f
    assert f["core"] == ("DAL", frozenset({"q1", "w1", "w2"})), f

    chosen, info = select_portfolio(df, 5, ["Wins", "Top10", "Top100"])
    assert info["chosen"] == 5 and info["max_core"] == 5, info

    chosen, info = select_portfolio(df, 5, ["Wins", "Top10", "Top100"], core_cap=0.8)
    assert info["max_core"] <= 4 and info["distinct_cores"] >= 2, info

    # secondary stack detection on the alternate build (KC has 2 skill players)
    fa = lineup_features(df.iloc[-1])
    assert fa["primary"] == "PHI" and fa["secondary"] == "KC", fa

    # EV selection: two lineups win sim0, one wins sim1 -> Kelly spreads coverage
    ev = pd.DataFrame([
        {"QB": "a (AAA)", "RB1": "a2 (AAA)", "RB2": "x (XX)", "WR1": "a3 (AAA)",
         "WR2": "a4 (AAA)", "WR3": "y (YY)", "TE": "a5 (AAA)", "FLEX": "z (ZZ)",
         "DST": "da (D1)"},
        {"QB": "b (BBB)", "RB1": "b2 (BBB)", "RB2": "x (XX)", "WR1": "b3 (BBB)",
         "WR2": "b4 (BBB)", "WR3": "y (YY)", "TE": "b5 (BBB)", "FLEX": "z (ZZ)",
         "DST": "db (D2)"},
        {"QB": "c (CCC)", "RB1": "c2 (CCC)", "RB2": "x (XX)", "WR1": "c3 (CCC)",
         "WR2": "c4 (CCC)", "WR3": "y (YY)", "TE": "c5 (CCC)", "FLEX": "z (ZZ)",
         "DST": "dc (D3)"},
    ])
    from portfolio_ev import utility
    pay = np.array([[100., 100., 0.], [0., 0., 100.]])
    _, iC, WC = select_portfolio_ev(ev, 2, pay, utility("Conservative (consistent cashing)"))
    assert iC["cash_rate"] == 1.0, iC
    _, iL, WL = select_portfolio_ev(ev, 2, pay, utility("Aggressive (max ceiling)"))
    assert iL["cash_rate"] == 0.5, iL
    print("portfolio.py self-test passed:", info)
