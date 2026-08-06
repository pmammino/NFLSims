#!/usr/bin/env python3
"""
STAGE 4 / 4 — CONTEST SCORING
=============================
Place the CANDIDATES against the FIELD across every simulation and report how
often each candidate finishes 1st / top-10 / top-100, plus (optionally) select a
payout-aware export set and write a DK-uploadable CSV.

This is the stage that turns "a pile of lineups + a pile of sims" into an answer.
It reads the three earlier stages' file outputs and re-simulates NOTHING — a
lineup's per-sim score is just the sum of its nine players' saved DK arrays.

HOW A CONTEST IS SCORED
-----------------------
For each simulation, every field lineup and every candidate gets a fantasy total
(sum of its players' points that sim). A candidate's PLACE that sim is its rank
inside the sorted field. Win% / Top10% / Top100% are the share of sims a
candidate lands in those places. Because the sims are correlated, the same chalk
stack tends to win the same sims — which is why the optional EV selection uses a
concave utility to spread the exported set's winning sims across slate outcomes
(``portfolio`` / ``portfolio_ev``).

INPUTS  (from ``--indir``, produced by the earlier stages)
----------------------------------------------------------
  player_dk_sims.npy   {key: array[n_sims]} — from simulate.py
  candidates.csv       our lineups          — from build_candidates.py
  field_<N>.csv        opponent field(s)    — from build_field.py (one per size)
  player_pool.csv      key -> contest_id / name (for the DK upload + readable EV)

OUTPUTS  (to ``--outdir``, default same as indir)
-------------------------------------------------
  candidate_results_<N>.csv  candidates + Win% / Top10% / Top100% / AvgPlace vs
                             the N-entry field (sorted best-first)
  DK_upload_<N>.csv          [only with --select] the chosen export set as DK
                             contest ids, ready to import

USAGE
-----
  python3 score_contest.py                                   # score every field_<N>.csv found
  python3 score_contest.py --sizes 6000
  python3 score_contest.py --sizes 6000 --select 20 --objective ev --entry-fee 20
  python3 score_contest.py --sizes 6000 --select 20 --objective top100 \
          --skill-cap 0.5 --max-overlap 0.7
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

import contest_sim
import portfolio
import portfolio_ev as pev
from portfolio import SLOT_COLS, _split


def load_sims(indir):
    path = os.path.join(indir, "player_dk_sims.npy")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run simulate.py first")
    dk = np.load(path, allow_pickle=True).item()
    n = len(next(iter(dk.values()))) if dk else 0
    return dk, int(n)


def lineups_from_df(df):
    """Rebuild builder-style lineup dicts ({'players': [{'key':...}, ...]}) from a
    field/candidate CSV whose slot cells are ``KEY (TEAM)``."""
    out = []
    for row in df.to_dict("records"):
        out.append({"players": [{"key": _split(row[c])[0]} for c in SLOT_COLS]})
    return out


def _sizes_present(indir, sizes):
    if sizes:
        return sizes
    found = []
    for p in glob.glob(os.path.join(indir, "field_*.csv")):
        m = re.search(r"field_(\d+)\.csv$", p)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def _select_and_upload(a, indir, outdir, cand_df, cand_lineups, cand_mat,
                       field_mat, size):
    """Optional export: pick `--select` lineups maximizing the objective under
    exposure caps, then write them as DK contest ids."""
    res = cand_df.copy()
    res.insert(0, "Candidate", np.arange(1, len(cand_df) + 1))
    # attach the finish rates so the ranked objectives have something to sort on
    n_sim = cand_mat.shape[0]
    wins, t10, t100, avg = contest_sim.run_contest(field_mat, cand_mat, n_sim, size)
    res["Wins"], res["Top10"], res["Top100"] = wins, t10, t100
    if a.objective == "ev":
        prize = pev.make_payout_curve(size, a.entry_fee)
        cut = pev.field_place_cutpoints(size)
        fs_desc = -np.sort(-field_mat, axis=1)
        field_cut = fs_desc[:, np.clip(cut - 1, 0, fs_desc.shape[1] - 1)]
        pay = pev.candidate_payout_matrix(cand_mat, field_cut, cut, prize)
        chosen, info, W = portfolio.select_portfolio_ev(
            res, a.select, pay, pev.utility(a.utility),
            skill_cap=a.skill_cap, dst_cap=a.dst_cap, team_cap=a.team_cap,
            max_overlap=a.max_overlap)
        n = max(info["chosen"], 1)
        print(f"  EV export: {info['chosen']} lineups  exp ${info['exp_return']:.0f} "
              f"(${info['exp_return']/n:.2f}/entry vs ${a.entry_fee:.0f})  "
              f"cash {100*info['cash_rate']:.1f}%  p90 ${info['ceiling_p90']:.0f}")
    else:
        keymap = {"win": ["Wins", "Top10", "Top100"],
                  "top10": ["Top10", "Top100", "Wins"],
                  "top100": ["Top100", "Top10", "Wins"]}[a.objective]
        chosen, info = portfolio.select_portfolio(
            res, a.select, keymap, skill_cap=a.skill_cap, dst_cap=a.dst_cap,
            team_cap=a.team_cap, max_overlap=a.max_overlap)
        print(f"  ranked export ({a.objective}): {info['chosen']} lineups  "
              f"max team {info['max_team']}  distinct stacks {info['distinct_cores']}")

    # map keys -> DK contest ids from player_pool.csv
    pool = pd.read_csv(os.path.join(indir, "player_pool.csv"))
    cid = {str(r["key"]): str(r.get("contest_id", "")) for r in pool.to_dict("records")}
    header = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    rows = [[cid.get(_split(row[c])[0], _split(row[c])[0]) for c in SLOT_COLS]
            for row in chosen]
    up_path = os.path.join(outdir, f"DK_upload_{size}.csv")
    pd.DataFrame(rows, columns=header).to_csv(up_path, index=False)
    print(f"  wrote {up_path}  (unmet mins: {info['unmet_mins']})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="*", type=int, default=None,
                    help="contest sizes to score (default: every field_<N>.csv found)")
    ap.add_argument("--select", type=int, default=0, help="export N lineups (0=skip)")
    ap.add_argument("--objective", choices=["win", "top10", "top100", "ev"],
                    default="top100")
    ap.add_argument("--from-size", type=int, default=None,
                    help="which field size to select the export from (default: first)")
    ap.add_argument("--entry-fee", type=float, default=20.0)
    ap.add_argument("--utility", default="Balanced")
    ap.add_argument("--skill-cap", type=float, default=1.0)
    ap.add_argument("--dst-cap", type=float, default=1.0)
    ap.add_argument("--team-cap", type=float, default=1.0)
    ap.add_argument("--max-overlap", type=float, default=1.0)
    ap.add_argument("--indir", default="out")
    ap.add_argument("--outdir", default=None, help="defaults to --indir")
    a = ap.parse_args()
    outdir = a.outdir or a.indir
    os.makedirs(outdir, exist_ok=True)

    dk, n_sims = load_sims(a.indir)
    cand_path = os.path.join(a.indir, "candidates.csv")
    if not os.path.exists(cand_path):
        raise SystemExit(f"missing {cand_path} — run build_candidates.py first")
    cand_df = pd.read_csv(cand_path)
    cand_lineups = lineups_from_df(cand_df)
    cand_mat = contest_sim.score_matrix(cand_lineups, dk, n_sims)
    print(f"scoring {len(cand_lineups)} candidates over {n_sims} sims")

    sizes = _sizes_present(a.indir, a.sizes)
    if not sizes:
        raise SystemExit("no field_<N>.csv found — run build_field.py first")

    field_mats = {}
    for N in sizes:
        fpath = os.path.join(a.indir, f"field_{N}.csv")
        if not os.path.exists(fpath):
            print(f"  (skip {N}: {fpath} not found)")
            continue
        field_lineups = lineups_from_df(pd.read_csv(fpath))
        field_mat = contest_sim.score_matrix(field_lineups, dk, n_sims)
        field_mats[N] = field_mat
        wins, t10, t100, avg = contest_sim.run_contest(
            field_mat, cand_mat, n_sims, len(field_lineups))
        res = cand_df.copy()
        res.insert(0, "Candidate", np.arange(1, len(cand_df) + 1))
        res["Win%"] = np.round(100 * wins / n_sims, 3)
        res["Top10%"] = np.round(100 * t10 / n_sims, 2)
        res["Top100%"] = np.round(100 * t100 / n_sims, 2)
        res["AvgPlace"] = np.round(avg, 1)
        res = res.sort_values(["Win%", "Top10%", "Top100%", "AvgPlace"],
                              ascending=[False, False, False, True])
        rpath = os.path.join(outdir, f"candidate_results_{N}.csv")
        res.to_csv(rpath, index=False)
        print(f"[field {N:>6}] best Win% {res['Win%'].max():.2f}  "
              f"best Top100% {res['Top100%'].max():.1f}  -> {rpath}")

    if a.select > 0 and field_mats:
        size = a.from_size if (a.from_size in field_mats) else sorted(field_mats)[0]
        _select_and_upload(a, a.indir, outdir, cand_df, cand_lineups, cand_mat,
                           field_mats[size], size)

    print(f"done -> {outdir}/")


if __name__ == "__main__":
    main()
