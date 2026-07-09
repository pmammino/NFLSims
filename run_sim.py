#!/usr/bin/env python3
"""
run_sim.py
==========
End-to-end NFL DFS simulation — the orchestrator (analog of the MLB engine's
``run_full`` + ``stage_d``). Wires the stages:

  ingest -> correlated sim -> candidates + field(s) -> contest scoring ->
  portfolio/EV selection -> exports

Outputs land in ``out/``:
  player_dk_sims.npy         {entity_key: DK points array}  (stage boundary)
  player_projections.csv     the player table
  candidates.csv             the candidate lineup set
  field_<N>.csv              a simulated opponent field per contest size
  candidate_results_<N>.csv  candidate finish rates vs that field
  DK_upload_<N>.csv          the selected export set (DK contest ids)

Example:
  python3 run_sim.py --n-sims 10000 --contest-sizes 1000 6000 20000 \
          --num-candidates 10000 --select 20 --objective ev
"""
import argparse
import os

import numpy as np
import pandas as pd

import nfl_ingest
import sim_engine
import field_builder as fb
import field_simulator as fs
import contest_sim
import exports
import portfolio
import portfolio_ev as pev


def _build_set(builder, n, cap_mult=40):
    lus, fails = [], 0
    limit = n * cap_mult + 1000
    while len(lus) < n and fails < limit:
        lu = builder.build_one()
        if lu is None:
            fails += 1
            continue
        lus.append(lu)
    return lus, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--contest-sizes", nargs="+", type=int, default=[1000, 6000, 20000])
    ap.add_argument("--num-candidates", type=int, default=10000)
    ap.add_argument("--medium", type=int, default=6000)
    ap.add_argument("--chalk-sensitivity", type=float, default=0.30)
    ap.add_argument("--stack-tilt", type=float, default=0.12)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="lognormal shock on candidate selection weights (diversify)")
    ap.add_argument("--params", default="field_params_nfl.json")
    ap.add_argument("--seed-field", type=int, default=101)
    ap.add_argument("--seed-candidates", type=int, default=2025)
    ap.add_argument("--outdir", default="out")
    # portfolio / export
    ap.add_argument("--select", type=int, default=0, help="export N lineups (0=skip)")
    ap.add_argument("--objective", choices=["win", "top10", "top100", "ev"],
                    default="top100")
    ap.add_argument("--from-size", type=int, default=None)
    ap.add_argument("--entry-fee", type=float, default=20.0)
    ap.add_argument("--utility", default="Balanced")
    ap.add_argument("--skill-cap", type=float, default=1.0)
    ap.add_argument("--dst-cap", type=float, default=1.0)
    ap.add_argument("--team-cap", type=float, default=1.0)
    ap.add_argument("--max-overlap", type=float, default=1.0)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    # ---- ingest + field params (learn from standings if not yet built) ----
    slate = nfl_ingest.build_slate()
    if not os.path.exists(a.params):
        import glob
        import learn_field
        standings = sorted(glob.glob("contest-standings-*.csv"))
        if standings:
            learn_field.learn(standings, a.params)
    params = fb.load_params(a.params)
    print(f"slate: {len(slate.players)} offense "
          f"({sum(p['matched'] for p in slate.players)} matched), "
          f"{len(slate.dst)} DST, {len(slate.teams)} teams, {len(slate.games)} games")

    # ---- correlated simulation ----
    sim = sim_engine.simulate(slate, n_sims=a.n_sims, seed=a.seed)
    np.save(os.path.join(a.outdir, "player_dk_sims.npy"),
            {k: v for k, v in sim.dk.items()}, allow_pickle=True)
    ptab = exports.player_table(sim, slate)
    ptab.to_csv(os.path.join(a.outdir, "player_projections.csv"), index=False)
    rc = sim_engine.realized_correlations(sim, slate)
    print(f"sim: {a.n_sims} sims  QB-WR corr {rc['qb_wr_same']:.2f}  "
          f"WR-WR corr {rc['wr_wr_same']:.2f}")

    # ---- candidate set (uniform, stack-structured, ownership-blind) ----
    cpool = fb.Pool(slate.entities)
    cb = fb.Builder(cpool, params, seed=a.seed_candidates, uniform=True, jitter=a.jitter)
    cands, cfails = _build_set(cb, a.num_candidates)
    cand_df = fb.lineups_to_df(cands)
    cand_df.to_csv(os.path.join(a.outdir, "candidates.csv"), index=False)
    cand_mat = contest_sim.score_matrix(cands, sim.dk, a.n_sims)
    print(f"candidates: {len(cands)} built ({cfails} fails), scored")

    # ---- field per contest size + contest scoring ----
    results_by_size, field_mat_by_size = {}, {}
    for N in a.contest_sizes:
        adj, p_sz, beta = fs.prepare_field_pool(
            slate.entities, params, N, n_med=a.medium,
            chalk_sensitivity=a.chalk_sensitivity, stack_tilt=a.stack_tilt)
        fbuild = fb.Builder(fb.Pool(adj), p_sz, seed=a.seed_field + N, uniform=False)
        field, ffails = _build_set(fbuild, N)
        fb.lineups_to_df(field).to_csv(os.path.join(a.outdir, f"field_{N}.csv"), index=False)
        field_mat = contest_sim.score_matrix(field, sim.dk, a.n_sims)
        field_mat_by_size[N] = field_mat
        wins, t10, t100, avg = contest_sim.run_contest(field_mat, cand_mat, a.n_sims, N)
        res = cand_df.copy()
        res.insert(0, "Candidate", np.arange(1, len(cands) + 1))
        res["Wins"] = wins
        res["Win%"] = np.round(100 * wins / a.n_sims, 3)
        res["Top10"] = t10
        res["Top10%"] = np.round(100 * t10 / a.n_sims, 2)
        res["Top100"] = t100
        res["Top100%"] = np.round(100 * t100 / a.n_sims, 2)
        res["AvgPlace"] = np.round(avg, 1)
        res = res.sort_values(["Wins", "Top10", "Top100", "AvgPlace"],
                              ascending=[False, False, False, True])
        res.to_csv(os.path.join(a.outdir, f"candidate_results_{N}.csv"), index=False)
        results_by_size[N] = res
        print(f"[field {N:>6}] beta {beta:.2f}  fails {ffails}  "
              f"best Win% {res['Win%'].max():.2f}  best Top100% {res['Top100%'].max():.1f}")

    # ---- optional portfolio selection / DK upload ----
    if a.select > 0:
        size = a.from_size or a.medium
        if size not in results_by_size:
            size = a.contest_sizes[0]
        _select_and_upload(a, slate, cands, cand_mat, results_by_size[size],
                           field_mat_by_size[size], size)

    print(f"done -> {a.outdir}/")


def _select_and_upload(a, slate, cands, cand_mat, res, field_mat, size):
    if a.objective == "ev":
        prize = pev.make_payout_curve(size, a.entry_fee)
        cut_places = pev.field_place_cutpoints(size)
        fs_desc = -np.sort(-field_mat, axis=1)                 # (n_sim, size) desc
        field_cut = fs_desc[:, np.clip(cut_places - 1, 0, fs_desc.shape[1] - 1)]
        # align payout matrix rows to res's candidate order
        order = res["Candidate"].to_numpy() - 1
        cand_scores = cand_mat[:, order]
        pay = pev.candidate_payout_matrix(cand_scores, field_cut, cut_places, prize)
        chosen, info, W = portfolio.select_portfolio_ev(
            res, a.select, pay, pev.utility(a.utility),
            skill_cap=a.skill_cap, dst_cap=a.dst_cap, team_cap=a.team_cap,
            max_overlap=a.max_overlap)
        n = max(info["chosen"], 1)
        print(f"EV export: {info['chosen']} lineups  exp ${info['exp_return']:.0f} total "
              f"(${info['exp_return']/n:.2f}/entry vs ${a.entry_fee:.0f})  "
              f"portfolio-cash {100*info['cash_rate']:.1f}%  p90 ${info['ceiling_p90']:.0f}")
    else:
        keymap = {"win": ["Wins", "Top10", "Top100"],
                  "top10": ["Top10", "Top100", "Wins"],
                  "top100": ["Top100", "Top10", "Wins"]}[a.objective]
        chosen, info = portfolio.select_portfolio(
            res, a.select, keymap, skill_cap=a.skill_cap, dst_cap=a.dst_cap,
            team_cap=a.team_cap, max_overlap=a.max_overlap)
        print(f"ranked export ({a.objective}): {info['chosen']} lineups  "
              f"max team {info['max_team']}  distinct stacks {info['distinct_cores']}")
    up = exports.dk_upload(chosen, slate)
    path = os.path.join(a.outdir, f"DK_upload_{size}.csv")
    up.to_csv(path, index=False)
    print(f"  wrote {path}  (unmet mins: {info['unmet_mins']})")


if __name__ == "__main__":
    main()
