#!/usr/bin/env python3
"""
app.py — Streamlit interface for the NFL DFS contest simulator
==============================================================
A point-and-click front end over the NFL engine (``run_sim.py``). Unlike the
MLB app it runs the whole pipeline in-process (the sim is seconds, not a
multi-stage external rebuild):

  ingest (projections.csv + ownership.csv + schedule.csv)
    -> correlated sim -> candidates + ownership-weighted field
    -> DraftKings contest scoring -> portfolio / EV export

Tabbed workspace (RotoWire full-dark theme):
  Setup   — slate summary, sim + contest controls, Run
  Players — the player projection table + a player's score distribution
  Results — candidate finish rates vs the simulated field + place charts
  Export  — diversity / payout-EV lineup selection + DK upload download

Launch:  streamlit run app.py
"""
import json
import os
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import nfl_ingest
import sim_engine
import field_builder as fb
import field_simulator as fs
import contest_sim
import exports
import portfolio
import portfolio_ev as pev

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
LOCKUP = os.path.join(ASSETS, "rotowire_lockup.svg")
LOGO = os.path.join(ASSETS, "logo.svg")
PROJ_CSV = os.path.join(HERE, "projections.csv")
OWN_CSV = os.path.join(HERE, "ownership.csv")
SCHED_CSV = os.path.join(HERE, "schedule.csv")
PARAMS_PATH = os.path.join(HERE, "field_params_nfl.json")
OUTDIR = os.path.join(HERE, "out")

# RotoWire palette (mirror of the MLB app tokens)
RW_PURPLE = "#a020fe"
RW_TURF = "#00e657"
RW_LEMON = "#d9fc07"
RW_KETCHUP = "#ff4537"
RW_MUT = "#8f8f99"
RW_LINE = "#26262c"
RW_SURFACE = "#0f0f12"

st.set_page_config(page_title="NFL DFS Contest Simulator",
                   page_icon="🏈", layout="wide")

try:
    st.logo(LOCKUP if os.path.exists(LOCKUP) else LOGO)
except Exception:
    pass

st.markdown("""
<style>
  @font-face{font-family:'Integral CF';src:url('app/static/fonts/Integral-Heavy.otf') format('opentype'),url('app/static/fonts/Integral-Heavy.ttf') format('truetype');font-weight:700;font-display:swap;}
  @font-face{font-family:'Cosmica';src:url('app/static/fonts/Cosmica-Regular.otf') format('opentype');font-weight:400;font-display:swap;}
  @font-face{font-family:'Cosmica';src:url('app/static/fonts/Cosmica-Semibold.otf') format('opentype');font-weight:600;font-display:swap;}
  @font-face{font-family:'Cosmica';src:url('app/static/fonts/Cosmica-Heavy.otf') format('opentype');font-weight:700;font-display:swap;}
  @font-face{font-family:'Cosmica Mono';src:url('app/static/fonts/CosmicaMono-Semibold.otf') format('opentype');font-weight:600;font-display:swap;}

  :root{
    --rw-purple:#a020fe; --rw-purple-400:#b34dfe; --rw-purple-700:#7217b4;
    --rw-ink:#0b0b0d; --rw-surface:#0f0f12; --rw-raised:#16161a; --rw-line:#26262c;
    --rw-mut:#8f8f99; --rw-turf:#00e657; --rw-ketchup:#ff4537; --rw-lemon:#d9fc07;
    --font-body:'Cosmica',ui-sans-serif,system-ui,'Segoe UI',sans-serif;
    --font-display:'Integral CF','Impact',system-ui,sans-serif;
    --font-mono:'Cosmica Mono',ui-monospace,Menlo,Consolas,monospace;
  }
  html, body, .stApp, [data-testid="stAppViewContainer"],
  p, span, div, label, input, textarea, button, select, li, td, th { font-family: var(--font-body); }
  .stApp { background: var(--rw-ink); }
  h1, h2, h3, [data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
    font-family: var(--font-display) !important; text-transform: uppercase; letter-spacing:.02em; color:#fff; }
  [data-testid="stMetric"]{ background: var(--rw-surface); border:1px solid var(--rw-line); border-radius:12px; padding:14px 16px; }
  [data-testid="stMetricLabel"] p{ font-family: var(--font-mono); text-transform:uppercase; letter-spacing:.06em; font-size:10px !important; color: var(--rw-mut); }
  [data-testid="stMetricValue"]{ font-family: var(--font-display); color:#fff; font-size:30px; }
  .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--rw-line); }
  .stTabs [data-baseweb="tab"]{ font-family: var(--font-display); text-transform:uppercase; letter-spacing:.04em; font-size:13px; color: var(--rw-mut); padding:6px 14px; }
  .stTabs [aria-selected="true"]{ color:#fff !important; }
  .stTabs [data-baseweb="tab-highlight"]{ background-color: var(--rw-purple) !important; height:3px; }
  .stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"]>button{ font-family: var(--font-body); font-weight:600; border-radius:8px; border:1px solid var(--rw-line); }
  [data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"]>button{ background: var(--rw-purple) !important; border-color: var(--rw-purple) !important; color:#fff !important; }
  [data-testid="stBaseButton-primary"]:hover, [data-testid="stFormSubmitButton"]>button:hover{ background: var(--rw-purple-400) !important; border-color: var(--rw-purple-400) !important; }
  [data-testid="stExpander"]{ background: var(--rw-surface); border:1px solid var(--rw-line); border-radius:12px; }
  [data-baseweb="input"], [data-baseweb="select"]>div, .stTextInput input, .stNumberInput input{ background: var(--rw-raised) !important; border-radius:8px !important; }
  [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"]{ border:1px solid var(--rw-line); border-radius:10px; }
  .stProgress > div > div > div > div { background-color: var(--rw-purple); }
  a, a:visited { color: var(--rw-purple-400); }
  hr { border-top:1px solid var(--rw-line); }
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:#2b2b31;border-radius:8px}
  ::-webkit-scrollbar-track{background:transparent}
  .rw-header{display:flex;align-items:center;gap:16px;padding:14px 18px;margin:2px 0 6px;
    background:var(--rw-surface);border:1px solid var(--rw-line);border-radius:14px;}
  .rw-header .rw-wordmark{height:30px;flex-shrink:0;display:flex;align-items:center;}
  .rw-header .rw-wordmark svg{height:30px;width:auto;display:block;}
  .rw-header .rw-divider{width:1px;height:34px;background:var(--rw-line);flex-shrink:0;}
  .rw-title{font-family:var(--font-display);text-transform:uppercase;letter-spacing:.02em;font-size:26px;line-height:1;color:#fff;}
  .rw-eyebrow{font-family:var(--font-mono);text-transform:uppercase;letter-spacing:.08em;font-size:10px;color:var(--rw-mut);margin-top:4px;}
  .rw-badge{margin-left:auto;font-family:var(--font-mono);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
    background:var(--rw-lemon);color:#3a3800;padding:6px 12px;border-radius:9999px;display:inline-flex;align-items:center;gap:7px;}
  .rw-badge .dot{width:7px;height:7px;border-radius:9999px;background:#3a3800;display:inline-block;animation:rwspin 2s linear infinite;}
  @keyframes rwspin{to{transform:rotate(360deg)}}
</style>
""", unsafe_allow_html=True)


def _lockup_svg():
    for p in (LOCKUP, LOGO):
        if p and os.path.exists(p):
            try:
                return open(p, encoding="utf-8").read()
            except Exception:
                pass
    return ""


# --------------------------------------------------------------------------- #
# Cached loaders (keyed on file mtimes so edits bust the cache)
# --------------------------------------------------------------------------- #
def _mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


@st.cache_resource(show_spinner=False)
def cached_slate(proj_m, own_m, sched_m):
    return nfl_ingest.build_slate(PROJ_CSV, OWN_CSV, SCHED_CSV)


@st.cache_resource(show_spinner=False)
def cached_sim(n_sims, seed, proj_m, own_m, sched_m):
    slate = cached_slate(proj_m, own_m, sched_m)
    return slate, sim_engine.simulate(slate, n_sims=n_sims, seed=seed)


@st.cache_data(show_spinner=False)
def cached_params(path, mtime):
    if os.path.exists(path):
        return json.load(open(path))
    return fb.DEFAULT_PARAMS


@st.cache_data(show_spinner=False)
def cached_player_table(n_sims, seed, proj_m, own_m, sched_m):
    slate, sim = cached_sim(n_sims, seed, proj_m, own_m, sched_m)
    return exports.player_table(sim, slate)


def build_many(builder, target, label, hard_cap_mult=45):
    out, attempts = [], 0
    cap = target * hard_cap_mult + 1000
    bar = st.progress(0.0, text=f"{label}: 0 / {target}")
    step = max(1, target // 100)
    while len(out) < target and attempts < cap:
        lu = builder.build_one()
        attempts += 1
        if lu is not None:
            out.append(lu)
            if len(out) % step == 0:
                bar.progress(len(out) / target, text=f"{label}: {len(out)} / {target}")
    bar.progress(1.0, text=f"{label}: {len(out)} / {target}")
    return out, attempts


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def player_score_chart(arr, title=""):
    df = pd.DataFrame({"pts": np.asarray(arr, float)})
    med = float(np.percentile(arr, 50))
    ceil = float(np.percentile(arr, 90))
    base = alt.Chart(df).mark_bar(color=RW_PURPLE, opacity=0.85).encode(
        x=alt.X("pts:Q", bin=alt.Bin(maxbins=45), title="DK points"),
        y=alt.Y("count()", title="sims"),
    )
    rules = alt.Chart(pd.DataFrame({
        "v": [med, ceil], "lab": ["median", "p90"],
        "c": [RW_TURF, RW_LEMON]})).mark_rule(strokeWidth=2).encode(
        x="v:Q", color=alt.Color("c:N", scale=None, legend=None))
    return (base + rules).properties(height=240, title=title)


def place_distribution_chart(places, n_field):
    df = pd.DataFrame({"place": np.asarray(places, float)})
    return alt.Chart(df).mark_bar(color=RW_TURF, opacity=0.85).encode(
        x=alt.X("place:Q", bin=alt.Bin(maxbins=50),
                title=f"finishing place (of {n_field:,})"),
        y=alt.Y("count()", title="sims"),
    ).properties(height=240)


def portfolio_return_chart(W, entry_cost):
    df = pd.DataFrame({"ret": np.asarray(W, float)})
    base = alt.Chart(df).mark_bar(color=RW_PURPLE, opacity=0.85).encode(
        x=alt.X("ret:Q", bin=alt.Bin(maxbins=50), title="portfolio $ return / slate"),
        y=alt.Y("count()", title="sims"))
    rule = alt.Chart(pd.DataFrame({"v": [entry_cost]})).mark_rule(
        color=RW_KETCHUP, strokeWidth=2, strokeDash=[4, 3]).encode(x="v:Q")
    return (base + rule).properties(height=240)


def stack_dist_chart(field):
    c = Counter(lu["stack"] for lu in field)
    tot = sum(c.values()) or 1
    df = pd.DataFrame([{"stack": f"QB+{k}", "pct": 100 * v / tot}
                       for k, v in sorted(c.items())])
    return alt.Chart(df).mark_bar(color=RW_PURPLE).encode(
        x=alt.X("stack:N", title="primary stack size", sort=None),
        y=alt.Y("pct:Q", title="% of field"),
    ).properties(height=220)


# --------------------------------------------------------------------------- #
# Header + slate load
# --------------------------------------------------------------------------- #
st.markdown(
    f"""<div class="rw-header">
      <div class="rw-wordmark">{_lockup_svg()}</div>
      <div class="rw-divider"></div>
      <div>
        <div class="rw-title">NFL DFS Contest Sims</div>
        <div class="rw-eyebrow">DraftKings NFL Classic · correlated GPP simulator</div>
      </div>
      <span class="rw-badge"><span class="dot"></span>WEEK SLATE</span>
    </div>""", unsafe_allow_html=True)
st.caption(
    "Simulate DraftKings NFL contest outcomes for machine-developed candidate "
    "lineups against an ownership- and stack-aware field, using correlated "
    "player sims built from the slate's floor/median/ceiling projections. You "
    "choose the contest size, the number of sim runs, and how many candidates "
    "to develop.")

if not (os.path.exists(PROJ_CSV) and os.path.exists(OWN_CSV)):
    st.error("Missing `projections.csv` and/or `ownership.csv` in the app folder.")
    st.stop()

mt = (_mtime(PROJ_CSV), _mtime(OWN_CSV), _mtime(SCHED_CSV))
slate = cached_slate(*mt)
params = cached_params(PARAMS_PATH, _mtime(PARAMS_PATH))
n_matched = sum(p["matched"] for p in slate.players)
KEY_NAME = {e["key"]: e.get("name", e["key"]) for e in slate.entities}
KEY_META = {e["key"]: e for e in slate.entities}


def _cell_to_name(cell):
    """'O123 (TEAM)' -> 'Real Name (TEAM)' for display."""
    key, team = portfolio._split(cell)
    return f"{KEY_NAME.get(key, key)} ({team})"


def relabel_cells(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(_cell_to_name)
    return out


def _caps_from_editor(edf, id_col):
    """Turn a min/max editor table into (caps, mins) dicts keyed by `id_col`,
    as fractions; omit entries left at the defaults (max 100 / min 0)."""
    caps, mins = {}, {}
    for _, r in edf.iterrows():
        mx, mn = float(r["Max%"]) / 100.0, float(r["Min%"]) / 100.0
        if mx < 1.0:
            caps[r[id_col]] = mx
        if mn > 0.0:
            mins[r[id_col]] = mn
    return caps, mins

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offense (matched)", f"{len(slate.players)} ({n_matched})")
c2.metric("Team defenses", len(slate.dst))
c3.metric("Teams · games", f"{len(slate.teams)} · {len(slate.games)}")
c4.metric("Field params",
          "learned" if params.get("learned") else "prior")

tabs = st.tabs(["⚙️  Setup", "📊  Players", "🏆  Results", "⬇️  Export"])

# =========================================================================== #
# SETUP
# =========================================================================== #
with tabs[0]:
    st.subheader("Simulation settings")
    a1, a2, a3 = st.columns(3)
    n_sims = a1.select_slider("Sim runs", options=[2000, 5000, 10000, 20000],
                              value=10000)
    seed = a2.number_input("Random seed", value=20260709, step=1)
    num_candidates = a3.select_slider(
        "Candidate lineups", options=[1000, 2000, 5000, 10000], value=5000)

    st.subheader("Contest field")
    b1, b2, b3 = st.columns(3)
    sizes = b1.multiselect("Contest sizes (entries)",
                           [1000, 3000, 6000, 20000, 50000],
                           default=[1000, 6000, 20000])
    medium = b2.select_slider("Baseline (medium) size",
                              options=[1000, 3000, 6000, 20000], value=6000)
    chalk = b3.slider("Chalk sensitivity", 0.0, 0.8, 0.30, 0.05,
                      help="How much ownership concentrates in small fields / "
                           "flattens in large fields.")
    c1b, c2b = st.columns(2)
    stack_tilt = c1b.slider("Stack tilt", 0.0, 0.4, 0.12, 0.02,
                            help="Lean the field toward bigger QB stacks in large fields.")
    jitter = c2b.slider("Candidate jitter", 0.0, 0.6, 0.0, 0.05,
                        help="Lognormal shock on candidate selection to diversify the pool.")

    if params.get("learned"):
        lw = params["learned"]
        st.caption(
            f"Field grammar learned from {lw.get('n_contests', 0)} contest file(s), "
            f"{lw.get('n_entries_total', 0):,} entries · FLEX mix "
            f"({lw.get('flex_pos_source')}) · stacks: {lw.get('stack_source','prior')}")

    run = st.button("▶  Run simulation & contests", type="primary",
                    width="stretch")

    # sim is cheap; compute (cached) every run so Players is always live
    slate, sim = cached_sim(int(n_sims), int(seed), *mt)
    rc = sim_engine.realized_correlations(sim, slate)
    st.caption(f"sim ready · {n_sims:,} runs · realized QB–WR corr "
               f"{rc['qb_wr_same']:.2f}, WR–WR {rc['wr_wr_same']:.2f}")

    if run:
        if not sizes:
            st.warning("Pick at least one contest size.")
        else:
            with st.status("Building candidates and scoring contests…",
                           expanded=True) as status:
                st.write("Building candidate lineups…")
                cb = fb.Builder(fb.Pool(slate.entities), params,
                                seed=2025, uniform=True, jitter=float(jitter))
                cands, _ = build_many(cb, int(num_candidates), "candidates")
                cand_df = fb.lineups_to_df(cands)
                cand_mat = contest_sim.score_matrix(cands, sim.dk, int(n_sims))

                results, fields, field_mats = {}, {}, {}
                for N in sorted(sizes):
                    st.write(f"Field for {N:,}-entry contest…")
                    adj, p_sz, beta = fs.prepare_field_pool(
                        slate.entities, params, N, n_med=int(medium),
                        chalk_sensitivity=float(chalk), stack_tilt=float(stack_tilt))
                    fbuild = fb.Builder(fb.Pool(adj), p_sz, seed=101 + N, uniform=False)
                    field, _ = build_many(fbuild, N, f"field {N:,}")
                    fmat = contest_sim.score_matrix(field, sim.dk, int(n_sims))
                    wins, t10, t100, avg = contest_sim.run_contest(
                        fmat, cand_mat, int(n_sims), N)
                    res = cand_df.copy()
                    res.insert(0, "Candidate", np.arange(1, len(cands) + 1))
                    res["Win%"] = np.round(100 * wins / n_sims, 3)
                    res["Top10%"] = np.round(100 * t10 / n_sims, 2)
                    res["Top100%"] = np.round(100 * t100 / n_sims, 2)
                    res["AvgPlace"] = np.round(avg, 1)
                    res["_wins"] = wins
                    res = res.sort_values(
                        ["Win%", "Top10%", "Top100%"], ascending=False)
                    results[N] = res
                    fields[N] = field
                    field_mats[N] = fmat
                    status.write(f"  {N:,}: best Win% {res['Win%'].max():.2f}, "
                                 f"best Top100% {res['Top100%'].max():.1f} (beta {beta:.2f})")
                status.update(label="Done — see Results and Export.", state="complete")

            st.session_state["run"] = {
                "cands": cands, "cand_df": cand_df, "cand_mat": cand_mat,
                "results": results, "fields": fields, "field_mats": field_mats,
                "n_sims": int(n_sims), "sizes": sorted(sizes),
            }

# =========================================================================== #
# PLAYERS
# =========================================================================== #
with tabs[1]:
    st.subheader("Player projections")
    ptab = exports.player_table(sim, slate)
    fcol1, fcol2 = st.columns([1, 2])
    pos_filter = fcol1.multiselect("Position", ["QB", "RB", "WR", "TE", "DST"],
                                   default=["QB", "RB", "WR", "TE", "DST"])
    only_matched = fcol2.checkbox(
        "Only players with full stat projections (hide replacement-level)",
        value=True)
    view = ptab[ptab.Pos.isin(pos_filter)]
    if only_matched:
        view = view[view.Matched | (view.Pos == "DST")]
    view = view.reset_index(drop=True)
    show_cols = ["Name", "Pos", "Team", "Opp", "Salary", "Ownership", "Proj",
                 "Floor_p25", "Median_p50", "Ceiling_p75", "p90", "p99",
                 "Std", "Val", "Bust%", "3x%", "5x%"]
    st.dataframe(view[show_cols], width="stretch", height=460,
                 hide_index=True)

    st.markdown("##### Player score distribution")
    pick = st.selectbox(
        "Player (by projection)", list(view.index),
        format_func=lambda i: f"{view.at[i, 'Name']} · {view.at[i, 'Pos']} "
                              f"{view.at[i, 'Team']}",
        index=0 if len(view) else None)
    if pick is not None:
        row = view.loc[pick]
        key = (f"DST_{row['Team']}" if row["Pos"] == "DST"
               else f"O{row['PlayerID']}")
        arr = sim.dk.get(key)
        if arr is not None:
            st.markdown(f"**{row['Name']}** · {row['Pos']} {row['Team']} "
                        f"vs {row['Opp']} · ${int(row['Salary'])}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Proj", f"{row['Proj']:.1f}")
            m2.metric("Ceiling p75", f"{row['Ceiling_p75']:.1f}")
            m3.metric("Value (pts/$1k)", f"{row['Val']:.2f}")
            m4.metric("5x boom%", f"{row['5x%']:.0f}%")
            st.altair_chart(player_score_chart(arr), width="stretch")

# =========================================================================== #
# RESULTS
# =========================================================================== #
with tabs[2]:
    st.subheader("Contest results")
    run_state = st.session_state.get("run")
    if not run_state:
        st.info("Run the simulation on the **Setup** tab to see contest results.")
    else:
        size = st.selectbox("Contest size", run_state["sizes"],
                            index=len(run_state["sizes"]) // 2)
        res = run_state["results"][size]
        field = run_state["fields"][size]
        nsim = run_state["n_sims"]
        d1, d2, d3 = st.columns(3)
        d1.metric("Best Win%", f"{res['Win%'].max():.2f}%")
        d2.metric("Candidates w/ a win", int((res["_wins"] > 0).sum()))
        d3.metric("Best Top100%", f"{res['Top100%'].max():.1f}%")

        rcols = ["Candidate", "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE",
                 "FLEX", "DST", "QBstack", "Salary", "Win%", "Top10%",
                 "Top100%", "AvgPlace"]
        slot_cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
        st.dataframe(relabel_cells(res[rcols].head(200), slot_cols),
                     width="stretch", height=400, hide_index=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("##### Field stack composition")
            st.altair_chart(stack_dist_chart(field), width="stretch")
        with cc2:
            st.markdown("##### Finishing-place distribution")
            cand_no = st.number_input(
                "Candidate #", min_value=1, max_value=len(run_state["cands"]),
                value=int(res.iloc[0]["Candidate"]))
            fmat = run_state["field_mats"][size]
            cscore = run_state["cand_mat"][:, int(cand_no) - 1]
            fs_desc = np.sort(fmat, axis=1)
            places = size - np.array([np.searchsorted(fs_desc[s], cscore[s], "right")
                                      for s in range(nsim)]) + 1
            st.altair_chart(place_distribution_chart(places, size),
                            width="stretch")

# =========================================================================== #
# EXPORT
# =========================================================================== #
with tabs[3]:
    st.subheader("Build your upload")
    run_state = st.session_state.get("run")
    if not run_state:
        st.info("Run the simulation on the **Setup** tab first.")
    else:
        e1, e2, e3 = st.columns(3)
        size = e1.selectbox("Score from contest size", run_state["sizes"],
                            index=len(run_state["sizes"]) // 2, key="exp_size")
        objective = e2.selectbox(
            "Objective", ["ev", "top100", "top10", "win"],
            format_func=lambda x: {"ev": "Payout EV (recommended)",
                                   "top100": "Top-100 rate", "top10": "Top-10 rate",
                                   "win": "Win rate"}[x])
        n_sel = e3.number_input("Lineups to export", 1, 150, 20)

        g1, g2, g3 = st.columns(3)
        max_overlap = g1.slider("Max lineup overlap", 0.3, 1.0, 1.0, 0.05)
        utility = g2.selectbox("Risk posture (EV only)", list(pev.UTILITIES.keys()),
                               index=1)
        entry_fee = g3.number_input("Entry fee ($, for EV payout curve)",
                                    1.0, 10000.0, 20.0)

        # ---- exposure control: global sliders OR per-entity min/max editors ---
        slot_cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
        res = run_state["results"][size].copy()
        cand_mat = run_state["cand_mat"]
        nsim = run_state["n_sims"]

        # universe of keys / primary teams that actually appear in candidates
        cand_keys = set()
        for lu in run_state["cands"]:
            for pl in lu["players"]:
                cand_keys.add(pl["key"])
        prim_teams = sorted({portfolio.lineup_features(res.iloc[i])["primary"]
                             for i in range(min(len(res), 4000))} - {""})

        skill_cap = dst_cap = team_cap = 1.0
        player_caps = team_caps = player_mins = team_mins = None
        mode = st.radio("Exposure control",
                        ["Global caps", "Per-player / per-team min–max"],
                        horizontal=True, key="exp_mode")
        if mode == "Global caps":
            gc1, gc2, gc3 = st.columns(3)
            skill_cap = gc1.slider("Max player exposure", 0.1, 1.0, 1.0, 0.05)
            dst_cap = gc2.slider("Max DST exposure", 0.1, 1.0, 1.0, 0.05)
            team_cap = gc3.slider("Max stack-team exposure", 0.1, 1.0, 1.0, 0.05)
        else:
            with st.expander("Per-player exposure (min / max %)", expanded=True):
                prows = []
                for k in cand_keys:
                    e = KEY_META.get(k)
                    if not e:
                        continue
                    prows.append({"key": k, "Name": e.get("name", k),
                                  "Pos": e["pos"], "Team": e.get("team", ""),
                                  "Own%": round(e.get("own", 0.0), 1),
                                  "Min%": 0, "Max%": 100})
                pdf = pd.DataFrame(prows).sort_values(
                    ["Own%"], ascending=False).reset_index(drop=True)
                ped = st.data_editor(
                    pdf, hide_index=True, height=330, key="player_expo_editor",
                    disabled=["key", "Name", "Pos", "Team", "Own%"],
                    column_order=["Name", "Pos", "Team", "Own%", "Min%", "Max%"],
                    column_config={
                        "Min%": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
                        "Max%": st.column_config.NumberColumn(min_value=0, max_value=100, step=5)})
                player_caps, player_mins = _caps_from_editor(ped, "key")
            with st.expander("Per-team (primary stack) exposure (min / max %)"):
                tdf = pd.DataFrame([{"Team": t, "Min%": 0, "Max%": 100}
                                    for t in prim_teams])
                ted = st.data_editor(
                    tdf, hide_index=True, height=280, key="team_expo_editor",
                    disabled=["Team"],
                    column_config={
                        "Min%": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
                        "Max%": st.column_config.NumberColumn(min_value=0, max_value=100, step=5)})
                team_caps, team_mins = _caps_from_editor(ted, "Team")

        if st.button("Build export set", type="primary"):
            if objective == "ev":
                prize = pev.make_payout_curve(size, entry_fee)
                cut = pev.field_place_cutpoints(size)
                fmat = run_state["field_mats"][size]
                fs_desc = -np.sort(-fmat, axis=1)
                field_cut = fs_desc[:, np.clip(cut - 1, 0, fs_desc.shape[1] - 1)]
                order = res["Candidate"].to_numpy() - 1
                pay = pev.candidate_payout_matrix(cand_mat[:, order], field_cut, cut, prize)
                chosen, info, W = portfolio.select_portfolio_ev(
                    res, int(n_sel), pay, pev.utility(utility),
                    skill_cap=skill_cap, dst_cap=dst_cap, team_cap=team_cap,
                    max_overlap=max_overlap, player_caps=player_caps,
                    team_caps=team_caps, player_mins=player_mins,
                    team_mins=team_mins)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Lineups", info["chosen"])
                m2.metric("Exp $ / entry", f"${info['exp_return']/max(info['chosen'],1):.2f}")
                m3.metric("Portfolio cash%", f"{100*info['cash_rate']:.1f}%")
                m4.metric("Ceiling p90", f"${info['ceiling_p90']:.0f}")
                st.altair_chart(
                    portfolio_return_chart(W, entry_fee * info["chosen"]),
                    width="stretch")
            else:
                keymap = {"win": ["Win%", "Top10%", "Top100%"],
                          "top10": ["Top10%", "Top100%", "Win%"],
                          "top100": ["Top100%", "Top10%", "Win%"]}[objective]
                chosen, info = portfolio.select_portfolio(
                    res, int(n_sel), keymap, skill_cap=skill_cap, dst_cap=dst_cap,
                    team_cap=team_cap, max_overlap=max_overlap,
                    player_caps=player_caps, team_caps=team_caps,
                    player_mins=player_mins, team_mins=team_mins)
                m1, m2, m3 = st.columns(3)
                m1.metric("Lineups", info["chosen"])
                m2.metric("Distinct stacks", info["distinct_cores"])
                m3.metric("Max team", info["max_team"])

            # name-annotated lineup preview
            st.markdown("##### Selected lineups")
            prev = pd.DataFrame(list(chosen))
            prev_cols = [c for c in slot_cols if c in prev.columns]
            keep = prev_cols + [c for c in ("QBstack", "Salary") if c in prev.columns]
            st.dataframe(relabel_cells(prev[keep], prev_cols),
                         width="stretch", height=300, hide_index=True)

            up = exports.dk_upload(chosen, slate)   # duplicate DK headers (CSV)
            st.download_button(
                "⬇  Download DK_upload.csv (DraftKings IDs)", up.to_csv(index=False),
                file_name=f"DK_upload_{size}.csv", mime="text/csv",
                type="primary")

            # exposure breakdown (with player names)
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("##### Player exposure")
                pe = sorted(info["player_expo"].items(), key=lambda kv: -kv[1])
                pe_df = pd.DataFrame(
                    [{"Player": KEY_NAME.get(k, k),
                      "Pos": KEY_META.get(k, {}).get("pos", ""),
                      "Lineups": v, "Exposure%": round(100 * v / max(info["chosen"], 1), 1)}
                     for k, v in pe])
                st.dataframe(pe_df, width="stretch", height=260, hide_index=True)
            with b2:
                st.markdown("##### Primary-stack team exposure")
                te_df = pd.DataFrame(
                    [{"Stack team": t, "Lineups": v,
                      "Exposure%": round(100 * v / max(info["chosen"], 1), 1)}
                     for t, v in sorted(info["team_expo"].items(), key=lambda kv: -kv[1])])
                st.dataframe(te_df, width="stretch", height=260, hide_index=True)
            if info["unmet_mins"]:
                labeled = [{**u, "name": KEY_NAME.get(u["name"], u["name"])}
                           for u in info["unmet_mins"]]
                st.warning(f"Unmet minimums (pool couldn't supply): {labeled}")
