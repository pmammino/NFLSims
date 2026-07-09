#!/usr/bin/env python3
"""
exports.py
==========
Player-projection table and DK upload writer — mirrors the MLB engine's player
export ("Proj / Floor / Median / Ceiling / p99 / Std / Bust% / 2x% / ...") with
NFL-appropriate value metrics.

The player table is the human-facing read on the sim: mean projection, the
floor/median/ceiling percentiles, tail (p99), volatility, and salary-relative
value / boom / bust rates. The DK upload maps lineup cells (``KEY (TEAM)``) to
DraftKings contest IDs so the CSV imports directly.
"""
import numpy as np
import pandas as pd

PLAYER_COLS = [
    "PlayerID", "ContestID", "Pos", "Team", "Opp", "Salary", "Ownership",
    "Proj", "Floor_p25", "Median_p50", "Ceiling_p75", "p10", "p90", "p99",
    "Max", "Std", "Val", "Bust%", "3x%", "5x%", "Matched",
    "mean_pass_yds", "mean_rush_yds", "mean_rec_yds", "mean_rec",
]


def player_table(sim, slate):
    """Per-entity projection table sorted by mean projection (desc)."""
    rows = []
    for e in slate.entities:
        arr = sim.dk[e["key"]]
        salary = e["salary"]
        per1k = salary / 1000.0 if salary else 1.0
        mean = float(arr.mean())
        med = float(np.percentile(arr, 50))
        val = arr / per1k
        sm = sim.stat_means.get(e["key"], {})
        rows.append({
            "PlayerID": e.get("rid", ""),
            "ContestID": e.get("contest_id", ""),
            "Pos": e["pos"], "Team": e.get("team", ""), "Opp": e.get("opp", ""),
            "Salary": salary, "Ownership": round(e.get("own", 0.0), 3),
            "Proj": round(mean, 2),
            "Floor_p25": round(float(np.percentile(arr, 25)), 2),
            "Median_p50": round(med, 2),
            "Ceiling_p75": round(float(np.percentile(arr, 75)), 2),
            "p10": round(float(np.percentile(arr, 10)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "Max": round(float(arr.max()), 2),
            "Std": round(float(arr.std()), 2),
            "Val": round(mean / per1k, 2),
            "Bust%": round(100 * float(np.mean(arr < 0.5 * max(med, 1e-9))), 1),
            "3x%": round(100 * float(np.mean(val >= 3)), 1),
            "5x%": round(100 * float(np.mean(val >= 5)), 1),
            "Matched": bool(e.get("matched", e["pos"] == "DST")),
            "mean_pass_yds": round(sm.get("pass_yards", 0.0), 1),
            "mean_rush_yds": round(sm.get("rush_yards", 0.0), 1),
            "mean_rec_yds": round(sm.get("rec_yards", 0.0), 1),
            "mean_rec": round(sm.get("receptions", 0.0), 1),
        })
    df = pd.DataFrame(rows, columns=PLAYER_COLS)
    return df.sort_values("Proj", ascending=False).reset_index(drop=True)


UPLOAD_HEADER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]


def dk_upload(chosen_rows, slate, cols=None):
    """Build a DK-importable upload DataFrame from selected lineup rows.

    `chosen_rows` are result rows whose slot cells are ``"KEY (TEAM)"``; each key
    is mapped to its DraftKings contest id."""
    from portfolio import SLOT_COLS, _split
    cols = cols or SLOT_COLS
    cid = {e["key"]: e.get("contest_id", "") for e in slate.entities}
    out = []
    for row in chosen_rows:
        ids = []
        for c in cols:
            key, _ = _split(row[c])
            ids.append(cid.get(key, key))
        out.append(ids)
    return pd.DataFrame(out, columns=UPLOAD_HEADER)


if __name__ == "__main__":
    import nfl_ingest
    import sim_engine
    slate = nfl_ingest.build_slate()
    sim = sim_engine.simulate(slate, n_sims=3000, seed=3)
    df = player_table(sim, slate)
    cols = ["Pos", "Team", "Opp", "Salary", "Ownership", "Proj",
            "Floor_p25", "Median_p50", "Ceiling_p75", "p99", "Val", "5x%"]
    print(df[df.Matched | (df.Pos == "DST")][cols].head(15).to_string(index=False))
