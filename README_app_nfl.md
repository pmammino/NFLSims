# NFL DFS Simulator — Streamlit app

`app.py` is the point-and-click front end over the NFL engine, in the RotoWire
full-dark theme (brand fonts from `static/fonts/`, theme tokens in
`.streamlit/config.toml`). Unlike the MLB app it runs the whole pipeline
**in-process** — the correlated sim is seconds, so there's no external
multi-stage rebuild, no live feeds, and no shared-store/freshness machinery to
manage.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the slate straight from the repo files: `projections.csv`,
`ownership.csv`, `schedule.csv`, and the learned `field_params_nfl.json`
(regenerate with `python3 learn_field.py`).

> **Deploying (Streamlit Community Cloud):** `requirements.txt` pins
> `streamlit>=1.49` — the app uses the `width="stretch"` layout API, and an
> older Streamlit raises `TypeError: 'str' object cannot be interpreted as an
> integer` and prompts you to upgrade. If a deploy is stuck on an old version,
> reboot it (or clear the cache) so the pinned requirements reinstall.
>
> **Memory:** contest scoring builds an `(n_sims × field_size)` matrix per size,
> which is freed immediately after use; only small per-place cut scores are kept.
> On a hosted free tier keep `field size × sim runs` modest (the defaults —
> 1k/6k fields at 10k sims — are safe); very large fields (20k+) at high sim
> counts can exceed the memory limit while a run is in progress.

## Tabbed workspace

- **⚙️ Setup** — slate summary (offense matched, team defenses, teams/games,
  field-params source). Controls: sim runs, seed, candidate count, contest
  sizes, medium baseline, chalk sensitivity, stack tilt, candidate jitter.
  **▶ Run** builds the candidate pool + an ownership/stack-aware field per size,
  scores every contest, and stores the results. (The sim itself is computed and
  cached continuously, so the Players tab is live even before you Run.)
- **📊 Players** — the player projection table (Proj / Floor p25 / Median /
  Ceiling p75 / p90 / p99 / Std / Value / Bust% / 3x% / 5x%), filterable by
  position, plus a selected player's DK-point distribution (median + p90 marked).
- **🏆 Results** — per-contest candidate finish rates (Win% / Top10% / Top100% /
  AvgPlace), the field's QB-stack composition, and any candidate's
  finishing-place distribution across all sims.
- **⬇️ Export** — diversity-aware or **payout-EV** lineup selection. Exposure
  control is either **global caps** (max player / DST / stack-team exposure) or
  **per-player and per-team min–max editors** (set a floor/ceiling on any
  player's or stack team's share of the exported set, like the MLB app). A
  **value-groups** control detects near-twin players (same position, close
  salary + projection) and caps combined exposure across each group, so the set
  spreads over interchangeable plays instead of forcing one. Plus max overlap,
  risk posture, and entry fee. Shows a name-annotated lineup
  preview, a portfolio-return chart, player + stack-team exposure breakdowns,
  and a one-click **DK upload CSV** download (header
  `QB,RB,RB,WR,WR,WR,TE,FLEX,DST`, DraftKings contest IDs).

## Notes

- Real player names come from `player_names.csv` (`ID,firstname,lastname`, where
  `ID` = `RotoPlayerID`). Delete/replace it and the app falls back to id labels;
  DST always displays as `TEAM DST`. Internally the engine still keys on the
  stable entity id, and uploads use DraftKings contest IDs, so they import
  directly regardless.
- Caching is keyed on input-file mtimes and the sim settings, so editing
  `projections.csv` / `ownership.csv` / `schedule.csv` busts the cache.
- Windows-portable (pure Python + numpy/pandas/scipy + streamlit/altair).
