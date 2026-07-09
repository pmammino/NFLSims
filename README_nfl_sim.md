# NFL DFS Simulation Engine

A full DraftKings **NFL Classic** GPP simulator, built in the same spirit as the
MLB engine in `DFSSimsFull`: learn how the real field constructs lineups, build a
realistic ownership- and stacking-aware opponent field, generate a candidate set,
run a correlated player simulation off range-of-outcomes projections, score every
lineup by DraftKings rules, and select an EV-optimal portfolio for upload.

The stages are deliberately decoupled the same way the MLB engine is. The
contract between the slow projection/sim layer and the fast contest layer is one
simple artifact: a dict `{player_key: np.ndarray[N_SIMS]}` of DK fantasy points
(`out/player_dk_sims.npy`). Everything downstream (fields, candidates, contest
scoring, portfolio EV) operates on that matrix and is sport-agnostic.

```
projections.csv ─┐
ownership.csv  ──┤ nfl_ingest.build_slate()  ─► Slate (players, DST, schedule)
schedule.csv  ──┘            │
                             ▼
              sim_engine.simulate()  ─►  {key: DK points [N_SIMS]}   (out/player_dk_sims.npy)
                             │                     │
   contest-standings ─► learn_field ─► field_params_nfl.json         │
   (aggregate only)          │                                       │
                             ▼                                       ▼
              field_builder + field_simulator          contest_sim.score_matrix
              (realistic opponent field per size)      (per-sim lineup totals)
                             │                                       │
                             └──────────────► contest_sim.run_contest ◄──── candidate lineups
                                                       │
                                                       ▼
                              portfolio / portfolio_ev  ─►  DK_upload_<N>.csv
                                                       │
                                                       ▼
                                       exports.player_table  ─► player_projections.csv
```

## Roster / rules (DK NFL Classic)

* Slots: `QB, RB, RB, WR, WR, WR, TE, FLEX, DST` (FLEX = RB/WR/TE). 9 players.
* Salary cap: `$50,000`.
* Scoring: full PPR. See `dk_scoring.py` for the exact rule set (passing 0.04/yd
  + 4/TD − 1/INT + 3 @300yd; rushing 0.1/yd + 6/TD + 3 @100yd; receiving 1/rec +
  0.1/yd + 6/TD + 3 @100yd; −1 fumble lost; +2 two-point; return/recovery TD +6;
  DST sacks/INT/fumble-rec/TD/safety/block + points-allowed tiers).

## Data model (the four input files)

* **`projections.csv`** — one row per `PlayerID` per `Split` (`C`=ceiling≈75th,
  `M`=median≈50th, `F`=floor≈25th), stat by stat, for `GameWeek`. These are
  *individual* players including IDP-style defenders — there are no team-DST rows.
* **`ownership.csv`** — the playable DK slate pool (`SlateID`, `Salary`,
  `Position`, projected `Ownership`, `PlayerID`, `RotoPlayerID`). The join key to
  projections is **`RotoPlayerID` = projections `PlayerID`**.
* **`contest-standings-*.csv`** — real DK NFL contest results (dual-column: entry
  lineups + realized `%Drafted`/`FPTS`), keyed by **player name only**. Used for
  *aggregate* field learning (stack-shape distribution, bring-back rate, FLEX
  mix, chalk temperature) — not per-player joins.

### Modeling decisions forced by the data

1. **Offense join** — `ownership.RotoPlayerID → projections.PlayerID`. ~150/253
   pool players match. Unmatched entries are low-salary/low-ownership skill
   players; they stay in the field pool but get a **replacement-level** marginal
   (salary-scaled, near-zero ceiling) so field lineups remain realistic without
   inventing projections.
2. **DST** — projections have no team-defense rows, so a team DST projection is
   **aggregated from that team's individual defenders** (sacks, INT, fumble
   recoveries, defensive/return TDs, safeties, blocks). Points-allowed is modeled
   from the *opponent's* simulated offense when a schedule is available (so DST
   correlates negatively with the offense it faces), else from a Vegas/neutral
   prior. Pool DST rows are mapped to teams via `dst_teams.csv` when present, else
   by a documented salary-rank heuristic.
3. **Schedule** — no opponent column exists in any input. `schedule.csv`
   (`Team,Opp[,Total,Implied]`) supplies game pairings and (optional) Vegas
   totals. It unlocks bring-back correlation, game-stack (shootout) correlation,
   and DST-vs-opponent scoring. A slate `schedule.csv` is generated as a
   **starting point you should verify/edit** — pairings only affect correlation
   structure, not the marginal projections.

## The correlated simulation (`sim_engine.py`)

For each player and each DK-relevant stat we fit a **lognormal** to the
`(floor=p25, median=p50, ceiling=p75)` triple (median-anchored; IQR sets the log
spread). We then draw one standard-normal latent per player from a **Gaussian
copula** whose target correlation matrix encodes NFL structure:

| relationship | target ρ |
|---|---|
| QB ↔ own WR/TE | +0.55 / +0.50 |
| WR ↔ WR (same team) | +0.30 |
| WR/TE ↔ own RB | −0.05 |
| RB ↔ RB (same team) | −0.20 |
| QB ↔ opp pass-catcher (game stack) | +0.18 |
| skill ↔ opp skill (same game) | +0.10 |
| DST ↔ opponent offense | via points-allowed (negative) |
| DST ↔ own offense | +0.10 |
| different game | 0 |

The player latent is shared across that player's own stats (with a small
per-stat idiosyncratic component, so yards and TDs move together but not
identically). Sampled stats are scored per-sim by `dk_scoring`, so yardage
bonuses fire on the realized yardage and the whole thing is internally
consistent. The output is `{key: DK points [N_SIMS]}` plus per-stat sims for the
player table. Optional Vegas `total_scale` (from `schedule.csv`) reshapes each
team's mean and ceiling, exactly like the MLB engine's `total_scale`.

## Modules

| file | role | ported / new |
|---|---|---|
| `dk_scoring.py` | DK NFL scoring (offense + DST) | new |
| `nfl_ingest.py` | build the Slate from the 4 files (+ schedule) | new |
| `contest_ingest.py` | parse DK NFL standings CSVs | new (analog of `contest_review.parse_contest_csv`) |
| `learn_field.py` | derive `field_params_nfl.json` from standings | new (analog of the MLB field-params derivation) |
| `sim_engine.py` | correlated player sim → DK points | new (analog of `sim_proj.py`) |
| `field_builder.py` | QB-centric field/candidate lineup builder | new (analog of `mlb_lineup_builder.py`) |
| `field_simulator.py` | contest-size chalk/tilt model | ported (`normalize_to_slots`, `beta_for_size`, `adjust_ownership`, `tilt_structures`) |
| `contest_sim.py` | `score_matrix` + `run_contest` | ported from `stage_d.py` |
| `portfolio.py` | diversity-aware selection (NFL stack semantics) | adapted from `DFSSimsFull/portfolio.py` |
| `portfolio_ev.py` | payout curve + concave-utility EV selection | verbatim from `DFSSimsFull/portfolio_ev.py` |
| `exports.py` | player projection table + DK upload CSV | new (mirrors the MLB player export) |
| `run_sim.py` | end-to-end orchestrator | new (analog of `run_full.py`/`stage_d.py`) |

## Running it

```bash
python3 run_sim.py --n-sims 10000 --contest-sizes 1000 6000 20000 \
        --num-candidates 10000 --select 20 --objective ev
```

Outputs land in `out/`:

* `player_dk_sims.npy` / `player_stat_sims.npy` — the sim artifacts (stage boundary).
* `player_projections.csv` — the player table (Proj, Floor p25, Median, Ceiling
  p75, p10, p90, p99, Std, Bust%, 2x%, 3x%, plus pos/team/salary/ownership).
* `candidates.csv`, `field_<N>.csv`, `candidate_results_<N>.csv`.
* `DK_upload_<N>.csv` — ranked or EV-optimal export set.

Everything defaults to reproducing a simple baseline (no jitter, caps off, EV
off) — every diversity/EV lever is additive, matching the MLB engine's discipline.

## Not in this pass (follow-up)

The Streamlit app parity (Setup/Players/Results/Export tabs, RotoWire theme) and
live NFL feeds (inactives/Vegas) are deferred to a follow-up, per the agreed
"plan + core engine first" scope. The engine writes the same artifacts the app
would consume.
