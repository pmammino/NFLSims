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

## Showdown (single-game) support

Everything above is DK **Classic**. The same engine also runs DK **Showdown**
(one game) via `--slate-type showdown` on `run_sim.py`, or the *Slate format*
toggle in the app. Classic is the default; nothing about it changes.

**Roster.** 1 `CPT` (Captain) + 5 `FLEX`, six players, all from the one game;
any position (QB/RB/WR/TE/K/DST) fills any slot. `$50,000` cap. The Captain
scores **1.5×** DK points and costs **1.5×** salary — the points multiplier is
applied in `contest_sim.score_matrix` off each lineup's `captain_key`; the salary
comes from each entity's `cpt_salary`. Kickers are scored (`dk_scoring.score_kicker`:
XP +1, FG 0-39 +3, 40-49 +4, 50+ +5).

**Data.** Feed the same four files, scoped to the single game. `ownership.csv`
carries **one row per player** — the player's **overall** rostered ownership
across both slots. If the file also carries explicit Captain-priced rows
(a `Position`/`Roster Position` of `CPT`), the ingest reads their exact salary
and DK contest id; otherwise it derives Captain salary = round(1.5× FLEX) and
reuses the base contest id (swap in a CPT-id crosswalk later if your contest
needs distinct Captain ids to import).

**Ownership split** (`showdown.split_ownership`). Six roster spots per entry, so
across the field overall ownership sums to 600%, of which the Captain slot is
100% and FLEX is 500%. Each player's overall exposure `O_i` is split into a
Captain-slot rate and a FLEX-slot rate via a per-player Captain *share*
(`cpt_i = s_i·O_i`, so `cpt_i ≤ O_i`), with `s_i` tilted toward ceiling so the
scarce Captain spot skews to high-upside players. Captain mass sums to exactly 1,
FLEX to exactly 5.

**Two builders** (`showdown.py`):

* **Ownership-aware field** (`FieldBuilder` / `build_field`) — draws the Captain
  ∝ Captain-slot ownership and the five FLEX ∝ FLEX-slot ownership, i.e. the
  crowd you actually face. It keeps the Classic realism guards: chalk temperature
  for the field size, a sharp fraction drawn from the smart candidate builder so
  the field isn't a too-soft pure-ownership draw, and overbuild-and-trim to the
  top lineups by projection (Captain at 1.5×).
* **Ownership-blind candidates** (`CandidateBuilder` / `build_candidates`) —
  ignores ownership entirely and builds off single-game construction rules: pick
  a team to build around, Captain a ceiling player, keep the QB with ≥1
  pass-catcher (the intra-team passing stack), force a bring-back from the other
  team (winning single-game lineups carry both sides — team split is sampled from
  balanced 3-3 / onslaught 4-2 / 2-4 / rare 5-1, never 6-0), guard against
  rostering a DST behind a big opposing stack, and weight every remaining pick by
  simulated ceiling.

Portfolio selection, EV, and the DK upload are all Showdown-aware: the Captain
is the specially-capped slot (`dst_cap` is reused as the Captain-exposure cap),
`core_cap` diversifies which player you Captain, and `showdown.dk_upload` writes
the `CPT,FLEX,FLEX,FLEX,FLEX,FLEX` header mapping the Captain cell to its Captain
contest id.

**Seeing the two flows on their own.** Two standalone scripts run each Showdown
build in isolation, write a name-annotated CSV, and print a summary (ownership
split, team splits, captain distribution, stack/bring-back rates):

```bash
# ownership-aware opponent field
python3 gen_showdown_field.py      --ownership ownership.csv --n 20000 --out out/showdown_field.csv
# ownership-blind, correlation-driven candidates (the lineups you'd enter)
python3 gen_showdown_candidates.py --ownership ownership.csv --n 10000 --out out/showdown_candidates.csv
```

Both take a single-game slate's files; `--help` lists the knobs (chalk/sharp/
overbuild for the field; QB-stack / bring-back / captain-skill probabilities for
the candidates).

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

The sim is **hierarchical and game-consistent** rather than a flat copula, so a
QB and his receivers can't independently boom in the same sim (which would
double-count the same passing yards and fatten the tail).

1. **Latents.** Per sim: a game latent (shootout), a team-offense latent tied to
   it (this is what creates bring-back / game-stack correlation), and team pass /
   rush latents beneath it.
2. **Marginals.** Every quantity is drawn from its *own* `(floor p25, median p50,
   ceiling p75)` triple via `q_from_triple` — a quantile map that is
   piecewise-linear in standard-normal space with **damped tails** so p25/p75 are
   honored exactly but the deep tails stay realistic (elite-RB p99 ≈ 70, QB/WR/TE
   max ≈ 60–70).
3. **Allocation (the key step).** The starting QB's passing line and each
   pass-catcher's receiving line are each sampled from their own ranges, then the
   receivers are **rescaled so their receptions / rec yards / rec TDs sum exactly
   to the QB's completions / pass yards / pass TDs in every sim.** So team
   receptions == QB completions, team rec yards == QB pass yards, team rec TDs ==
   QB pass TDs — game-consistent, and each receiver's ceiling is bounded by the
   team's realized passing total. Rushing is sampled per player from a team-rush
   latent; DST points-allowed is derived from the opponent's simulated offense
   (so DST anti-correlates with the offense it faces).
4. **Scoring.** Final stats are scored per-sim by `dk_scoring`, so yardage bonuses
   fire on realized yardage.

The realized QB↔own-WR correlation lands ≈ 0.5 and WR↔WR ≈ 0.2. Output is `{key:
DK points [N_SIMS]}` plus per-stat means for the player table. Optional Vegas
`total_scale` (from `schedule.csv`) reshapes each team's offensive output.

## Modules

| file | role | ported / new |
|---|---|---|
| `dk_scoring.py` | DK NFL scoring (offense + DST) | new |
| `nfl_ingest.py` | build the Slate from the 4 files (+ schedule) | new |
| `contest_ingest.py` | parse DK NFL standings CSVs | new (analog of `contest_review.parse_contest_csv`) |
| `learn_field.py` | derive `field_params_nfl.json` from standings | new (analog of the MLB field-params derivation) |
| `sim_engine.py` | correlated player sim → DK points | new (analog of `sim_proj.py`) |
| `field_builder.py` | QB-centric field/candidate lineup builder (+ candidate upside/stack-tilt/bring-back controls) | new (analog of `mlb_lineup_builder.py`) |
| `field_simulator.py` | contest-size chalk/tilt model + realistic-field build (`build_field`) | ported (`normalize_to_slots`, `beta_for_size`, `adjust_ownership`, `tilt_structures`) |
| `stack_signal.py` | stack-ownership ceiling bump on the sim scores | ported from `DFSSimsFull/stack_signal.py` |
| `ownership_model.py` | ownership-uncertainty resampling for the field | ported (subset of `DFSSimsFull/ownership_model.py`) |
| `contest_sim.py` | `score_matrix` (+ Captain 1.5× via `captain_key`) + `run_contest` | ported from `stage_d.py` |
| `showdown.py` | Showdown ownership split + ownership-aware field + ownership-blind candidate builders + DK upload | new |
| `portfolio.py` | diversity-aware selection (NFL stack semantics) | adapted from `DFSSimsFull/portfolio.py` |
| `portfolio_ev.py` | payout curve + concave-utility EV selection | verbatim from `DFSSimsFull/portfolio_ev.py` |
| `exports.py` | player projection table + DK upload CSV | new (mirrors the MLB player export) |
| `run_sim.py` | end-to-end orchestrator | new (analog of `run_full.py`/`stage_d.py`) |

## Running it

```bash
python3 run_sim.py --n-sims 10000 --contest-sizes 1000 6000 20000 \
        --num-candidates 10000 --select 20 --objective ev

# Showdown (single game): same flags + --slate-type showdown
python3 run_sim.py --slate-type showdown --n-sims 10000 \
        --contest-sizes 1000 6000 --num-candidates 10000 --select 20 --objective ev
```

Outputs land in `out/`:

* `player_dk_sims.npy` / `player_stat_sims.npy` — the sim artifacts (stage boundary).
* `player_projections.csv` — the player table (Proj, Floor p25, Median, Ceiling
  p75, p10, p90, p99, Std, Bust%, 2x%, 3x%, plus pos/team/salary/ownership).
* `candidates.csv`, `field_<N>.csv`, `candidate_results_<N>.csv`.
* `DK_upload_<N>.csv` — ranked or EV-optimal export set.

Every diversity/EV lever is additive (caps off, EV off by default), matching the
MLB engine's discipline.

### Field realism & candidate sharpness (ported from the MLB engine)

A pure ownership-imitation field is too soft — it hands a no-skill lineup a
positive edge, which is impossible. Four levers, all tunable on the CLI and in
the app's *Field realism & candidate sharpness* panel, close that gap:

* **Sharp/chalk field mix + overbuild** (`--sharp-frac`, `--overbuild`): a share
  of the field is built sharp (bigger stacks, ceiling-weighted players,
  bring-backs) on the ownership base; the field is overbuilt and trimmed to the
  highest-projection lineups, so you face the lineups people *submit*, not raw
  draws.
* **Candidate sharpness** (`--cand-stack-strength`, `--cand-bringback`,
  `--no-cand-upside`): candidates lean toward bigger QB stacks, force more
  bring-backs, and weight skill/FLEX picks by sim ceiling (p90) rather than
  ownership.
* **Stack-ownership ceiling signal** (`--stack-boost`): popular offenses get a
  small bump to their high-end sims (tied to projected stack ownership), applied
  to both the field and candidates so it's a coherent re-weighting of reality.
* **Ownership uncertainty** (`--own-uncertainty`): the field mixes over fresh
  ownership draws instead of one point estimate.

### Learned field weights

`learn_field.py` now builds a name→team crosswalk (`player_names.csv` ⋈
`projections.csv`) and learns the QB-stack-size distribution from
**fully-resolved** real standings lineups (unbiased per lineup), in addition to
the FLEX mix / ownership / duplication it already reported. On the shipped
standings this moved `stack_sizes` off the prior to a data-derived
`[0:8%, 1:34%, 2:50%, 3:7%, 4:0.5%]` (the 2-man QB stack is the modal build).
Bring-back / DST-vs-own-stack stay on documented priors — labelling an opponent
bring-back needs each standings slate's schedule, which the files don't carry;
the learned secondary-cluster distribution is recorded as a corroborating
diagnostic.

## Not in this pass (follow-up)

The Streamlit app parity (Setup/Players/Results/Export tabs, RotoWire theme) and
live NFL feeds (inactives/Vegas) are deferred to a follow-up, per the agreed
"plan + core engine first" scope. The engine writes the same artifacts the app
would consume.
