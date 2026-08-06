# NFL DFS pipeline — the four stages

The engine is split into four standalone scripts, one per stage. Each reads the
previous stage's files and writes its own, so every stage's **inputs and outputs
are explicit files on disk** — you can run, inspect, tune, or re-run any stage on
its own.

```
   raw slate files                    ┌──────────────────────────────┐
 projections.csv                      │ 1. simulate.py                │
 ownership.csv        ───────────────▶│    correlated simulations     │
 schedule.csv                         └───────────────┬──────────────┘
 player_names.csv                                     │ player_dk_sims.npy
 dst_teams.csv (opt)                                  │ player_pool.csv
                                                      │ player_projections.csv
                        ┌─────────────────────────────┼─────────────────────────┐
                        ▼                             ▼                          │
        ┌──────────────────────────┐   ┌──────────────────────────┐             │
        │ 2. build_field.py        │   │ 3. build_candidates.py    │             │
        │    the opponent field    │   │    our sharp lineups      │             │
        └────────────┬─────────────┘   └────────────┬─────────────┘             │
                     │ field_<N>.csv                 │ candidates.csv            │
                     └───────────────┬───────────────┘                          │
                                     ▼                                           │
                     ┌──────────────────────────────┐   player_dk_sims.npy ◀────┘
                     │ 4. score_contest.py           │   player_pool.csv
                     │    finish rates + export      │
                     └───────────────┬──────────────┘
                                     │ candidate_results_<N>.csv
                                     ▼ DK_upload_<N>.csv
```

## Quickstart

```bash
python3 simulate.py         --n-sims 10000
python3 build_field.py      --sizes 1000 6000 20000
python3 build_candidates.py --num-candidates 10000
python3 score_contest.py    --sizes 1000 6000 20000 --select 20 --objective ev
```

All four default to reading/writing the `out/` directory (`--indir` / `--outdir`
to change). `run_sim.py` still does all four in one process if you want a single
command; these scripts are the same logic broken out so each stage is inspectable.

---

## 1. `simulate.py` — Correlated simulations

Monte-Carlo of DraftKings points for every player, with real teammate
correlation (shared game / team-offense / pass / rush latents, so a QB and his
receivers boom together in the same sims). Everything downstream reasons about
these same arrays.

**Inputs** (raw slate files in the working directory)

| file | what it provides |
|---|---|
| `projections.csv` | per-player per-split (floor/median/ceiling) stat projections |
| `ownership.csv` | the DK playable pool: salary, position, projected ownership, ids |
| `schedule.csv` | `Team,Opp[,Total,Implied]` game pairings (+ Vegas totals) |
| `player_names.csv` | id → name crosswalk *(optional but recommended)* |
| `dst_teams.csv` | DST id → team crosswalk *(optional)* |

**Outputs** (to `out/`)

| file | contents |
|---|---|
| `player_dk_sims.npy` | dict `{entity_key: float32[n_sims]}` of DK points — **the stage-boundary artifact**. Includes the stack-ownership ceiling boost. |
| `player_pool.csv` | one row per entity: `key,name,pos,team,opp,salary,own,contest_id,proj,ceiling` (proj = mean sim pts, ceiling = p90). The compact contract the build/scoring stages read. |
| `player_projections.csv` | full human-readable player table (Proj, Floor p25, Median, Ceiling p75, p10/p90/p99, Std, value, boom/bust). |

**Key knobs:** `--n-sims`, `--seed`, `--stack-boost` (ceiling bump for popular
offenses, baked into the saved arrays; `0` disables).

---

## 2. `build_field.py` — Field

Manufactures the opponent field — the lineups the crowd is expected to submit.
A naive ownership-imitation field is too soft (it would hand a no-skill lineup a
positive edge), so three realism layers are applied: a contest-size chalk/stack
model, a **sharp/chalk mix** (part of the field is stacked/ceiling-built with
bring-backs, the rest is chalk), and **overbuild-and-keep-top-by-projection**
(the field you face is the better lineups people submit, not raw draws).

**Inputs** (from `--indir`, i.e. stage 1's output)

| file | used for |
|---|---|
| `player_pool.csv` | the player pool (`own` drives selection), `proj` ranks the overbuild, `ceiling` weights the sharp portion |
| `field_params_nfl.json` | learned QB-stack / FLEX / bring-back grammar (`--params`) |

**Outputs**

| file | contents |
|---|---|
| `field_<N>.csv` | one row per opponent lineup for each `--sizes` entry: `Lineup, Salary, QBstack, QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST, Proj`. Slot cells are `KEY (TEAM)` (the entity key) so stage 4 can map each slot back to its sim array. |

**Key knobs:** `--sizes`, `--sharp-frac`, `--overbuild`, `--chalk-sensitivity`,
`--stack-tilt`, `--own-uncertainty`, `--medium`. (`--sharp-frac 0 --overbuild 1`
reproduces the old naive/soft field.)

---

## 3. `build_candidates.py` — Candidate

Builds the lineups *we* are considering. Same builder as the field, opposite
posture: stack teams chosen **ownership-blind**, skill/FLEX picks weighted by
**simulated ceiling (p90)** instead of ownership, QB stacks tilted **bigger**,
and bring-backs **forced** more often — the construction edges the win region
shows over the crowd.

**Inputs** (from `--indir`)

| file | used for |
|---|---|
| `player_pool.csv` | the player pool; `ceiling` is the candidate skill-pick weight |
| `player_dk_sims.npy` | to compute each lineup's `Proj` / `Ceiling` / `Std` (a lineup ceiling needs the summed per-sim arrays, not just per-player numbers) |
| `field_params_nfl.json` | the QB-stack / FLEX grammar (`--params`) |

**Outputs**

| file | contents |
|---|---|
| `candidates.csv` | one row per candidate, sorted by ceiling: `Lineup, Salary, QBstack, QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST, Proj, Ceiling, Std`. Slot cells are `KEY (TEAM)`. |

**Key knobs:** `--num-candidates`, `--cand-stack-strength` (how much bigger our
stacks lean), `--cand-bringback`, `--no-cand-upside` (turn off ceiling
weighting), `--jitter` (diversify near-equivalent players).

---

## 4. `score_contest.py` — Contest Scoring

Places the candidates against each field across every sim and reports finish
rates; optionally selects a payout-aware export set. Re-simulates nothing — a
lineup's per-sim score is the sum of its nine players' saved arrays; its place
that sim is its rank in the sorted field.

**Inputs** (from `--indir`)

| file | used for |
|---|---|
| `player_dk_sims.npy` | per-player per-sim points (to score every lineup) |
| `candidates.csv` | our lineups (stage 3) |
| `field_<N>.csv` | the opponent field(s) (stage 2) — one per size scored |
| `player_pool.csv` | `key → contest_id` for the DK upload |

**Outputs**

| file | contents |
|---|---|
| `candidate_results_<N>.csv` | candidates + `Win% / Top10% / Top100% / AvgPlace` vs the N-entry field, best-first |
| `DK_upload_<N>.csv` | *(only with `--select`)* the chosen export set as DK contest ids, ready to import |

**Key knobs:** `--sizes` (defaults to every `field_<N>.csv` found), `--select N`,
`--objective {win,top10,top100,ev}`, `--entry-fee`, `--utility`
(`Aggressive` / `Balanced` / `Conservative`), and exposure caps `--skill-cap`,
`--dst-cap`, `--team-cap`, `--max-overlap`.

---

## The shared artifact contract

The stages talk to each other through three formats:

- **`player_dk_sims.npy`** — `np.save` of a dict `{entity_key: float32[n_sims]}`.
  Load with `np.load(path, allow_pickle=True).item()`. `n_sims` is inferred from
  the array length. The stack-ownership boost from stage 1 is already applied.
- **`player_pool.csv`** — the entity table. `key` is the join key used
  everywhere; `contest_id` is the DK id for uploads; `own` / `proj` / `ceiling`
  drive selection, overbuild ranking, and ceiling weighting.
- **lineup CSVs** (`field_<N>.csv`, `candidates.csv`) — nine slot columns
  (`QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST`) whose cells are `KEY (TEAM)`.
  The key (not the display name) is stored so scoring maps each slot straight to
  its sim array; extra columns like `Proj`/`Ceiling` are ignored by the scorer.

## Where the field grammar comes from

`field_params_nfl.json` (the QB-stack-size distribution, FLEX mix, bring-back
rates) is **learned from real DK NFL contest standings** by `learn_field.py`,
which builds a name→team crosswalk (`player_names.csv` ⋈ `projections.csv`) and
reads the QB-stack sizes off fully-resolved real lineups. Regenerate it with
`python3 learn_field.py`; override the file any stage reads with `--params`.
