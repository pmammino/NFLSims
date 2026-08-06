# Field & Candidate build scripts

Two standalone scripts split the two halves of lineup construction into their
own entry points:

| script | question it answers | what it produces |
|---|---|---|
| `build_field.py` | *Who am I playing against?* | a realistic simulated opponent **field** (`out/field_<N>.csv`) |
| `build_candidates.py` | *What should I enter?* | a pool of sharp **candidate** lineups (`out/candidates.csv`) |

Both build on the shared engine modules (`nfl_ingest`, `sim_engine`,
`field_builder`, `field_simulator`, `stack_signal`, `ownership_model`) so there
is one source of truth for the logic — the scripts are thin, documented
orchestrators dedicated to one job each. The full pipeline
(`run_sim.py` / the Streamlit app) does both plus scoring and export; these
scripts let you run, inspect, and tune each half on its own.

---

## The core idea: field vs candidate are opposite postures

Both halves use the **same** builder and the **same** DK NFL construction grammar
(QB + pass-catcher primary stack + a bring-back from the opponent + fills + DST,
under the $50k cap). They differ only in *how they choose*:

| | **Field** (`build_field.py`) | **Candidate** (`build_candidates.py`) |
|---|---|---|
| stack **team** / QB | by projected **ownership** (models the crowd) | **ownership-blind** (`uniform`) — not anchored to chalk |
| **player** picks (skill/FLEX) | by ownership | by simulated **ceiling** (p90) — elite upside, not chalk filler |
| QB-stack **size** | the field-shaped learned distribution | **tilted bigger** (winners stack bigger than the crowd) |
| **bring-back** | the field's sampled rate | **forced more often** (locks in game correlation) |
| goal | be a *believable opponent* | be a *tournament-winning* build |

The field is what you must beat; the candidates are your attempts to beat it.
Keeping them in separate scripts makes that asymmetry explicit.

---

## `build_field.py`

A pure ownership-imitation field is **too soft** — it would hand a no-skill
random lineup a positive edge, which is impossible. The script layers three
kinds of realism on top of the raw ownership imitation:

1. **Ingest** the slate (DK pool + projected ownership + range-of-outcomes
   projections + schedule).
2. **Simulate** the slate (`--n-sims`): correlated, game-consistent DK points
   for every player. Needed to rank the overbuild by projection and to weight
   the sharp builds by each player's ceiling (p90).
3. **Stack-ownership ceiling signal** (`--stack-boost`): popular offenses get a
   small bump to their high-end sims, tied to projected stack ownership. The
   same boosted scores grade candidates downstream, so it's a coherent
   re-weighting of reality, not a thumb on the scale.
4. **Build the field** per contest size with three realism layers:
   - **Size model** — a chalk "temperature" (beta) concentrates chalk in small
     fields and flattens it in large ones (`--chalk-sensitivity`), and the
     stack-size distribution tilts toward bigger stacks for bigger fields
     (`--stack-tilt`).
   - **Sharp / chalk mix** (`--sharp-frac`) — this share of the field is built
     "sharp" (bigger stacks, ceiling-weighted players, forced bring-backs) on
     the ownership base, so it still stacks popular teams; the rest is a soft
     chalk tail.
   - **Submitted, not random** (`--overbuild`) — overbuild N× and keep the
     highest-projection lineups, because the field you face is the *better*
     lineups people actually submit, not raw builder draws.
   - `--own-uncertainty` additionally mixes the field over fresh ownership draws.
5. **Write** `out/field_<N>.csv` (one row per opponent lineup: slot columns as
   `NAME (TEAM)`, plus `Salary`, `QBstack`, `Proj`) and print the realized
   stack-size distribution, salary, and chalk concentration.

```bash
# realistic fields for three contest sizes
python3 build_field.py --sizes 1000 6000 20000 --n-sims 8000

# one size, dial the realism knobs
python3 build_field.py --sizes 6000 --sharp-frac 0.75 --overbuild 2.0 --stack-boost 0.05

# naive ownership field (no sim) — the "too soft" baseline, for comparison
python3 build_field.py --sizes 6000 --fast
```

Key flags: `--sizes`, `--n-sims`, `--sharp-frac`, `--overbuild`, `--stack-boost`,
`--chalk-sensitivity`, `--stack-tilt`, `--own-uncertainty`, `--fast`, `--params`,
`--outdir`.

---

## `build_candidates.py`

Deliberately builds sharp, ceiling-seeking lineups — the shapes the win region
favors over the crowd.

1. **Ingest** the slate.
2. **Simulate** (`--n-sims`): used to derive each player's ceiling (p90, the
   candidate selection weight) and to report each finished lineup's projection
   and ceiling.
3. **Stack-ownership ceiling signal** (`--stack-boost`): the same bump used for
   the field, so the ceilings the builder chases match how candidates are graded.
4. **Build candidates** with the sharp posture:
   - `uniform=True` — ownership-blind stack-team / QB choice.
   - ceiling-weighted skill/FLEX picks (`--no-cand-upside` turns this off).
   - bigger QB stacks (`--cand-stack-strength`).
   - forced primary-opponent bring-back (`--cand-bringback`).
   - optional `--jitter` to diversify near-equivalent players across the pool.
5. **Write** `out/candidates.csv` (slot columns as `NAME (TEAM)`, plus `Salary`,
   `QBstack`, and the simulated `Proj` / `Ceiling` / `Std`), sorted by ceiling,
   and print the pool's stack-shape mix and projection spread.

```bash
# a sharp candidate pool
python3 build_candidates.py --num-candidates 10000 --n-sims 8000

# push stacks bigger, more bring-backs, diversify
python3 build_candidates.py --num-candidates 5000 --cand-stack-strength 0.8 \
        --cand-bringback 0.4 --jitter 0.2

# ownership-blind only (no ceiling weighting) — closer to a plain random pool
python3 build_candidates.py --num-candidates 5000 --no-cand-upside
```

Key flags: `--num-candidates`, `--n-sims`, `--cand-stack-strength`,
`--cand-bringback`, `--no-cand-upside`, `--jitter`, `--stack-boost`, `--fast`,
`--params`, `--outdir`.

---

## Where the field "grammar" comes from

Both scripts read the QB-stack-size distribution, FLEX mix, and bring-back rates
from `field_params_nfl.json`. Those weights are **learned from real DK NFL
contest standings** by `learn_field.py` (which builds a name→team crosswalk from
`player_names.csv` ⋈ `projections.csv` and reads the QB-stack sizes off
fully-resolved real lineups). Regenerate them with:

```bash
python3 learn_field.py
```

Override the file either script reads with `--params <path>`.

## Fast (no-sim) mode

`--fast` on either script skips the simulation. It's a quick way to see the
naive, ownership-only behaviour — a soft field, or ownership-blind candidates
with no ceiling weighting — and makes the value of the sim-driven realism
features obvious by contrast. For any real analysis, run with the sim.
