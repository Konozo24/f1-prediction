# F1 Grand Prix Podium Prediction

BMDS2114 Machine Learning — group assignment. Predicts, before each Grand Prix
starts, the probability that each driver finishes on the podium (top 3).

**This repo is M1's workstream: the shared data pipeline + the instance-based
(kNN) model family. The pipeline blocks M2, M3 and M4.**

---

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash)
pip install -r requirements.txt
```

## Get the data

Dataset: [jtrotman/formula-1-race-data](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)

> Provenance note: data **to 2024 comes from Ergast** (retired 2024);
> **2025 onward comes from Jolpica**, the community successor. Our test set is
> 2022–2025, so it straddles that boundary. `verify.py` Q3 checked this:
> **2025 is complete** and in line with earlier seasons.

Download the zip from the dataset page and extract the 14 CSVs **flat** into
`data/raw/` — so `data/raw/results.csv`, not
`data/raw/formula-1-race-data/results.csv`.

No Kaggle API token needed. We download once and freeze, so there is nothing to
script.

Then record the date. **Nobody re-downloads mid-project** — the dataset is
refreshed after every Grand Prix, and different data means incomparable results
between members:

```bash
date -u > data/raw/DOWNLOADED_AT.txt
```

The code reads 8 of the 14 tables (see `config.TABLES`). The rest are left in
place but unused — `qualifying.csv` in particular, in case the team revisits
the decision to exclude it.

## Step 1 — verify the dataset supports the plan

```bash
python -m src.f1.verify | tee data/raw/VERIFY.txt
pytest -q
```

`verify.py` answers six questions before any feature code gets written:

| | Question | Why it must be answered first |
|---|---|---|
| Q1 | Real rows + podiums per season | Replaces the plan's estimates |
| Q2 | Per split, plus events-per-predictor | If train podiums ≈ 360 against 33 features, EPV ≈ 11 — at the floor |
| Q3 | Is 2025 as complete as earlier seasons? | 4 features depend on `lap_times`/`pit_stops`; thin 2025 breaks a test season |
| Q4 | Pit-lane starts (`grid == 0`) | Confirms the recode is needed; gives the count to report |
| Q5 | dtypes — did `\N` parse? | If these are `object`, nothing downstream works |
| Q6 | Which `status` values occur | Makes the DNF rule exact instead of guessed |

Already run — results below under *Verified dataset facts*, full output in
`data/raw/VERIFY.txt`. Re-run it after any change to `config.py` scope or splits.

---

## Layout

```
src/f1/
  config.py      every shared decision: paths, splits, folds, banned columns
  schema.py      the frozen 30-feature contract  <- M2/M3/M4 import this
  features.py    past_mean / past_sum / recode_grid  <- leakage discipline
  verify.py      step 1: prove the data supports the plan
  stub.py        synthetic table so M2-M4 can start before the real one lands
tests/
  test_leakage.py   shift(1), grid encoding, split ordering
  test_schema.py    the 30-feature contract
data/raw/           gitignored, frozen after download
data/processed/      features.parquet (the artifact M2-M4 depend on)
                     features_stub.parquet (plumbing only, no conclusions)
```

## The three rules that matter

1. **`shift(1)` before `rolling()`.** Every rolling feature goes through
   `past_mean` so this is implemented once. Without it the current race sits
   inside its own feature and the answer leaks.
2. **`grid == 0` is a pit-lane start, not pole.** Left raw, a model reads it as
   better than pole — corrupting the strongest feature in the direction that
   looks like good performance. `recode_grid` handles it and keeps the flag.
3. **Splits are by season, never random.** Driver and car performance barely
   change week to week, so a random split tells the model the answer.
   `test_leakage.py` asserts train < validation < test chronologically.

## Shared interfaces (frozen — M2/M3/M4 depend on these)

- **Feature table**: `data/processed/features.parquet`, fixed column names,
  plus `year`, `round`, `split`, `fold`
- **Model wrapper**: `fit(X, y)` and `predict_proba(X)`
- **Out-of-fold file**: `raceId, driverId, fold, model_name, proba`

## Verified dataset facts (step 1, see `data/raw/VERIFY.txt`)

| | |
|---|---|
| Scope 2014–2025 | 5,105 rows · 252 races · 756 podiums |
| Train 2014–19 | 2,487 rows · **363 podiums** |
| Validation 2020–21 | 780 rows · 117 podiums |
| Test 2022–25 | 1,838 rows · 276 podiums |
| **EPV at 33 features** | **11.0** — at the accepted floor of 10 |
| Pit-lane starts (`grid == 0`) | 79 |
| DNF rate | 16.9% |
| 2025 `lap_times` / `pit_stops` | Complete — in line with earlier seasons |

Two data quirks confirmed and handled:

- Real grids reach **22** (2014–16 ran 22-car fields), so `PIT_LANE_GRID = 23`.
- The **2025 Qatar GP (round 23)** has no grid data at all. `recode_grid` refuses
  to impute grid; `drop_races_missing_grid` removes such races whole and returns
  their ids for the methodology.

## Status

- [x] Repo scaffold, venv, pinned dependencies
- [x] `config.py` — splits, folds, banned columns, DNF rule, grid constants
- [x] `features.py` — `past_mean`, `past_sum`, `recode_grid`, `drop_races_missing_grid`
- [x] `schema.py` — frozen 30-feature contract
- [x] `stub.py` — day-3 hand-off table for M2/M3/M4
- [x] Dataset downloaded and frozen
- [x] **Step 1: `verify.py` run, 35 tests passing**
- [ ] Step 2: spine + target + the 30 features → `features.parquet`
