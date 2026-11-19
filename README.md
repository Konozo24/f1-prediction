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
> 2022–2025, so it straddles that boundary — `verify.py` Q3 checks whether 2025
> is as complete as earlier seasons.

**Option A — Kaggle CLI** (needs an API token: kaggle.com → Settings → Create
New API Token, then save `kaggle.json` to `~/.kaggle/`):

```bash
pip install kaggle
kaggle datasets download -d jtrotman/formula-1-race-data -p data/raw --unzip
```

**Option B — manual**: download the zip from the dataset page and extract the
CSVs directly into `data/raw/` (flat, no subfolder).

Then freeze it. **Nobody re-downloads mid-project** — different data means
incomparable results:

```bash
date -u > data/raw/DOWNLOADED_AT.txt
```

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

Paste Q1–Q3 into the team chat. Q3 may change the split design, and that
decision is far cheaper now than after M2–M4 have trained.

---

## Layout

```
src/f1/
  config.py      every shared decision: paths, splits, folds, banned columns
  features.py    past_mean / past_sum / recode_grid  <- the leakage discipline
  verify.py      step 1
tests/
  test_leakage.py
data/raw/        gitignored, frozen after download
data/processed/   features.parquet (the artifact M2-M4 depend on)
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

## Status

- [x] Repo scaffold, venv, pinned dependencies
- [x] `config.py` — splits, folds, banned columns, DNF rule, pit-lane constant
- [x] `features.py` — `past_mean`, `past_sum`, `recode_grid`
- [x] `tests/test_leakage.py` — 17 passing
- [ ] **Download the dataset** ← only remaining blocker on step 1
- [ ] Run `verify.py`, post Q1–Q3 to the team
- [ ] Step 2: spine + target + the 30 features → frozen parquet
