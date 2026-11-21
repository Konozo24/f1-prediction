"""
Optuna hyperparameter search for the three kNN rungs.

Every trial is scored by mean PR-AUC across the SAME walk-forward folds used
in knn.py (train on 2014-2016/17/18, validate on 2017/18/19) - never on the
real validation (2020-21) or test (2022-25) splits. Those stay untouched until
a config is already chosen, which is the whole point of having walk-forward
folds in the first place: picking hyperparameters by looking at validation
performance IS a form of fitting to validation, so a separate set of folds
exists so real validation can still do its job later (threshold + calibration).

Search spaces are deliberately narrow and justified, not thrown wide open:

  * p in {1, 2} only. Aggarwal et al. (2001) found L1 (Manhattan) distances
    stay more meaningful than L2 (Euclidean) as dimensionality grows. We
    expect p=1 to win; the search either confirms or overturns that, and
    either result is reportable.
  * weights="distance" only, never tried against "uniform". Uniform weights
    with k=25 can only produce probabilities that are multiples of 1/25,
    which quantises the calibration curve into visible steps - already ruled
    out on theoretical grounds in knn.py, not re-litigated here.
  * algorithm="brute" is fixed, not searched. KD-tree/ball-tree degrade to
    brute force above ~20 dimensions regardless of what Optuna picks.

Usage:
    python -m src.models.tune_knn              # all three rungs, 40 trials each
    python -m src.models.tune_knn --trials 100  # more trials
"""

from __future__ import annotations

import argparse
import json

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import PROCESSED
from ..evaluate import summarise, walk_forward

optuna.logging.set_verbosity(optuna.logging.WARNING)

PRE = [("impute", SimpleImputer(strategy="median")),
       ("scale", StandardScaler())]


def _score(pipe: Pipeline, df: pd.DataFrame) -> float:
    per_fold, _ = walk_forward(pipe, df)
    return float(summarise(per_fold)["pr_auc"])


def objective_base(trial: optuna.Trial, df: pd.DataFrame) -> float:
    k = trial.suggest_int("n_neighbors", 5, 150)
    p = trial.suggest_categorical("p", [1, 2])
    pipe = Pipeline(PRE + [("knn", KNeighborsClassifier(
        n_neighbors=k, weights="distance", p=p, algorithm="brute"))])
    return _score(pipe, df)


def objective_subspace(trial: optuna.Trial, df: pd.DataFrame) -> float:
    k = trial.suggest_int("n_neighbors", 5, 100)
    p = trial.suggest_categorical("p", [1, 2])
    n_estimators = trial.suggest_int("n_estimators", 20, 100, step=10)
    max_features = trial.suggest_float("max_features", 0.2, 0.8)
    pipe = Pipeline(PRE + [("bag", BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=k, weights="distance",
                                       p=p, algorithm="brute"),
        n_estimators=n_estimators, max_features=max_features,
        bootstrap_features=False, random_state=0, n_jobs=-1))])
    return _score(pipe, df)


def objective_nca(trial: optuna.Trial, df: pd.DataFrame) -> float:
    k = trial.suggest_int("n_neighbors", 5, 100)
    n_components = trial.suggest_int("n_components", 2, 20)
    pipe = Pipeline(PRE + [
        ("nca", NeighborhoodComponentsAnalysis(
            n_components=n_components, random_state=0, max_iter=100)),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance",
                                     algorithm="brute")),
    ])
    return _score(pipe, df)


OBJECTIVES = {"rung1_base": objective_base,
              "rung2_subspace": objective_subspace,
              "rung3_nca": objective_nca}


def tune(name: str, df: pd.DataFrame, n_trials: int, seed: int = 0) -> optuna.Study:
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed),
                                study_name=name)
    fn = OBJECTIVES[name]
    study.optimize(lambda t: fn(t, df), n_trials=n_trials, show_progress_bar=False)
    return study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"tuning on walk-forward folds only (2017/18/19) - "
          f"{args.trials} trials per rung\n")

    results = {}
    for name in OBJECTIVES:
        study = tune(name, df, args.trials)
        results[name] = {"best_pr_auc": study.best_value,
                         "best_params": study.best_params,
                         "n_trials": len(study.trials)}
        print(f"  {name:<16} best PR-AUC {study.best_value:.4f}  "
              f"params {study.best_params}")

    out = PROCESSED / "m1_tuning.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print("\nNEXT: plug study.best_params into knn.py's RUNGS, re-run walk_forward,")
    print("and ONLY THEN evaluate once on the real validation split (2020-21).")


if __name__ == "__main__":
    main()
