"""
Metrics and the walk-forward cross-validation runner.

NOTE ON OWNERSHIP: the *shared* evaluate() harness is M4's component. This is
M1's local version so that kNN work is not blocked waiting for it. When M4's
lands, delete `metrics()` here and import theirs - the column names below are
deliberately the ones the team plan specifies, so the swap is a one-line change.

Why a fold runner at all: kNN with weights="distance" scores a perfect 1.000 on
its own training data, because every training point sits at distance 0 from
itself and receives infinite weight. The train/validation gap that M2 and M3
will use to detect overfitting is therefore meaningless for this family. Only
out-of-fold scores mean anything here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

from .config import FOLDS
from .schema import FEATURES, TARGET


def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """One row of the shared comparison table.

    PR-AUC is the headline: podiums are ~15% of rows, so accuracy is dominated
    by the majority class. On this dataset a Random Forest scores exactly the
    same accuracy (0.908) as the one-line "podium if grid <= 3" rule, while
    PR-AUC separates them 0.682 to 0.525.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    yhat = (proba >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "f1": f1_score(y_true, yhat, zero_division=0),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "accuracy": accuracy_score(y_true, yhat),
        "no_skill_pr_auc": float(y_true.mean()),
        "majority_accuracy": float(1 - y_true.mean()),
    }


def race_top3_hit_rate(df: pd.DataFrame, proba, target: str = TARGET) -> float:
    """Of the 3 real podium finishers, how many are in our top 3 for that race?

    A domain metric the generic ones miss: exactly three drivers podium per
    race, so ranking within a race is the thing that actually matters.
    """
    tmp = df[["raceId", target]].copy()
    tmp["proba"] = np.asarray(proba)
    hits = (tmp.sort_values("proba", ascending=False)
               .groupby("raceId")
               .head(3)
               .groupby("raceId")[target]
               .sum())
    return float(hits.mean() / 3.0)


def walk_forward(estimator, df: pd.DataFrame, features: list[str] | None = None,
                 target: str = TARGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit/score the estimator across the walk-forward folds.

    Fold k trains on its seasons and predicts the next one, so nothing is ever
    scored on data it was fitted on. Scalers and imputers live inside the
    estimator pipeline, so they are refitted per fold - never on the full table.

    Returns (per_fold_metrics, out_of_fold_predictions). The OOF frame is in the
    schema the team plan specifies for stacking:
        raceId, driverId, fold, model_name, proba
    """
    features = features or FEATURES
    name = getattr(estimator, "_name", type(estimator).__name__)

    rows, oof = [], []
    for k, ((first, last), validate_on) in enumerate(FOLDS):
        tr = df[(df.year >= first) & (df.year <= last)]
        va = df[df.year == validate_on]
        if va.empty or tr.empty:
            continue

        model = clone(estimator).fit(tr[features], tr[target])
        proba = model.predict_proba(va[features])[:, 1]

        rows.append({"model_name": name, "fold": k,
                     "train": f"{first}-{last}", "validate_on": validate_on,
                     "n_train": len(tr), "n_val": len(va),
                     **metrics(va[target], proba),
                     "top3_hit_rate": race_top3_hit_rate(va, proba, target)})

        oof.append(pd.DataFrame({"raceId": va.raceId.values,
                                 "driverId": va.driverId.values,
                                 "fold": k, "model_name": name,
                                 "proba": proba}))

    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def summarise(per_fold: pd.DataFrame) -> pd.Series:
    """Mean across folds - the number used to choose between models."""
    num = per_fold.select_dtypes("number").drop(columns=["fold"], errors="ignore")
    out = num.mean()
    out["pr_auc_std"] = per_fold["pr_auc"].std()
    out["model_name"] = per_fold["model_name"].iloc[0]
    return out


def best_threshold(y_true, proba, metric=f1_score) -> tuple[float, float]:
    """Pick a decision threshold on the PR curve.

    kNN cannot use class_weight="balanced" - KNeighborsClassifier has no such
    parameter - so threshold tuning is this family's main lever against the 15%
    class imbalance. Must be chosen on validation folds, never on test.
    """
    y_true = np.asarray(y_true)
    grid = np.quantile(np.asarray(proba), np.linspace(0.5, 0.999, 120))
    scores = [metric(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])
