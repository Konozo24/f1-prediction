"""
M1 - instance-based family: kNN -> random-subspace bagging -> NCA metric.

The three rungs are not three arbitrary models. kNN is the family most damaged
by a 30-feature table: as dimensionality grows, distances concentrate and the
nearest and farthest neighbours become nearly equidistant, so "nearest" stops
carrying information. Each rung attacks that specific failure:

    Rung 1  base kNN                 establishes the degradation
    Rung 2  random-subspace bagging  each learner sees ~40% of dimensions
    Rung 3  NCA learned metric       learns which dimensions matter, and reduces

Note on rung 2: the team plan assumes bagging barely helps kNN because kNN is a
stable learner. That is true of *sample* bagging. Random *subspace* bagging is a
different mechanism - the problem here is dimensionality, not variance - so it
should help where plain bagging would not.

Four family-specific constraints, all deliberate:

  * weights="distance", not uniform. With uniform and k=25 the probabilities can
    only take multiples of 1/25 = 0.04, which turns the calibration curve into
    steps and inflates Brier. Distance weighting makes them continuous.
  * p=1 (Manhattan), not Euclidean. Aggarwal et al. (2001) show L1 stays more
    meaningful than L2 as dimension grows. Tuned anyway, but L1 is expected.
  * algorithm="brute". KD-tree and ball-tree degrade to brute force above ~20
    dimensions regardless, so this is honest rather than pretending otherwise.
  * ID columns are excluded. driverId=825 is not "greater than" driverId=3;
    raw integer identifiers are meaningless under any distance metric.

Usage:
    python -m src.models.knn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import PROCESSED
from ..evaluate import summarise, walk_forward
from ..schema import FEATURES, TARGET

K = 25
SEED = 0


def _pre() -> list:
    """Imputer + scaler, fitted INSIDE each fold by the Pipeline.

    Scaling is not optional for this family. constructor_avg_points_5 spans
    0-44 while driver_podium_rate_5 spans 0-1; unscaled, Euclidean distance is
    almost entirely the former and the rest of the table is ignored.
    """
    return [("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]


def rung1_base(k: int = K, p: int = 1) -> Pipeline:
    m = Pipeline(_pre() + [
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance",
                                     p=p, algorithm="brute")),
    ])
    m._name = f"knn_base_k{k}_p{p}"
    return m


def rung2_subspace(k: int = K, n_estimators: int = 50,
                   max_features: float = 0.4) -> Pipeline:
    m = Pipeline(_pre() + [
        ("bag", BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=k, weights="distance",
                                           p=1, algorithm="brute"),
            n_estimators=n_estimators,
            max_features=max_features,      # <- the point: fewer dimensions each
            bootstrap_features=False,
            random_state=SEED, n_jobs=-1)),
    ])
    m._name = f"knn_subspace_mf{max_features}"
    return m


def rung3_nca(k: int = K, n_components: int = 8) -> Pipeline:
    m = Pipeline(_pre() + [
        ("nca", NeighborhoodComponentsAnalysis(n_components=n_components,
                                               random_state=SEED, max_iter=120)),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance",
                                     algorithm="brute")),
    ])
    m._name = f"knn_nca_d{n_components}"
    return m


RUNGS = {"rung1_base": rung1_base,
         "rung2_subspace": rung2_subspace,
         "rung3_nca": rung3_nca}


# --------------------------------------------------------------------------
# Feature ranking for the dimensionality curve
# --------------------------------------------------------------------------

def rank_features(df: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    """Order features by univariate signal, TRAIN SPLIT ONLY.

    Provisional: M2 owns feature selection and this will be replaced by their
    ranking. Computed on train only - ranking features using validation or test
    performance would leak.
    """
    features = features or FEATURES
    tr = df[df.split == "train"]
    corr = tr[features].corrwith(tr[TARGET]).abs().sort_values(ascending=False)
    return corr.index.tolist()


def dimensionality_curve(df: pd.DataFrame, sizes=(3, 5, 8, 12, 20, 30)) -> pd.DataFrame:
    """PR-AUC against feature count - the headline figure for M1's section.

    The expectation to state BEFORE running it: kNN should peak at a small
    number of features and then decline, because adding dimensions dilutes the
    distance metric. The tree family's curve should keep rising. That contrast
    is the most informative comparison in the report.
    """
    order = rank_features(df)
    rows = []
    for n in sizes:
        feats = order[:n]
        for label, make in [("base", rung1_base), ("subspace", rung2_subspace)]:
            per_fold, _ = walk_forward(make(), df, features=feats)
            s = summarise(per_fold)
            rows.append({"rung": label, "n_features": n,
                         "pr_auc": s["pr_auc"], "pr_auc_std": s["pr_auc_std"],
                         "brier": s["brier"], "top3_hit_rate": s["top3_hit_rate"]})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"loaded {len(df):,} rows; walk-forward folds validate on 2017, 2018, 2019\n")

    print("=" * 74)
    print("THREE RUNGS - mean across walk-forward folds, all 30 features")
    print("=" * 74)
    summaries, oofs = [], []
    for name, make in RUNGS.items():
        per_fold, oof = walk_forward(make(), df)
        s = summarise(per_fold)
        summaries.append({"rung": name, "model": s["model_name"],
                          "pr_auc": s["pr_auc"], "sd": s["pr_auc_std"],
                          "roc_auc": s["roc_auc"], "brier": s["brier"],
                          "top3": s["top3_hit_rate"]})
        oofs.append(oof)
        print(f"  {name:<16} done")

    res = pd.DataFrame(summaries)
    print()
    print(res.round(4).to_string(index=False))
    print(f"\n  no-skill PR-AUC = {df[df.split == 'train'][TARGET].mean():.3f}")

    print("\n" + "=" * 74)
    print("DIMENSIONALITY CURVE - does adding features help or hurt kNN?")
    print("=" * 74)
    curve = dimensionality_curve(df)
    print(curve.round(4).pivot(index="n_features", columns="rung",
                               values="pr_auc").to_string())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    res.to_csv(PROCESSED / "m1_rungs.csv", index=False)
    curve.to_csv(PROCESSED / "m1_dimensionality_curve.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(
        PROCESSED / "oof_m1_knn.parquet", index=False)
    print(f"\nwrote m1_rungs.csv, m1_dimensionality_curve.csv, oof_m1_knn.parquet")


if __name__ == "__main__":
    main()
