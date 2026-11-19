"""Schema contract tests. M2/M3/M4 depend on these column names not moving."""

import pandas as pd
import pytest

from src.config import BANNED_COLUMNS, PROCESSED
from src.schema import (ALL_COLUMNS, FEATURE_GROUPS, FEATURES, ID_COLUMNS,
                           TARGET, assert_valid)
from src.stub import make_stub


def test_exactly_thirty_features():
    assert len(FEATURES) == 30, f"plan promises 30, schema has {len(FEATURES)}"


def test_no_duplicate_feature_names():
    assert len(FEATURES) == len(set(FEATURES))


def test_groups_sum_to_all_features():
    flat = [c for g in FEATURE_GROUPS.values() for c in g]
    assert sorted(flat) == sorted(FEATURES)


def test_group_counts_match_the_plan():
    expected = {"starting_position": 3, "driver_form": 11, "team_form": 4,
                "circuit": 4, "season_context": 4, "lap_and_pit": 4}
    assert {k: len(v) for k, v in FEATURE_GROUPS.items()} == expected


def test_no_feature_is_a_banned_column():
    assert not (set(FEATURES) & BANNED_COLUMNS)


def test_ids_are_not_features():
    """kNN and SVM must never see raw integer ids under a distance metric."""
    assert not (set(ID_COLUMNS) & set(FEATURES))


def test_stub_honours_the_contract():
    assert_valid(make_stub())


def test_stub_has_realistic_class_balance():
    df = make_stub()
    assert 0.13 < df[TARGET].mean() < 0.17, "podium rate should be ~15%"


def test_stub_has_exactly_three_podiums_per_race():
    df = make_stub()
    assert (df.groupby("raceId")[TARGET].sum() == 3).all()


def test_stub_splits_are_chronological():
    df = make_stub()
    yr = df.groupby("split")["year"].agg(["min", "max"])
    assert yr.loc["train", "max"] < yr.loc["validation", "min"]
    assert yr.loc["validation", "max"] < yr.loc["test", "min"]


def test_stub_folds_only_inside_train():
    df = make_stub()
    assert (df[df.fold >= 0]["split"] == "train").all()


def test_assert_valid_rejects_a_missing_column():
    bad = make_stub().drop(columns=["grid"])
    with pytest.raises(ValueError, match="missing columns"):
        assert_valid(bad)


def test_assert_valid_rejects_a_banned_column():
    bad = make_stub()
    bad["positionOrder"] = 1
    with pytest.raises(ValueError, match="banned columns"):
        assert_valid(bad)


def test_stub_parquet_roundtrips():
    path = PROCESSED / "features_stub.parquet"
    if not path.exists():
        pytest.skip("run: python -m src.stub")
    df = pd.read_parquet(path)
    assert list(df.columns) == ALL_COLUMNS
    assert_valid(df)
