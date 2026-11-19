"""Invariants the real feature table must satisfy. Skips until it is built."""

import pandas as pd
import pytest

from src.f1.config import (BANNED_COLUMNS, MAX_OBSERVED_GRID, PIT_LANE_GRID,
                           PROCESSED, YEAR_MAX, YEAR_MIN)
from src.f1.schema import ALL_COLUMNS, FEATURES, TARGET, assert_valid

PATH = PROCESSED / "features.parquet"


@pytest.fixture(scope="module")
def df():
    if not PATH.exists():
        pytest.skip("run: python -m src.f1.build_features")
    return pd.read_parquet(PATH)


def test_contract(df):
    assert list(df.columns) == ALL_COLUMNS
    assert_valid(df)


def test_no_banned_columns(df):
    assert not (set(df.columns) & BANNED_COLUMNS)


def test_one_row_per_driver_per_race(df):
    assert not df.duplicated(["raceId", "driverId"]).any()


def test_exactly_three_podiums_per_race(df):
    """Structural truth of the sport. If this fails, the join dropped rows."""
    counts = df.groupby("raceId")[TARGET].sum()
    assert (counts == 3).all(), counts[counts != 3]


def test_in_scope_years_only(df):
    assert df.year.min() >= YEAR_MIN
    assert df.year.max() <= YEAR_MAX


def test_grid_has_no_zeros_and_pit_lane_is_last(df):
    assert (df.grid > 0).all(), "grid == 0 must have been recoded"
    assert df.grid.max() == PIT_LANE_GRID
    assert (df.grid[df.pit_lane_start == 1] == PIT_LANE_GRID).all()
    real = df.grid[df.pit_lane_start == 0]
    assert real.max() <= MAX_OBSERVED_GRID


def test_pit_lane_flag_matches_recoded_grid(df):
    assert int(df.pit_lane_start.sum()) == int((df.grid == PIT_LANE_GRID).sum())


def test_front_row_is_consistent_with_grid(df):
    assert (df.front_row == (df.grid <= 2).astype(int)).all()


def test_no_feature_too_correlated_with_target(df):
    """The automated leakage assertion our literature review relies on."""
    c = df[FEATURES].corrwith(df[TARGET]).abs()
    assert c.max() < 0.95, c.sort_values(ascending=False).head()


def test_first_race_of_a_career_has_no_history(df):
    """shift(1) must leave the first appearance empty, not filled."""
    first = df.sort_values(["year", "round"]).groupby("driverId").head(1)
    assert first.driver_avg_finish_3.isna().all()
    assert (first.driver_career_races == 0).all()
    assert (first.driver_prior_wins == 0).all()


def test_splits_are_chronological(df):
    yr = df.groupby("split")["year"].agg(["min", "max"])
    assert yr.loc["train", "max"] < yr.loc["validation", "min"]
    assert yr.loc["validation", "max"] < yr.loc["test", "min"]


def test_folds_only_label_training_seasons(df):
    assert (df.loc[df.fold >= 0, "split"] == "train").all()


def test_rates_are_within_bounds(df):
    for col in ["driver_podium_rate_5", "driver_dnf_rate_10",
                "constructor_dnf_rate_10", "circuit_dnf_rate",
                "circuit_grid_hold"]:
        s = df[col].dropna()
        assert s.between(0, 1).all(), f"{col} outside [0,1]"


def test_driver_age_is_plausible(df):
    s = df.driver_age.dropna()
    assert s.between(17, 50).all(), (s.min(), s.max())


def test_class_balance_is_about_fifteen_percent(df):
    assert 0.14 < df[TARGET].mean() < 0.16
