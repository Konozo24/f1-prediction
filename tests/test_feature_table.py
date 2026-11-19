"""Invariants the real feature table must satisfy. Skips until it is built."""

import pandas as pd
import pytest

from src.config import (BANNED_COLUMNS, MAX_OBSERVED_GRID, PIT_LANE_GRID,
                           PROCESSED, YEAR_MAX, YEAR_MIN)
from src.schema import ALL_COLUMNS, FEATURES, TARGET, assert_valid

PATH = PROCESSED / "features.parquet"


@pytest.fixture(scope="module")
def df():
    if not PATH.exists():
        pytest.skip("run: python -m src.build_features")
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


def test_continuing_drivers_have_form_at_the_2014_opener(df):
    """2012-13 are loaded as lookback, so returning drivers start with history.

    Before this, every driver had NaN form at the 2014 opener and 11.3% of all
    rows had a "last 10" window built from fewer than 10 races.
    """
    opener = df[(df.year == 2014) & (df["round"] == 1)]
    assert opener.driver_avg_finish_10.notna().mean() > 0.8, (
        "most of the 2014 grid raced in 2013 and should have form"
    )
    ham = opener[opener.driverId == 1]
    assert ham.driver_avg_finish_10.notna().all(), "Hamilton raced all of 2013"


def test_rolling_form_is_nan_only_for_drivers_with_no_lookback(df):
    """NaN form must be rare and confined to genuine newcomers."""
    assert df.driver_avg_finish_10.isna().mean() < 0.02


def test_lookback_seasons_are_not_in_the_table(df):
    """2012-13 fill the windows, then must be dropped - never predicted."""
    from src.config import YEAR_MIN
    assert df.year.min() == YEAR_MIN


def test_career_totals_come_from_full_history_not_the_window(df):
    """Regression: these were counted inside 2014+ only.

    That made Hamilton - 128 starts and 22 wins before our data begins - look
    identical to an actual rookie at the 2014 opener.
    """
    ham = df[df.driverId == 1].sort_values(["year", "round"]).iloc[0]
    assert ham.driver_career_races > 100, "pre-2014 starts must be counted"
    assert ham.driver_prior_wins > 20, "pre-2014 wins must be counted"


def test_true_debutants_still_start_at_zero(df):
    """A driver whose first ever race is inside the window must have 0."""
    first = df.sort_values(["year", "round"]).groupby("driverId").head(1)
    debutants = first[first.driver_career_races == 0]
    assert len(debutants) > 0, "some drivers must debut inside 2014-2025"
    assert (debutants.driver_prior_wins == 0).all()


def test_pit_stop_times_are_physically_plausible(df):
    """Regression: red-flag stops (up to 3069s) were being averaged in."""
    s = df.constructor_avg_pitstop_time.dropna()
    assert s.max() < 60, f"max {s.max():.1f}s is not a pit stop"
    assert 15 < s.median() < 40, s.median()


def test_laptime_consistency_excludes_non_racing_laps(df):
    """Regression: safety-car and in/out laps inflated std to 285s."""
    s = df.driver_laptime_consistency.dropna()
    assert s.max() < 60, f"max {s.max():.1f}s std suggests safety-car laps"


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
