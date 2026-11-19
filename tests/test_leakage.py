"""
Leakage tests. These are the project's credibility.

Our literature review criticises published F1 models for using post-race
columns and random splits. These tests are what entitle us to make that
criticism. They also satisfy the rubric's requirement that evaluation steps be
automated.

Run:  pytest -q
"""

import pandas as pd
import pytest

from src.f1.config import BANNED_COLUMNS, FOLDS, split_of
from src.f1.features import past_mean, past_sum, recode_grid


# ---------------------------------------------------------------------------
# The shift(1) discipline - the single most dangerous line in the project
# ---------------------------------------------------------------------------

def _toy() -> pd.DataFrame:
    """Two drivers, three races, known finishing positions."""
    return pd.DataFrame({
        "year":     [2014] * 6,
        "round":    [1, 1, 2, 2, 3, 3],
        "driverId": [1, 2, 1, 2, 1, 2],
        "pos":      [1, 10, 3, 12, 5, 14],
        "pts":      [25, 1, 15, 0, 10, 0],
    }).sort_values(["year", "round"]).reset_index(drop=True)


def test_past_mean_excludes_current_race():
    """A driver's rolling mean must never contain the race being predicted."""
    df = _toy()
    got = past_mean(df, "driverId", "pos", 5)

    # driver 1 finished 1st, 3rd, 5th across rounds 1-3
    d1 = got[df.driverId == 1].tolist()
    assert pd.isna(d1[0]), "first race must be NaN - no history exists"
    assert d1[1] == 1.0, "round 2 sees only round 1"
    assert d1[2] == 2.0, "round 3 sees rounds 1-2 (mean of 1 and 3)"

    # the failure mode we are guarding against: including the current race
    # would give round 2 the mean of (1, 3) = 2.0, not 1.0
    assert d1[1] != 2.0, "LEAK: current race is inside its own feature"


def test_past_mean_first_row_per_group_is_nan():
    df = _toy()
    got = past_mean(df, "driverId", "pos", 3)
    firsts = df.groupby("driverId").head(1).index
    assert got.loc[firsts].isna().all()


def test_past_sum_excludes_current_race():
    """Season points must exclude points scored in the race being predicted."""
    df = _toy()
    got = past_sum(df, ["year", "driverId"], "pts")
    d1 = got[df.driverId == 1].tolist()
    assert d1 == [0.0, 25.0, 40.0], "points-before must lag by one race"


def test_past_mean_respects_grouping():
    """One driver's history must never bleed into another's."""
    df = _toy()
    got = past_mean(df, "driverId", "pos", 5)
    d2 = got[df.driverId == 2].tolist()
    assert pd.isna(d2[0])
    assert d2[1] == 10.0, "driver 2 round 2 must see only driver 2 round 1"


def test_past_mean_empty_input():
    empty = _toy().iloc[0:0]
    assert past_mean(empty, "driverId", "pos", 5).empty


# ---------------------------------------------------------------------------
# Pit-lane grid encoding
# ---------------------------------------------------------------------------

def test_recode_grid_moves_pit_lane_to_the_back():
    grid = pd.Series([1, 0, 5, 0, 20])
    recoded, flag = recode_grid(grid)

    assert (recoded > 0).all(), "no zeros may survive"
    assert recoded.iloc[1] > recoded.iloc[4], "pit lane must be worse than P20"
    assert flag.tolist() == [0, 1, 0, 1, 0], "flag must mark the pit-lane rows"


def test_recode_grid_leaves_normal_starts_untouched():
    grid = pd.Series([1, 2, 3, 20])
    recoded, flag = recode_grid(grid)
    assert recoded.tolist() == [1, 2, 3, 20]
    assert flag.sum() == 0


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("year,expected", [
    (2014, "train"), (2019, "train"),
    (2020, "validation"), (2021, "validation"),
    (2022, "test"), (2025, "test"),
])
def test_split_assignment(year, expected):
    assert split_of(year) == expected


def test_splits_are_chronological():
    """Train must end before validation starts, which must end before test."""
    years = range(2014, 2026)
    by = {s: [y for y in years if split_of(y) == s]
          for s in ("train", "validation", "test")}
    assert max(by["train"]) < min(by["validation"])
    assert max(by["validation"]) < min(by["test"])


def test_walk_forward_folds_never_look_ahead():
    """Each fold validates on a season strictly after its training seasons."""
    for (first, last), validate_on in FOLDS:
        assert first <= last < validate_on, f"fold {first}-{last} -> {validate_on}"


def test_folds_stay_inside_training_years():
    """Walk-forward validation must not touch the real validation or test sets."""
    for (_, last), validate_on in FOLDS:
        assert split_of(validate_on) == "train", (
            f"fold validates on {validate_on}, which is not a training season"
        )
        assert split_of(last) == "train"


# ---------------------------------------------------------------------------
# Banned columns
# ---------------------------------------------------------------------------

def test_banned_set_covers_the_obvious_outcomes():
    for col in ["positionOrder", "points", "laps", "statusId", "fastestLapTime"]:
        assert col in BANNED_COLUMNS


def test_feature_table_has_no_banned_columns():
    """Runs once the real feature table exists; skips until then."""
    from src.f1.config import PROCESSED
    path = PROCESSED / "features.parquet"
    if not path.exists():
        pytest.skip("feature table not built yet")

    cols = set(pd.read_parquet(path).columns)
    # the label is allowed to exist in the file; it just must not be a FEATURE
    leaked = (cols & BANNED_COLUMNS) - {"podium"}
    assert not leaked, f"banned columns present: {sorted(leaked)}"
