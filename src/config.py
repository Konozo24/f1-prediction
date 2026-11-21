"""
Shared configuration for the F1 podium prediction project.

Every decision that the whole team depends on lives here, in one place, so that
nobody re-derives it slightly differently in their own module.

BMDS2114 Machine Learning - M1 (pipeline + kNN)
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
# Ergast CSVs encode nulls as the two-character string \N, NOT as empty fields.
# Without this, every numeric column silently loads as object dtype.
NA_VALUES = ["\\N", ""]

# The 8 tables this project uses. Grand Prix only - sprint_results is
# deliberately never loaded, which is what makes "one row per driver per Grand
# Prix" fall out of results.csv with no filtering.
TABLES = [
    "results",
    "races",
    "drivers",
    "constructors",
    "circuits",
    "status",
    "lap_times",
    "pit_stops",
]

# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------
YEAR_MIN = 2014  # hybrid-engine era begins - first season KEPT in the table
YEAR_MAX = 2025  # 2026 excluded: new regulations

# Seasons loaded ONLY to fill rolling-window lookback. These rows are dropped
# before the table is written, so they never become training examples and no
# pre-hybrid race is ever predicted. Without them, "driver_avg_finish_10" at
# the 2014 opener averages zero races, and 11.3% of all rows had a "last 10"
# window computed over fewer than 10 races - a silent quality problem, because
# the column name promises 10 either way.
HISTORY_YEAR_MIN = 2012

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
PODIUM_CUTOFF = 3

# --------------------------------------------------------------------------
# Season splits - always by season, never random.
# --------------------------------------------------------------------------
TRAIN_YEARS = (2014, 2019)
VALIDATION_YEARS = (2020, 2021)
TEST_YEARS = (2022, 2025)


def split_of(year: int) -> str:
    """Map a season to its split. Single source of truth for the whole team."""
    if year <= TRAIN_YEARS[1]:
        return "train"
    if year <= VALIDATION_YEARS[1]:
        return "validation"
    return "test"


# Walk-forward folds inside training: ((first, last), validate_on)
# Out-of-fold predictions from these folds train the stacking meta-learner.
FOLDS = [
    ((2014, 2016), 2017),
    ((2014, 2017), 2018),
    ((2014, 2018), 2019),
]

# Written into the `fold` column for rows that no walk-forward fold validates on
# (every training season before the first fold, plus validation and test).
FOLD_SENTINEL = -1


def fold_of(year: int) -> int:
    """Index of the walk-forward fold that validates on this season.

    Returns FOLD_SENTINEL if no fold validates on it.
    """
    for i, (_, validate_on) in enumerate(FOLDS):
        if year == validate_on:
            return i
    return FOLD_SENTINEL

# --------------------------------------------------------------------------
# Data quirks
# --------------------------------------------------------------------------
# Ergast encodes a pit-lane start as grid = 0. Left raw, a model reads that as
# BETTER than pole position, corrupting the strongest feature in the project in
# the direction that looks like good performance.
#
# Hardcoded constant on purpose: df["grid"].max() + 1 would be a statistic
# computed over the full dataset, including test.
#
# Verified against the data (verify.py Q4): the largest real grid slot in
# 2014-2025 is 22 (2014-2016 ran 22-car fields). 23 therefore places a
# pit-lane start behind every legitimate slot. An earlier value of 21 was a
# bug - it ranked pit-lane starts ahead of 70 real grid-21 and grid-22 rows.
MAX_OBSERVED_GRID = 22
PIT_LANE_GRID = 23

# 79 rows in scope start from the pit lane (verify.py Q4).
N_PIT_LANE_EXPECTED = 79

# A driver is classified as having finished if their status is "Finished" or
# "+N Lap(s)". Everything else - Accident, Engine, Collision, Disqualified - is
# a DNF. Stated explicitly because an undocumented DNF rule makes every
# reliability feature unreproducible.
FINISHED_PATTERN = r"^(Finished|\+\d+ Lap)"

# --------------------------------------------------------------------------
# Outlier thresholds - verified against the raw data
# --------------------------------------------------------------------------
# pit_stops.milliseconds has a median of 25.1s but a 99th percentile of 1523s
# and a max of 3069s (51 minutes). Those are cars parked during a red flag, not
# pit stops. 830 stops exceed 60s. Taking a MEAN over them made
# constructor_avg_pitstop_time range 21-1089s, with 38% of rows above twice the
# median. We drop stops over this threshold and aggregate with a MEDIAN.
MAX_PIT_STOP_SECONDS = 60

# Lap-time standard deviation is meant to measure a driver's consistency, but
# safety-car laps, in/out laps and red flags inflate it enormously (median 11s,
# 95th percentile 104s). We keep only laps within this multiple of the driver's
# own median lap for that race, which removes the non-racing laps.
GREEN_LAP_MULTIPLE = 1.5

# --------------------------------------------------------------------------
# Leakage control
# --------------------------------------------------------------------------
# Recorded during or after the race. These may NEVER appear in X.
#
# positionOrder is the one exception, and only for building the label: creating
# the target is post-race by definition. It must never reach the feature matrix.
BANNED_COLUMNS = {
    "position",
    "positionText",
    "positionOrder",
    "points",
    "laps",
    "status",
    "statusId",
    "time",
    "milliseconds",
    "fastestLap",
    "fastestLapTime",
    "fastestLapSpeed",
    "rank",
}
