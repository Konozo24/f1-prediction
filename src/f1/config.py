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
ROOT = Path(__file__).resolve().parents[2]
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
YEAR_MIN = 2014  # hybrid-engine era begins
YEAR_MAX = 2025  # 2026 excluded: new regulations

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

# --------------------------------------------------------------------------
# Data quirks
# --------------------------------------------------------------------------
# Ergast encodes a pit-lane start as grid = 0. Left raw, a model reads that as
# BETTER than pole position, corrupting the strongest feature in the project in
# the direction that looks like good performance.
#
# Hardcoded constant on purpose: df["grid"].max() + 1 would be a statistic
# computed over the full dataset, including test.
PIT_LANE_GRID = 21

# A driver is classified as having finished if their status is "Finished" or
# "+N Lap(s)". Everything else - Accident, Engine, Collision, Disqualified - is
# a DNF. Stated explicitly because an undocumented DNF rule makes every
# reliability feature unreproducible.
FINISHED_PATTERN = r"^(Finished|\+\d+ Lap)"

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
