"""
The frozen feature-table schema.

This is the contract between M1 and everyone else. The plan says interfaces are
agreed on day 1 and then frozen; this file IS that interface. Column names here
are what M2's feature selection, M3's dashboard and M4's evaluate() harness all
import, so nobody hardcodes a string and nobody renames anything unilaterally.

30 engineered features + 3 ID columns, per the project plan.

BMDS2114 Machine Learning - M1 (pipeline + kNN)
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 30 features, grouped exactly as the plan groups them.
# Every one uses only information available BEFORE the race starts.
# --------------------------------------------------------------------------

STARTING_POSITION = [
    "grid",                             # starting slot, after penalties
    "front_row",                        # grid <= 2
    "pit_lane_start",                   # Ergast encodes this as grid == 0
]

DRIVER_FORM = [
    "driver_avg_finish_3",
    "driver_avg_finish_5",
    "driver_avg_finish_10",
    "driver_podium_rate_3",
    "driver_podium_rate_5",
    "driver_podium_rate_10",
    "driver_points_last_5",
    "driver_dnf_rate_10",
    "driver_career_races",
    "driver_prior_wins",
    "driver_age",
]

TEAM_FORM = [
    "constructor_avg_points_5",
    "constructor_avg_finish_5",
    "constructor_dnf_rate_10",
    "constructor_podium_rate_10",
]

CIRCUIT = [
    "driver_avg_at_circuit",
    "constructor_avg_at_circuit",
    "circuit_grid_hold",                # how strongly grid order survives here
    "circuit_dnf_rate",
]

SEASON_CONTEXT = [
    "round_number",
    "driver_points_before",
    "driver_championship_pos",
    "driver_gap_to_leader",
]

LAP_AND_PIT = [
    "driver_avg_laptime_rank",
    "driver_laptime_consistency",
    "driver_avg_positions_gained_lap1",
    "constructor_avg_pitstop_time",
]

FEATURES: list[str] = (
    STARTING_POSITION
    + DRIVER_FORM
    + TEAM_FORM
    + CIRCUIT
    + SEASON_CONTEXT
    + LAP_AND_PIT
)

FEATURE_GROUPS: dict[str, list[str]] = {
    "starting_position": STARTING_POSITION,
    "driver_form": DRIVER_FORM,
    "team_form": TEAM_FORM,
    "circuit": CIRCUIT,
    "season_context": SEASON_CONTEXT,
    "lap_and_pit": LAP_AND_PIT,
}

# Kept for the tree and neural-network families only. kNN and SVM must NOT use
# these: an arbitrary integer id is meaningless under a distance metric.
ID_COLUMNS = ["driverId", "constructorId", "circuitId"]

# Bookkeeping columns. Present in the parquet, never passed as features.
META_COLUMNS = ["raceId", "year", "round", "split", "fold"]

TARGET = "podium"

ALL_COLUMNS = META_COLUMNS + ID_COLUMNS + FEATURES + [TARGET]

# These four depend on lap_times / pit_stops. If verify.py Q3 shows 2025 is
# thin, these are the features at risk across a whole test season.
LAP_PIT_DEPENDENT = LAP_AND_PIT


def assert_valid(df) -> None:
    """Fail loudly if a feature table does not honour the contract."""
    from .config import BANNED_COLUMNS

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")

    leaked = (set(df.columns) & BANNED_COLUMNS) - {TARGET}
    if leaked:
        raise ValueError(f"feature table contains banned columns: {sorted(leaked)}")

    if len(FEATURES) != 30:
        raise ValueError(f"expected 30 features, schema defines {len(FEATURES)}")
