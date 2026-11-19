"""
Generate a STUB feature table so M2, M3 and M4 can start today.

This is M1's day-3 hand-off, built with no real data. It has the exact frozen
schema, realistic value ranges, the correct ~15% podium rate, the real season
splits and the real walk-forward fold labels. Every column name matches what
the real table will have.

What teammates CAN do against it:
    - write fit / predict_proba model wrappers
    - build the evaluate() harness and the metrics table
    - build the out-of-fold prediction writer
    - build dashboard layout and chart scaffolding
    - check their pipeline runs end to end without shape errors

What teammates must NOT do against it:
    - draw any conclusion about model performance

The relationships in here are deliberately crude. A model will score far too
well on it, because grid is generated almost deterministically from the label.
That is fine - the point is to exercise the plumbing, not to learn anything.

Usage:
    python -m src.stub
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROCESSED, YEAR_MAX, YEAR_MIN, fold_of, split_of
from .schema import ALL_COLUMNS, FEATURES, ID_COLUMNS, TARGET, assert_valid

RACES_PER_SEASON = 20
DRIVERS_PER_RACE = 20
PODIUM_SLOTS = 3
SEED = 20261118


def make_stub(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    race_id = 1

    for year in range(YEAR_MIN, YEAR_MAX + 1):
        for rnd in range(1, RACES_PER_SEASON + 1):
            # a "true" running order for this race, podium = first 3
            order = rng.permutation(DRIVERS_PER_RACE)

            for slot, driver_idx in enumerate(order):
                podium = int(slot < PODIUM_SLOTS)

                # grid correlates with the outcome, loosely, so that a model
                # trained on the stub does not simply fail to converge
                grid = int(np.clip(slot + 1 + rng.normal(0, 3), 1, 20))

                rows.append({
                    "raceId": race_id,
                    "year": year,
                    "round": rnd,
                    "split": split_of(year),
                    "fold": fold_of(year),

                    "driverId": int(driver_idx) + 1,
                    "constructorId": int(driver_idx) // 2 + 1,   # 2 drivers per team
                    "circuitId": rnd,

                    # starting position
                    "grid": grid,
                    "front_row": int(grid <= 2),
                    "pit_lane_start": int(rng.random() < 0.01),

                    # driver form
                    "driver_avg_finish_3": rng.uniform(1, 20),
                    "driver_avg_finish_5": rng.uniform(1, 20),
                    "driver_avg_finish_10": rng.uniform(1, 20),
                    "driver_podium_rate_3": rng.uniform(0, 1),
                    "driver_podium_rate_5": rng.uniform(0, 1),
                    "driver_podium_rate_10": rng.uniform(0, 1),
                    "driver_points_last_5": rng.uniform(0, 120),
                    "driver_dnf_rate_10": rng.uniform(0, 0.4),
                    "driver_career_races": rng.integers(0, 350),
                    "driver_prior_wins": rng.integers(0, 60),
                    "driver_age": rng.uniform(18, 43),

                    # team form
                    "constructor_avg_points_5": rng.uniform(0, 80),
                    "constructor_avg_finish_5": rng.uniform(1, 20),
                    "constructor_dnf_rate_10": rng.uniform(0, 0.4),
                    "constructor_podium_rate_10": rng.uniform(0, 1),

                    # circuit
                    "driver_avg_at_circuit": rng.uniform(1, 20),
                    "constructor_avg_at_circuit": rng.uniform(1, 20),
                    "circuit_grid_hold": rng.uniform(0, 1),
                    "circuit_dnf_rate": rng.uniform(0, 0.4),

                    # season context
                    "round_number": rnd,
                    "driver_points_before": rng.uniform(0, 450),
                    "driver_championship_pos": rng.integers(1, 21),
                    "driver_gap_to_leader": rng.uniform(0, 400),

                    # lap and pit  (<- at risk if verify.py Q3 shows 2025 thin)
                    "driver_avg_laptime_rank": rng.uniform(1, 20),
                    "driver_laptime_consistency": rng.uniform(0.1, 3.0),
                    "driver_avg_positions_gained_lap1": rng.normal(0, 2),
                    "constructor_avg_pitstop_time": rng.uniform(2.0, 4.5),

                    TARGET: podium,
                })
            race_id += 1

    df = pd.DataFrame(rows)[ALL_COLUMNS]

    # ~2% missing in the lap/pit block, mirroring rookies and new circuits, so
    # teammates exercise their imputation code rather than discovering NaNs later
    for col in ["driver_avg_laptime_rank", "constructor_avg_pitstop_time"]:
        mask = rng.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    df = make_stub()
    assert_valid(df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "features_stub.parquet"
    df.to_parquet(out, index=False)

    print(f"wrote {out}")
    print(f"  rows      : {len(df):,}")
    print(f"  features  : {len(FEATURES)}  (+{len(ID_COLUMNS)} id columns)")
    print(f"  podiums   : {df[TARGET].sum():,}  ({df[TARGET].mean():.1%})")
    print("\nrows and podiums per split:")
    print(df.groupby("split").agg(rows=(TARGET, "size"), podiums=(TARGET, "sum"))
            .reindex(["train", "validation", "test"]))
    print("\nwalk-forward folds (inside train only):")
    print(df[df.fold >= 0].groupby(["fold", "year"]).size())
    print("\n*** STUB DATA. Plumbing only - no performance conclusions. ***")


if __name__ == "__main__":
    main()
