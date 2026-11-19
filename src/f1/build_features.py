"""
STEP 2: build the frozen feature table.

Produces data/processed/features.parquet - one row per driver per Grand Prix,
30 features + 3 id columns, all computed from information available BEFORE the
race starts. This is the artifact M2, M3 and M4 depend on.

Design rules, in priority order:

1.  Every rolling feature goes through the helpers in features.py, so the
    shift(1) discipline is implemented once rather than thirty times.

2.  Constructor and circuit features are rolled at the CONSTRUCTOR-RACE and
    CIRCUIT-RACE level, then merged back. Rolling them at driver-row level
    would make "last 5" mean the last 5 driver-rows, i.e. 2.5 races for a
    two-car team. This is a silent, plausible-looking bug.

3.  positionOrder, points, laps and statusId are used ONLY to build the label
    and shifted history. They never appear in the output table.

4.  NaNs are left in. Imputation belongs inside each member's fold pipeline -
    filling here would compute the fill value across train+validation+test and
    leak it into everyone's features.

Usage:
    python -m src.f1.build_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FINISHED_PATTERN,
    NA_VALUES,
    FOLD_SENTINEL,
    PODIUM_CUTOFF,
    PROCESSED,
    RAW,
    YEAR_MAX,
    YEAR_MIN,
    fold_of,
    split_of,
)
from .features import (
    drop_races_missing_grid,
    past_count,
    past_cumsum,
    past_expanding_mean,
    past_mean,
    past_roll_sum,
    past_sum,
    recode_grid,
)
from .schema import ALL_COLUMNS, FEATURES, TARGET, assert_valid


# ==========================================================================
# Load
# ==========================================================================

def load() -> dict[str, pd.DataFrame]:
    def rd(name: str) -> pd.DataFrame:
        return pd.read_csv(RAW / f"{name}.csv", na_values=NA_VALUES, low_memory=False)

    return {n: rd(n) for n in
            ["results", "races", "drivers", "constructors", "circuits",
             "status", "lap_times", "pit_stops"]}


# ==========================================================================
# Spine + label
# ==========================================================================

def build_spine(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per driver per Grand Prix, in scope, with label and raw history."""
    races = d["races"][["raceId", "year", "round", "circuitId", "date"]]

    df = (
        d["results"][["raceId", "driverId", "constructorId", "grid",
                      "positionOrder", "points", "statusId"]]
        .merge(races, on="raceId", how="inner")
        .query("@YEAR_MIN <= year <= @YEAR_MAX")
    )

    # Races with no grid data at all cannot be predicted. Drop them whole and
    # report which, rather than imputing the strongest feature in the project.
    df, dropped = drop_races_missing_grid(df)
    if dropped:
        print(f"  dropped {len(dropped)} race(s) with no grid data: {dropped}")

    df = df.sort_values(["year", "round", "positionOrder"]).reset_index(drop=True)

    # --- label. positionOrder is banned as a FEATURE; building the target is
    # --- post-race by definition, so this is its one legitimate use.
    df[TARGET] = (df["positionOrder"] <= PODIUM_CUTOFF).astype("int8")
    df["points_finish"] = (df["positionOrder"] <= 10).astype("int8")

    # --- DNF, from the documented status rule
    st = d["status"].copy()
    st["finished"] = st["status"].str.match(FINISHED_PATTERN).fillna(False)
    df = df.merge(st[["statusId", "finished"]], on="statusId", how="left")
    df["dnf"] = (~df["finished"].fillna(False)).astype("int8")

    # --- Grand Prix points, used only via shifted history
    df["gp_points"] = df["points"].fillna(0.0)
    df["win"] = (df["positionOrder"] == 1).astype("int8")

    return df


# ==========================================================================
# Feature blocks
# ==========================================================================

def add_starting_position(df: pd.DataFrame) -> pd.DataFrame:
    df["grid"], df["pit_lane_start"] = recode_grid(df["grid"])
    df["front_row"] = (df["grid"] <= 2).astype("int8")
    return df


def add_driver_form(df: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    for w in (3, 5, 10):
        df[f"driver_avg_finish_{w}"] = past_mean(df, "driverId", "positionOrder", w)
        df[f"driver_podium_rate_{w}"] = past_mean(df, "driverId", TARGET, w)

    df["driver_points_last_5"] = past_roll_sum(df, "driverId", "gp_points", 5)
    df["driver_dnf_rate_10"] = past_mean(df, "driverId", "dnf", 10)
    df["driver_career_races"] = past_count(df, "driverId")
    df["driver_prior_wins"] = past_cumsum(df, "driverId", "win")

    dob = drivers[["driverId", "dob"]].copy()
    dob["dob"] = pd.to_datetime(dob["dob"], errors="coerce")
    df = df.merge(dob, on="driverId", how="left")
    df["driver_age"] = (
        (pd.to_datetime(df["date"], errors="coerce") - df["dob"]).dt.days / 365.25
    )
    return df.drop(columns="dob")


def add_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Rolled at constructor-race level, NOT driver-row level.

    A two-car team contributes two rows per race. Rolling over driver-rows
    would make a 5-race window cover 2.5 actual races.
    """
    cr = (
        df.groupby(["raceId", "year", "round", "constructorId"], as_index=False)
          .agg(c_points=("gp_points", "sum"),
               c_finish=("positionOrder", "mean"),
               c_dnf=("dnf", "mean"),
               c_podium=(TARGET, "max"))
          .sort_values(["year", "round"])
          .reset_index(drop=True)
    )

    cr["constructor_avg_points_5"] = past_mean(cr, "constructorId", "c_points", 5)
    cr["constructor_avg_finish_5"] = past_mean(cr, "constructorId", "c_finish", 5)
    cr["constructor_dnf_rate_10"] = past_mean(cr, "constructorId", "c_dnf", 10)
    cr["constructor_podium_rate_10"] = past_mean(cr, "constructorId", "c_podium", 10)

    cols = ["raceId", "constructorId", "constructor_avg_points_5",
            "constructor_avg_finish_5", "constructor_dnf_rate_10",
            "constructor_podium_rate_10"]
    return df.merge(cr[cols], on=["raceId", "constructorId"], how="left")


def add_circuit(df: pd.DataFrame) -> pd.DataFrame:
    # driver / constructor history at this circuit - expanding, there are only
    # ever a handful of prior visits
    df["driver_avg_at_circuit"] = past_expanding_mean(
        df, ["driverId", "circuitId"], "positionOrder")

    cc = (
        df.groupby(["raceId", "year", "round", "circuitId", "constructorId"],
                   as_index=False)
          .agg(c_finish=("positionOrder", "mean"))
          .sort_values(["year", "round"])
          .reset_index(drop=True)
    )
    cc["constructor_avg_at_circuit"] = past_expanding_mean(
        cc, ["constructorId", "circuitId"], "c_finish")
    df = df.merge(
        cc[["raceId", "constructorId", "constructor_avg_at_circuit"]],
        on=["raceId", "constructorId"], how="left")

    # circuit character, from prior races at that circuit only
    df["_grid_delta"] = (df["positionOrder"] - df["grid"]).abs()
    cir = (
        df.groupby(["raceId", "year", "round", "circuitId"], as_index=False)
          .agg(r_dnf=("dnf", "mean"), r_delta=("_grid_delta", "mean"))
          .sort_values(["year", "round"])
          .reset_index(drop=True)
    )
    cir["circuit_dnf_rate"] = past_expanding_mean(cir, "circuitId", "r_dnf")
    # higher = starting order survives the race more strongly
    cir["circuit_grid_hold"] = 1.0 / (
        1.0 + past_expanding_mean(cir, "circuitId", "r_delta"))

    df = df.merge(cir[["raceId", "circuit_dnf_rate", "circuit_grid_hold"]],
                  on="raceId", how="left")
    return df.drop(columns="_grid_delta")


def add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    df["round_number"] = df["round"]
    df["driver_points_before"] = past_sum(df, ["year", "driverId"], "gp_points")

    # Derived from points_before, so still strictly pre-race.
    g = df.groupby(["year", "round"])["driver_points_before"]
    df["driver_championship_pos"] = g.rank(method="min", ascending=False)
    df["driver_gap_to_leader"] = g.transform("max") - df["driver_points_before"]
    return df


def add_lap_and_pit(df: pd.DataFrame, lap_times: pd.DataFrame,
                    pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Per-race lap/pit summaries, then rolled over PREVIOUS races only."""
    # --- per race-driver pace summary
    lt = (
        lap_times.groupby(["raceId", "driverId"], as_index=False)
                 .agg(_median_ms=("milliseconds", "median"),
                      _std_ms=("milliseconds", "std"))
    )
    lt["_laptime_rank"] = lt.groupby("raceId")["_median_ms"].rank(method="min")
    # seconds, so the feature is readable in SHAP plots
    lt["_consistency"] = lt["_std_ms"] / 1000.0

    # --- positions gained on lap 1
    lap1 = (lap_times.loc[lap_times["lap"] == 1, ["raceId", "driverId", "position"]]
                     .rename(columns={"position": "_lap1_pos"}))

    df = (df.merge(lt[["raceId", "driverId", "_laptime_rank", "_consistency"]],
                   on=["raceId", "driverId"], how="left")
            .merge(lap1, on=["raceId", "driverId"], how="left"))
    df["_gained_lap1"] = df["grid"] - df["_lap1_pos"]

    # must re-sort: merges do not guarantee order, and rolling depends on it
    df = df.sort_values(["year", "round", "positionOrder"]).reset_index(drop=True)

    df["driver_avg_laptime_rank"] = past_mean(df, "driverId", "_laptime_rank", 5)
    df["driver_laptime_consistency"] = past_mean(df, "driverId", "_consistency", 5)
    df["driver_avg_positions_gained_lap1"] = past_mean(df, "driverId", "_gained_lap1", 5)

    # --- pit stops, at constructor-race level
    ps = (
        pit_stops.groupby(["raceId", "driverId"], as_index=False)
                 .agg(_pit_ms=("milliseconds", "mean"))
    )
    tmp = df[["raceId", "driverId", "constructorId", "year", "round"]].merge(
        ps, on=["raceId", "driverId"], how="left")
    cr = (
        tmp.groupby(["raceId", "year", "round", "constructorId"], as_index=False)
           .agg(_pit=("_pit_ms", "mean"))
           .sort_values(["year", "round"])
           .reset_index(drop=True)
    )
    cr["_pit"] = cr["_pit"] / 1000.0          # seconds
    cr["constructor_avg_pitstop_time"] = past_mean(cr, "constructorId", "_pit", 5)

    df = df.merge(cr[["raceId", "constructorId", "constructor_avg_pitstop_time"]],
                  on=["raceId", "constructorId"], how="left")

    return df.drop(columns=["_laptime_rank", "_consistency", "_lap1_pos", "_gained_lap1"])


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    df["split"] = df["year"].map(split_of)
    df["fold"] = df["year"].map(fold_of).fillna(FOLD_SENTINEL).astype(int)
    return df


# ==========================================================================
# Report
# ==========================================================================

def report(df: pd.DataFrame) -> None:
    print("\nrows and podiums per split")
    print(df.groupby("split").agg(races=("raceId", "nunique"),
                                  rows=(TARGET, "size"),
                                  podiums=(TARGET, "sum"))
            .reindex(["train", "validation", "test"]))

    n_train = int(df.loc[df.split == "train", TARGET].sum())
    print(f"\ntraining events: {n_train}   EPV at {len(FEATURES)} features: "
          f"{n_train / len(FEATURES):.1f}")

    miss = df[FEATURES].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\nmissingness ({len(miss)} of {len(FEATURES)} features affected)")
    print((miss * 100).round(2).to_string() if len(miss) else "  none")

    corr = df[FEATURES].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    print("\ntop |correlation| with podium  (any >= 0.95 means leakage)")
    print(corr.head(8).round(3).to_string())


def main() -> None:
    print("STEP 2 - BUILD FEATURE TABLE")
    d = load()

    df = build_spine(d)
    print(f"  spine: {len(df):,} rows, {df['raceId'].nunique()} races")

    df = add_starting_position(df)
    df = add_driver_form(df, d["drivers"])
    df = add_team_form(df)
    df = add_circuit(df)
    df = add_season_context(df)
    df = add_lap_and_pit(df, d["lap_times"], d["pit_stops"])
    df = add_splits(df)

    out = df[ALL_COLUMNS].copy()
    assert_valid(out)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "features.parquet"
    out.to_parquet(path, index=False)

    report(out)
    print(f"\nwrote {path}  ({out.shape[0]:,} x {out.shape[1]})")
    print("NaNs are deliberate - impute inside your own fold pipeline.")


if __name__ == "__main__":
    main()
