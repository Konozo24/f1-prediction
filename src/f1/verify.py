"""
STEP 1 of the data pipeline: verify the dataset supports the project plan.

Run this BEFORE writing any feature code. The plan rests on roughly six
unverified numbers; this script turns each of them into a fact. Finding out on
day 1 that (say) 2025 has no lap_times costs an afternoon. Finding out on day 10
costs the test-set design, after three people have already trained against it.

Six questions:
    Q1  Real rows and podiums per season - do they match the plan's estimates?
    Q2  Rows and podiums per split, and events-per-predictor headroom.
    Q3  Is 2025 as complete as earlier seasons? (2025 comes from Jolpica, not
        Ergast, so its provenance differs from 2014-2024.)
    Q4  Are pit-lane starts (grid == 0) present, and how many?
    Q5  Did the \\N null handling work, or is everything object dtype?
    Q6  Which status values actually occur, so the DNF rule can be exact?

Usage:
    python -m src.f1.verify
    python -m src.f1.verify | tee data/raw/VERIFY.txt
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import (
    FINISHED_PATTERN,
    NA_VALUES,
    PODIUM_CUTOFF,
    RAW,
    TABLES,
    YEAR_MAX,
    YEAR_MIN,
    split_of,
)

pd.set_option("display.width", 100)


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_tables() -> dict[str, pd.DataFrame]:
    """Load the 8 tables, reporting any that are missing rather than crashing."""
    rule("LOADING")

    if not RAW.exists():
        sys.exit(f"FATAL: {RAW} does not exist. Download the dataset first.")

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for name in TABLES:
        path = RAW / f"{name}.csv"
        try:
            df = pd.read_csv(path, na_values=NA_VALUES, low_memory=False)
        except FileNotFoundError:
            print(f"  {name:<16} *** MISSING: {path.name} ***")
            missing.append(name)
            continue
        except pd.errors.EmptyDataError:
            print(f"  {name:<16} *** EMPTY FILE ***")
            missing.append(name)
            continue

        frames[name] = df
        print(f"  {name:<16} {len(df):>9,} rows   {len(df.columns):>2} cols")

    if missing:
        sys.exit(f"\nFATAL: cannot continue without {missing}")

    return frames


def build_spine(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per driver per Grand Prix, in scope, with the label attached."""
    races = d["races"][["raceId", "year", "round", "circuitId", "date"]]

    spine = (
        d["results"]
        .merge(races, on="raceId", how="inner")
        .query("@YEAR_MIN <= year <= @YEAR_MAX")
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )

    # positionOrder is a BANNED input column. Legitimate here and only here:
    # building the label is post-race by definition. It must never reach X.
    spine["podium"] = (spine["positionOrder"] <= PODIUM_CUTOFF).astype(int)
    spine["split"] = spine["year"].map(split_of)
    return spine


def q1_per_season(spine: pd.DataFrame) -> None:
    rule("Q1  Rows and podiums per season")
    out = spine.groupby("year").agg(
        races=("raceId", "nunique"),
        rows=("raceId", "size"),
        podiums=("podium", "sum"),
    )
    out["rows_per_race"] = (out["rows"] / out["races"]).round(1)
    print(out)
    print("\n  Expect ~3 podiums per race and ~20 rows per race.")
    print("  Deviations mean the join dropped rows - investigate before continuing.")


def q2_per_split(spine: pd.DataFrame) -> None:
    rule("Q2  Rows and podiums per split")
    out = spine.groupby("split").agg(
        seasons=("year", "nunique"),
        races=("raceId", "nunique"),
        rows=("raceId", "size"),
        podiums=("podium", "sum"),
    )
    out["positive_rate"] = (out["podiums"] / out["rows"]).round(3)
    print(out.reindex(["train", "validation", "test"]))

    train_events = int(out.loc["train", "podiums"]) if "train" in out.index else 0
    print(f"\n  Training events (podiums): {train_events}")
    print("  Events-per-predictor at various feature counts (floor is 10):")
    for k in (10, 20, 30, 33):
        print(f"    {k:>3} features -> EPV {train_events / k:5.1f}"
              f"{'   <-- AT OR BELOW FLOOR' if train_events / k < 12 else ''}")
    print("\n  Report these numbers to the team. They replace every estimate in the plan.")


def q3_lap_pit_coverage(d: dict[str, pd.DataFrame]) -> None:
    rule("Q3  lap_times / pit_stops coverage by season  (2025 provenance check)")
    years = d["races"][["raceId", "year"]]

    for name in ("lap_times", "pit_stops"):
        counts = (
            d[name]
            .merge(years, on="raceId", how="inner")
            .query("@YEAR_MIN <= year <= @YEAR_MAX")
            .groupby("year")
            .size()
        )
        print(f"\n  {name}:")
        if counts.empty:
            print("    *** NO ROWS IN SCOPE ***")
            continue
        median = counts.median()
        for year, n in counts.items():
            flag = "  <-- THIN" if n < 0.5 * median else ""
            print(f"    {year}  {n:>8,}{flag}")
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            if year not in counts.index:
                print(f"    {year}  *** ABSENT ***")

    print("\n  2025 comes from Jolpica; 2014-2024 from Ergast. If 2025 is thin or")
    print("  absent, the 4 lap/pit features are unusable across a whole test season.")
    print("  That is a team decision and it needs making today, not on day 10.")


def q4_pit_lane_starts(spine: pd.DataFrame) -> None:
    rule("Q4  Pit-lane starts (grid == 0)")
    n = int((spine["grid"] == 0).sum())
    print(f"  rows with grid == 0: {n}")
    if n:
        print("\n  by season:")
        print(spine[spine["grid"] == 0].groupby("year").size().to_string())
        print(f"\n  Confirmed: the grid recode is required. Report this count in the")
        print(f"  methodology - left raw, these {n} rows read as better than pole.")
    else:
        print("  None found. Verify independently before skipping the recode.")


def q5_dtypes(d: dict[str, pd.DataFrame]) -> None:
    rule("Q5  dtypes  (did the \\N handling work?)")
    results = d["results"]
    print(results.dtypes.to_string())

    should_be_numeric = ["grid", "positionOrder", "points", "laps"]
    bad = [c for c in should_be_numeric
           if c in results.columns and results[c].dtype == "object"]
    if bad:
        print(f"\n  *** PROBLEM: {bad} loaded as object dtype. ***")
        print("  The \\N null handling is wrong; nothing downstream will work.")
    else:
        print("\n  OK: key numeric columns parsed correctly.")

    print("\n  Null counts in results:")
    nulls = results.isna().sum()
    print(nulls[nulls > 0].to_string() if nulls.any() else "  (none)")


def q6_status_values(d: dict[str, pd.DataFrame], spine: pd.DataFrame) -> None:
    rule("Q6  status values in scope  (for the DNF rule)")
    merged = spine[["statusId"]].merge(d["status"], on="statusId", how="left")
    counts = merged["status"].value_counts()
    print(counts.head(25).to_string())

    finished = counts.index.to_series().str.match(FINISHED_PATTERN)
    n_fin = int(counts[finished.values].sum())
    n_dnf = int(counts[~finished.values].sum())
    total = n_fin + n_dnf

    print(f"\n  Pattern: {FINISHED_PATTERN}")
    print(f"    finished : {n_fin:>6,}  ({n_fin / total:.1%})")
    print(f"    DNF      : {n_dnf:>6,}  ({n_dnf / total:.1%})")
    print("\n  Classified as FINISHED:")
    print("   ", sorted(counts.index[finished.values].tolist())[:12])
    print("\n  Sanity check: DNF rate should be roughly 10-20%. If it is wildly")
    print("  off, the regex is wrong - fix it here before any feature uses it.")


def main() -> None:
    print("STEP 1 - DATASET VERIFICATION")
    print(f"raw data: {RAW}")
    print(f"scope   : {YEAR_MIN}-{YEAR_MAX}, Grand Prix only")

    d = load_tables()
    spine = build_spine(d)

    print(f"\nspine: {len(spine):,} rows "
          f"({spine['raceId'].nunique()} races, {int(spine['podium'].sum())} podiums)")

    q1_per_season(spine)
    q2_per_split(spine)
    q3_lap_pit_coverage(d)
    q4_pit_lane_starts(spine)
    q5_dtypes(d)
    q6_status_values(d, spine)

    rule("DONE")
    print("  Paste Q1, Q2 and Q3 into the team chat. Those are the first real")
    print("  numbers the project has, and Q3 may change the split design.")


if __name__ == "__main__":
    main()
