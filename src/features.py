"""
Feature construction helpers.

Every rolling feature in the project goes through `past_mean`. One rule,
implemented once, so the leakage discipline cannot drift between the 30
features or between team members.

BMDS2114 Machine Learning - M1 (pipeline + kNN)
"""

from __future__ import annotations

import pandas as pd

from .config import PIT_LANE_GRID


def past_mean(
    df: pd.DataFrame,
    group: str | list[str],
    col: str,
    window: int,
) -> pd.Series:
    """Rolling mean of `col` over the PREVIOUS `window` races within `group`.

    The `.shift(1)` is the single most important call in this project. Without
    it the current race is included in its own feature, which leaks the answer
    directly and produces scores that look excellent and mean nothing.

    `df` MUST already be sorted by ["year", "round"], otherwise "previous"
    is meaningless.

    Returns NaN for a group's first appearance, by design - there is genuinely
    no history available. Those rows are handled by the imputation experiment.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    return df.groupby(group)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def past_roll_sum(
    df: pd.DataFrame,
    group: str | list[str],
    col: str,
    window: int,
) -> pd.Series:
    """Rolling SUM of `col` over the PREVIOUS `window` races within `group`.

    Same shift(1) discipline as past_mean. Used where a total rather than an
    average is wanted, e.g. points scored in the last 5 races.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    return df.groupby(group)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )


def past_expanding_mean(
    df: pd.DataFrame,
    group: str | list[str],
    col: str,
) -> pd.Series:
    """Mean of `col` over ALL PREVIOUS rows within `group` (no fixed window).

    Used for "how has this driver historically gone at this circuit", where the
    whole available history is wanted rather than a recent window - there are
    only ever a handful of prior visits.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    return df.groupby(group)[col].transform(
        lambda s: s.shift(1).expanding().mean()
    )


def past_count(df: pd.DataFrame, group: str | list[str]) -> pd.Series:
    """Number of PREVIOUS rows within `group`. Zero on first appearance.

    cumcount() is already exclusive of the current row, so no shift is needed.
    """
    if df.empty:
        return pd.Series(dtype="int64", index=df.index)
    return df.groupby(group).cumcount()


def past_cumsum(
    df: pd.DataFrame,
    group: str | list[str],
    col: str,
) -> pd.Series:
    """Cumulative sum of `col` over ALL PREVIOUS rows within `group`."""
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)
    return (
        df.groupby(group)[col]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )


def past_sum(
    df: pd.DataFrame,
    group: str | list[str],
    col: str,
) -> pd.Series:
    """Cumulative sum of `col` over PREVIOUS races within `group`.

    Used for season-to-date championship points, which must exclude points
    scored in the race being predicted.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    return (
        df.groupby(group)[col]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )


def recode_grid(grid: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fix the Ergast pit-lane encoding.

    Ergast records a pit-lane start as grid == 0. Treated as a number, that
    reads as BETTER than pole position - corrupting the strongest feature in
    the project, in the direction that looks like good performance.

    Returns (recoded_grid, pit_lane_start_flag) so the information is kept
    rather than thrown away.

    Raises if any grid is NaN. Grid is the strongest feature in the project, so
    silently imputing it is not acceptable - call drop_races_missing_grid first
    and record which races were removed.
    """
    if grid.isna().any():
        n = int(grid.isna().sum())
        raise ValueError(
            f"{n} rows have a missing grid. Grid cannot be imputed - it is the "
            f"strongest predictor. Call drop_races_missing_grid() first."
        )

    pit_lane = (grid == 0).astype("int8")
    return grid.replace(0, PIT_LANE_GRID), pit_lane


def drop_races_missing_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Remove entire races whose starting grid is missing.

    Found in verification: the 2025 Qatar Grand Prix (round 23) has no grid data
    at all for any of its 20 drivers - a gap in the Jolpica-sourced 2025 data,
    not scattered non-starters. 2014-2024 come from Ergast and are complete.

    Dropped whole-race rather than row-wise, because a race missing its grid
    cannot be predicted at all, and keeping a partial field would also corrupt
    any per-race normalisation or top-3 hit-rate metric.

    Returns (clean_df, dropped_race_ids) so the count can be reported in the
    methodology rather than disappearing silently.
    """
    bad = df.loc[df["grid"].isna(), "raceId"].unique().tolist()
    if not bad:
        return df, []

    clean = df[~df["raceId"].isin(bad)].copy()
    return clean, sorted(int(r) for r in bad)
