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
    """
    pit_lane = (grid == 0).astype("int8")
    return grid.replace(0, PIT_LANE_GRID), pit_lane
