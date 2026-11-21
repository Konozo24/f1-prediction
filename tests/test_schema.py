"""Schema contract tests. M2/M3/M4 depend on these column names not moving."""

import pytest

from src.config import BANNED_COLUMNS
from src.schema import FEATURE_GROUPS, FEATURES, ID_COLUMNS, assert_valid


def test_exactly_thirty_features():
    assert len(FEATURES) == 30, f"plan promises 30, schema has {len(FEATURES)}"


def test_no_duplicate_feature_names():
    assert len(FEATURES) == len(set(FEATURES))


def test_groups_sum_to_all_features():
    flat = [c for g in FEATURE_GROUPS.values() for c in g]
    assert sorted(flat) == sorted(FEATURES)


def test_group_counts_match_the_plan():
    expected = {"starting_position": 3, "driver_form": 11, "team_form": 4,
                "circuit": 4, "season_context": 4, "lap_and_pit": 4}
    assert {k: len(v) for k, v in FEATURE_GROUPS.items()} == expected


def test_no_feature_is_a_banned_column():
    assert not (set(FEATURES) & BANNED_COLUMNS)


def test_ids_are_not_features():
    """kNN and SVM must never see raw integer ids under a distance metric."""
    assert not (set(ID_COLUMNS) & set(FEATURES))


def test_assert_valid_rejects_a_missing_column():
    import pandas as pd
    from src.schema import ALL_COLUMNS
    bad = pd.DataFrame({c: [0] for c in ALL_COLUMNS if c != "grid"})
    with pytest.raises(ValueError, match="missing columns"):
        assert_valid(bad)


def test_assert_valid_rejects_a_banned_column():
    import pandas as pd
    from src.schema import ALL_COLUMNS
    bad = pd.DataFrame({c: [0] for c in ALL_COLUMNS})
    bad["positionOrder"] = 1
    with pytest.raises(ValueError, match="banned columns"):
        assert_valid(bad)
