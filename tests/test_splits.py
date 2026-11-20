import numpy as np
import pandas as pd

from ml2.config import load_config
from ml2.splits import assign_partition, issuer_group_folds, walk_forward_folds

MONTH_ENDS = pd.Series(pd.date_range("2024-01-31", "2026-03-31", freq="ME"))


def test_partitions_follow_config_with_gaps():
    parts = assign_partition(MONTH_ENDS, load_config()["split"])
    by_date = dict(zip(MONTH_ENDS.dt.strftime("%Y-%m"), parts))
    assert by_date["2025-05"] == "train"
    assert by_date["2025-06"] == "gap"
    assert by_date["2025-07"] == "valid"
    assert by_date["2025-10"] == "valid"
    assert by_date["2025-11"] == "gap"
    assert by_date["2025-12"] == "test"
    assert by_date["2026-03"] == "test"
    assert (parts == "train").sum() == 17


def test_walk_forward_has_gap_and_no_overlap():
    train_dates = MONTH_ENDS[MONTH_ENDS <= "2025-05-31"].repeat(3).reset_index(drop=True)
    folds = walk_forward_folds(train_dates, n_folds=3, gap_months=1, min_train_months=6)
    assert len(folds) == 3
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
        last_train = train_dates.iloc[tr].max()
        first_valid = train_dates.iloc[va].min()
        assert (first_valid.to_period("M") - last_train.to_period("M")).n >= 2


def test_issuer_folds_never_split_an_issuer():
    groups = pd.Series(["A", "A", "B", "B", "C", "C", "D", "E", "E", "F"])
    for tr, va in issuer_group_folds(groups, n_splits=3):
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va]))


def test_issuer_holdout_folds_hold_size_and_months_fixed():
    from ml2.splits import issuer_holdout_folds

    months = pd.date_range("2024-01-31", periods=12, freq="ME")
    dates = pd.Series(np.repeat(months, 50))
    groups = pd.Series([f"C{i % 25}" for i in range(len(dates))])
    for seen, unseen, held in issuer_holdout_folds(dates, groups, n_folds=3, min_train_months=4):
        held_companies = set(groups.iloc[held])
        assert held_companies and not held_companies & set(groups.iloc[unseen])  # never seen
        assert held_companies <= set(groups.iloc[seen])                          # seen
        assert len(seen) == len(unseen)                                          # same amount of data
        assert dates.iloc[seen].max() == dates.iloc[unseen].max()                # same months
        assert dates.iloc[held].min() > dates.iloc[unseen].max()                 # scored later
