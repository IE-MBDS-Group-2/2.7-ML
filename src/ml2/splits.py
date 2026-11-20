"""Validation splits. No random splits anywhere in this project.

Why: bonds of the same issuer on the same date move together, and spreads persist
from one month to the next. A random split would leak both.

    assign_partition      train / gap / valid / test by snapshot date (config.yaml)
    walk_forward_folds    expanding-window folds by month inside the training window
    issuer_group_folds    folds where an issuer's bonds are never on both sides
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def assign_partition(dates: pd.Series, split_cfg: dict) -> pd.Series:
    """Label each snapshot date as train, gap, valid or test."""
    d = pd.to_datetime(pd.Series(dates))
    train_end = pd.Timestamp(split_cfg["train_end"])
    valid_start = pd.Timestamp(split_cfg["valid_start"])
    valid_end = pd.Timestamp(split_cfg["valid_end"])
    test_start = pd.Timestamp(split_cfg["test_start"])
    test_end = pd.Timestamp(split_cfg["test_end"])

    out = pd.Series("gap", index=d.index, dtype=object)
    out[d <= train_end] = "train"
    out[(d >= valid_start) & (d <= valid_end)] = "valid"
    out[(d >= test_start) & (d <= test_end)] = "test"
    out[d > test_end] = "out_of_range"
    return out


def walk_forward_folds(
    snapshot_dates: pd.Series, n_folds: int = 3, gap_months: int = 1, min_train_months: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over distinct snapshot dates.

    Each fold validates on one block of months; training uses only months that end
    at least `gap_months` before the block. Returns (train_index, valid_index) row positions.
    """
    d = pd.to_datetime(pd.Series(snapshot_dates)).reset_index(drop=True)
    months = np.sort(d.unique())
    n = len(months)
    usable = n - min_train_months - gap_months
    if usable < n_folds:
        raise ValueError(f"Only {n} months: too few for {n_folds} folds with min_train_months={min_train_months}")
    block = usable // n_folds
    folds = []
    for k in range(n_folds):
        v_start = min_train_months + gap_months + k * block
        v_end = n if k == n_folds - 1 else v_start + block
        train_months = months[: v_start - gap_months]
        valid_months = months[v_start:v_end]
        train_idx = np.flatnonzero(d.isin(train_months).to_numpy())
        valid_idx = np.flatnonzero(d.isin(valid_months).to_numpy())
        folds.append((train_idx, valid_idx))
    return folds


def issuer_group_folds(groups: pd.Series, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by issuer: "price a bond from an issuer the model has never seen"."""
    g = pd.Series(groups).reset_index(drop=True)
    dummy = np.zeros(len(g))
    yield from GroupKFold(n_splits=n_splits).split(dummy, dummy, groups=g)
