"""Treasury par curve interpolation.

The spread target is the bond's yield minus the Treasury par yield at the bond's
remaining maturity. The curve comes from LDP series UST_PAR_* (percent).
Interpolation is linear in maturity, flat beyond the shortest and longest tenor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TENOR_YEARS: dict[str, float] = {
    "UST_PAR_1M": 1 / 12,
    "UST_PAR_1_5M": 1.5 / 12,
    "UST_PAR_2M": 2 / 12,
    "UST_PAR_3M": 3 / 12,
    "UST_PAR_4M": 4 / 12,
    "UST_PAR_6M": 6 / 12,
    "UST_PAR_1Y": 1.0,
    "UST_PAR_2Y": 2.0,
    "UST_PAR_3Y": 3.0,
    "UST_PAR_5Y": 5.0,
    "UST_PAR_7Y": 7.0,
    "UST_PAR_10Y": 10.0,
    "UST_PAR_20Y": 20.0,
    "UST_PAR_30Y": 30.0,
}


def interpolate_yield(tenors: np.ndarray, yields: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """Linear interpolation of one curve at several maturities. Missing tenors are skipped."""
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    mask = ~np.isnan(yields)
    if mask.sum() < 2:
        return np.full(maturities.shape, np.nan)
    order = np.argsort(tenors[mask])
    x = tenors[mask][order]
    y = yields[mask][order]
    return np.interp(maturities, x, y)


def treasury_at_maturity(
    curves: pd.DataFrame, snapshot_dates: pd.Series, maturities: pd.Series
) -> np.ndarray:
    """Par yield (percent) for each (snapshot date, years to maturity) pair.

    curves: one row per snapshot date (index), one column per UST_PAR_* series.
    """
    cols = [c for c in curves.columns if c in TENOR_YEARS]
    tenors = np.array([TENOR_YEARS[c] for c in cols])
    dates = pd.to_datetime(pd.Series(snapshot_dates)).reset_index(drop=True)
    mats = pd.Series(maturities).reset_index(drop=True).to_numpy(dtype=float)
    curve_idx = pd.to_datetime(curves.index)
    lookup = pd.DataFrame(curves[cols].to_numpy(dtype=float), index=curve_idx)

    out = np.full(len(dates), np.nan)
    for date, positions in dates.groupby(dates).groups.items():
        if date not in lookup.index:
            continue
        row = lookup.loc[date].to_numpy(dtype=float)
        pos = np.asarray(list(positions))
        out[pos] = interpolate_yield(tenors, row, mats[pos])
    return out
