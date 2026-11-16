"""Issuer equity features, computed from daily prices and balance sheets.

Every feature uses only information available on or before the snapshot date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_price_features(adj_close: pd.DataFrame, market: str) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) features from adjusted closes.

    Returns annualized volatility (20, 60, 252 days), 252-day return, 12-1 momentum
    and 252-day beta against the market column.
    """
    px = adj_close.sort_index()
    rets = px.pct_change(fill_method=None)
    ann = np.sqrt(TRADING_DAYS)

    feats = {
        "vol_20d": rets.rolling(20, min_periods=15).std() * ann,
        "vol_60d": rets.rolling(60, min_periods=40).std() * ann,
        "vol_252d": rets.rolling(252, min_periods=180).std() * ann,
        "ret_252d": px / px.shift(252) - 1,
        "mom_12_1": px.shift(21) / px.shift(252) - 1,
    }

    if market in rets.columns:
        m = rets[market]
        window, minp = 252, 120
        mean_x = rets.rolling(window, min_periods=minp).mean()
        mean_m = m.rolling(window, min_periods=minp).mean()
        mean_xm = rets.mul(m, axis=0).rolling(window, min_periods=minp).mean()
        cov = mean_xm - mean_x.mul(mean_m, axis=0)
        var_m = m.rolling(window, min_periods=minp).var(ddof=0)
        feats["beta_252d"] = cov.div(var_m.replace(0, np.nan), axis=0)
    else:
        feats["beta_252d"] = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    return feats


def rows_asof(wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows of a date-indexed frame as of each date (last row on or before the date)."""
    idx = wide.index.searchsorted(dates, side="right") - 1
    out = pd.DataFrame(np.nan, index=dates, columns=wide.columns)
    valid = idx >= 0
    out.loc[dates[valid]] = wide.iloc[idx[valid]].to_numpy()
    return out


def features_at_snapshots(
    adj_close: pd.DataFrame, snapshot_dates: pd.DatetimeIndex, market: str
) -> pd.DataFrame:
    """Long frame (snapshot_date, ticker, features...) of price-based features."""
    dates = pd.DatetimeIndex(snapshot_dates)
    long_parts = []
    for name, wide in rolling_price_features(adj_close, market).items():
        snap = rows_asof(wide, dates)
        snap.index.name = "snapshot_date"
        s = snap.stack(future_stack=True).rename(name)
        long_parts.append(s)
    out = pd.concat(long_parts, axis=1).reset_index()
    out = out.rename(columns={"level_1": "ticker"})
    if "ticker" not in out.columns:
        out = out.rename(columns={out.columns[1]: "ticker"})
    return out


def naive_distance_to_default(
    equity_value: pd.Series | np.ndarray,
    debt_face: pd.Series | np.ndarray,
    sigma_equity: pd.Series | np.ndarray,
    expected_return: pd.Series | np.ndarray,
    horizon_years: float = 1.0,
) -> np.ndarray:
    """Bharath and Shumway (2008) naive distance to default.

    sigma_D = 0.05 + 0.25 * sigma_E
    sigma_V = E/(E+F) * sigma_E + F/(E+F) * sigma_D
    DD = (ln((E+F)/F) + (mu - 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
    Higher DD = further from default. NaN when debt or equity is missing or not positive.
    """
    e = np.asarray(equity_value, dtype=float)
    f = np.asarray(debt_face, dtype=float)
    s_e = np.asarray(sigma_equity, dtype=float)
    mu = np.asarray(expected_return, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (e > 0) & (f > 0) & (s_e > 0)
        s_d = 0.05 + 0.25 * s_e
        v = e + f
        s_v = (e / v) * s_e + (f / v) * s_d
        dd = (np.log(v / f) + (mu - 0.5 * s_v**2) * horizon_years) / (s_v * np.sqrt(horizon_years))
    return np.where(valid, dd, np.nan)
