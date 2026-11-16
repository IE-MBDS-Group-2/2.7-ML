import numpy as np
import pandas as pd

from ml2.equity import features_at_snapshots, naive_distance_to_default, rolling_price_features, rows_asof


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    m = rng.normal(0, 0.01, n)
    stock = 2 * m + rng.normal(0, 0.001, n)
    px = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + m), "AAA": 50 * np.cumprod(1 + stock), "FLAT": np.full(n, 10.0)},
        index=idx,
    )
    px.columns.name = "ticker"
    return px


def test_beta_and_flat_volatility():
    feats = rolling_price_features(_prices(), "SPY")
    assert abs(feats["beta_252d"]["AAA"].iloc[-1] - 2.0) < 0.1
    assert feats["vol_60d"]["FLAT"].iloc[-1] == 0.0


def test_rows_asof_takes_last_row_on_or_before():
    wide = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    out = rows_asof(wide, pd.DatetimeIndex(["2025-01-04", "2025-01-01", "2025-01-06"]))
    assert out["x"].iloc[0] == 2.0
    assert np.isnan(out["x"].iloc[1])
    assert out["x"].iloc[2] == 3.0


def test_features_at_snapshots_long_shape():
    px = _prices()
    snaps = pd.DatetimeIndex([px.index[-1], px.index[-30]])
    out = features_at_snapshots(px, snaps, "SPY")
    assert set(["snapshot_date", "ticker", "vol_20d", "beta_252d", "mom_12_1"]).issubset(out.columns)
    assert len(out) == 2 * 3


def test_no_lookahead_in_snapshot_features():
    px = _prices()
    snap = pd.DatetimeIndex([px.index[200]])
    before = features_at_snapshots(px, snap, "SPY")
    changed = px.copy()
    changed.iloc[201:] *= 3  # change the future only
    after = features_at_snapshots(changed, snap, "SPY")
    pd.testing.assert_frame_equal(before, after)


def test_naive_dd_falls_with_leverage_and_handles_zero_debt():
    dd = naive_distance_to_default(
        np.array([100.0, 100.0, 100.0]), np.array([10.0, 100.0, 0.0]), np.array([0.3, 0.3, 0.3]),
        np.array([0.05, 0.05, 0.05]),
    )
    assert dd[0] > dd[1]
    assert np.isnan(dd[2])
