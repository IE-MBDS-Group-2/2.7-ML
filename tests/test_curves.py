import numpy as np
import pandas as pd

from ml2.curves import interpolate_yield, treasury_at_maturity


def test_exact_at_tenor_and_linear_between():
    tenors = np.array([1.0, 2.0, 5.0])
    yields = np.array([4.0, 4.2, 4.8])
    out = interpolate_yield(tenors, yields, np.array([2.0, 3.5]))
    assert out[0] == 4.2
    assert np.isclose(out[1], 4.5)


def test_flat_extrapolation():
    out = interpolate_yield(np.array([1.0, 30.0]), np.array([4.0, 5.0]), np.array([0.5, 40.0]))
    assert list(out) == [4.0, 5.0]


def test_missing_tenor_is_skipped():
    out = interpolate_yield(np.array([1.0, 2.0, 3.0]), np.array([4.0, np.nan, 5.0]), np.array([2.0]))
    assert np.isclose(out[0], 4.5)


def test_treasury_at_maturity_uses_each_dates_curve():
    curves = pd.DataFrame(
        {"UST_PAR_2Y": [4.0, 3.0], "UST_PAR_10Y": [5.0, 4.0]},
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )
    dates = pd.Series(pd.to_datetime(["2025-02-28", "2025-01-31", "2025-03-31"]))
    out = treasury_at_maturity(curves, dates, pd.Series([2.0, 6.0, 5.0]))
    assert out[0] == 3.0
    assert np.isclose(out[1], 4.5)
    assert np.isnan(out[2])  # no curve on that date
