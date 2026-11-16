import numpy as np
import pandas as pd

from ml2.evaluate import metrics_by_stratum, passes_gate, regression_metrics


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    assert m["mae_bps"] == (10 + 10 + 30) / 3
    assert np.isclose(m["rmse_bps"], np.sqrt((100 + 100 + 900) / 3))
    assert m["medae_bps"] == 10
    assert m["n"] == 3


def test_nan_rows_are_ignored():
    m = regression_metrics([100, np.nan, 300], [110, 5, 300])
    assert m["n"] == 2


def test_gate_requires_every_stratum():
    df = pd.DataFrame(
        {
            "grade_bucket": ["IG", "IG", "HY", "HY"],
            "y": [100.0, 120.0, 400.0, 500.0],
            "model": [102.0, 118.0, 380.0, 520.0],
            "base": [110.0, 110.0, 390.0, 510.0],
        }
    )
    model = metrics_by_stratum(df, "y", "model", "grade_bucket")
    base = metrics_by_stratum(df, "y", "base", "grade_bucket")
    ok, table = passes_gate(model, base, {"metric": "mae_bps", "strata": ["IG", "HY"], "min_improvement": 0.10})
    assert table.loc["IG", "pass"]
    assert not table.loc["HY", "pass"]  # HY: 20 vs 10 bps, worse
    assert not ok
