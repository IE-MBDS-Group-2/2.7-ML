import numpy as np
import pandas as pd

from ml2.evaluate import gate_report, improvement_ci, metrics_by_stratum, passes_gate, regression_metrics


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


def _two_grade_errors(n_issuers=40, seed=0, model_noise=5.0, base_noise=10.0):
    rng = np.random.default_rng(seed)
    rows = []
    for grade in ["IG", "HY"]:
        for i in range(n_issuers):
            for _ in range(3):
                y = 100.0 if grade == "IG" else 400.0
                rows.append(
                    {
                        "grade_bucket": grade,
                        "issuer_ticker": f"{grade}{i}",
                        "y": y,
                        "model": y + rng.normal(0, model_noise),
                        "base": y + rng.normal(0, base_noise),
                    }
                )
    return pd.DataFrame(rows)


def test_improvement_ci_brackets_the_point_estimate():
    ci = improvement_ci(_two_grade_errors(), "y", "model", "base", cluster="issuer_ticker", n_boot=300)
    assert ci["ci_low"] < ci["improvement"] < ci["ci_high"]
    assert ci["clusters"] == 80
    assert ci["improvement"] > 0.3  # model noise is half the baseline's


def test_gate_with_interval_fails_when_noise_dominates():
    df = _two_grade_errors(n_issuers=5, model_noise=9.0, base_noise=10.0)
    cfg = {"metric": "mae_bps", "strata": ["IG", "HY"], "min_improvement": 0.0, "ci_excludes_zero": True,
           "bootstrap": {"n_boot": 300}}
    ok, table = gate_report(df, "y", "model", "base", cfg)
    assert {"ci_low", "ci_high"} <= set(table.columns)
    assert not ok  # 5 companies per grade and similar errors: the interval includes zero


def test_gate_with_interval_passes_a_clear_win():
    df = _two_grade_errors(n_issuers=60, model_noise=2.0, base_noise=10.0)
    cfg = {"metric": "mae_bps", "strata": ["IG", "HY"], "min_improvement": 0.10, "ci_excludes_zero": True,
           "bootstrap": {"n_boot": 300}}
    ok, table = gate_report(df, "y", "model", "base", cfg)
    assert ok
    assert (table["ci_low"] > 0.10).all()
