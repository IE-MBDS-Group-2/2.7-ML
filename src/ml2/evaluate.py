"""Model evaluation: metrics in basis points, by stratum, and the quality gate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, median absolute error (all in the target's units, bps) and R squared."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "n": int(mask.sum()),
        "mae_bps": float(np.mean(np.abs(err))),
        "rmse_bps": float(np.sqrt(np.mean(err**2))),
        "medae_bps": float(np.median(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def metrics_by_stratum(df: pd.DataFrame, y_true: str, y_pred: str, by: str | list[str]) -> pd.DataFrame:
    """regression_metrics for every group of `by`, plus an 'ALL' row."""
    rows = []
    for key, g in df.groupby(by, dropna=False, observed=True):
        rows.append({"stratum": key, **regression_metrics(g[y_true], g[y_pred])})
    rows.append({"stratum": "ALL", **regression_metrics(df[y_true], df[y_pred])})
    return pd.DataFrame(rows)


def passes_gate(model: pd.DataFrame, baseline: pd.DataFrame, gate_cfg: dict) -> tuple[bool, pd.DataFrame]:
    """Check the quality gate from config.yaml.

    model, baseline: outputs of metrics_by_stratum on the same rows.
    The model passes only if it improves the gate metric by at least min_improvement
    in every listed stratum.
    """
    metric = gate_cfg["metric"]
    strata = gate_cfg["strata"]
    m = model.set_index("stratum")[metric]
    b = baseline.set_index("stratum")[metric]
    table = pd.DataFrame({"model": m, "baseline": b}).loc[strata]
    table["improvement"] = 1 - table["model"] / table["baseline"]
    table["pass"] = table["improvement"] >= gate_cfg["min_improvement"]
    return bool(table["pass"].all()), table
