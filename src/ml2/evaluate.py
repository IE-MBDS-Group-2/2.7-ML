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


def improvement_ci(
    df: pd.DataFrame,
    y_true: str,
    y_model: str,
    y_base: str,
    cluster: str = "issuer_ticker",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Improvement in MAE (1 - MAE_model / MAE_base) with a cluster-bootstrap interval.

    Whole clusters (companies) are resampled, because a company's bonds move together: resampling rows would
    make the interval far too narrow.
    """
    d = df[[cluster, y_true, y_model, y_base]].dropna(subset=[y_true, y_model, y_base])
    errors = pd.DataFrame(
        {
            # a missing company is its own cluster, never one big "nan" cluster
            "cluster": d[cluster].astype(object).where(d[cluster].notna(), "row" + d.index.astype(str)).astype(str),
            "model": (d[y_model] - d[y_true]).abs().to_numpy(),
            "base": (d[y_base] - d[y_true]).abs().to_numpy(),
        }
    )
    sums = errors.groupby("cluster")[["model", "base"]].sum()
    s_model, s_base = sums["model"].to_numpy(), sums["base"].to_numpy()
    k = len(sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    draws = 1 - s_model[idx].sum(axis=1) / s_base[idx].sum(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {
        "improvement": float(1 - s_model.sum() / s_base.sum()),
        "ci_low": float(low),
        "ci_high": float(high),
        "clusters": int(k),
    }


def passes_gate(
    model: pd.DataFrame, baseline: pd.DataFrame, gate_cfg: dict, ci: dict[str, dict] | None = None
) -> tuple[bool, pd.DataFrame]:
    """Check the quality gate from config.yaml.

    model, baseline: outputs of metrics_by_stratum on the same rows.
    ci: optional {stratum: improvement_ci(...)}; required when gate_cfg["ci_excludes_zero"] is true.
    The model passes only if, in every listed stratum, it improves the gate metric by at least
    min_improvement and (when required) the interval of the improvement lies above zero.
    """
    metric = gate_cfg["metric"]
    strata = gate_cfg["strata"]
    m = model.set_index("stratum")[metric]
    b = baseline.set_index("stratum")[metric]
    table = pd.DataFrame({"model": m, "baseline": b}).loc[strata]
    table["improvement"] = 1 - table["model"] / table["baseline"]
    table["pass"] = table["improvement"] >= gate_cfg["min_improvement"]
    if gate_cfg.get("ci_excludes_zero", False):
        if ci is None:
            raise ValueError("This gate needs ci={stratum: improvement_ci(...)} for every stratum")
        table["ci_low"] = [ci[s]["ci_low"] for s in strata]
        table["ci_high"] = [ci[s]["ci_high"] for s in strata]
        table["pass"] = table["pass"] & (table["ci_low"] > 0)
    return bool(table["pass"].all()), table


def gate_report(
    df: pd.DataFrame, y_true: str, y_model: str, y_base: str, gate_cfg: dict, by: str = "grade_bucket"
) -> tuple[bool, pd.DataFrame]:
    """Everything the gate needs in one call: metrics by stratum, bootstrap intervals and the verdict.

    ok, table = gate_report(test, "spread_bps", "pred_best", "pred_matrix_same_date", cfg["quality_gate"])
    """
    # The point estimate and the interval use the same rows: those where the target and both predictions exist
    df = df.dropna(subset=[y_true, y_model, y_base])
    missing = [s for s in gate_cfg["strata"] if not (df[by] == s).any()]
    if missing:
        raise ValueError(f"No rows with a target and both predictions in stratum {missing}")
    boot = gate_cfg.get("bootstrap", {})
    ci = {
        s: improvement_ci(
            df[df[by] == s],
            y_true,
            y_model,
            y_base,
            cluster=boot.get("cluster", "issuer_ticker"),
            n_boot=boot.get("n_boot", 1000),
            level=boot.get("level", 0.95),
            seed=boot.get("seed", 42),
        )
        for s in gate_cfg["strata"]
    }
    model = metrics_by_stratum(df, y_true, y_model, by)
    base = metrics_by_stratum(df, y_true, y_base, by)
    return passes_gate(model, base, gate_cfg, ci)
