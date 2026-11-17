"""Baselines every model is compared with. One definition for everyone, because the quality gate depends on it.

    median_baseline        the class baseline: one number, the median training spread
    fit_matrix_frozen      matrix pricing frozen at the training months (reference row only)
    predict_matrix_frozen
    matrix_same_date       matrix pricing re-marked on every snapshot date: the gate baseline

Matrix pricing = the median spread of bonds in the same grade x maturity bucket x sector cell. When a cell holds
fewer than `min_bonds` bonds it falls back to grade x maturity, then grade, then all bonds.

The gate baseline works like a pricing desk: on each snapshot date, a bond is priced from the same-date spreads of
other companies' bonds in its cell. Its own company is left out, so a bond never prices itself or its sister bonds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "spread_bps"
LEVELS = [["grade_bucket", "maturity_bucket", "sector"], ["grade_bucket", "maturity_bucket"], ["grade_bucket"]]
MIN_BONDS = 20


def median_baseline(train: pd.DataFrame, df: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """Predict the median training spread for every row (Regression_Improve, "0. Baseline model")."""
    return pd.Series(float(train[target].median()), index=df.index, name="pred_median")


def fit_matrix_frozen(train: pd.DataFrame, target: str = TARGET, min_bonds: int = MIN_BONDS) -> dict:
    """Cell medians over the training months, keeping only cells with at least `min_bonds` distinct bonds."""
    tables = []
    for keys in LEVELS:
        g = train.groupby(keys, observed=True).agg(med=(target, "median"), bonds=("cusip_id", "nunique"))
        tables.append((keys, g.loc[g["bonds"] >= min_bonds, "med"]))
    return {"tables": tables, "overall": float(train[target].median())}


def predict_matrix_frozen(model: dict, df: pd.DataFrame) -> pd.Series:
    """Look each row up in the finest cell that has enough bonds."""
    pred = pd.Series(np.nan, index=df.index)
    for keys, med in model["tables"]:
        hit = df[keys].merge(med.rename("m"), left_on=keys, right_index=True, how="left")["m"]
        pred = pred.fillna(pd.Series(hit.to_numpy(), index=df.index))
    return pred.fillna(model["overall"]).rename("pred_matrix_frozen")


def matrix_same_date(
    df: pd.DataFrame, target: str = TARGET, issuer: str = "issuer_ticker", min_bonds: int = MIN_BONDS
) -> pd.Series:
    """Matrix pricing re-marked on each snapshot date from other companies' bonds (the gate baseline).

    Uses only same-date spreads of other issuers, which a desk observes when it marks an untraded bond.
    Rows with a missing target never serve as peers.
    """
    data = df[df[target].notna()]
    out = pd.Series(np.nan, index=df.index, name="pred_matrix_same_date")
    for _, day in df.groupby("snapshot_date"):
        peers_day = data.loc[data.index.intersection(day.index)]
        for iss, rows in day.groupby(issuer, dropna=False):
            peers = peers_day[peers_day[issuer] != iss] if pd.notna(iss) else peers_day
            pred = pd.Series(np.nan, index=rows.index)
            for keys in LEVELS:
                g = peers.groupby(keys, observed=True)[target].agg(["median", "size"])
                med = g.loc[g["size"] >= min_bonds, "median"].rename("m")
                hit = rows[keys].merge(med, left_on=keys, right_index=True, how="left")["m"]
                pred = pred.fillna(pd.Series(hit.to_numpy(), index=rows.index))
            out.loc[rows.index] = pred.fillna(peers[target].median()).to_numpy()
    return out
