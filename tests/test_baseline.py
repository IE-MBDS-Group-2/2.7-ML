import numpy as np
import pandas as pd

from ml2.baseline import fit_matrix_frozen, matrix_same_date, median_baseline, predict_matrix_frozen


def _cell_panel(dates=("2025-07-31", "2025-08-29"), issuers=30, bonds_per_issuer=1, level=100.0):
    """One IG cell: every bond in the same grade, maturity and sector; spread = level + issuer number."""
    rows = []
    for d in dates:
        for i in range(issuers):
            for b in range(bonds_per_issuer):
                rows.append(
                    {
                        "snapshot_date": pd.Timestamp(d),
                        "issuer_ticker": f"I{i:02d}",
                        "cusip_id": f"C{i:02d}{b}",
                        "grade_bucket": "IG",
                        "maturity_bucket": "5-7y",
                        "sector": "Energy",
                        "spread_bps": level + i,
                    }
                )
    return pd.DataFrame(rows)


def test_same_date_excludes_own_issuer():
    df = _cell_panel(issuers=31)  # spreads 100..130, median of the other 30 depends on who is left out
    pred = matrix_same_date(df)
    lowest = df.index[(df["issuer_ticker"] == "I00")][0]
    highest = df.index[(df["issuer_ticker"] == "I30")][0]
    assert pred[lowest] == np.median(np.arange(101, 131))  # without 100
    assert pred[highest] == np.median(np.arange(100, 130))  # without 130


def test_same_date_uses_only_the_same_date():
    july = _cell_panel(dates=("2025-07-31",), level=100.0)
    august = _cell_panel(dates=("2025-08-29",), level=300.0)
    pred = matrix_same_date(pd.concat([july, august], ignore_index=True))
    assert pred.iloc[: len(july)].between(100, 130).all()
    assert pred.iloc[len(july):].between(300, 330).all()


def test_same_date_falls_back_when_cell_is_thin():
    df = _cell_panel(issuers=30)
    thin = df["issuer_ticker"].isin(["I00", "I01", "I02"])
    df.loc[thin, "sector"] = "Utilities"  # a 3-bond cell: falls back to grade x maturity
    pred = matrix_same_date(df)
    assert pred.notna().all()
    assert pred[thin].between(100, 129).all()


def test_frozen_matrix_and_median():
    train = _cell_panel(dates=("2024-01-31",), issuers=25)
    valid = _cell_panel(dates=("2025-07-31",), issuers=25, level=500.0)
    assert (median_baseline(train, valid) == 112.0).all()
    assert (predict_matrix_frozen(fit_matrix_frozen(train), valid) == 112.0).all()
