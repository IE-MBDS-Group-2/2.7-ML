"""Build the bond-month modelling table from the loader's extracts.

One row per bond per month-end snapshot. Pure pandas: the Dagster loader handles
the database reads and writes, this module holds the logic so it can be tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml2.curves import treasury_at_maturity
from ml2.equity import naive_distance_to_default
from ml2.schema import COLUMN_DOCS, column_dictionary  # noqa: F401  (re-exported for the loader)
from ml2.splits import assign_partition

MATURITY_BINS = [1, 3, 5, 7, 10, 20, np.inf]
MATURITY_LABELS = ["1-3y", "3-5y", "5-7y", "7-10y", "10-20y", "20y+"]


def issuer_consistent_grade(df: pd.DataFrame) -> pd.Series:
    """IG or HY per row: the TRACE grade flag, unless the flag comes from a master record newer than the bond's trades.

    The bond master keeps one record per bond, the latest, with its grade flag. When that record starts on or after
    every snapshot the bond appears in, it was written after the bond traded (typically as the bond nears maturity
    and its ratings are withdrawn; FINRA may then flag an unrated bond as non-investment grade), so its flag says
    nothing about how the bond traded. Those bonds take the majority grade of their company's other bonds whose
    record predates their trades. A company with no such bond keeps the flag.
    """
    raw = pd.Series(np.where(df["grade"] == "I", "IG", np.where(df["grade"] == "H", "HY", "NR")), index=df.index)
    newer = (df["stdt"] >= df["snapshot_date"]).groupby(df["cusip_id"]).transform("all")
    trusted = (
        pd.DataFrame({"issuer": df["issuer_ticker"], "cusip": df["cusip_id"], "g": raw})[~newer]
        .drop_duplicates("cusip")
        .sort_values("cusip")
    )
    company = df["issuer_ticker"].map(trusted.groupby("issuer")["g"].agg(lambda s: s.value_counts().index[0]))
    return raw.where(~newer | company.isna(), company)


def build_panel(
    bonds: pd.DataFrame,
    curves: pd.DataFrame,
    macro: pd.DataFrame,
    equity: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Join the extracts into the modelling panel and compute target and features.

    bonds:  one row per (snapshot_date, cusip_id) from TRACE plus the bond master
    curves: index snapshot_date, columns UST_PAR_* (percent)
    macro:  snapshot_date, hy_oas_bps, bbb_oas_bps, vix, sofr_pct
    equity: snapshot_date, ticker, equity features, market_cap, total_debt, sector, sector_is_pit
    """
    ex, pr = cfg["extract"], cfg["problem"]
    df = bonds.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["mtrty_dt"] = pd.to_datetime(df["mtrty_dt"])
    df["stdt"] = pd.to_datetime(df["stdt"])

    # --- target ---------------------------------------------------------------
    df["years_to_maturity"] = (df["mtrty_dt"] - df["snapshot_date"]).dt.days / 365.25

    # Every reported spread before the row filters. The annex label needs next month's spread even when that row is
    # dropped below (a spread past the bound, a yield past the bound, no stock data): those are the bonds that blew
    # out, so dropping them would hide most of the widenings.
    raw = df[(df["years_to_maturity"] > 0) & (df["vw_yield"] > 0)]
    next_spread = pd.DataFrame({
        "cusip_id": raw["cusip_id"],
        "snapshot_date": raw["snapshot_date"],
        "spread_next_bps": (raw["vw_yield"].astype(float)
                            - treasury_at_maturity(curves, raw["snapshot_date"], raw["years_to_maturity"])) * 100,
    })
    snaps = np.sort(df["snapshot_date"].unique())

    df = df[df["years_to_maturity"].between(ex["min_years_to_maturity"], ex["max_years_to_maturity"])]
    y_lo, y_hi = ex["yield_bounds_pct"]
    df = df[(df["vw_yield"] > y_lo) & (df["vw_yield"] < y_hi)].copy()

    df["yield_pct"] = df["vw_yield"].astype(float)
    df["ust_yield_at_maturity_pct"] = treasury_at_maturity(curves, df["snapshot_date"], df["years_to_maturity"])
    df["spread_bps"] = (df["yield_pct"] - df["ust_yield_at_maturity_pct"]) * 100
    s_lo, s_hi = ex["spread_bounds_bps"]
    df = df[df["spread_bps"].between(s_lo, s_hi)].copy()

    # --- bond terms -----------------------------------------------------------
    df["maturity_bucket"] = pd.cut(df["years_to_maturity"], MATURITY_BINS, labels=MATURITY_LABELS, right=False)
    df["maturity_bucket"] = df["maturity_bucket"].astype(str)
    df["coupon_pct"] = df["cpn_rt"].astype(float)
    df["bond_age_years"] = ((df["snapshot_date"] - df["stdt"]).dt.days / 365.25).clip(lower=0)
    df["grade_bucket"] = np.where(df["grade"] == "I", "IG", np.where(df["grade"] == "H", "HY", "NR"))
    df["is_144a"] = df["ind_144a"].fillna("N").eq("Y")
    df["debt_type"] = df["debt_type_cd"].fillna("UNKNOWN")
    df["is_subordinated"] = df["debt_type"].str.contains("SUB", na=False)

    # --- liquidity ------------------------------------------------------------
    df["log_par_volume"] = np.log1p(df["par_volume"].astype(float))
    df["has_inst_trade"] = df["n_trades_inst"].fillna(0) > 0
    df["price"] = np.where(df["has_inst_trade"], df["vwap_price_inst"], df["vwap_price"])
    df["log_par_volume_month"] = np.log1p(df["par_volume_month"].astype(float))

    # --- issuer structure -----------------------------------------------------
    grp = df.groupby(["snapshot_date", "issuer_ticker"])
    df["n_bonds_issuer"] = grp["cusip_id"].transform("count")
    df["issuer_log_par_volume_month"] = np.log1p(grp["par_volume_month"].transform("sum").astype(float))

    # --- curve shape and macro (same for every bond on a date) ----------------
    c = curves.copy()
    c.index = pd.to_datetime(c.index)
    curve_feats = pd.DataFrame(
        {"ust_2y_pct": c.get("UST_PAR_2Y"), "ust_10y_pct": c.get("UST_PAR_10Y")}, index=c.index
    )
    curve_feats["slope_2s10s_bps"] = (curve_feats["ust_10y_pct"] - curve_feats["ust_2y_pct"]) * 100
    curve_feats = curve_feats.rename_axis("snapshot_date").reset_index()
    df = df.merge(curve_feats, on="snapshot_date", how="left")
    m = macro.copy()
    m["snapshot_date"] = pd.to_datetime(m["snapshot_date"])
    df = df.merge(m, on="snapshot_date", how="left")

    # --- issuer equity (bonds of issuers without equity data are dropped) -----
    eq = equity.copy()
    eq["snapshot_date"] = pd.to_datetime(eq["snapshot_date"])
    df = df.merge(eq, left_on=["snapshot_date", "issuer_ticker"], right_on=["snapshot_date", "ticker"], how="inner")
    df["log_market_cap"] = np.log(df["market_cap"].where(df["market_cap"] > 0))
    debt = df["total_debt"].where(df["total_debt"] >= 0)
    df["leverage"] = debt / (debt + df["market_cap"])
    df["naive_dd"] = naive_distance_to_default(df["market_cap"], debt, df["vol_252d"], df["ret_252d"])
    df["sector"] = df["sector"].fillna("Unknown")
    df["sector_is_pit"] = df["sector_is_pit"].fillna(False).astype(bool)

    # --- grade: the TRACE flag, corrected where it comes from a later record ---
    df["grade_bucket"] = issuer_consistent_grade(df)

    # --- annex label: does the spread widen by the next month-end? ------------
    snap_idx = {d: i for i, d in enumerate(snaps)}
    df["snap_idx"] = df["snapshot_date"].map(snap_idx)
    nxt = next_spread.assign(snap_idx=next_spread["snapshot_date"].map(snap_idx) - 1)
    nxt = nxt[["cusip_id", "snap_idx", "spread_next_bps"]].drop_duplicates(["cusip_id", "snap_idx"])
    df = df.merge(nxt, on=["cusip_id", "snap_idx"], how="left")
    df["spread_change_next_bps"] = df["spread_next_bps"] - df["spread_bps"]
    widens = (df["spread_change_next_bps"] >= pr["widen_threshold_bps"]).astype("Int8")
    df["widens_25bp"] = widens.where(df["spread_change_next_bps"].notna())

    # --- partition and final shape --------------------------------------------
    df["partition"] = assign_partition(df["snapshot_date"], cfg["split"]).to_numpy()
    df = df.rename(columns={"issuer_nm": "issuer_name"})
    cols = [c for c, _, _ in COLUMN_DOCS]
    out = df[cols].sort_values(["snapshot_date", "issuer_ticker", "cusip_id"]).reset_index(drop=True)
    return out
