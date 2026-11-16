"""Dagster assets: extract from LDP, build the panel, publish to the group database.

    snapshot_dates -> trace_bonds_month_end -> equity_month_end --+
                   -> ust_par_month_end ----------------------------+-> bond_month_panel -> group_database
                   -> macro_month_end ------------------------------+
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

from ml2.config import load_config, resolve
from ml2.curves import TENOR_YEARS
from ml2.equity import features_at_snapshots, rows_asof
from ml2.panel import build_panel, column_dictionary
from ml2_pipeline import queries
from ml2_pipeline.resources import PostgresResource

SCHEMA = "ml2"
MACRO_MAP = {  # LDP series id -> panel column, and a unit multiplier
    "BAMLH0A0HYM2": ("hy_oas_bps", 100.0),
    "BAMLC0A4CBBB": ("bbb_oas_bps", 100.0),
    "VIXCLS": ("vix", 1.0),
    "NYFED_SOFR": ("sofr_pct", 1.0),
}


def _cfg() -> dict:
    return load_config()


def _dates(snapshot_dates: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(snapshot_dates["snapshot_date"])).sort_values()


def _series_asof(long: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long (series_id, date, value) -> one row per snapshot date, last value on or before it."""
    wide = long.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().ffill(limit=10)
    return rows_asof(wide, dates)


@asset(group_name="extract", compute_kind="sql")
def snapshot_dates(context: AssetExecutionContext, ldp: PostgresResource) -> pd.DataFrame:
    """Month-end snapshot dates: the last TRACE trading day of each month."""
    ex = _cfg()["extract"]
    df = ldp.read(queries.SNAPSHOT_DATES, {"start": ex["start_date"], "end": ex["end_date"]})
    context.add_output_metadata({"n_snapshots": len(df), "first": str(df.iloc[0, 0]), "last": str(df.iloc[-1, 0])})
    return df


@asset(group_name="extract", compute_kind="sql")
def trace_bonds_month_end(
    context: AssetExecutionContext, ldp: PostgresResource, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    """Straight fixed-rate corporate bonds traded on each month-end, with month-level liquidity."""
    ex = _cfg()["extract"]
    snaps = [d.date().isoformat() for d in _dates(snapshot_dates)]
    df = ldp.read(
        queries.TRACE_BONDS_MONTH_END,
        {"snapshots": snaps, "start": ex["start_date"], "end": ex["end_date"], "coupon_type": ex["coupon_type"]},
    )
    context.add_output_metadata(
        {"rows": len(df), "bonds": int(df["cusip_id"].nunique()), "issuers": int(df["issuer_ticker"].nunique())}
    )
    return df


@asset(group_name="extract", compute_kind="sql")
def ust_par_month_end(
    context: AssetExecutionContext, ldp: PostgresResource, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    """Treasury par curve (percent) as of each snapshot date. Index: snapshot_date."""
    dates = _dates(snapshot_dates)
    long = ldp.read(
        queries.MACRO_SERIES,
        {"series": list(TENOR_YEARS), "start": (dates.min() - pd.Timedelta(days=15)).date(), "end": dates.max().date()},
    )
    out = _series_asof(long, dates)
    out.index.name = "snapshot_date"
    context.add_output_metadata({"tenors": len(out.columns), "missing_cells": int(out.isna().sum().sum())})
    return out


@asset(group_name="extract", compute_kind="sql")
def macro_month_end(
    context: AssetExecutionContext, ldp: PostgresResource, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    """Credit spreads, VIX and SOFR as of each snapshot date."""
    dates = _dates(snapshot_dates)
    long = ldp.read(
        queries.MACRO_SERIES,
        {"series": list(MACRO_MAP), "start": (dates.min() - pd.Timedelta(days=15)).date(), "end": dates.max().date()},
    )
    wide = _series_asof(long, dates)
    out = pd.DataFrame(index=dates)
    for sid, (col, mult) in MACRO_MAP.items():
        out[col] = wide[sid] * mult if sid in wide.columns else np.nan
    out = out.rename_axis("snapshot_date").reset_index()
    context.add_output_metadata({"rows": len(out), "missing_cells": int(out.isna().sum().sum())})
    return out


@asset(group_name="extract", compute_kind="sql")
def equity_month_end(
    context: AssetExecutionContext,
    ldp: PostgresResource,
    snapshot_dates: pd.DataFrame,
    trace_bonds_month_end: pd.DataFrame,
) -> pd.DataFrame:
    """Issuer equity features, market cap, debt and sector as of each snapshot date."""
    ex = _cfg()["extract"]
    dates = _dates(snapshot_dates)
    market = ex["market_ticker"]
    tickers = sorted(set(trace_bonds_month_end["issuer_ticker"].dropna()) | {market})

    px = ldp.read(queries.PRICES, {"start": ex["equity_history_start"], "end": dates.max().date(), "tickers": tickers})
    px["date"] = pd.to_datetime(px["date"])
    adj = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    raw_close = px.pivot_table(index="date", columns="ticker", values="raw_close", aggfunc="last").sort_index()

    feats = features_at_snapshots(adj, dates, market)
    close_snap = rows_asof(raw_close, dates).rename_axis("snapshot_date")
    close_long = close_snap.stack(future_stack=True).rename("close_unadj").reset_index()
    close_long.columns = ["snapshot_date", "ticker", "close_unadj"]
    feats = feats.merge(close_long, on=["snapshot_date", "ticker"], how="left")
    feats = feats[feats["ticker"] != market]

    # Shares outstanding (by entity id), as of the snapshot, at most N days stale
    ent = px.dropna(subset=["entity_id"]).sort_values("date").groupby("ticker")["entity_id"].last().astype("int64")
    shares = ldp.read(
        queries.SHARES,
        {"entity_ids": ent.unique().tolist(), "start": (pd.Timestamp(ex["equity_history_start"])).date()},
    )
    shares["date"] = pd.to_datetime(shares["date"])
    shares = shares.merge(ent.rename("entity_id").reset_index(), on="entity_id")
    feats = _merge_asof_by_ticker(
        feats, shares[["ticker", "date", "shares_outstanding"]], "date", ex["shares_max_staleness_days"]
    )
    feats["market_cap"] = feats["close_unadj"] * feats["shares_outstanding"]

    # Total debt from the latest quarterly balance sheet already filed
    bs = ldp.read(queries.BALANCE_SHEET, {"tickers": tickers, "start": "2022-01-01"})
    bs["period_end"] = pd.to_datetime(bs["period_end"])
    bs["filing_date"] = pd.to_datetime(bs["filing_date"])
    lag = pd.Timedelta(days=ex["filing_lag_days_if_missing"])
    bs["available_date"] = bs["filing_date"].fillna(bs["period_end"] + lag)
    bs = bs.dropna(subset=["total_debt"]).sort_values(["ticker", "available_date", "period_end"])
    bs = bs.drop_duplicates(["ticker", "available_date"], keep="last")
    feats = _merge_asof_by_ticker(feats, bs[["ticker", "available_date", "total_debt"]], "available_date", 400)

    # Sector: point-in-time GICS where available, else today's profile
    feats = feats.merge(_sector_asof(ldp, tickers, dates), on=["snapshot_date", "ticker"], how="left")

    keep = ["snapshot_date", "ticker", "vol_20d", "vol_60d", "vol_252d", "ret_252d", "mom_12_1", "beta_252d",
            "market_cap", "total_debt", "sector", "sector_is_pit"]
    out = feats[keep].reset_index(drop=True)
    context.add_output_metadata(
        {
            "rows": len(out),
            "tickers_requested": len(tickers) - 1,
            "tickers_with_prices": int(out.dropna(subset=["vol_60d"])["ticker"].nunique()),
            "share_with_market_cap": float(out["market_cap"].notna().mean()),
            "share_with_debt": float(out["total_debt"].notna().mean()),
            "share_with_pit_sector": float(out["sector_is_pit"].fillna(False).mean()),
        }
    )
    return out


def _merge_asof_by_ticker(left: pd.DataFrame, right: pd.DataFrame, right_on: str, max_days: int) -> pd.DataFrame:
    left = left.sort_values("snapshot_date")
    right = right.dropna(subset=[right_on]).sort_values(right_on)
    return pd.merge_asof(
        left,
        right,
        left_on="snapshot_date",
        right_on=right_on,
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=max_days),
    ).drop(columns=[right_on])


def _sector_asof(ldp: PostgresResource, tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    hist = ldp.read(queries.SECTOR_HISTORY, {"tickers": tickers})
    prof = ldp.read(queries.SECTOR_PROFILE, {"tickers": tickers})
    grid = pd.MultiIndex.from_product([dates, tickers], names=["snapshot_date", "ticker"]).to_frame(index=False)

    hist["valid_from"] = pd.to_datetime(hist["valid_from"])
    hist["valid_to"] = pd.to_datetime(hist["valid_to"]).fillna(pd.Timestamp("2100-01-01"))
    hist["rank"] = np.where(hist["source"].eq("compustat_hgic"), 0, 1)
    pit = grid.merge(hist, on="ticker", how="inner")
    pit = pit[(pit["snapshot_date"] >= pit["valid_from"]) & (pit["snapshot_date"] <= pit["valid_to"])]
    pit = pit.sort_values("rank").drop_duplicates(["snapshot_date", "ticker"])
    pit = pit[["snapshot_date", "ticker", "gics_sector_name"]].rename(columns={"gics_sector_name": "sector"})
    pit["sector_is_pit"] = True

    out = grid.merge(pit, on=["snapshot_date", "ticker"], how="left")
    fallback = prof.drop_duplicates("ticker").set_index("ticker")["sector"]
    missing = out["sector"].isna()
    out.loc[missing, "sector"] = out.loc[missing, "ticker"].map(fallback)
    out.loc[missing, "sector_is_pit"] = False
    return out


@asset(group_name="panel", compute_kind="pandas")
def bond_month_panel(
    context: AssetExecutionContext,
    trace_bonds_month_end: pd.DataFrame,
    ust_par_month_end: pd.DataFrame,
    macro_month_end: pd.DataFrame,
    equity_month_end: pd.DataFrame,
) -> pd.DataFrame:
    """The modelling table. Also written to data/processed/ as parquet with a manifest."""
    cfg = _cfg()
    panel = build_panel(trace_bonds_month_end, ust_par_month_end, macro_month_end, equity_month_end, cfg)

    path = resolve(cfg["paths"]["panel_parquet"])
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(panel),
        "bonds": int(panel["cusip_id"].nunique()),
        "issuers": int(panel["issuer_ticker"].nunique()),
        "snapshots": int(panel["snapshot_date"].nunique()),
        "first_snapshot": str(panel["snapshot_date"].min().date()),
        "last_snapshot": str(panel["snapshot_date"].max().date()),
        "rows_by_partition": panel["partition"].value_counts().to_dict(),
        "rows_by_grade": panel["grade_bucket"].value_counts().to_dict(),
        "parquet": str(cfg["paths"]["panel_parquet"]),
        "sha256": sha,
        "extract_config": cfg["extract"],
        "split_config": cfg["split"],
    }
    mpath = resolve(cfg["paths"]["manifest"])
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    context.add_output_metadata(
        {
            "rows": manifest["rows"],
            "bonds": manifest["bonds"],
            "issuers": manifest["issuers"],
            "snapshots": manifest["snapshots"],
            "rows_by_partition": MetadataValue.json(manifest["rows_by_partition"]),
            "rows_by_grade": MetadataValue.json(manifest["rows_by_grade"]),
            "sha256": sha,
            "preview": MetadataValue.md(panel.head(5).to_markdown() if _has_tabulate() else str(panel.head(5))),
        }
    )
    return panel


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


@asset(group_name="publish", compute_kind="postgres")
def group_database(
    context: AssetExecutionContext,
    groupdb: PostgresResource,
    bond_month_panel: pd.DataFrame,
    ust_par_month_end: pd.DataFrame,
    macro_month_end: pd.DataFrame,
) -> MaterializeResult:
    """Publish schema ml2 to the group database and grant read access to role ml2_read."""
    cfg = _cfg()
    groupdb.execute(f"create schema if not exists {SCHEMA}")

    counts = {
        "bond_month_panel": groupdb.replace_table(bond_month_panel, SCHEMA, "bond_month_panel"),
        "ust_par_month_end": groupdb.replace_table(ust_par_month_end.reset_index(), SCHEMA, "ust_par_month_end"),
        "macro_month_end": groupdb.replace_table(macro_month_end, SCHEMA, "macro_month_end"),
        "column_dictionary": groupdb.replace_table(column_dictionary(), SCHEMA, "column_dictionary"),
    }
    manifest = json.loads(resolve(cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    load = pd.DataFrame([{"key": k, "value": json.dumps(v, default=str)} for k, v in manifest.items()])
    counts["load_manifest"] = groupdb.replace_table(load, SCHEMA, "load_manifest")

    groupdb.execute(
        f"""
        create index if not exists bond_month_panel_date_idx on {SCHEMA}.bond_month_panel (snapshot_date);
        create index if not exists bond_month_panel_issuer_idx on {SCHEMA}.bond_month_panel (issuer_ticker);
        """
    )
    granted = True
    try:
        groupdb.execute(
            f"""
            grant usage on schema {SCHEMA} to ml2_read;
            grant select on all tables in schema {SCHEMA} to ml2_read;
            """
        )
    except Exception as exc:  # role missing on a fresh managed host: run docker/initdb/01-init.sql first
        granted = False
        context.log.warning(f"Could not grant read access to ml2_read: {exc}")

    return MaterializeResult(
        metadata={"tables": MetadataValue.json(counts), "read_grant": granted, "sha256": manifest["sha256"]}
    )
