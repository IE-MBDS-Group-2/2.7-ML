"""Dagster assets: extract from the data lake, build the panel, publish to the group database.

    snapshot_dates -> trace_bonds_month_end -> issuer_resolution -> equity_month_end --+
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
from ml2.equity import (
    EQUITY_COLUMNS,
    SECTOR_TO_GICS,
    features_at_snapshots,
    issuer_keys,
    issuer_link,
    resolve_issuers,
    rows_asof,
    share_count_jumps,
)
from ml2.panel import build_panel, column_dictionary
from ml2_pipeline import queries
from ml2_pipeline.resources import PostgresResource

SCHEMA = "ml2"
MACRO_MAP = {  # data lake series id -> panel column, and a unit multiplier
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
def snapshot_dates(context: AssetExecutionContext, lake: PostgresResource) -> pd.DataFrame:
    """Month-end snapshot dates: the last TRACE trading day of each month."""
    ex = _cfg()["extract"]
    df = lake.read(
        queries.SNAPSHOT_DATES,
        {"start": ex["start_date"], "end": ex["end_date"], "min_bonds": ex["min_bonds_per_snapshot_day"],
         "min_share": ex["min_share_of_month_median_day"]},
    )
    context.add_output_metadata({"n_snapshots": len(df), "first": str(df.iloc[0, 0]), "last": str(df.iloc[-1, 0])})
    return df


@asset(group_name="extract", compute_kind="sql")
def trace_bonds_month_end(
    context: AssetExecutionContext, lake: PostgresResource, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    """Straight fixed-rate corporate bonds traded on each month-end, with month-level liquidity."""
    ex = _cfg()["extract"]
    snaps = [d.date().isoformat() for d in _dates(snapshot_dates)]
    df = lake.read(
        queries.TRACE_BONDS_MONTH_END,
        {"snapshots": snaps, "start": ex["start_date"], "end": ex["end_date"], "coupon_type": ex["coupon_type"]},
    )
    context.add_output_metadata(
        {"rows": len(df), "bonds": int(df["cusip_id"].nunique()), "issuers": int(df["issuer_ticker"].nunique())}
    )
    return df


@asset(group_name="extract", compute_kind="sql")
def ust_par_month_end(
    context: AssetExecutionContext, lake: PostgresResource, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    """Treasury par curve (percent) as of each snapshot date. Index: snapshot_date."""
    dates = _dates(snapshot_dates)
    long = lake.read(
        queries.MACRO_SERIES,
        {"series": list(TENOR_YEARS), "start": (dates.min() - pd.Timedelta(days=15)).date(), "end": dates.max().date()},
    )
    out = _series_asof(long, dates)
    out.index.name = "snapshot_date"
    context.add_output_metadata({"tenors": len(out.columns), "missing_cells": int(out.isna().sum().sum())})
    return out


@asset(group_name="extract", compute_kind="sql")
def macro_month_end(
    context: AssetExecutionContext, lake: PostgresResource, snapshot_dates: pd.DataFrame
) -> pd.DataFrame:
    """Credit spreads, VIX and SOFR as of each snapshot date."""
    dates = _dates(snapshot_dates)
    long = lake.read(
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
def issuer_resolution(
    context: AssetExecutionContext,
    lake: PostgresResource,
    snapshot_dates: pd.DataFrame,
    trace_bonds_month_end: pd.DataFrame,
) -> pd.DataFrame:
    """Which listed company each bond-month's stock data comes from (ticker match, or the issuer-code map)."""
    dates = _dates(snapshot_dates)
    bonds = trace_bonds_month_end
    tmap = lake.read(queries.TICKER_MAP, {"c6": sorted(set(bonds["cusip_id"].str[:6]))})
    tickers = sorted(set(bonds["issuer_ticker"].dropna()) | set(tmap["ticker"].dropna()))
    start = (dates.min() - pd.Timedelta(days=10)).date()
    px = lake.read(queries.PRICES, {"start": start, "end": dates.max().date(), "tickers": tickers})
    res = resolve_issuers(bonds, px, tmap)
    res["issuer_key"] = issuer_keys(res)
    moved = (res["source"] == "map") & (res["map_entity"] != res["ticker_entity"])
    context.add_output_metadata(
        {
            "rows": len(res),
            "rows_by_source": MetadataValue.json(res["source"].value_counts().to_dict()),
            "rows_moved_to_another_company": int(moved.sum()),
            "rows_without_listed_company": int(res["entity_id"].isna().sum()),
            "keys_with_entity_suffix": int(res["issuer_key"].str.contains(".", regex=False).sum()),
        }
    )
    return res


@asset(group_name="extract", compute_kind="sql")
def equity_month_end(
    context: AssetExecutionContext,
    lake: PostgresResource,
    snapshot_dates: pd.DataFrame,
    trace_bonds_month_end: pd.DataFrame,
    issuer_resolution: pd.DataFrame,
) -> pd.DataFrame:
    """Issuer equity features, market cap, debt and sector per company key and snapshot date.

    Everything is computed per listed company (data lake entity), so a ticker reused by another company never mixes
    two price histories; the result is keyed by issuer_key, the panel's issuer_ticker.
    """
    ex = _cfg()["extract"]
    dates = _dates(snapshot_dates)
    market = ex["market_ticker"]
    res = issuer_resolution.dropna(subset=["entity_id"]).copy()
    res["entity_id"] = res["entity_id"].astype("int64")
    entities = [int(e) for e in sorted(res["entity_id"].unique())]
    tickers = sorted(set(res["equity_ticker"].dropna()) | set(res["issuer_ticker"].dropna()) | {market})

    px = lake.read(queries.PRICES, {"start": ex["equity_history_start"], "end": dates.max().date(), "tickers": tickers})
    px["date"] = pd.to_datetime(px["date"])
    mkt = px[px["ticker"] == market].groupby("date")["adj_close"].last()
    px = px.dropna(subset=["entity_id"])
    px["entity_id"] = px["entity_id"].astype("int64")
    px = px[px["entity_id"].isin(entities)]
    # One listing per company and day: a company with two listings (ADR and ordinary shares, two share classes)
    # would otherwise flip between their price levels. Keep the listing with the longest price history.
    n_rows = px.groupby(["entity_id", "ticker"])["date"].transform("size")
    px = px.assign(n_rows=n_rows).sort_values(["date", "entity_id", "n_rows", "ticker"])
    px = px.drop_duplicates(["date", "entity_id"], keep="last").drop(columns="n_rows")
    adj = px.pivot_table(index="date", columns="entity_id", values="adj_close", aggfunc="last").sort_index()
    raw_close = px.pivot_table(index="date", columns="entity_id", values="raw_close", aggfunc="last").sort_index()
    adj.columns = adj.columns.astype(str)
    raw_close.columns = raw_close.columns.astype(str)
    adj["__market__"] = mkt.reindex(adj.index)

    feats = features_at_snapshots(adj, dates, "__market__").rename(columns={"ticker": "entity"})
    feats = feats[feats["entity"] != "__market__"]

    def at_snapshots(wide: pd.DataFrame, name: str) -> pd.DataFrame:
        snap = rows_asof(wide, dates).rename_axis("snapshot_date")
        long = snap.stack(future_stack=True).rename(name).reset_index()
        long.columns = ["snapshot_date", "entity", name]
        return long

    feats = feats.merge(at_snapshots(raw_close, "close_unadj"), on=["snapshot_date", "entity"], how="left")
    feats = feats.merge(at_snapshots(adj.drop(columns="__market__"), "adj_close"), on=["snapshot_date", "entity"],
                        how="left")
    feats["entity_id"] = feats["entity"].astype("int64")

    # Shares outstanding by company, as of the snapshot, at most N days stale
    shares = lake.read(
        queries.SHARES, {"entity_ids": entities, "start": pd.Timestamp(ex["equity_history_start"]).date()}
    )
    shares["date"] = pd.to_datetime(shares["date"])
    feats = _merge_asof_by(feats, shares[["entity_id", "date", "shares_outstanding"]], "entity_id", "date",
                           ex["shares_max_staleness_days"])
    feats["market_cap"] = feats["close_unadj"] * feats["shares_outstanding"]
    jumps = share_count_jumps(feats[["entity_id", "snapshot_date", "market_cap", "adj_close"]],
                              factor=ex["share_jump_factor"])
    feats.loc[jumps, "market_cap"] = np.nan

    # Total debt from the latest quarterly balance sheet already filed, reported in USD only
    bs = lake.read(queries.BALANCE_SHEET, {"tickers": tickers, "entity_ids": entities, "start": "2022-01-01"})
    ticker_entity = res.groupby("equity_ticker")["entity_id"].agg(lambda s: s.value_counts().index[0])
    bs["entity_id"] = bs["entity_id"].fillna(bs["ticker"].map(ticker_entity))
    other_ccy = bs["currency"].notna() & (bs["currency"] != ex["debt_currency"])
    n_other_ccy = int(other_ccy.sum())
    bs = bs[~other_ccy & bs["entity_id"].notna()].copy()
    bs["entity_id"] = bs["entity_id"].astype("int64")
    bs["period_end"] = pd.to_datetime(bs["period_end"])
    bs["filing_date"] = pd.to_datetime(bs["filing_date"])
    lag = pd.Timedelta(days=ex["filing_lag_days_if_missing"])
    bs["available_date"] = bs["filing_date"].fillna(bs["period_end"] + lag)
    # The newest period known on each date: an old period filed late never replaces a newer one
    bs = bs.dropna(subset=["total_debt"]).sort_values(["entity_id", "available_date", "period_end"])
    bs["newest_period"] = bs.groupby("entity_id")["period_end"].cummax()
    bs = bs[bs["period_end"] == bs["newest_period"]]
    bs = bs.drop_duplicates(["entity_id", "available_date"], keep="last")
    feats = _merge_asof_by(feats, bs[["entity_id", "available_date", "total_debt"]], "entity_id",
                           "available_date", 400)

    # From company to issuer key (one key per company), then sector by the key's ticker on each date
    keys = res.groupby(["snapshot_date", "issuer_key"]).agg(
        entity_id=("entity_id", "first"), equity_ticker=("equity_ticker", "first"),
        via_map=("source", lambda s: bool((s == "map").any()))).reset_index()
    keys["snapshot_date"] = pd.to_datetime(keys["snapshot_date"])
    out = keys.merge(feats.drop(columns=["entity"]), on=["snapshot_date", "entity_id"], how="left")
    sector = _sector_asof(lake, sorted(set(keys["equity_ticker"].dropna())), dates)
    out = out.merge(sector.rename(columns={"ticker": "equity_ticker"}), on=["snapshot_date", "equity_ticker"],
                    how="left")

    # Blank the stock data where its company is not confirmed as the bonds' issuer. Bond-months matched through
    # the issuer-code map are confirmed by construction; the rest need the issuer code or a distinctive name word.
    codes = lake.read(queries.ISSUER_CODES, {})
    names = lake.read(queries.COMPANY_NAMES, {"entity_ids": entities})
    bonds_k = trace_bonds_month_end.assign(snapshot_date=pd.to_datetime(trace_bonds_month_end["snapshot_date"])).merge(
        issuer_resolution[["snapshot_date", "cusip_id", "issuer_key"]], on=["snapshot_date", "cusip_id"])
    bonds_k = bonds_k.assign(issuer_ticker=bonds_k["issuer_key"])
    key_entity = res.groupby("issuer_key")["entity_id"].first()
    link = issuer_link(bonds_k, key_entity, codes, names.set_index("entity_id")["name"])
    via_map = set(keys.loc[keys["via_map"], "issuer_key"])
    out["issuer_confirmed"] = out["issuer_key"].map(link).fillna(False).astype(bool) | out["issuer_key"].isin(via_map)
    unconfirmed = ~out["issuer_confirmed"]
    out.loc[unconfirmed, EQUITY_COLUMNS + ["sector"]] = np.nan
    out.loc[unconfirmed, "sector_is_pit"] = False

    out = out.rename(columns={"issuer_key": "ticker"})
    keep = ["snapshot_date", "ticker", "vol_20d", "vol_60d", "vol_252d", "ret_252d", "mom_12_1", "beta_252d",
            "market_cap", "total_debt", "sector", "sector_is_pit", "issuer_confirmed"]
    out = out[keep].reset_index(drop=True)
    context.add_output_metadata(
        {
            "rows": len(out),
            "companies": len(entities),
            "share_with_market_cap": float(out["market_cap"].notna().mean()),
            "market_cap_blanked_share_jumps": int(jumps.sum()),
            "share_with_debt": float(out["total_debt"].notna().mean()),
            "debt_filings_other_currency_dropped": n_other_ccy,
            "share_with_pit_sector": float(out["sector_is_pit"].fillna(False).mean()),
            "keys_unconfirmed": int((~out.drop_duplicates("ticker")["issuer_confirmed"]).sum()),
        }
    )
    return out


def _merge_asof_by(left: pd.DataFrame, right: pd.DataFrame, by: str, right_on: str, max_days: int) -> pd.DataFrame:
    left = left.sort_values("snapshot_date")
    right = right.dropna(subset=[right_on]).sort_values(right_on)
    return pd.merge_asof(
        left,
        right,
        left_on="snapshot_date",
        right_on=right_on,
        by=by,
        direction="backward",
        tolerance=pd.Timedelta(days=max_days),
    ).drop(columns=[right_on])


def _sector_asof(lake: PostgresResource, tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    hist = lake.read(queries.SECTOR_HISTORY, {"tickers": tickers})
    prof = lake.read(queries.SECTOR_PROFILE, {"tickers": tickers})
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
    out.loc[missing, "sector"] = out.loc[missing, "ticker"].map(fallback).replace(SECTOR_TO_GICS)
    out.loc[missing, "sector_is_pit"] = False
    return out


@asset(group_name="panel", compute_kind="pandas")
def bond_month_panel(
    context: AssetExecutionContext,
    trace_bonds_month_end: pd.DataFrame,
    issuer_resolution: pd.DataFrame,
    ust_par_month_end: pd.DataFrame,
    macro_month_end: pd.DataFrame,
    equity_month_end: pd.DataFrame,
) -> pd.DataFrame:
    """The modelling table. Also written to data/processed/ as parquet with a manifest."""
    cfg = _cfg()
    bonds = trace_bonds_month_end.assign(snapshot_date=pd.to_datetime(trace_bonds_month_end["snapshot_date"])).merge(
        issuer_resolution[["snapshot_date", "cusip_id", "issuer_key", "source", "ticker_entity", "map_entity"]],
        on=["snapshot_date", "cusip_id"], how="left")
    moved = (bonds["source"] == "map") & (bonds["map_entity"] != bonds["ticker_entity"])
    bonds["issuer_ticker"] = bonds["issuer_key"].fillna(bonds["issuer_ticker"])
    panel = build_panel(bonds, ust_par_month_end, macro_month_end, equity_month_end, cfg)
    moved_keys = bonds.loc[moved, ["snapshot_date", "cusip_id"]].assign(moved=True)
    moved_keys["snapshot_date"] = pd.to_datetime(moved_keys["snapshot_date"])
    panel_moved = panel[["snapshot_date", "cusip_id"]].merge(moved_keys, how="left")["moved"].fillna(False)
    unconfirmed = set(equity_month_end.loc[~equity_month_end["issuer_confirmed"], "ticker"])

    path = resolve(cfg["paths"]["panel_parquet"])
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    relabelled = panel["grade"].map({"I": "IG", "H": "HY"}).fillna("NR") != panel["grade_bucket"]
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
        "equity_blanked": {
            "rule": "stock data blanked where its company is not confirmed as the issuer (issuer code or name)",
            "issuers": int(panel["issuer_ticker"].isin(unconfirmed).groupby(panel["issuer_ticker"]).any().sum()),
            "rows": int(panel["issuer_ticker"].isin(unconfirmed).sum()),
        },
        "issuer_from_code_map": {
            "rule": "the issuer-code map's company when unique, not ambiguous and listed on the date; else the ticker",
            "rows_moved_to_another_company": int(panel_moved.sum()),
            "issuers_moved": int(panel.loc[panel_moved.to_numpy(), "issuer_ticker"].nunique()),
        },
        "grade_from_issuer": {
            "rule": "bonds whose master record is newer than all their trades take their company's majority grade",
            "bonds": int(panel.loc[relabelled, "cusip_id"].nunique()),
            "rows": int(relabelled.sum()),
        },
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
