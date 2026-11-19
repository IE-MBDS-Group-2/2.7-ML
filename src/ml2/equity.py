"""Issuer equity features, computed from daily prices and balance sheets.

Every feature uses only information available on or before the snapshot date.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_price_features(adj_close: pd.DataFrame, market: str) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) features from adjusted closes.

    Returns annualized volatility (20, 60, 252 days), 252-day return, 12-1 momentum
    and 252-day beta against the market column.
    """
    px = adj_close.sort_index()
    rets = px.pct_change(fill_method=None)
    ann = np.sqrt(TRADING_DAYS)

    feats = {
        "vol_20d": rets.rolling(20, min_periods=15).std() * ann,
        "vol_60d": rets.rolling(60, min_periods=40).std() * ann,
        "vol_252d": rets.rolling(252, min_periods=180).std() * ann,
        "ret_252d": px / px.shift(252) - 1,
        "mom_12_1": px.shift(21) / px.shift(252) - 1,
    }

    if market in rets.columns:
        m = rets[market]
        window, minp = 252, 120
        mean_x = rets.rolling(window, min_periods=minp).mean()
        mean_m = m.rolling(window, min_periods=minp).mean()
        mean_xm = rets.mul(m, axis=0).rolling(window, min_periods=minp).mean()
        cov = mean_xm - mean_x.mul(mean_m, axis=0)
        var_m = m.rolling(window, min_periods=minp).var(ddof=0)
        feats["beta_252d"] = cov.div(var_m.replace(0, np.nan), axis=0)
    else:
        feats["beta_252d"] = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    return feats


def rows_asof(wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows of a date-indexed frame as of each date (last row on or before the date)."""
    idx = wide.index.searchsorted(dates, side="right") - 1
    out = pd.DataFrame(np.nan, index=dates, columns=wide.columns)
    valid = idx >= 0
    out.loc[dates[valid]] = wide.iloc[idx[valid]].to_numpy()
    return out


def features_at_snapshots(
    adj_close: pd.DataFrame, snapshot_dates: pd.DatetimeIndex, market: str
) -> pd.DataFrame:
    """Long frame (snapshot_date, ticker, features...) of price-based features."""
    dates = pd.DatetimeIndex(snapshot_dates)
    long_parts = []
    for name, wide in rolling_price_features(adj_close, market).items():
        snap = rows_asof(wide, dates)
        snap.index.name = "snapshot_date"
        s = snap.stack(future_stack=True).rename(name)
        long_parts.append(s)
    out = pd.concat(long_parts, axis=1).reset_index()
    out = out.rename(columns={"level_1": "ticker"})
    if "ticker" not in out.columns:
        out = out.rename(columns={out.columns[1]: "ticker"})
    return out


def naive_distance_to_default(
    equity_value: pd.Series | np.ndarray,
    debt_face: pd.Series | np.ndarray,
    sigma_equity: pd.Series | np.ndarray,
    expected_return: pd.Series | np.ndarray,
    horizon_years: float = 1.0,
) -> np.ndarray:
    """Bharath and Shumway (2008) naive distance to default.

    sigma_D = 0.05 + 0.25 * sigma_E
    sigma_V = E/(E+F) * sigma_E + F/(E+F) * sigma_D
    DD = (ln((E+F)/F) + (mu - 0.5 * sigma_V^2) * T) / (sigma_V * sqrt(T))
    Higher DD = further from default. NaN when debt or equity is missing or not positive.
    """
    e = np.asarray(equity_value, dtype=float)
    f = np.asarray(debt_face, dtype=float)
    s_e = np.asarray(sigma_equity, dtype=float)
    mu = np.asarray(expected_return, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (e > 0) & (f > 0) & (s_e > 0)
        s_d = 0.05 + 0.25 * s_e
        v = e + f
        s_v = (e / v) * s_e + (f / v) * s_d
        dd = (np.log(v / f) + (mu - 0.5 * s_v**2) * horizon_years) / (s_v * np.sqrt(horizon_years))
    return np.where(valid, dd, np.nan)


EQUITY_COLUMNS = ["vol_20d", "vol_60d", "vol_252d", "ret_252d", "mom_12_1", "beta_252d", "market_cap", "total_debt"]
_NAME_STOP = {"inc", "corp", "corporation", "company", "the", "holdings", "holding", "group", "ltd", "llc", "plc",
              "and", "co", "lp", "international", "intl", "trust", "new", "of", "de", "sa", "nv", "ag", "se", "spa"}
# Words many unrelated companies share: a match on one of them alone does not confirm a company
_NAME_GENERIC = {"bank", "bancorp", "bk", "american", "america", "amer", "first", "energy", "capital", "financial",
                 "finance", "funding", "national", "general", "united", "global", "resources", "partners", "services",
                 "systems", "industries", "technologies", "technology", "us", "usa", "north", "south", "east", "west",
                 "power", "health", "healthcare", "communications", "air", "lines", "airlines", "motor", "motors",
                 "insurance", "properties", "realty", "fund", "pharmaceuticals", "oil", "gas", "electric", "water",
                 "brands", "foods", "credit", "investment", "investments", "securities", "markets", "solutions",
                 "enterprises", "mortgage", "life", "royal", "pacific", "atlantic", "southern", "northern", "western",
                 "eastern", "central", "mid", "inds", "svcs", "fin", "hldgs", "grp", "natl"}

# Company profiles use their own sector names; map them to the GICS names used by the sector history
SECTOR_TO_GICS = {
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Healthcare": "Health Care",
    "Technology": "Information Technology",
    "Basic Materials": "Materials",
    "Consumer Defensive": "Consumer Staples",
}


def name_tokens(name: object) -> set[str]:
    """Distinctive lower-case words of a company name (legal suffixes and fillers removed)."""
    words = re.findall(r"[a-z0-9]+", str(name).lower().replace("&", ""))
    return {w for w in words if len(w) >= 2 and w not in _NAME_STOP and w not in _NAME_GENERIC}


def issuer_link(
    bonds: pd.DataFrame,
    price_entity: pd.Series,
    issuer_codes: pd.DataFrame,
    entity_names: pd.Series,
) -> pd.Series:
    """Is the company behind each ticker's stock data confirmed as the issuer of the ticker's bonds?

    The bond master gives each bond one ticker (its latest), and stock data is looked up by that ticker. A ticker
    is confirmed when at least one of its bonds' issuer codes (first 6 CUSIP characters) belongs to the stock's
    company, or when the bond issuer name and the stock company's name share a word. Otherwise the stock data may
    belong to another company, and the caller blanks it.

    bonds:        issuer_ticker, cusip_id, issuer_nm
    price_entity: ticker -> entity id behind the ticker's prices
    issuer_codes: c6, entity_id (from equity CUSIPs)
    entity_names: entity id -> company name
    Returns a boolean Series indexed by ticker.
    """
    b = bonds[["issuer_ticker", "cusip_id", "issuer_nm"]].drop_duplicates()
    b = b.assign(c6=b["cusip_id"].str[:6], price_entity=b["issuer_ticker"].map(price_entity))
    codes = b.merge(issuer_codes, on="c6", how="inner")
    by_code = set(codes.loc[codes["entity_id"] == codes["price_entity"], "issuer_ticker"])
    company = b["price_entity"].map(entity_names)
    shared = [bool(name_tokens(n) & name_tokens(c)) if pd.notna(c) else False for n, c in zip(b["issuer_nm"], company)]
    by_name = set(b.loc[shared, "issuer_ticker"])
    tickers = pd.Index(sorted(b["issuer_ticker"].dropna().unique()), name="ticker")
    return pd.Series(tickers.isin(by_code | by_name), index=tickers, name="issuer_confirmed")


def resolve_issuers(
    bonds: pd.DataFrame, prices: pd.DataFrame, ticker_map: pd.DataFrame, max_stale_days: int = 7
) -> pd.DataFrame:
    """Which company's stock each bond-month uses.

    By default the stock behind the bond master's ticker, as listed on the snapshot date. The issuer-code map
    (first 6 CUSIP characters to company) wins when, on that date, it gives exactly one company, is not flagged
    ambiguous, and that company has a listed price; otherwise the ticker match stays. This fixes bonds of a
    separately listed company joined to another company's ticker, and tickers reused by another company.

    bonds:      snapshot_date, cusip_id, issuer_ticker
    prices:     date, ticker, entity_id (listed prices)
    ticker_map: issuer_cusip6, entity_id, cusip6_valid_from, cusip6_valid_to, cusip6_ambiguous, ticker,
                valid_from, valid_to, ticker_precedence
    Returns one row per bond-month: snapshot_date, cusip_id, issuer_ticker, ticker_entity, map_entity,
    entity_id, equity_ticker, source ("map" or "ticker").
    """
    b = bonds[["snapshot_date", "cusip_id", "issuer_ticker"]].drop_duplicates(["snapshot_date", "cusip_id"]).copy()
    b["snapshot_date"] = pd.to_datetime(b["snapshot_date"])
    px = prices.dropna(subset=["entity_id", "ticker"])[["date", "ticker", "entity_id"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px["entity_id"] = px["entity_id"].astype("int64")

    # Listings as of each snapshot: the last price row of each ticker within max_stale_days
    parts = []
    for d in sorted(b["snapshot_date"].unique()):
        w = px[(px["date"] <= d) & (px["date"] > d - pd.Timedelta(days=max_stale_days))]
        w = w.sort_values(["date", "entity_id"]).drop_duplicates("ticker", keep="last")
        parts.append(w.assign(snapshot_date=d)[["snapshot_date", "ticker", "entity_id"]])
    listed = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"snapshot_date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=object),
         "entity_id": pd.Series(dtype="int64")})
    by_ticker = listed.rename(columns={"ticker": "issuer_ticker", "entity_id": "ticker_entity"})
    b = b.merge(by_ticker, on=["snapshot_date", "issuer_ticker"], how="left")
    listed_entities = listed[["snapshot_date", "entity_id"]].drop_duplicates()

    # Map candidates valid on the snapshot date
    m = ticker_map.copy()
    for c in ["cusip6_valid_from", "valid_from"]:
        m[c] = pd.to_datetime(m[c]).fillna(pd.Timestamp("1900-01-01"))
    for c in ["cusip6_valid_to", "valid_to"]:
        m[c] = pd.to_datetime(m[c]).fillna(pd.Timestamp("2100-01-01"))
    m = m[~m["cusip6_ambiguous"].fillna(False).astype(bool)].dropna(subset=["entity_id"]).copy()
    m["entity_id"] = m["entity_id"].astype("int64")
    cand = b[["snapshot_date", "cusip_id"]].assign(c6=b["cusip_id"].str[:6]).merge(
        m, left_on="c6", right_on="issuer_cusip6", how="inner")
    d = cand["snapshot_date"]
    cand = cand[(cand["cusip6_valid_from"] <= d) & (d <= cand["cusip6_valid_to"])
                & (cand["valid_from"] <= d) & (d <= cand["valid_to"])]
    n_ent = cand.groupby(["snapshot_date", "cusip_id"])["entity_id"].transform("nunique")
    cand = cand[n_ent == 1].merge(listed_entities, on=["snapshot_date", "entity_id"], how="inner")
    cand = cand.sort_values(["ticker_precedence", "valid_from", "ticker"], ascending=[True, False, True])
    best = cand.drop_duplicates(["snapshot_date", "cusip_id"])[["snapshot_date", "cusip_id", "entity_id", "ticker"]]
    best = best.rename(columns={"entity_id": "map_entity", "ticker": "map_ticker"})
    b = b.merge(best, on=["snapshot_date", "cusip_id"], how="left")

    use_map = b["map_entity"].notna()
    b["entity_id"] = b["map_entity"].where(use_map, b["ticker_entity"])
    b["equity_ticker"] = b["map_ticker"].where(use_map, b["issuer_ticker"])
    b["source"] = np.where(use_map, "map", "ticker")
    return b.drop(columns=["map_ticker"])


def issuer_keys(resolved: pd.DataFrame) -> pd.Series:
    """One key per company: its ticker, plus ".<entity id>" when a ticker belongs to more than one company.

    The company with the most bond-months under a ticker keeps the plain ticker, so most keys are ordinary tickers.
    """
    r = resolved[["equity_ticker", "entity_id"]]
    pairs = r.dropna().groupby(["equity_ticker", "entity_id"]).size().rename("n").reset_index()
    pairs = pairs.sort_values(["equity_ticker", "n", "entity_id"], ascending=[True, False, True])
    main = pairs.drop_duplicates("equity_ticker").set_index("equity_ticker")["entity_id"]
    is_main = r["entity_id"].isna() | (r["equity_ticker"].map(main) == r["entity_id"])
    suffix = r["entity_id"].astype("Int64").astype(str)
    return pd.Series(np.where(is_main, r["equity_ticker"], r["equity_ticker"] + "." + suffix),
                     index=resolved.index, name="issuer_key")


def share_count_jumps(frame: pd.DataFrame, factor: float = 2.5, window: int = 3) -> pd.Series:
    """Flag snapshots where market cap implies a share count far from the company's recent past.

    frame: entity_id, snapshot_date, market_cap, adj_close (one row per company and snapshot). Market cap divided by
    the split-adjusted price is a split-free share count; a jump of more than `factor` against the median of the
    previous `window` snapshots is a data error (wrong share class, a stale or mis-scaled count), not a market move.
    Only past snapshots are used. Returns a boolean Series aligned with frame.
    """
    f = frame.sort_values(["entity_id", "snapshot_date"])
    implied = f["market_cap"] / f["adj_close"]
    trailing = implied.groupby(f["entity_id"]).transform(lambda s: s.shift(1).rolling(window, min_periods=1).median())
    ratio = implied / trailing
    flag = (ratio > factor) | (ratio < 1 / factor)
    return flag.reindex(frame.index).fillna(False).astype(bool)
