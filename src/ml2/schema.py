"""Columns of the modelling table `bond_month_panel`: name, group and meaning.

The single source for docs/DATA_DICTIONARY.md and for ml2.data.load_column_dictionary().
"""

from __future__ import annotations

import pandas as pd

# (column, group, description). Order here is the column order of the panel.
COLUMN_DOCS: list[tuple[str, str, str]] = [
    ("snapshot_date", "id", "Month-end snapshot: the last bond trading day of the month"),
    ("cusip_id", "id", "9-character CUSIP: the bond's unique identifier"),
    ("issuer_ticker", "id", "Stock ticker of the company that issued the bond; grouping key for issuer-grouped validation"),
    ("issuer_name", "id", "Issuer name"),
    ("partition", "id", "train / gap / valid / test by snapshot date (see config.yaml split)"),
    ("spread_bps", "target", "TARGET. Bond yield minus the Treasury yield of the same maturity, in basis points (1 bp = 0.01%)"),
    ("yield_pct", "target", "The bond's yield on the snapshot day (volume-weighted over trades), percent. Contains the target: not a feature"),
    ("ust_yield_at_maturity_pct", "target", "Treasury yield at the bond's remaining maturity, percent. Contains the target: not a feature"),
    ("spread_change_next_bps", "annex", "Spread at the next month-end minus spread now (empty if the bond does not trade then). Future info: not a feature"),
    ("widens_25bp", "annex", "ANNEX TARGET. 1 if the spread widens by 25 bp or more by the next month-end, 0 if not, empty if unknown"),
    ("years_to_maturity", "bond", "Years until the bond repays its principal"),
    ("maturity_bucket", "bond", "Maturity bucket: 1-3y, 3-5y, 5-7y, 7-10y, 10-20y, 20y+"),
    ("coupon_pct", "bond", "Annual interest the bond pays, percent of face value"),
    ("bond_age_years", "bond", "Years since the start of the bond's latest master record, not its issue date. Not a feature: a record rewritten after the bond traded is future information"),
    ("grade", "bond", "TRACE credit grade flag: I = investment grade, H = high yield (the bond's latest master record, not as of the snapshot)"),
    ("grade_bucket", "bond", "IG or HY: the two strata for metrics and the quality gate. The TRACE flag, except for bonds whose master record is newer than all their trades: those take the majority grade of their company's other bonds"),
    ("is_144a", "bond", "True for a privately placed bond (Rule 144A), usually less liquid"),
    ("debt_type", "bond", "Debt type code (seniority and structure, e.g. S-NT senior note)"),
    ("is_subordinated", "bond", "True if the bond is subordinated (paid after senior debt in a default)"),
    ("n_trades", "liquidity", "Number of trades on the snapshot day"),
    ("log_par_volume", "liquidity", "log(1 + face value traded on the snapshot day, USD)"),
    ("has_inst_trade", "liquidity", "True if at least one trade of $1 million face value or more that day (an institutional trade)"),
    ("price", "liquidity", "Price in percent of face value (institutional trades if any, else all trades). Contains the target: not a feature"),
    ("trade_days_in_month", "liquidity", "Number of days in the month the bond traded"),
    ("log_par_volume_month", "liquidity", "log(1 + face value traded over the month, USD)"),
    ("n_bonds_issuer", "issuer", "Number of the issuer's bonds in this snapshot"),
    ("issuer_log_par_volume_month", "issuer", "log(1 + face value traded over the month across all the issuer's bonds)"),
    ("sector", "issuer", "GICS sector of the issuer, as of the snapshot where available"),
    ("sector_is_pit", "issuer", "True if the sector is as of the snapshot date, False if it is today's sector"),
    ("vol_20d", "equity", "Issuer's stock: annualized volatility of daily returns over 20 trading days"),
    ("vol_60d", "equity", "Issuer's stock: annualized volatility over 60 trading days"),
    ("vol_252d", "equity", "Issuer's stock: annualized volatility over 252 trading days (one year)"),
    ("ret_252d", "equity", "Issuer's stock: total return over the last year"),
    ("mom_12_1", "equity", "Issuer's stock: return from 12 months ago to 1 month ago (momentum)"),
    ("beta_252d", "equity", "Issuer's stock: sensitivity to the S&P 500 (SPY) over one year"),
    ("log_market_cap", "equity", "log(market value of the issuer's shares, USD)"),
    ("total_debt", "equity", "Issuer's total debt from the latest balance sheet published before the snapshot, USD"),
    ("leverage", "equity", "Market leverage: total debt / (total debt + market cap)"),
    ("naive_dd", "equity", "Distance to default (Bharath-Shumway): how many standard deviations the firm is from default; higher = safer"),
    ("ust_2y_pct", "macro", "2-year Treasury yield, percent"),
    ("ust_10y_pct", "macro", "10-year Treasury yield, percent"),
    ("slope_2s10s_bps", "macro", "10-year minus 2-year Treasury yield, basis points (shape of the yield curve)"),
    ("hy_oas_bps", "macro", "Average spread of the US high-yield bond market (ICE BofA index), basis points"),
    ("bbb_oas_bps", "macro", "Average spread of BBB-rated corporate bonds (ICE BofA index), basis points"),
    ("vix", "macro", "VIX: expected volatility of the S&P 500 (market fear gauge)"),
    ("sofr_pct", "macro", "SOFR: the overnight secured funding rate, percent"),
]

NOT_FEATURES = [
    # identifiers
    "snapshot_date", "cusip_id", "issuer_ticker", "issuer_name", "partition",
    # contain the target
    "spread_bps", "yield_pct", "ust_yield_at_maturity_pct", "price",
    # contain the future
    "spread_change_next_bps", "widens_25bp",
    # duplicate of grade_bucket
    "grade",
    # dated by the latest master record, which can postdate the trade
    "bond_age_years",
]


def column_dictionary() -> pd.DataFrame:
    """The panel's columns with their group and description."""
    return pd.DataFrame(COLUMN_DOCS, columns=["column", "group", "description"])


def feature_columns() -> list[str]:
    """Columns allowed as model inputs (everything except ids, targets and leaky columns)."""
    return [c for c, _, _ in COLUMN_DOCS if c not in NOT_FEATURES]
