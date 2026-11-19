import numpy as np
import pandas as pd

from ml2.config import load_config
from ml2.panel import COLUMN_DOCS, build_panel

SNAPS = pd.to_datetime(["2025-01-31", "2025-02-28"])


def _inputs():
    bonds = pd.DataFrame(
        {
            "snapshot_date": [SNAPS[0], SNAPS[1], SNAPS[0], SNAPS[0]],
            "cusip_id": ["B1", "B1", "B2", "B3"],
            "issuer_ticker": ["AAA", "AAA", "AAA", "ZZZ"],
            "issuer_nm": ["Alpha", "Alpha", "Alpha", "Zed"],
            "scrty_ds": ["x"] * 4,
            "debt_type_cd": ["S-NT", "S-NT", "SUB-NT", None],
            "cpn_rt": [5.0, 5.0, 6.0, 4.0],
            "mtrty_dt": pd.to_datetime(["2030-01-31", "2030-01-31", "2025-06-30", "2032-01-31"]),
            "ind_144a": ["N", "N", "Y", "N"],
            "grade": ["I", "I", "H", "I"],
            "stdt": pd.to_datetime(["2020-01-31"] * 4),
            "n_trades": [3, 4, 2, 1],
            "par_volume": [1e6, 2e6, 5e5, 1e5],
            "vwap_price": [99.0, 98.0, 100.0, 97.0],
            "last_price": [99.0, 98.0, 100.0, 97.0],
            "vw_yield": [5.0, 5.5, 6.0, 5.2],
            "n_trades_inst": [1, 0, 0, 0],
            "par_volume_inst": [1e6, 0, 0, 0],
            "vwap_price_inst": [99.1, np.nan, np.nan, np.nan],
            "n_trades_customer": [2, 2, 1, 1],
            "trade_days_in_month": [10, 12, 3, 1],
            "par_volume_month": [1e7, 2e7, 1e6, 1e5],
            "n_trades_month": [30, 40, 5, 1],
        }
    )
    curves = pd.DataFrame({"UST_PAR_2Y": [4.0, 4.0], "UST_PAR_10Y": [4.0, 4.5]}, index=SNAPS)
    macro = pd.DataFrame(
        {"snapshot_date": SNAPS, "hy_oas_bps": [300.0, 320.0], "bbb_oas_bps": [100.0, 110.0],
         "vix": [15.0, 18.0], "sofr_pct": [4.3, 4.3]}
    )
    equity = pd.DataFrame(
        {
            "snapshot_date": SNAPS,
            "ticker": ["AAA", "AAA"],
            "vol_20d": [0.2, 0.25], "vol_60d": [0.2, 0.22], "vol_252d": [0.25, 0.25],
            "ret_252d": [0.1, 0.05], "mom_12_1": [0.08, 0.04], "beta_252d": [1.1, 1.1],
            "market_cap": [1e10, 1e10], "total_debt": [5e9, 5e9],
            "sector": ["Industrials", "Industrials"], "sector_is_pit": [True, True],
        }
    )
    return bonds, curves, macro, equity


def test_panel_target_filters_and_columns():
    panel = build_panel(*_inputs(), load_config())
    assert list(panel.columns) == [c for c, _, _ in COLUMN_DOCS]
    # B2 matures in < 1 year: dropped. B3's issuer has no equity data: dropped.
    assert set(panel["cusip_id"]) == {"B1"}
    jan = panel[panel["snapshot_date"] == SNAPS[0]].iloc[0]
    assert np.isclose(jan["spread_bps"], (5.0 - 4.0) * 100)  # curve is flat at 4.0 beyond 2y in January
    assert jan["price"] == 99.1  # institutional VWAP preferred
    assert jan["grade_bucket"] == "IG"


def test_next_month_widening_label():
    panel = build_panel(*_inputs(), load_config())
    jan = panel[panel["snapshot_date"] == SNAPS[0]].iloc[0]
    feb = panel[panel["snapshot_date"] == SNAPS[1]].iloc[0]
    assert np.isclose(jan["spread_change_next_bps"], feb["spread_bps"] - jan["spread_bps"])
    assert jan["widens_25bp"] == int(jan["spread_change_next_bps"] >= 25)
    assert pd.isna(feb["widens_25bp"])  # no later snapshot


def test_partition_assigned():
    panel = build_panel(*_inputs(), load_config())
    assert set(panel["partition"]) == {"train"}


def test_grade_from_a_later_master_record_follows_the_company():
    bonds, curves, macro, equity = _inputs()
    bonds.loc[2, "mtrty_dt"] = pd.Timestamp("2028-06-30")  # B2: H flag, company AAA's other bond B1 is I
    panel = build_panel(bonds, curves, macro, equity, load_config())
    assert panel.loc[panel.cusip_id == "B2", "grade_bucket"].item() == "HY"  # record older than its trade: kept

    bonds.loc[2, "stdt"] = pd.Timestamp("2025-03-31")  # record written after the bond's only trade
    panel = build_panel(bonds, curves, macro, equity, load_config())
    b2 = panel.loc[panel.cusip_id == "B2"].iloc[0]
    assert b2["grade"] == "H" and b2["grade_bucket"] == "IG"


def test_label_uses_next_spread_even_when_that_row_is_filtered_out():
    bonds, curves, macro, equity = _inputs()
    bonds.loc[1, "vw_yield"] = 45.0  # B1 blows out in February: the row is dropped by the yield bound
    panel = build_panel(bonds, curves, macro, equity, load_config())
    b1 = panel[panel.cusip_id == "B1"]
    assert len(b1) == 1  # only January is left
    assert b1["widens_25bp"].item() == 1  # but January still knows it widened
