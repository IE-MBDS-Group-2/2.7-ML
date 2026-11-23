import numpy as np
import pandas as pd

from ml2.equity import features_at_snapshots, naive_distance_to_default, rolling_price_features, rows_asof


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    m = rng.normal(0, 0.01, n)
    stock = 2 * m + rng.normal(0, 0.001, n)
    px = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + m), "AAA": 50 * np.cumprod(1 + stock), "FLAT": np.full(n, 10.0)},
        index=idx,
    )
    px.columns.name = "ticker"
    return px


def test_beta_and_flat_volatility():
    feats = rolling_price_features(_prices(), "SPY")
    assert abs(feats["beta_252d"]["AAA"].iloc[-1] - 2.0) < 0.1
    assert feats["vol_60d"]["FLAT"].iloc[-1] == 0.0


def test_rows_asof_takes_last_row_on_or_before():
    wide = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    out = rows_asof(wide, pd.DatetimeIndex(["2025-01-04", "2025-01-01", "2025-01-06"]))
    assert out["x"].iloc[0] == 2.0
    assert np.isnan(out["x"].iloc[1])
    assert out["x"].iloc[2] == 3.0


def test_features_at_snapshots_long_shape():
    px = _prices()
    snaps = pd.DatetimeIndex([px.index[-1], px.index[-30]])
    out = features_at_snapshots(px, snaps, "SPY")
    assert set(["snapshot_date", "ticker", "vol_20d", "beta_252d", "mom_12_1"]).issubset(out.columns)
    assert len(out) == 2 * 3


def test_no_lookahead_in_snapshot_features():
    px = _prices()
    snap = pd.DatetimeIndex([px.index[200]])
    before = features_at_snapshots(px, snap, "SPY")
    changed = px.copy()
    changed.iloc[201:] *= 3  # change the future only
    after = features_at_snapshots(changed, snap, "SPY")
    pd.testing.assert_frame_equal(before, after)


def test_naive_dd_falls_with_leverage_and_handles_zero_debt():
    dd = naive_distance_to_default(
        np.array([100.0, 100.0, 100.0]), np.array([10.0, 100.0, 0.0]), np.array([0.3, 0.3, 0.3]),
        np.array([0.05, 0.05, 0.05]),
    )
    assert dd[0] > dd[1]
    assert np.isnan(dd[2])


def test_issuer_link_by_code_or_name():
    from ml2.equity import issuer_link

    bonds = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB", "CCC", "DDD"],
            "cusip_id": ["111111AA1", "222222AA2", "333333AA3", "444444AA4"],
            "issuer_nm": ["ALPHA CORP", "BK OF BETA", "GAMMA INC", "DELTA HLDGS"],
        }
    )
    price_entity = pd.Series({"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4})
    codes = pd.DataFrame({"c6": ["111111", "333333"], "entity_id": [1, 99]})  # CCC's code points elsewhere
    names = pd.Series({1: "Something Else Inc", 2: "Beta Bancorp", 3: "Unrelated Co", 4: "Delta Air Lines"})
    link = issuer_link(bonds, price_entity, codes, names)
    assert link["AAA"]          # issuer code matches the stock's company
    assert link["BBB"]          # names share "beta"
    assert not link["CCC"]      # code points to another company and names differ
    assert link["DDD"]          # names share "delta"


def test_resolve_issuers_uses_the_code_map_only_when_safe():
    from ml2.equity import resolve_issuers

    d = pd.Timestamp("2025-01-31")
    bonds = pd.DataFrame({
        "snapshot_date": [d, d, d, d],
        "cusip_id": ["111111AA1", "222222AA2", "333333AA3", "444444AA4"],
        "issuer_ticker": ["PAR", "PAR", "OLD", "XYZ"],
    })
    prices = pd.DataFrame({
        "date": [d, d, d, d],
        "ticker": ["PAR", "SUB", "OLD", "XYZ"],
        "entity_id": [1, 2, 3, 4],
    })
    tmap = pd.DataFrame({
        "issuer_cusip6": ["222222", "333333", "333333", "444444"],
        "entity_id": [2, 7, 8, 4],
        "cusip6_valid_from": [None] * 4, "cusip6_valid_to": [None] * 4,
        "cusip6_ambiguous": [False, False, False, False],
        "ticker": ["SUB", "OLD", "NEW", "XYZ"],
        "valid_from": [None] * 4, "valid_to": [None] * 4,
        "ticker_precedence": [1, 1, 1, 1],
    })
    r = resolve_issuers(bonds, prices, tmap).set_index("cusip_id")
    assert r.loc["111111AA1", "source"] == "ticker" and r.loc["111111AA1", "entity_id"] == 1  # no map row
    assert r.loc["222222AA2", "source"] == "map" and r.loc["222222AA2", "entity_id"] == 2      # listed subsidiary
    assert r.loc["222222AA2", "equity_ticker"] == "SUB"
    assert r.loc["333333AA3", "source"] == "ticker" and r.loc["333333AA3", "entity_id"] == 3   # two companies
    assert r.loc["444444AA4", "source"] == "map" and r.loc["444444AA4", "entity_id"] == 4


def test_resolve_issuers_needs_a_listed_price_on_the_date():
    from ml2.equity import resolve_issuers

    d = pd.Timestamp("2025-01-31")
    bonds = pd.DataFrame({"snapshot_date": [d], "cusip_id": ["222222AA2"], "issuer_ticker": ["PAR"]})
    prices = pd.DataFrame({"date": [d, d - pd.Timedelta(days=40)], "ticker": ["PAR", "SUB"], "entity_id": [1, 2]})
    tmap = pd.DataFrame({
        "issuer_cusip6": ["222222"], "entity_id": [2], "cusip6_valid_from": [None], "cusip6_valid_to": [None],
        "cusip6_ambiguous": [False], "ticker": ["SUB"], "valid_from": [None], "valid_to": [None],
        "ticker_precedence": [1],
    })
    r = resolve_issuers(bonds, prices, tmap)
    assert r["source"].item() == "ticker" and r["entity_id"].item() == 1  # SUB stopped trading 40 days ago


def test_issuer_keys_split_a_reused_ticker():
    from ml2.equity import issuer_keys

    res = pd.DataFrame({"equity_ticker": ["ABC", "ABC", "ABC", "XYZ"], "entity_id": [1, 1, 9, None]})
    assert issuer_keys(res).tolist() == ["ABC", "ABC", "ABC.9", "XYZ"]


def test_share_count_jumps_flag_a_spike_not_a_split():
    from ml2.equity import share_count_jumps

    dates = pd.date_range("2025-01-31", periods=5, freq="ME")
    frame = pd.DataFrame({
        "entity_id": [1] * 5 + [2] * 5,
        "snapshot_date": list(dates) * 2,
        # company 1: a 10-for-1 split in month 3 (price /10, shares x10): market cap and adjusted price unchanged
        # company 2: market cap 40x in month 3 only, price unchanged: a wrong share count
        "market_cap": [100.0, 102, 101, 103, 104, 50, 51, 2000, 52, 53],
        "adj_close": [10.0, 10.2, 10.1, 10.3, 10.4, 5, 5.1, 5.0, 5.2, 5.3],
    })
    flags = share_count_jumps(frame)
    assert flags.sum() == 1 and frame.loc[flags, "entity_id"].item() == 2


def test_generic_words_do_not_confirm_a_company():
    from ml2.equity import name_tokens

    assert not (name_tokens("BANK OF AMERICA CORP") & name_tokens("Bank of Montreal"))
    assert not (name_tokens("AMERICAN EXPRESS CO") & name_tokens("American Airlines Group Inc"))
    assert name_tokens("DELTA AIR LINES INC") & name_tokens("Delta Air Lines, Inc.")
