"""SQL against Miguel's data lake. All read-only."""

# The last day of each month with a normal trading volume. Thin days are skipped: Good Friday (bond market
# closed, a few trades still reported) and half-days such as the Friday after Thanksgiving.
SNAPSHOT_DATES = """
with daily as (
    select trade_date, count(*) as n_bonds
    from credit.trace_daily
    where trade_date between :start and :end
    group by trade_date
),
monthly as (
    select date_trunc('month', trade_date) as month,
           percentile_cont(0.5) within group (order by n_bonds) as median_bonds
    from daily
    group by 1
)
select max(d.trade_date) as snapshot_date
from daily d
join monthly m on m.month = date_trunc('month', d.trade_date)
where d.n_bonds >= :min_bonds
  and d.n_bonds >= :min_share * m.median_bonds
group by date_trunc('month', d.trade_date)
order by 1
"""

# One row per bond per month-end snapshot, with month-level liquidity.
TRACE_BONDS_MONTH_END = """
with me as (
    select unnest(cast(:snapshots as date[])) as snapshot_date
),
ref as (
    select cusip_id, upper(company_symbol) as issuer_ticker, issuer_nm, scrty_ds, debt_type_cd,
           cpn_rt, mtrty_dt, ind_144a, grade, stdt
    from credit.bond_reference
    where sub_prdct_type = 'CORP'
      and cpn_type_cd = :coupon_type
      and coalesce(cnvrb_fl, 'N') = 'N'
      and company_symbol is not null
      and cpn_rt is not null
      and mtrty_dt is not null
),
monthly as (
    select t.cusip_id,
           cast(date_trunc('month', t.trade_date) as date) as month,
           count(*) as trade_days_in_month,
           sum(t.par_volume) as par_volume_month,
           sum(t.n_trades) as n_trades_month
    from credit.trace_daily t
    join ref r on r.cusip_id = t.cusip_id
    where t.trade_date between :start and :end
    group by 1, 2
)
select me.snapshot_date, r.cusip_id, r.issuer_ticker, r.issuer_nm, r.scrty_ds, r.debt_type_cd,
       r.cpn_rt, r.mtrty_dt, r.ind_144a, r.grade, r.stdt,
       t.n_trades, t.par_volume, t.vwap_price, t.last_price, t.vw_yield,
       t.n_trades_inst, t.par_volume_inst, t.vwap_price_inst, t.n_trades_customer,
       m.trade_days_in_month, m.par_volume_month, m.n_trades_month
from credit.trace_daily t
join me on t.trade_date = me.snapshot_date
join ref r on r.cusip_id = t.cusip_id
left join monthly m
       on m.cusip_id = t.cusip_id
      and m.month = cast(date_trunc('month', me.snapshot_date) as date)
where t.vw_yield is not null
"""

MACRO_SERIES = """
select series_id, date, value
from macro.series_daily
where series_id = any(:series)
  and date between :start and :end
"""

# Filter on ts_event (indexed), not on date: see the view comment in the data lake.
PRICES = """
select date, ticker, adj_close, raw_close, entity_id
from mart.prices_daily_v2
where ts_event >= cast(:start as timestamptz)
  and ts_event < cast(:end as timestamptz) + interval '1 day'
  and ticker = any(:tickers)
"""

SHARES = """
select entity_id, date, shares_outstanding
from entities.shares_daily
where entity_id = any(:entity_ids)
  and date >= cast(:start as date)
"""

BALANCE_SHEET = """
select ticker, entity_id, date as period_end, filing_date, total_debt,
       raw_json::jsonb ->> 'reportedCurrency' as currency
from fundamentals.balance_sheet
where period = 'quarter'
  and (ticker = any(:tickers) or entity_id = any(:entity_ids))
  and date >= cast(:start as date)
"""

# Issuer code (first 6 CUSIP characters) to listed company, with validity spans (data lake fixed_income views)
TICKER_MAP = """
select issuer_cusip6, entity_id, cusip6_valid_from, cusip6_valid_to, cusip6_ambiguous,
       ticker, valid_from, valid_to, ticker_precedence
from fixed_income.issuer_ticker_map
where issuer_cusip6 = any(:c6)
"""

SECTOR_HISTORY = """
select ticker, valid_from, valid_to, gics_sector_name, source
from universes.sector_history
where ticker = any(:tickers)
  and gics_sector_name is not null
"""

SECTOR_PROFILE = """
select ticker, sector
from fundamentals.company_profile
where ticker = any(:tickers)
"""

# Issuer codes: the first 6 characters of each company's equity CUSIPs, to confirm the bond-to-stock link.
ISSUER_CODES = """
select distinct left(id_value, 6) as c6, entity_id
from entities.security_identifiers
where id_type in ('cusip8', 'cusip9')
  and entity_id is not null
"""

COMPANY_NAMES = """
select entity_id, name
from entities.companies
where entity_id = any(:entity_ids)
"""
