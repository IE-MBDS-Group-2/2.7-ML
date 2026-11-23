# Data loader and group database (data admin)

A self-contained Dagster project in Docker. It reads Miguel's data lake through a read-only connection, builds the
panel with `src/ml2/panel.py`, writes `data/processed/bond_month_panel.parquet` with a sha256 manifest, and
publishes schema `ml2` to the group Postgres database.

```text
snapshot_dates -> trace_bonds_month_end -> equity_month_end --+
               -> ust_par_month_end ----------------------------+-> bond_month_panel -> group_database
               -> macro_month_end ------------------------------+
```

## Sources

The loader reads Miguel's data lake (`LAKE_DB_URL`), a Postgres store of the sources below, and never writes to it.

| Data lake table | What it holds | Where it comes from |
|---|---|---|
| `credit.trace_daily` | Each bond's trades, yield, price and volume per day | TRACE Enhanced (FINRA corporate bond trade reports), through IE's WRDS access |
| `credit.bond_reference` | Issuer, ticker, coupon, maturity, 144A flag, grade, debt type | TRACE corporate bond master file, through IE's WRDS access |
| `macro.series_daily` | US Treasury par yields; ICE BofA HY and BBB spread indices; VIX; SOFR | treasury.gov; FRED; the New York Fed |
| `mart.prices_daily_v2`, `entities.shares_daily` | Daily stock prices and shares outstanding | Commercial equity market data |
| `fundamentals.balance_sheet`, `fundamentals.company_profile`, `universes.sector_history` | Quarterly balance sheets, company profiles and sectors | Commercial company fundamentals |
| `entities.companies`, `entities.security_identifiers`, `fixed_income.issuer_ticker_map` | Company names, issuer codes and the issuer-to-stock map | Built in the data lake from the sources above |

TRACE is licensed through IE's WRDS subscription, so the panel stays inside the group and the submission.

## Run

```bash
docker compose up -d --build              # Postgres (127.0.0.1:5433) + Dagster UI (http://localhost:3000)
```

Materialize all assets (job `refresh_ml2_data`). Then:

```bash
python scripts/add_member.py <name> --host <public host> --port <public port>   # one read-only login each
python scripts/write_data_dictionary.py                                          # refresh "Last build"
```

## Files

| Path | Role |
|---|---|
| `pipeline/ml2_pipeline/` | Dagster assets, resources (read-only source, owner target), SQL |
| `src/ml2/panel.py`, `curves.py`, `equity.py` | Panel logic: spread target, equity features, labels (tested) |
| `src/ml2/db.py` | Table load into the group database: drop, recreate and COPY in one transaction, so a failed load keeps the previous table (tested) |
| `docker/Dockerfile`, `docker/requirements.txt` | Loader image (Dagster 1.13.27) |
| `docker/initdb/01-init.sql` | Schema `ml2`, role `ml2_read`, default grants |
| `docker-compose.yml` | `groupdb` (Postgres 16) and `dagster` |
| `scripts/add_member.py` | Create, rotate or revoke personal read-only logins |

## Hosting the group database (decision pending)

The container binds to `127.0.0.1`, so nothing is exposed until we choose:

| Option | Teammate setup | Notes |
|---|---|---|
| A. Managed Postgres (Neon or Supabase free tier) | None: paste the URL | TLS by default; point `GROUP_DB_URL` at it and run `docker/initdb/01-init.sql` once. Data sits with a third-party host |
| B. Self-hosted on a private network (Tailscale) | Install Tailscale, accept an invite | Data stays on our infrastructure; one install per person |
| C. Self-hosted, public port | None | Needs TLS, strong passwords and an IP allowlist. Not recommended |

Only `GROUP_DB_URL` changes between options.
