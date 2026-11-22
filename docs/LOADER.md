# Data loader and group database (data admin)

A self-contained Dagster project in Docker. It reads the source mart through a read-only connection, builds the
panel with `src/ml2/panel.py`, writes `data/processed/bond_month_panel.parquet` with a sha256 manifest, and
publishes schema `ml2` to the group Postgres database.

```text
snapshot_dates -> trace_bonds_month_end -> equity_month_end --+
               -> ust_par_month_end ----------------------------+-> bond_month_panel -> group_database
               -> macro_month_end ------------------------------+
```

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
