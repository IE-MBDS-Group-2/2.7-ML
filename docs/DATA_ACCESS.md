# Data access

The modelling table is `bond_month_panel`: one row per corporate bond per month-end, January 2024 to March 2026.
It is built from licensed data (TRACE, via IE's WRDS subscription), so it stays inside the group and the
submission zip. **Never commit it, upload it anywhere else, or share your login.**

## Option 1: the parquet file (default)

1. Download `bond_month_panel.parquet` and `manifest.json` from the private group folder (Miguel shares the
   link). Do not forward the link or the file.
2. Put both files in `data/processed/` (gitignored). Leave `ML2_DB_URL` empty in `.env`.
3. In Python:

   ```python
   from ml2.data import check_panel, load_panel
   check_panel()            # raises if your copy is not the shared version
   df = load_panel()
   ```

`notebooks/00_data_access.ipynb` does exactly this and shows the row counts by partition. If it runs, you are set.
When the data is rebuilt, a new pair of files replaces the old one: download both again.

## Option 2: the group database (if we set it up)

1. Miguel sends you a personal line `ML2_DB_URL=postgresql+psycopg2://ml2_<you>:<password>@<host>:<port>/ml2`.
2. Copy `.env.example` to `.env` at the repo root and paste the line in. `.env` is gitignored.
3. In Python:

   ```python
   from ml2.data import load_panel
   df = load_panel()
   ```

Your login is **read-only**. Tables in schema `ml2`:

| Table | What it is |
|---|---|
| `bond_month_panel` | The modelling table (see [DATA_DICTIONARY.md](DATA_DICTIONARY.md)) |
| `column_dictionary` | Column, group, description |
| `ust_par_month_end` | Treasury yield curve at each snapshot (handy for plots) |
| `macro_month_end` | High-yield and BBB spreads, VIX, SOFR at each snapshot |
| `load_manifest` | When the data was built, row counts, checksum |

Any SQL client works too (DBeaver, pgAdmin, the VS Code PostgreSQL extension) with the same host, port, user and password.

## Problems

| Symptom | Fix |
|---|---|
| `FileNotFoundError ... bond_month_panel.parquet` | The files are not in `data/processed/`: download both (or set `ML2_DB_URL`) |
| `ValueError ... does not match manifest.json` | Old or partial download: get both files again |
| `password authentication failed` | Copy the whole line again from Miguel's message, without spaces |
| `could not connect to server` / timeout | Check the host and port in the line; ask Miguel whether you need the group VPN |
| `permission denied` | You tried to write. Your login is read-only by design: save results in your notebook or `reports/` |
