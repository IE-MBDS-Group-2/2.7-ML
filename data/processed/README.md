# data/processed

Written by the loader, gitignored:

- `bond_month_panel.parquet`: the modelling table (same as `ml2.bond_month_panel` in the group database)
- `manifest.json`: build time, row counts, extract and split settings, sha256 of the parquet file

Check your copy: `from ml2.data import check_panel; check_panel()` (compares the file's sha256 with the manifest).
