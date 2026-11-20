# data/processed

Written by the loader, gitignored:

- `bond_month_panel.parquet`: the modelling table (same as `ml2.bond_month_panel` in the group database)
- `manifest.json`: build time, row counts, extract and split settings, sha256 of the parquet file

Check your copy: `python -c "import hashlib;print(hashlib.sha256(open('data/processed/bond_month_panel.parquet','rb').read()).hexdigest())"`
must match `sha256` in the manifest.
