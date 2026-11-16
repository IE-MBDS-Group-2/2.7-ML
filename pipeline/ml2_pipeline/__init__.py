"""Self-contained Dagster loader for the ML II group database.

Reads the LDP mart (read-only), builds the bond-month panel with src/ml2,
writes data/processed/bond_month_panel.parquet and publishes schema ml2 to the group database.
"""
