"""Load the modelling panel.

Two sources, same columns:
    1. The group database (table ml2.bond_month_panel), when ML2_DB_URL is set in .env
    2. The parquet file in data/processed/, otherwise

Example
    from ml2.data import load_panel
    df = load_panel()                       # whole panel
    train = load_panel(partitions=["train"])
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

from ml2.config import load_config, repo_root, resolve

PANEL_TABLE = "ml2.bond_month_panel"
DICTIONARY_TABLE = "ml2.column_dictionary"


def _db_url() -> str | None:
    load_dotenv(repo_root() / ".env")
    url = os.environ.get("ML2_DB_URL", "").strip()
    return url or None


def load_panel(
    source: str = "auto",
    columns: Iterable[str] | None = None,
    partitions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the bond-month panel as a DataFrame.

    source: "auto" (database if ML2_DB_URL is set, else parquet), "db" or "parquet".
    columns: optional subset of columns.
    partitions: optional subset of {"train", "gap", "valid", "test"}.
    """
    url = _db_url()
    use_db = source == "db" or (source == "auto" and url is not None)
    cols = list(columns) if columns else None

    if use_db:
        if url is None:
            raise RuntimeError("ML2_DB_URL is not set in .env")
        from sqlalchemy import create_engine, text

        select = ", ".join(f'"{c}"' for c in cols) if cols else "*"
        sql = f"select {select} from {PANEL_TABLE}"
        params = {}
        if partitions:
            sql += " where partition = any(:parts)"
            params["parts"] = list(partitions)
        engine = create_engine(url)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    else:
        path = resolve(load_config()["paths"]["panel_parquet"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Set ML2_DB_URL in .env (see docs/DATA_ACCESS.md) or get the parquet file."
            )
        df = pd.read_parquet(path, columns=cols)
        if partitions:
            df = df[df["partition"].isin(list(partitions))]

    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.reset_index(drop=True)


def load_column_dictionary() -> pd.DataFrame:
    """Column name, group and description, as documented in docs/DATA_DICTIONARY.md."""
    from ml2.schema import column_dictionary

    return column_dictionary()
