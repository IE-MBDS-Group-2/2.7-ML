"""Database resources: the data lake source (read-only) and the group database (target)."""

from __future__ import annotations

import pandas as pd
from dagster import ConfigurableResource
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ml2.db import replace_table


class PostgresResource(ConfigurableResource):
    """A Postgres connection given as a SQLAlchemy URL."""

    url: str
    read_only: bool = False

    def engine(self) -> Engine:
        connect_args = {"options": "-c default_transaction_read_only=on"} if self.read_only else {}
        return create_engine(self.url, pool_pre_ping=True, connect_args=connect_args)

    def read(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with self.engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    def execute(self, sql: str) -> None:
        if self.read_only:
            raise RuntimeError("This resource is read-only")
        with self.engine().begin() as conn:
            conn.execute(text(sql))

    def replace_table(self, df: pd.DataFrame, schema: str, table: str) -> int:
        """Replace schema.table with df in one transaction (see ml2.db.replace_table). Returns the row count."""
        if self.read_only:
            raise RuntimeError("This resource is read-only")
        return replace_table(self.engine(), df, schema, table)
