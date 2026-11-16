"""Database resources: the LDP source (read-only) and the group database (target)."""

from __future__ import annotations

import io

import pandas as pd
from dagster import ConfigurableResource
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


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
        """Drop and recreate schema.table from df, loading rows with COPY. Returns the row count."""
        if self.read_only:
            raise RuntimeError("This resource is read-only")
        engine = self.engine()
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.date
        out.head(0).to_sql(table, engine, schema=schema, if_exists="replace", index=False)
        buf = io.StringIO()
        out.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        cols = ", ".join(f'"{c}"' for c in out.columns)
        raw = engine.raw_connection()
        try:
            with raw.cursor() as cur:
                cur.copy_expert(f'COPY "{schema}"."{table}" ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\')', buf)
            raw.commit()
        finally:
            raw.close()
        return len(out)
