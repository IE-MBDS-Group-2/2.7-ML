"""Writing tables to the group database (used by the loader in pipeline/)."""

from __future__ import annotations

import io

import pandas as pd
from sqlalchemy.engine import Engine


def replace_table(engine: Engine, df: pd.DataFrame, schema: str, table: str) -> int:
    """Replace schema.table with df and return the row count.

    The drop, the recreate and the COPY of the rows run on one connection in one transaction, so a failed
    load rolls back and readers keep the previous table. Readers that query during the load wait for it to
    finish instead of seeing an empty table.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.date
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in out.columns)
    copy_sql = f'COPY "{schema}"."{table}" ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\')'
    with engine.begin() as conn:
        out.head(0).to_sql(table, conn, schema=schema, if_exists="replace", index=False)
        with conn.connection.cursor() as cur:
            cur.copy_expert(copy_sql, buf)
    return len(out)
