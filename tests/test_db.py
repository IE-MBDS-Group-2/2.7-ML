"""ml2.db.replace_table: a reload either fully replaces a table or leaves the old one untouched.

Needs a scratch Postgres: set ML2_TEST_DB_URL (CI starts one). Each test works in its own schema and drops it.
"""

import os
import uuid

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from ml2.db import replace_table

DB_URL = os.environ.get("ML2_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="set ML2_TEST_DB_URL to a scratch Postgres to run")


@pytest.fixture
def db():
    engine = create_engine(DB_URL)
    schema = f"ml2_test_{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        conn.execute(text(f'create schema "{schema}"'))
    yield engine, schema
    with engine.begin() as conn:
        conn.execute(text(f'drop schema "{schema}" cascade'))
    engine.dispose()


def _read(engine, schema, table):
    return pd.read_sql(text(f'select * from "{schema}"."{table}" order by id'), engine)


def _old():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "snapshot_date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
            "spread_bps": [85.0, np.nan, 190.5],
            "issuer": ["Alpha", None, "Zed"],
        }
    )


def test_load_writes_all_rows_and_values(db):
    engine, schema = db
    assert replace_table(engine, _old(), schema, "t") == 3
    got = _read(engine, schema, "t")
    assert got["id"].tolist() == [1, 2, 3]
    assert got["spread_bps"].isna().tolist() == [False, True, False]
    assert got.loc[2, "spread_bps"] == 190.5
    assert got["issuer"].isna().tolist() == [False, True, False]
    assert str(got.loc[0, "snapshot_date"]) == "2025-01-31"


def test_reload_replaces_rows_and_columns(db):
    engine, schema = db
    replace_table(engine, _old(), schema, "t")
    new = pd.DataFrame({"id": [7], "label": ["only"]})
    assert replace_table(engine, new, schema, "t") == 1
    got = _read(engine, schema, "t")
    assert got.columns.tolist() == ["id", "label"]
    assert got["id"].tolist() == [7]


def test_failed_reload_keeps_the_previous_table(db):
    engine, schema = db
    replace_table(engine, _old(), schema, "t")
    # Postgres rejects a NUL byte in text, so COPY fails after the table has been recreated.
    bad = pd.DataFrame({"id": [9, 10], "label": ["fine", "bad\x00value"]})
    with pytest.raises(Exception):
        replace_table(engine, bad, schema, "t")
    got = _read(engine, schema, "t")
    assert got.columns.tolist() == ["id", "snapshot_date", "spread_bps", "issuer"]
    assert got["id"].tolist() == [1, 2, 3]
