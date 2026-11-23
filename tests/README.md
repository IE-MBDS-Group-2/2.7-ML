# tests

pytest tests for the code in `src/`: `tests/test_<module>.py`. Run with `pytest`.

`test_db.py` needs a scratch Postgres and is skipped without one. CI starts one; locally, point
`ML2_TEST_DB_URL` at a database you own (for example the local group database admin URL). Each test creates
its own schema and drops it afterwards.
