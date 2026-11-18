import hashlib
import json

import pandas as pd
import pytest

from ml2.data import check_panel
from ml2.final import open_test_once


def _write_panel(tmp_path, sha=None):
    parquet = tmp_path / "panel.parquet"
    pd.DataFrame({"spread_bps": [100.0, 200.0]}).to_parquet(parquet)
    digest = sha or hashlib.sha256(parquet.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": digest, "rows": 2}), encoding="utf-8")
    return parquet, manifest


def test_check_panel_accepts_matching_file(tmp_path):
    parquet, manifest = _write_panel(tmp_path)
    assert check_panel(parquet, manifest)["rows"] == 2


def test_check_panel_rejects_other_version(tmp_path):
    parquet, manifest = _write_panel(tmp_path, sha="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        check_panel(parquet, manifest)


def test_check_panel_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_panel(tmp_path / "none.parquet", tmp_path / "none.json")


def test_test_opens_once_per_model(tmp_path):
    lock = tmp_path / "test_opened.json"
    first = open_test_once("Tree(max_depth=8)", lock)
    assert open_test_once("Tree(max_depth=8)", lock) == first  # same model: rerun allowed
    with pytest.raises(RuntimeError, match="already opened"):
        open_test_once("Tree(max_depth=9)", lock)
