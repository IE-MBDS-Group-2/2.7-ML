"""Open the test partition once, for one model only.

The first time 99_final runs with OPEN_TEST = True, open_test_once writes a lock file with a fingerprint of the
chosen model. Re-running the same model is fine (the submission notebook must run top to bottom). Running a
different model against test raises, because choosing among models on test would make the test score optimistic.

The lock lives in notebooks/, which Git tracks: commit and push it as soon as it is written, so the whole group
shares one lock.

    spec = model_spec("XGBoost", model, ["train", "valid"])   # full settings, features and data version
    open_test_once(spec)                                       # call before loading the test partition
    test = load_panel(partitions=["test"], include_test=True)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ml2.config import load_config, resolve
from ml2.schema import feature_columns

LOCK = "notebooks/test_opened.json"


def _plain(value):
    """Settings as plain text, recursing into lists, tuples and dicts. Estimators are named by class only: their own
    settings appear separately in get_params(deep=True), so nothing is cut short the way a printed pipeline is."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "get_params"):
        return type(value).__name__
    return repr(value)


def model_spec(name: str, model, fit_partitions: list[str]) -> str:
    """Fingerprint text of a model: every setting, the feature list, the fit partitions and the data version."""
    manifest = json.loads(resolve(load_config()["paths"]["manifest"]).read_text(encoding="utf-8"))
    spec = {
        "name": name,
        "estimator": type(model).__name__,
        "params": _plain(model.get_params(deep=True)),
        "features": feature_columns(),
        "fit_partitions": list(fit_partitions),
        "panel_sha256": manifest["sha256"],
    }
    return json.dumps(spec, sort_keys=True)


def spec_hash(spec: str) -> str:
    """Short fingerprint of a model specification."""
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def open_test_once(spec: str, lock: str | Path = LOCK) -> dict:
    """Record the first opening of test, or check that this is the same model as the first time.

    Returns the lock record. Raises RuntimeError if test was already opened for a different model.
    """
    path = resolve(lock)
    h = spec_hash(spec)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["spec_hash"] != h:
            raise RuntimeError(
                f"Test was already opened on {record['opened_at_utc']} for another model:\n{record['spec']}\n"
                "Do not choose models on test. If the group decides to reopen it, log why in RUN_LOG.md and "
                f"delete {path} by hand."
            )
        return record
    record = {"opened_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spec_hash": h, "spec": spec}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Test opened and locked to this model. Commit and push {path.name} now so the whole group shares the lock.")
    return record
