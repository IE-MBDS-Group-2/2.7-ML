"""Open the test partition once, for one model only.

The first time 99_final runs with OPEN_TEST = True, open_test_once writes a lock file with a fingerprint of the
chosen model. Re-running the same model is fine (the submission notebook must run top to bottom). Running a
different model against test raises, because choosing among models on test would make the test score optimistic.

    spec = str(make_final_model())          # the estimator's repr: changed parameters only
    open_test_once(spec)                    # call before loading the test partition
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ml2.config import resolve

LOCK = "reports/test_opened.json"


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
    return record
