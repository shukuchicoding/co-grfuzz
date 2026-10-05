"""Evaluation-only seed provenance helpers for cooperative fuzzing.

Phase 5 records which worker first published a shared-corpus seed and how often
workers later use each seed as the direct parent of a mutation.  No provenance
state is fed back into corpus admission, mutation selection, coverage, rewards,
or policy updates.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict

SCHEMA_VERSION = 1
USAGE_SCHEMA_VERSION = 1


def seed_id(data):
    """Return the canonical SHA-256 identifier used by the shared corpus."""
    return hashlib.sha256(bytes(data)).hexdigest()


def normalize_seed_id(value):
    text = str(value or "").strip().lower()
    if len(text) != 64:
        raise ValueError("seed id must be a 64-character SHA-256 hex digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError("seed id must be hexadecimal") from exc
    return text


def safe_worker_id(worker_id, method):
    value = str(worker_id or "").strip()
    if value:
        return value
    return f"{method}-{os.getpid()}"


class SeedProvenanceLogger:
    """Append-only JSONL log of shared-corpus seed publication events."""

    def __init__(self, path, worker_id=None, method="unknown"):
        if not path:
            raise ValueError("seed event path must be non-empty")
        self.path = os.fspath(path)
        self.method = str(method or "unknown")
        self.worker_id = safe_worker_id(worker_id, self.method)
        self._closed = False

        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "event": "start",
                "worker_id": self.worker_id,
                "method": self.method,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
                "pid": os.getpid(),
            }
        )

    def _write(self, record):
        if self._closed:
            raise RuntimeError("seed provenance logger is closed")
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def record_publish(
        self,
        seed,
        parent_seed,
        execution,
        local_new_edge_count,
        seed_size,
    ):
        """Record that this worker admitted a generated seed to its shared corpus.

        A concurrent peer can generate the same bytes.  Offline analysis resolves
        ownership by the earliest publication timestamp, so duplicate publication
        records are deliberately retained rather than affecting runtime behavior.
        """
        if self._closed:
            raise RuntimeError("seed provenance logger is closed")
        execution = int(execution)
        local_new_edge_count = int(local_new_edge_count)
        seed_size = int(seed_size)
        if execution < 0:
            raise ValueError("execution must be >= 0")
        if local_new_edge_count < 1:
            raise ValueError("published seed must have at least one locally new edge")
        if seed_size < 1:
            raise ValueError("seed size must be >= 1")
        record = {
            "schema": SCHEMA_VERSION,
            "event": "publish",
            "worker_id": self.worker_id,
            "method": self.method,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            "execution": execution,
            "seed_id": normalize_seed_id(seed),
            "parent_seed_id": normalize_seed_id(parent_seed),
            "local_new_edge_count": local_new_edge_count,
            "seed_size": seed_size,
        }
        self._write(record)
        return record

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._file.flush()
        self._file.close()


class SeedUsageTracker:
    """Low-overhead per-parent counters; no per-execution file writes."""

    FIELDS = (
        "candidate_executions",
        "completed_executions",
        "local_discovery_events",
        "local_new_edges",
        "accepted_children",
    )

    def __init__(self):
        self._usage = defaultdict(lambda: {field: 0 for field in self.FIELDS})

    def _entry(self, parent_seed_id):
        return self._usage[normalize_seed_id(parent_seed_id)]

    def record_attempt(self, parent_seed_id):
        self._entry(parent_seed_id)["candidate_executions"] += 1

    def record_completed(self, parent_seed_id):
        self._entry(parent_seed_id)["completed_executions"] += 1

    def record_local_discovery(self, parent_seed_id, new_edge_count):
        count = int(new_edge_count)
        if count < 1:
            raise ValueError("new_edge_count must be >= 1")
        entry = self._entry(parent_seed_id)
        entry["local_discovery_events"] += 1
        entry["local_new_edges"] += count

    def record_accepted_child(self, parent_seed_id):
        self._entry(parent_seed_id)["accepted_children"] += 1

    def as_dict(self):
        return {seed: dict(self._usage[seed]) for seed in sorted(self._usage)}


def write_seed_usage(path, tracker, worker_id=None, method="unknown"):
    """Write one compact JSON usage snapshot for a completed worker run."""
    if not path:
        return None
    method = str(method or "unknown")
    worker = safe_worker_id(worker_id, method)
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(parent, exist_ok=True)
    payload = {
        "schema": USAGE_SCHEMA_VERSION,
        "event": "seed_usage_summary",
        "worker_id": worker,
        "method": method,
        "wall_time_ns": time.time_ns(),
        "seeds": tracker.as_dict(),
    }
    tmp = f"{os.fspath(path)}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, os.fspath(path))
    return payload
