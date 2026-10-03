"""Evaluation-only coverage event logging for cooperative fuzzing.

Workers keep their normal local coverage/reward state.  This module records only
*locally new* edges so multiple worker logs can be merged after a campaign into
an exact union-coverage timeline without feeding global state back into fuzzing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

SCHEMA_VERSION = 1


def _normalize_filename(value):
    """Return a stable filename representation shared by all local workers."""
    text = str(value)
    # Preserve Python pseudo filenames such as <stdin>/<string> verbatim.
    if len(text) >= 2 and text.startswith("<") and text.endswith(">"):
        return text
    try:
        # resolve(strict=False) normalizes relative paths and symlinks where
        # possible without requiring the source file to still exist.
        text = str(Path(text).resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError):
        text = os.path.abspath(text)
    return os.path.normcase(text)


def normalize_edge(edge):
    """Canonicalize one Python line-transition edge.

    Cooperative tracers use (src_file, src_line, dst_file, dst_line).  Rejecting
    malformed edges prevents silently merging unlike coverage representations.
    """
    if not isinstance(edge, (tuple, list)) or len(edge) != 4:
        raise ValueError("coverage edge must contain exactly 4 fields")
    src_file, src_line, dst_file, dst_line = edge
    try:
        src_line = int(src_line)
        dst_line = int(dst_line)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage edge line numbers must be integers") from exc
    if src_line < 1 or dst_line < 1:
        raise ValueError("coverage edge line numbers must be >= 1")
    return (_normalize_filename(src_file), src_line, _normalize_filename(dst_file), dst_line)


def edge_key(edge):
    """Return a collision-free-in-process canonical string key for an edge."""
    return json.dumps(normalize_edge(edge), ensure_ascii=False, separators=(",", ":"))


def _safe_worker_id(worker_id, method):
    value = str(worker_id or "").strip()
    if value:
        return value
    return f"{method}-{os.getpid()}"


class CoverageEventLogger:
    """Append-only JSONL logger for evaluation-only union coverage.

    The logger should be created *after* the target child process has started so
    the child does not inherit an open log handle.  One path per worker is
    recommended.  Events are line-buffered and fsync is intentionally avoided on
    every discovery to keep fuzzing overhead low; close() flushes the file.
    """

    def __init__(self, path, worker_id=None, method="unknown"):
        if not path:
            raise ValueError("coverage event path must be non-empty")
        self.path = os.fspath(path)
        self.method = str(method or "unknown")
        self.worker_id = _safe_worker_id(worker_id, self.method)
        self._closed = False
        self._last_local_coverage = 0

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
            raise RuntimeError("coverage event logger is closed")
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def record_discovery(self, edges, execution, local_coverage):
        """Record locally new edges and return the number written.

        Duplicate edges inside one call are removed.  Empty discovery sets do
        not produce an event.  ``local_coverage`` must be monotonic for a worker;
        this catches accidental reporting of frozen/group coverage as live state.
        """
        if self._closed:
            raise RuntimeError("coverage event logger is closed")
        execution = int(execution)
        local_coverage = int(local_coverage)
        if execution < 0:
            raise ValueError("execution must be >= 0")
        if local_coverage < 0:
            raise ValueError("local coverage must be >= 0")
        if local_coverage < self._last_local_coverage:
            raise ValueError("local coverage must be monotonic")

        canonical = {}
        for edge in edges or ():
            normalized = normalize_edge(edge)
            canonical[json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))] = normalized

        if not canonical:
            if local_coverage != self._last_local_coverage:
                raise ValueError("local coverage changed without discovery edges")
            return 0
        if local_coverage <= self._last_local_coverage:
            raise ValueError("discovery edges require increased local coverage")

        self._last_local_coverage = local_coverage
        ordered = [canonical[key] for key in sorted(canonical)]
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "event": "discover",
                "worker_id": self.worker_id,
                "method": self.method,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
                "execution": execution,
                "local_coverage": local_coverage,
                "edges": ordered,
            }
        )
        return len(ordered)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
