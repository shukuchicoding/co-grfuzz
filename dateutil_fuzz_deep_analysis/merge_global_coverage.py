#!/usr/bin/env python3
"""Merge cooperative worker edge-event logs into union/global coverage.

This is evaluation-only: it never writes state back to a running fuzzer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 1


class CoverageLogError(ValueError):
    pass


def _canonical_edge(edge):
    if not isinstance(edge, list) or len(edge) != 4:
        raise CoverageLogError("edge must be a four-element JSON array")
    try:
        normalized = [str(edge[0]), int(edge[1]), str(edge[2]), int(edge[3])]
    except (TypeError, ValueError, IndexError) as exc:
        raise CoverageLogError("malformed edge") from exc
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _int_field(record, name, minimum=None):
    try:
        value = int(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoverageLogError(f"missing/invalid {name}") from exc
    if minimum is not None and value < minimum:
        raise CoverageLogError(f"{name} must be >= {minimum}")
    return value


def _validate_record(record):
    if not isinstance(record, dict):
        raise CoverageLogError("record must be a JSON object")
    if int(record.get("schema", -1)) != SCHEMA_VERSION:
        raise CoverageLogError("unsupported schema")
    event = record.get("event")
    if event not in {"start", "discover"}:
        raise CoverageLogError("unsupported event type")
    worker_id = str(record.get("worker_id", "")).strip()
    method = str(record.get("method", "")).strip()
    if not worker_id or not method:
        raise CoverageLogError("worker_id and method are required")
    monotonic_ns = _int_field(record, "monotonic_ns", 0)
    wall_time_ns = _int_field(record, "wall_time_ns", 0)

    validated = dict(record)
    validated["worker_id"] = worker_id
    validated["method"] = method
    validated["monotonic_ns"] = monotonic_ns
    validated["wall_time_ns"] = wall_time_ns

    if event == "discover":
        validated["execution"] = _int_field(record, "execution", 0)
        validated["local_coverage"] = _int_field(record, "local_coverage", 0)
        edges = record.get("edges")
        if not isinstance(edges, list) or not edges:
            raise CoverageLogError("discover event must contain non-empty edges")
        validated["edge_keys"] = tuple(sorted({_canonical_edge(edge) for edge in edges}))
    else:
        validated["edge_keys"] = ()
    return validated


def read_logs(paths, strict=False):
    records = []
    invalid_lines = 0
    for file_index, raw_path in enumerate(paths):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(os.fspath(path))
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = _validate_record(json.loads(line))
                except (json.JSONDecodeError, CoverageLogError, TypeError, ValueError) as exc:
                    if strict:
                        raise CoverageLogError(f"{path}:{line_number}: {exc}") from exc
                    invalid_lines += 1
                    continue
                record["_file_index"] = file_index
                record["_line_number"] = line_number
                record["_source"] = os.fspath(path)
                records.append(record)
    return records, invalid_lines


def merge_records(records):
    if not records:
        return {
            "timeline": [],
            "global_coverage": 0,
            "max_local_coverage": 0,
            "worker_local_coverage": {},
            "worker_methods": {},
            "start_monotonic_ns": None,
            "start_wall_time_ns": None,
            "end_wall_time_ns": None,
            "discover_events": 0,
        }

    starts = [r for r in records if r["event"] == "start"]
    discoveries = [r for r in records if r["event"] == "discover"]

    # monotonic_ns is system-wide on the supported same-host cooperative setup.
    # Stable fallbacks make ties deterministic for repeatable post-processing.
    discoveries.sort(
        key=lambda r: (
            r["monotonic_ns"],
            r["wall_time_ns"],
            r["_file_index"],
            r["_line_number"],
        )
    )

    candidate_start_ns = [r["monotonic_ns"] for r in starts]
    candidate_start_wall = [r["wall_time_ns"] for r in starts]
    if not candidate_start_ns and discoveries:
        candidate_start_ns = [discoveries[0]["monotonic_ns"]]
        candidate_start_wall = [discoveries[0]["wall_time_ns"]]
    start_ns = min(candidate_start_ns) if candidate_start_ns else None
    start_wall = min(candidate_start_wall) if candidate_start_wall else None

    seen = set()
    worker_local = defaultdict(int)
    worker_methods = {}
    last_worker_local = defaultdict(int)
    workers_with_discovery = set()
    timeline = []

    if start_ns is not None:
        timeline.append(
            {
                "elapsed_seconds": 0.0,
                "global_coverage": 0,
                "new_global_edges": 0,
                "worker_id": "",
                "method": "",
                "execution": 0,
                "local_coverage": 0,
                "wall_time_ns": start_wall or 0,
            }
        )

    for record in discoveries:
        worker = record["worker_id"]
        local = record["local_coverage"]
        if worker in workers_with_discovery and local <= last_worker_local[worker]:
            raise CoverageLogError(
                f"non-increasing local coverage for discovery event from worker {worker!r}: "
                f"{local} <= {last_worker_local[worker]}"
            )
        if worker not in workers_with_discovery and local < 1:
            raise CoverageLogError(
                f"first discovery event for worker {worker!r} must have local coverage >= 1"
            )
        last_worker_local[worker] = local
        workers_with_discovery.add(worker)
        worker_local[worker] = max(worker_local[worker], local)
        worker_methods[worker] = record["method"]

        before = len(seen)
        seen.update(record["edge_keys"])
        new_global = len(seen) - before
        elapsed = 0.0 if start_ns is None else max(0.0, (record["monotonic_ns"] - start_ns) / 1e9)
        timeline.append(
            {
                "elapsed_seconds": elapsed,
                "global_coverage": len(seen),
                "new_global_edges": new_global,
                "worker_id": worker,
                "method": record["method"],
                "execution": record["execution"],
                "local_coverage": local,
                "wall_time_ns": record["wall_time_ns"],
            }
        )

    max_local = max(worker_local.values(), default=0)
    wall_values = [r["wall_time_ns"] for r in records]
    return {
        "timeline": timeline,
        "global_coverage": len(seen),
        "max_local_coverage": max_local,
        "global_gain_over_max": len(seen) - max_local,
        "worker_local_coverage": dict(sorted(worker_local.items())),
        "worker_methods": dict(sorted(worker_methods.items())),
        "start_monotonic_ns": start_ns,
        "start_wall_time_ns": start_wall,
        "end_wall_time_ns": max(wall_values) if wall_values else None,
        "discover_events": len(discoveries),
    }


def merge_files(paths, strict=False):
    records, invalid_lines = read_logs(paths, strict=strict)
    result = merge_records(records)
    result["invalid_lines"] = invalid_lines
    result["input_files"] = [os.fspath(Path(p)) for p in paths]
    return result


def write_timeline_csv(path, timeline):
    fields = [
        "elapsed_seconds",
        "global_coverage",
        "new_global_edges",
        "worker_id",
        "method",
        "execution",
        "local_coverage",
        "wall_time_ns",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timeline)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge evaluation-only cooperative coverage event logs into global union coverage."
    )
    parser.add_argument("logs", nargs="+", help="worker *.edges.jsonl files")
    parser.add_argument("--timeline-csv", help="write global coverage-over-time CSV")
    parser.add_argument("--summary-json", help="write machine-readable summary JSON")
    parser.add_argument("--strict", action="store_true", help="fail on malformed JSONL instead of skipping it")
    args = parser.parse_args(argv)

    try:
        result = merge_files(args.logs, strict=args.strict)
    except (OSError, CoverageLogError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.timeline_csv:
        write_timeline_csv(args.timeline_csv, result["timeline"])
    if args.summary_json:
        summary = {k: v for k, v in result.items() if k != "timeline"}
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    print(f"global_coverage={result['global_coverage']}")
    print(f"max_local_coverage={result['max_local_coverage']}")
    print(f"global_gain_over_max={result['global_gain_over_max']}")
    print(f"discover_events={result['discover_events']}")
    print(f"invalid_lines={result['invalid_lines']}")
    for worker, coverage in result["worker_local_coverage"].items():
        method = result["worker_methods"].get(worker, "unknown")
        print(f"worker={worker} method={method} local_coverage={coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
