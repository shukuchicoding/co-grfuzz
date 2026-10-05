#!/usr/bin/env python3
"""Offline Phase 5 analysis of two-worker shared-corpus seed utility.

The analysis is evaluation-only.  It combines:
  * Phase 4 coverage discovery logs (*.edges.jsonl),
  * Phase 5 seed publication logs (*.seeds.jsonl), and
  * Phase 5 compact seed usage summaries (*.seed_usage.json).

Credit is deliberately one-hop: an edge discovered from a mutation of seed S is
attributed to S as the direct parent.  Ancestors of S do not receive downstream
credit, which avoids ambiguous causal claims across long mutation lineages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SEED_SCHEMA = 1
USAGE_SCHEMA = 1
COVERAGE_SCHEMA = 1


class AnalysisError(ValueError):
    pass


def _seed_id(value):
    text = str(value or "").strip().lower()
    if len(text) != 64:
        raise AnalysisError("seed id must be a 64-character SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise AnalysisError("seed id must be hexadecimal") from exc
    return text


def _edge_key(edge):
    if not isinstance(edge, list) or len(edge) != 4:
        raise AnalysisError("coverage edge must contain four fields")
    try:
        normalized = [str(edge[0]), int(edge[1]), str(edge[2]), int(edge[3])]
    except (TypeError, ValueError, IndexError) as exc:
        raise AnalysisError("malformed coverage edge") from exc
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _int(record, name, minimum=0):
    try:
        value = int(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"missing/invalid {name}") from exc
    if value < minimum:
        raise AnalysisError(f"{name} must be >= {minimum}")
    return value


def _read_jsonl(paths, kind, strict=False):
    records = []
    invalid = 0
    for file_index, raw in enumerate(paths):
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(os.fspath(path))
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise AnalysisError("record must be a JSON object")
                    record["_file_index"] = file_index
                    record["_line_number"] = line_number
                    record["_source"] = os.fspath(path)
                    if kind == "coverage":
                        record = _validate_coverage(record)
                    elif kind == "seed":
                        record = _validate_seed_event(record)
                    else:
                        raise AssertionError(kind)
                except (json.JSONDecodeError, AnalysisError, TypeError, ValueError) as exc:
                    if strict:
                        raise AnalysisError(f"{path}:{line_number}: {exc}") from exc
                    invalid += 1
                    continue
                records.append(record)
    return records, invalid


def _base_fields(record, schema):
    if int(record.get("schema", -1)) != schema:
        raise AnalysisError("unsupported schema")
    worker = str(record.get("worker_id", "")).strip()
    method = str(record.get("method", "")).strip()
    if not worker or not method:
        raise AnalysisError("worker_id and method are required")
    record["worker_id"] = worker
    record["method"] = method
    record["monotonic_ns"] = _int(record, "monotonic_ns")
    record["wall_time_ns"] = _int(record, "wall_time_ns")
    return record


def _validate_seed_event(record):
    record = _base_fields(record, SEED_SCHEMA)
    event = record.get("event")
    if event not in {"start", "publish"}:
        raise AnalysisError("unsupported seed event")
    if event == "publish":
        record["execution"] = _int(record, "execution")
        record["seed_id"] = _seed_id(record.get("seed_id"))
        record["parent_seed_id"] = _seed_id(record.get("parent_seed_id"))
        record["local_new_edge_count"] = _int(record, "local_new_edge_count", 1)
        record["seed_size"] = _int(record, "seed_size", 1)
    return record


def _validate_coverage(record):
    record = _base_fields(record, COVERAGE_SCHEMA)
    event = record.get("event")
    if event not in {"start", "discover"}:
        raise AnalysisError("unsupported coverage event")
    if event == "discover":
        record["execution"] = _int(record, "execution")
        record["local_coverage"] = _int(record, "local_coverage")
        edges = record.get("edges")
        if not isinstance(edges, list) or not edges:
            raise AnalysisError("discover event must contain edges")
        record["edge_keys"] = tuple(sorted({_edge_key(e) for e in edges}))
        parent = record.get("parent_seed_id")
        candidate = record.get("candidate_seed_id")
        record["parent_seed_id"] = _seed_id(parent) if parent is not None else None
        record["candidate_seed_id"] = _seed_id(candidate) if candidate is not None else None
    return record


def _read_usage(paths):
    summaries = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(os.fspath(path))
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or int(payload.get("schema", -1)) != USAGE_SCHEMA:
            raise AnalysisError(f"{path}: unsupported usage schema")
        worker = str(payload.get("worker_id", "")).strip()
        method = str(payload.get("method", "")).strip()
        if not worker or not method:
            raise AnalysisError(f"{path}: worker_id and method are required")
        seeds = payload.get("seeds", {})
        if not isinstance(seeds, dict):
            raise AnalysisError(f"{path}: seeds must be an object")
        normalized = {}
        for seed, counters in seeds.items():
            sid = _seed_id(seed)
            if not isinstance(counters, dict):
                raise AnalysisError(f"{path}: usage counters must be objects")
            normalized[sid] = {
                name: int(counters.get(name, 0))
                for name in (
                    "candidate_executions",
                    "completed_executions",
                    "local_discovery_events",
                    "local_new_edges",
                    "accepted_children",
                )
            }
            if any(value < 0 for value in normalized[sid].values()):
                raise AnalysisError(f"{path}: negative usage counter")
        summaries.append({"worker_id": worker, "method": method, "seeds": normalized})
    return summaries


def _event_sort_key(record):
    return (
        record["monotonic_ns"],
        record["wall_time_ns"],
        record.get("_file_index", 0),
        record.get("_line_number", 0),
    )


def _blank_seed(seed_id):
    return {
        "seed_id": seed_id,
        "producer_worker": "initial",
        "producer_method": "initial",
        "parent_seed_id": "",
        "publish_monotonic_ns": None,
        "publish_execution": None,
        "birth_local_new_edges": 0,
        "birth_global_edges": 0,
        "seed_size": 0,
        "peer_candidate_executions": 0,
        "peer_completed_executions": 0,
        "peer_local_discovery_events": 0,
        "peer_local_new_edges": 0,
        "peer_accepted_children": 0,
        "same_worker_candidate_executions": 0,
        "same_worker_completed_executions": 0,
        "transfer_global_edges": 0,
        "same_worker_global_edges": 0,
        "eligible_by_birth_cutoff": True,
    }


def analyze(coverage_records, seed_records, usage_summaries, birth_cutoff_fraction=1.0):
    if not (0 < birth_cutoff_fraction <= 1.0):
        raise AnalysisError("birth cutoff fraction must be in (0, 1]")

    all_timestamps = [
        r["monotonic_ns"]
        for r in coverage_records + seed_records
        if "monotonic_ns" in r
    ]
    campaign_start = min(all_timestamps) if all_timestamps else 0
    campaign_end = max(all_timestamps) if all_timestamps else campaign_start
    cutoff_ns = campaign_start + int((campaign_end - campaign_start) * birth_cutoff_fraction)

    publishes = sorted(
        [r for r in seed_records if r.get("event") == "publish"],
        key=_event_sort_key,
    )
    owners = {}
    duplicate_publications = 0
    seed_stats = {}
    publish_lookup = set()
    for record in publishes:
        sid = record["seed_id"]
        if sid in owners:
            duplicate_publications += 1
            continue
        owners[sid] = {
            "worker_id": record["worker_id"],
            "method": record["method"],
            "monotonic_ns": record["monotonic_ns"],
            "execution": record["execution"],
        }
        stat = _blank_seed(sid)
        stat.update(
            {
                "producer_worker": record["worker_id"],
                "producer_method": record["method"],
                "parent_seed_id": record["parent_seed_id"],
                "publish_monotonic_ns": record["monotonic_ns"],
                "publish_execution": record["execution"],
                "birth_local_new_edges": record["local_new_edge_count"],
                "seed_size": record["seed_size"],
                "eligible_by_birth_cutoff": record["monotonic_ns"] <= cutoff_ns,
            }
        )
        seed_stats[sid] = stat
        publish_lookup.add((record["worker_id"], record["execution"], sid))

    worker_methods = {}
    for record in coverage_records + seed_records:
        worker_methods[record["worker_id"]] = record["method"]
    for summary in usage_summaries:
        worker_methods[summary["worker_id"]] = summary["method"]

    # Aggregate how each worker used direct parent seeds.
    for summary in usage_summaries:
        executor = summary["worker_id"]
        for sid, counters in summary["seeds"].items():
            if sid not in seed_stats:
                seed_stats[sid] = _blank_seed(sid)
            stat = seed_stats[sid]
            owner = owners.get(sid)
            if owner is not None and owner["worker_id"] != executor:
                stat["peer_candidate_executions"] += counters["candidate_executions"]
                stat["peer_completed_executions"] += counters["completed_executions"]
                stat["peer_local_discovery_events"] += counters["local_discovery_events"]
                stat["peer_local_new_edges"] += counters["local_new_edges"]
                stat["peer_accepted_children"] += counters["accepted_children"]
            elif owner is not None and owner["worker_id"] == executor:
                stat["same_worker_candidate_executions"] += counters["candidate_executions"]
                stat["same_worker_completed_executions"] += counters["completed_executions"]
            # Initial seeds are kept for completeness but excluded from generated
            # seed quality comparisons.

    discoveries = sorted(
        [r for r in coverage_records if r.get("event") == "discover"],
        key=_event_sort_key,
    )
    seen = set()
    exec_worker_global = defaultdict(int)
    exec_method_global = defaultdict(int)
    source_worker_global = defaultdict(int)
    source_method_global = defaultdict(int)
    global_timeline = []

    for record in discoveries:
        new_global = [edge for edge in record["edge_keys"] if edge not in seen]
        if new_global:
            seen.update(new_global)
        count = len(new_global)
        executor = record["worker_id"]
        method = record["method"]
        exec_worker_global[executor] += count
        exec_method_global[method] += count

        parent_sid = record.get("parent_seed_id")
        candidate_sid = record.get("candidate_seed_id")
        if parent_sid is None:
            source_worker = "unknown"
            source_method = "unknown"
        else:
            if parent_sid not in seed_stats:
                seed_stats[parent_sid] = _blank_seed(parent_sid)
            owner = owners.get(parent_sid)
            if owner is None:
                source_worker = "initial"
                source_method = "initial"
            else:
                source_worker = owner["worker_id"]
                source_method = owner["method"]
                if source_worker != executor:
                    seed_stats[parent_sid]["transfer_global_edges"] += count
                else:
                    seed_stats[parent_sid]["same_worker_global_edges"] += count
        source_worker_global[source_worker] += count
        source_method_global[source_method] += count

        if candidate_sid is not None and (
            executor,
            record["execution"],
            candidate_sid,
        ) in publish_lookup:
            if candidate_sid not in seed_stats:
                seed_stats[candidate_sid] = _blank_seed(candidate_sid)
            seed_stats[candidate_sid]["birth_global_edges"] += count

        global_timeline.append(
            {
                "monotonic_ns": record["monotonic_ns"],
                "worker_id": executor,
                "method": method,
                "execution": record["execution"],
                "new_global_edges": count,
                "global_coverage": len(seen),
                "parent_seed_id": parent_sid or "",
                "seed_source_worker": source_worker,
                "seed_source_method": source_method,
            }
        )

    max_local = defaultdict(int)
    for record in discoveries:
        max_local[record["worker_id"]] = max(
            max_local[record["worker_id"]], record["local_coverage"]
        )

    producer_groups = defaultdict(list)
    for stat in seed_stats.values():
        if stat["producer_worker"] not in {"initial", "unknown"} and stat["eligible_by_birth_cutoff"]:
            producer_groups[(stat["producer_worker"], stat["producer_method"])].append(stat)

    producer_summary = []
    for (worker, method), seeds in sorted(producer_groups.items()):
        peer_reused = [s for s in seeds if s["peer_candidate_executions"] > 0]
        productive = [s for s in peer_reused if s["transfer_global_edges"] > 0]
        peer_exec = sum(s["peer_candidate_executions"] for s in seeds)
        peer_completed = sum(s["peer_completed_executions"] for s in seeds)
        transfer_edges = sum(s["transfer_global_edges"] for s in seeds)
        producer_summary.append(
            {
                "producer_worker": worker,
                "producer_method": method,
                "published_seeds": len(seeds),
                "peer_reused_seeds": len(peer_reused),
                "productive_seeds": len(productive),
                "productive_seed_ratio": (
                    len(productive) / len(peer_reused) if peer_reused else 0.0
                ),
                "birth_local_new_edges": sum(s["birth_local_new_edges"] for s in seeds),
                "birth_global_edges": sum(s["birth_global_edges"] for s in seeds),
                "peer_candidate_executions": peer_exec,
                "peer_completed_executions": peer_completed,
                "peer_local_new_edges": sum(s["peer_local_new_edges"] for s in seeds),
                "peer_accepted_children": sum(s["peer_accepted_children"] for s in seeds),
                "transfer_global_edges": transfer_edges,
                "global_edges_per_100_peer_executions": (
                    100.0 * transfer_edges / peer_exec if peer_exec else 0.0
                ),
                "median_transfer_global_edges_per_reused_seed": (
                    statistics.median(s["transfer_global_edges"] for s in peer_reused)
                    if peer_reused
                    else 0.0
                ),
                "median_birth_local_new_edges": (
                    statistics.median(s["birth_local_new_edges"] for s in seeds)
                    if seeds
                    else 0.0
                ),
            }
        )

    global_coverage = len(seen)
    max_local_coverage = max(max_local.values(), default=0)
    summary = {
        "analysis": "one-hop direct-parent seed provenance",
        "birth_cutoff_fraction": birth_cutoff_fraction,
        "campaign_start_monotonic_ns": campaign_start,
        "campaign_end_monotonic_ns": campaign_end,
        "birth_cutoff_monotonic_ns": cutoff_ns,
        "global_coverage": global_coverage,
        "max_local_coverage": max_local_coverage,
        "global_gain_over_max": global_coverage - max_local_coverage,
        "worker_local_coverage": dict(sorted(max_local.items())),
        "worker_methods": dict(sorted(worker_methods.items())),
        "execution_worker_global_edges": dict(sorted(exec_worker_global.items())),
        "execution_method_global_edges": dict(sorted(exec_method_global.items())),
        "seed_source_worker_global_edges": dict(sorted(source_worker_global.items())),
        "seed_source_method_global_edges": dict(sorted(source_method_global.items())),
        "unique_generated_seeds": len(owners),
        "duplicate_publication_records": duplicate_publications,
        "producer_summary": producer_summary,
    }
    return summary, seed_stats, global_timeline


def _write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze one-hop seed provenance and global coverage for a two-worker campaign."
    )
    parser.add_argument("--coverage", nargs="+", required=True, help="worker *.edges.jsonl logs")
    parser.add_argument("--seeds", nargs="+", required=True, help="worker *.seeds.jsonl logs")
    parser.add_argument("--usage", nargs="+", required=True, help="worker *.seed_usage.json files")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--seed-csv", required=True)
    parser.add_argument("--producer-csv", required=True)
    parser.add_argument("--timeline-csv")
    parser.add_argument(
        "--birth-cutoff-fraction",
        type=float,
        default=1.0,
        help="include generated seeds born in the first fraction of campaign time (default: 1.0)",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    try:
        coverage, invalid_cov = _read_jsonl(args.coverage, "coverage", strict=args.strict)
        seeds, invalid_seed = _read_jsonl(args.seeds, "seed", strict=args.strict)
        usage = _read_usage(args.usage)
        summary, seed_stats, timeline = analyze(
            coverage,
            seeds,
            usage,
            birth_cutoff_fraction=args.birth_cutoff_fraction,
        )
    except (OSError, json.JSONDecodeError, AnalysisError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary["invalid_coverage_lines"] = invalid_cov
    summary["invalid_seed_lines"] = invalid_seed
    summary["coverage_logs"] = list(args.coverage)
    summary["seed_logs"] = list(args.seeds)
    summary["usage_files"] = list(args.usage)

    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    seed_fields = [
        "seed_id",
        "producer_worker",
        "producer_method",
        "parent_seed_id",
        "publish_execution",
        "birth_local_new_edges",
        "birth_global_edges",
        "seed_size",
        "peer_candidate_executions",
        "peer_completed_executions",
        "peer_local_discovery_events",
        "peer_local_new_edges",
        "peer_accepted_children",
        "transfer_global_edges",
        "same_worker_candidate_executions",
        "same_worker_completed_executions",
        "same_worker_global_edges",
        "eligible_by_birth_cutoff",
    ]
    generated_seed_rows = [
        stat for stat in seed_stats.values() if stat["producer_worker"] != "initial"
    ]
    generated_seed_rows.sort(key=lambda r: (r["producer_worker"], r["publish_execution"] or 0, r["seed_id"]))
    _write_csv(args.seed_csv, generated_seed_rows, seed_fields)

    producer_fields = [
        "producer_worker",
        "producer_method",
        "published_seeds",
        "peer_reused_seeds",
        "productive_seeds",
        "productive_seed_ratio",
        "birth_local_new_edges",
        "birth_global_edges",
        "peer_candidate_executions",
        "peer_completed_executions",
        "peer_local_new_edges",
        "peer_accepted_children",
        "transfer_global_edges",
        "global_edges_per_100_peer_executions",
        "median_transfer_global_edges_per_reused_seed",
        "median_birth_local_new_edges",
    ]
    _write_csv(args.producer_csv, summary["producer_summary"], producer_fields)

    if args.timeline_csv:
        timeline_fields = [
            "monotonic_ns",
            "worker_id",
            "method",
            "execution",
            "new_global_edges",
            "global_coverage",
            "parent_seed_id",
            "seed_source_worker",
            "seed_source_method",
        ]
        _write_csv(args.timeline_csv, timeline, timeline_fields)

    print(f"global_coverage={summary['global_coverage']}")
    print(f"max_local_coverage={summary['max_local_coverage']}")
    print(f"global_gain_over_max={summary['global_gain_over_max']}")
    print(f"unique_generated_seeds={summary['unique_generated_seeds']}")
    for row in summary["producer_summary"]:
        print(
            "producer={producer_worker} method={producer_method} published={published_seeds} "
            "peer_reused={peer_reused_seeds} productive={productive_seeds} "
            "transfer_global_edges={transfer_global_edges} "
            "edges_per_100_peer_exec={global_edges_per_100_peer_executions:.6f}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
