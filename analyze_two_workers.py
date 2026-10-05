#!/usr/bin/env python3
"""Analyze coverage overlap and runtime performance for two fuzzing workers.

Coverage is reconstructed from two ``coverage_log.txt.edges.jsonl`` files.
Runtime performance is read from the corresponding ``coverage_log.txt`` files,
whose expected row format is:

    timestamp,total_executions,coverage,execs_per_second,rss_mib

The primary throughput value reported here is an interval-weighted rate computed
from the cumulative execution counter and timestamps. The logged per-sample
exec/s values are also summarized for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

EDGE_LOG_NAME = "coverage_log.txt.edges.jsonl"
COVERAGE_LOG_NAME = "coverage_log.txt"


class AnalysisError(ValueError):
    pass


def _canonical_edge(edge: Any) -> tuple[str, int, str, int]:
    if not isinstance(edge, (list, tuple)) or len(edge) != 4:
        raise AnalysisError("edge must contain four fields")
    try:
        return str(edge[0]), int(edge[1]), str(edge[2]), int(edge[3])
    except (TypeError, ValueError, IndexError) as exc:
        raise AnalysisError("malformed edge") from exc


def load_edge_log(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    worker_id: str | None = None
    method: str | None = None
    edges: set[tuple[str, int, str, int]] = set()
    invalid_lines = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise AnalysisError("record is not an object")
                rid = str(record.get("worker_id", "")).strip()
                rmethod = str(record.get("method", "")).strip()
                if rid:
                    if worker_id is not None and rid != worker_id:
                        raise AnalysisError(
                            f"worker_id changes inside {path.name}: {worker_id!r} -> {rid!r}"
                        )
                    worker_id = rid
                if rmethod:
                    if method is not None and rmethod != method:
                        raise AnalysisError(
                            f"method changes inside {path.name}: {method!r} -> {rmethod!r}"
                        )
                    method = rmethod

                if record.get("event") == "discover":
                    raw_edges = record.get("edges", [])
                    if not isinstance(raw_edges, list):
                        raise AnalysisError("discover.edges must be a list")
                    for edge in raw_edges:
                        edges.add(_canonical_edge(edge))
            except (json.JSONDecodeError, AnalysisError, TypeError, ValueError):
                invalid_lines += 1

    if worker_id is None:
        worker_id = path.parent.name
    if method is None:
        method = path.parent.name
        if method.startswith("colabmode_"):
            method = method[len("colabmode_") :]
        method = method.rsplit("_", 1)[0] if method.rsplit("_", 1)[-1].isdigit() else method

    return {
        "path": path,
        "worker_id": worker_id,
        "method": method,
        "edges": edges,
        "invalid_lines": invalid_lines,
    }


def _parse_timestamp(value: str) -> datetime:
    value = value.strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise AnalysisError(f"invalid timestamp: {value!r}") from exc


def parse_coverage_log(path: str | Path) -> dict[str, Any]:
    """Parse ``coverage_log.txt`` and summarize throughput/RSS.

    ``overall_execs_per_second`` is computed from all consecutive samples with
    positive elapsed time and a non-decreasing cumulative execution counter:

        sum(delta executions) / sum(delta seconds)

    This is preferable to an unweighted mean when log intervals are irregular.
    If there is only one usable row (or no valid time delta), the mean of the
    logged exec/s samples is used as a fallback.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    invalid_lines = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                invalid_lines += 1
                continue
            try:
                timestamp = _parse_timestamp(parts[0])
                executions = int(parts[1])
                coverage = int(parts[2])
                sample_execs_per_second = float(parts[3])
                rss_mib = float(parts[4])
                if executions < 0 or coverage < 0:
                    raise ValueError("negative counter")
                if not math.isfinite(sample_execs_per_second) or sample_execs_per_second < 0:
                    raise ValueError("invalid exec/s")
                if not math.isfinite(rss_mib) or rss_mib < 0:
                    raise ValueError("invalid rss")
            except (ValueError, AnalysisError):
                invalid_lines += 1
                continue

            records.append(
                {
                    "timestamp": timestamp,
                    "executions": executions,
                    "coverage": coverage,
                    "sample_execs_per_second": sample_execs_per_second,
                    "rss_mib": rss_mib,
                }
            )

    records.sort(key=lambda r: r["timestamp"])

    if not records:
        return {
            "path": path,
            "samples": 0,
            "invalid_lines": invalid_lines,
            "overall_execs_per_second": None,
            "sample_mean_execs_per_second": None,
            "sample_median_execs_per_second": None,
            "rss_mean_mib": None,
            "rss_peak_mib": None,
            "final_executions": None,
            "final_logged_coverage": None,
            "elapsed_seconds": None,
        }

    total_delta_exec = 0
    total_delta_seconds = 0.0
    for prev, cur in zip(records, records[1:]):
        dt = (cur["timestamp"] - prev["timestamp"]).total_seconds()
        de = cur["executions"] - prev["executions"]
        if dt > 0 and de >= 0:
            total_delta_exec += de
            total_delta_seconds += dt

    sample_rates = [r["sample_execs_per_second"] for r in records]
    rss_values = [r["rss_mib"] for r in records]
    sample_mean = statistics.fmean(sample_rates)
    overall = (
        total_delta_exec / total_delta_seconds
        if total_delta_seconds > 0
        else sample_mean
    )

    return {
        "path": path,
        "samples": len(records),
        "invalid_lines": invalid_lines,
        "overall_execs_per_second": float(overall),
        "sample_mean_execs_per_second": float(sample_mean),
        "sample_median_execs_per_second": float(statistics.median(sample_rates)),
        "rss_mean_mib": float(statistics.fmean(rss_values)),
        "rss_peak_mib": float(max(rss_values)),
        "final_executions": int(records[-1]["executions"]),
        "final_logged_coverage": int(records[-1]["coverage"]),
        "elapsed_seconds": float(total_delta_seconds) if total_delta_seconds > 0 else 0.0,
    }


def _fmt(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def analyze_two_workers(
    edge_log_1: str | Path,
    edge_log_2: str | Path,
    coverage_log_1: str | Path | None = None,
    coverage_log_2: str | Path | None = None,
) -> dict[str, Any]:
    w1 = load_edge_log(edge_log_1)
    w2 = load_edge_log(edge_log_2)

    e1 = w1["edges"]
    e2 = w2["edges"]
    overlap = e1 & e2
    unique1 = e1 - e2
    unique2 = e2 - e1
    union = e1 | e2
    max_local = max(len(e1), len(e2))

    if coverage_log_1 is None:
        inferred = Path(edge_log_1).with_name(COVERAGE_LOG_NAME)
        coverage_log_1 = inferred if inferred.is_file() else None
    if coverage_log_2 is None:
        inferred = Path(edge_log_2).with_name(COVERAGE_LOG_NAME)
        coverage_log_2 = inferred if inferred.is_file() else None

    perf1 = parse_coverage_log(coverage_log_1) if coverage_log_1 is not None else None
    perf2 = parse_coverage_log(coverage_log_2) if coverage_log_2 is not None else None

    return {
        "workers": [w1, w2],
        "unique_edges": [len(unique1), len(unique2)],
        "overlap": len(overlap),
        "global_coverage": len(union),
        "max_local_coverage": max_local,
        "global_gain_over_max": len(union) - max_local,
        "jaccard": (len(overlap) / len(union)) if union else 1.0,
        "performance": [perf1, perf2],
    }


def render_report(result: dict[str, Any]) -> str:
    workers = result["workers"]
    uniques = result["unique_edges"]
    global_cov = result["global_coverage"]

    lines = [
        "=" * 72,
        "TWO-WORKER COVERAGE ANALYSIS",
        "=" * 72,
    ]

    for idx, (worker, unique) in enumerate(zip(workers, uniques), 1):
        lines.extend(
            [
                f"Worker {idx} : {worker['worker_id']} ({worker['method']})",
                f"  Local coverage : {len(worker['edges'])}",
                f"  Unique edges   : {unique}",
                "",
            ]
        )

    lines.extend(
        [
            f"Overlap                    : {result['overlap']}",
            f"Global / union coverage    : {global_cov}",
            f"Max local coverage         : {result['max_local_coverage']}",
            f"Global gain over max       : {result['global_gain_over_max']}",
            f"Jaccard similarity         : {result['jaccard']:.6f}",
            "",
        ]
    )

    for worker, unique in zip(workers, uniques):
        pct = (100.0 * unique / global_cov) if global_cov else 0.0
        lines.append(f"{worker['method']} unique contribution : {unique} ({pct:.2f}% global)")
    overlap_pct = (100.0 * result["overlap"] / global_cov) if global_cov else 0.0
    lines.append(f"Shared/overlap edges        : {result['overlap']} ({overlap_pct:.2f}% global)")

    perf = result.get("performance", [None, None])
    if any(p is not None for p in perf):
        lines.extend(
            [
                "",
                "=" * 72,
                "WORKER PERFORMANCE ANALYSIS (coverage_log.txt)",
                "=" * 72,
                (
                    f"{'Worker':<22} {'Method':<12} {'Samples':>7} "
                    f"{'Exec/s':>10} {'SampleMean':>11} {'RSS mean':>10} {'RSS peak':>10}"
                ),
                "-" * 88,
            ]
        )
        for worker, stats in zip(workers, perf):
            if stats is None:
                lines.append(
                    f"{worker['worker_id']:<22} {worker['method']:<12} {'N/A':>7} "
                    f"{'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>10}"
                )
                continue
            lines.append(
                f"{worker['worker_id']:<22} {worker['method']:<12} {stats['samples']:>7} "
                f"{_fmt(stats['overall_execs_per_second']):>10} "
                f"{_fmt(stats['sample_mean_execs_per_second']):>11} "
                f"{_fmt(stats['rss_mean_mib']):>10} "
                f"{_fmt(stats['rss_peak_mib']):>10}"
            )
        lines.extend(
            [
                "",
                "Exec/s = interval-weighted throughput from cumulative execution counters and timestamps.",
                "SampleMean = arithmetic mean of the exec/s values logged by the fuzzer.",
                "RSS mean/peak are MiB and include the fuzzer process plus its target executor child.",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze final edge overlap and runtime performance of two fuzzing workers."
    )
    parser.add_argument("edge_log_1", help="first worker coverage_log.txt.edges.jsonl")
    parser.add_argument("edge_log_2", help="second worker coverage_log.txt.edges.jsonl")
    parser.add_argument(
        "--coverage-log",
        nargs=2,
        metavar=("LOG1", "LOG2"),
        help=(
            "two coverage_log.txt files in the same worker order; if omitted, "
            "the analyzer tries to infer them beside each edge log"
        ),
    )
    args = parser.parse_args(argv)

    coverage_logs = args.coverage_log or (None, None)
    try:
        result = analyze_two_workers(
            args.edge_log_1,
            args.edge_log_2,
            coverage_logs[0],
            coverage_logs[1],
        )
    except (OSError, AnalysisError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sys.stdout.write(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
