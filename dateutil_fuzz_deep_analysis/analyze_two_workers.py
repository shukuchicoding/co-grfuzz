#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def canonical_edge(edge):
    """
    Edge schema from Phase 4:
        [src_file, src_line, dst_file, dst_line]
    """
    if not isinstance(edge, list) or len(edge) != 4:
        raise ValueError(f"Invalid edge: {edge!r}")

    return (
        str(edge[0]),
        int(edge[1]),
        str(edge[2]),
        int(edge[3]),
    )


def load_worker(path):
    path = Path(path)

    edges = set()
    worker_id = None
    method = None

    discover_events = 0
    max_reported_local = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{lineno}: invalid JSON: {exc}"
                ) from exc

            if worker_id is None:
                worker_id = record.get("worker_id")

            if method is None:
                method = record.get("method")

            if record.get("event") != "discover":
                continue

            discover_events += 1

            max_reported_local = max(
                max_reported_local,
                int(record.get("local_coverage", 0)),
            )

            for edge in record.get("edges", []):
                edges.add(canonical_edge(edge))

    return {
        "path": str(path),
        "worker_id": worker_id or path.stem,
        "method": method or "unknown",
        "edges": edges,
        "discover_events": discover_events,
        "reported_local_coverage": max_reported_local,
    }


def edge_to_json(edge):
    return list(edge)


def write_edges(path, edges):
    with open(path, "w", encoding="utf-8") as f:
        for edge in sorted(edges):
            f.write(
                json.dumps(
                    edge_to_json(edge),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local, union, overlap, and unique edge coverage "
            "from two Phase-4 cooperative worker logs."
        )
    )

    parser.add_argument("worker1")
    parser.add_argument("worker2")

    parser.add_argument(
        "--output-json",
        help="Write summary to JSON",
    )

    parser.add_argument(
        "--edge-dir",
        help=(
            "Optional directory for union/overlap/"
            "worker-unique edge lists"
        ),
    )

    args = parser.parse_args()

    w1 = load_worker(args.worker1)
    w2 = load_worker(args.worker2)

    e1 = w1["edges"]
    e2 = w2["edges"]

    union = e1 | e2
    overlap = e1 & e2

    unique1 = e1 - e2
    unique2 = e2 - e1

    local1 = len(e1)
    local2 = len(e2)

    global_coverage = len(union)
    max_coverage = max(local1, local2)

    jaccard = (
        len(overlap) / global_coverage
        if global_coverage
        else 0.0
    )

    complementarity = global_coverage - max_coverage

    summary = {
        "worker1": {
            "worker_id": w1["worker_id"],
            "method": w1["method"],
            "local_coverage": local1,
            "unique_edges": len(unique1),
            "discover_events": w1["discover_events"],
        },
        "worker2": {
            "worker_id": w2["worker_id"],
            "method": w2["method"],
            "local_coverage": local2,
            "unique_edges": len(unique2),
            "discover_events": w2["discover_events"],
        },
        "overlap_edges": len(overlap),
        "global_coverage": global_coverage,
        "max_local_coverage": max_coverage,
        "global_gain_over_max": complementarity,
        "jaccard_similarity": jaccard,
        "worker1_unique_fraction_global": (
            len(unique1) / global_coverage
            if global_coverage
            else 0.0
        ),
        "worker2_unique_fraction_global": (
            len(unique2) / global_coverage
            if global_coverage
            else 0.0
        ),
        "overlap_fraction_global": (
            len(overlap) / global_coverage
            if global_coverage
            else 0.0
        ),
    }

    print("=" * 72)
    print("TWO-WORKER COVERAGE ANALYSIS")
    print("=" * 72)

    print(
        f"Worker 1 : {w1['worker_id']} "
        f"({w1['method']})"
    )
    print(f"  Local coverage : {local1}")
    print(f"  Unique edges   : {len(unique1)}")

    print()

    print(
        f"Worker 2 : {w2['worker_id']} "
        f"({w2['method']})"
    )
    print(f"  Local coverage : {local2}")
    print(f"  Unique edges   : {len(unique2)}")

    print()
    print(f"Overlap                    : {len(overlap)}")
    print(f"Global / union coverage    : {global_coverage}")
    print(f"Max local coverage         : {max_coverage}")
    print(f"Global gain over max       : {complementarity}")
    print(f"Jaccard similarity         : {jaccard:.6f}")

    if global_coverage:
        print()
        print(
            f"{w1['method']} unique contribution : "
            f"{len(unique1)} "
            f"({100 * len(unique1) / global_coverage:.2f}% global)"
        )
        print(
            f"{w2['method']} unique contribution : "
            f"{len(unique2)} "
            f"({100 * len(unique2) / global_coverage:.2f}% global)"
        )
        print(
            f"Shared/overlap edges        : "
            f"{len(overlap)} "
            f"({100 * len(overlap) / global_coverage:.2f}% global)"
        )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

    if args.edge_dir:
        output_dir = Path(args.edge_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        write_edges(
            output_dir / "global_union.jsonl",
            union,
        )
        write_edges(
            output_dir / "overlap.jsonl",
            overlap,
        )
        write_edges(
            output_dir / "worker1_unique.jsonl",
            unique1,
        )
        write_edges(
            output_dir / "worker2_unique.jsonl",
            unique2,
        )


if __name__ == "__main__":
    main()