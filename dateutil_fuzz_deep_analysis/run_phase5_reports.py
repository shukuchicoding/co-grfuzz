#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EDGE_NAME = 'coverage_log.txt.edges.jsonl'
SEED_NAME = 'coverage_log.txt.seeds.jsonl'
USAGE_NAME = 'coverage_log.txt.seed_usage.json'
SCENARIO_GLOB = 'colabmode_*+colabmode_*__dateutil_fuzz__r1'


def parse_args():
    p = argparse.ArgumentParser(
        description='Run two-worker and Phase-5 seed-impact analysis for dateutil scenarios.'
    )
    p.add_argument(
        '--root',
        type=Path,
        default=Path(__file__).resolve().parent,
        help='Directory containing analyzers and scenario folders (default: script directory).',
    )
    return p.parse_args()


def scenario_label(name: str) -> str:
    pair = name.split('__', 1)[0]
    parts = pair.split('+')
    return '+'.join(p.replace('colabmode_', '').upper() for p in parts)


def expected_worker_dirs(scenario: Path):
    pair = scenario.name.split('__', 1)[0]
    left, right = pair.split('+', 1)

    if left == right:
        candidates = sorted(
            p for p in scenario.iterdir()
            if p.is_dir() and (p.name == left or p.name.startswith(left + '_'))
        )
        if len(candidates) != 2:
            return None, f'expected 2 worker directories for {left}, found {len(candidates)}'
        return candidates, None

    workers = []
    for base in (left, right):
        exact = scenario / base
        if exact.is_dir():
            workers.append(exact)
            continue
        candidates = sorted(
            p for p in scenario.iterdir()
            if p.is_dir() and p.name.startswith(base + '_')
        )
        if len(candidates) != 1:
            return None, f'expected 1 worker directory for {base}, found {len(candidates)}'
        workers.append(candidates[0])
    return workers, None


def collect_inputs(scenario: Path):
    workers, err = expected_worker_dirs(scenario)
    if err:
        return None, err

    result = {'workers': workers, 'coverage': [], 'seeds': [], 'usage': []}
    missing = []
    for worker in workers:
        edge = worker / EDGE_NAME
        seed = worker / SEED_NAME
        usage = worker / USAGE_NAME
        for path in (edge, seed, usage):
            if not path.is_file():
                missing.append(str(path.relative_to(scenario)))
        result['coverage'].append(edge)
        result['seeds'].append(seed)
        result['usage'].append(usage)

    if missing:
        return None, 'missing required file(s): ' + ', '.join(missing)
    return result, None


def run_command(cmd):
    cp = subprocess.run(cmd, text=True, capture_output=True)
    output = cp.stdout
    if cp.stderr:
        if output and not output.endswith('\n'):
            output += '\n'
        output += cp.stderr
    return cp.returncode, output


def main():
    args = parse_args()
    root = args.root.resolve()
    two_analyzer = root / 'analyze_two_workers.py'
    seed_analyzer = root / 'analyze_seed_impact.py'
    combined_path = root / 'analysis_all_phase5.txt'

    missing_tools = [str(p) for p in (two_analyzer, seed_analyzer) if not p.is_file()]
    if missing_tools:
        print('Missing analyzer(s): ' + ', '.join(missing_tools), file=sys.stderr)
        return 2

    scenarios = sorted(p for p in root.glob(SCENARIO_GLOB) if p.is_dir())
    processed = skipped = failed = 0
    combined_blocks = []

    for scenario in scenarios:
        label = scenario_label(scenario.name)
        header = '=' * 88 + f'\nSCENARIO: {scenario.name}\nPAIR: {label}\n' + '=' * 88 + '\n'
        inputs, err = collect_inputs(scenario)
        if err:
            skipped += 1
            msg = header + f'[SKIP] {err}\n'
            combined_blocks.append(msg)
            print(f'[SKIP] {scenario.name}: {err}')
            continue

        edge1, edge2 = inputs['coverage']
        two_cmd = [sys.executable, str(two_analyzer), str(edge1), str(edge2)]
        rc_two, out_two = run_command(two_cmd)
        (scenario / 'analysis_two_workers.txt').write_text(out_two, encoding='utf-8')

        seed_cmd = [
            sys.executable,
            str(seed_analyzer),
            '--coverage', *(str(p) for p in inputs['coverage']),
            '--seeds', *(str(p) for p in inputs['seeds']),
            '--usage', *(str(p) for p in inputs['usage']),
            '--summary-json', str(scenario / 'seed_summary.json'),
            '--seed-csv', str(scenario / 'seed_detail.csv'),
            '--producer-csv', str(scenario / 'producer_summary.csv'),
            '--timeline-csv', str(scenario / 'global_timeline.csv'),
            '--strict',
        ]
        rc_seed, out_seed = run_command(seed_cmd)
        (scenario / 'analysis_seed_impact.txt').write_text(out_seed, encoding='utf-8')

        block = (
            header
            + '\n--- TWO-WORKER COVERAGE ANALYSIS ---\n'
            + (out_two or '[no output]\n')
            + '\n--- SEED IMPACT ANALYSIS ---\n'
            + (out_seed or '[no output]\n')
        )
        combined_blocks.append(block)

        if rc_two == 0 and rc_seed == 0:
            processed += 1
            print(f'[OK]   {scenario.name}')
        else:
            failed += 1
            print(f'[FAIL] {scenario.name}: two_workers={rc_two}, seed_impact={rc_seed}')

    combined_path.write_text('\n'.join(combined_blocks), encoding='utf-8')

    print()
    print(f'Processed: {processed}')
    print(f'Skipped:   {skipped}')
    print(f'Failed:    {failed}')
    print(f'Combined:  {combined_path}')

    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
