"""Per-input Python line-transition coverage tracer.

The worker resets this module for every target execution and returns a serializable
snapshot to the parent process.  The parent owns cumulative campaign coverage, so
worker restarts after timeouts do not erase coverage state.
"""

from collections import Counter
from pathlib import Path

_current_counts = Counter()
_prev_filename = None
_prev_line = None
_excluded_files = set()


def _normalize_file(path):
    try:
        return str(Path(path).resolve())
    except (OSError, TypeError, ValueError):
        return str(path)


def reset_input(exclude_files=()):
    """Reset all per-input tracing state.

    ``exclude_files`` should contain the fuzzer/tracer implementation files so
    transitions back into the harness are not counted as target coverage.
    """
    global _current_counts, _prev_filename, _prev_line, _excluded_files
    _current_counts = Counter()
    _prev_filename = None
    _prev_line = None
    _excluded_files = {_normalize_file(p) for p in exclude_files if p}


def trace(frame, event, _arg):
    """Record consecutive Python source-line transitions for one testcase."""
    if event != "line":
        return trace

    global _prev_filename, _prev_line

    filename = _normalize_file(frame.f_code.co_filename)
    line = int(frame.f_lineno)

    if filename in _excluded_files:
        # Break the chain when execution enters the harness.  Otherwise the
        # return from target() would create a synthetic target->fuzzer edge.
        _prev_filename = None
        _prev_line = None
        return trace

    if _prev_filename is not None:
        edge = (_prev_filename, int(_prev_line), filename, line)
        _current_counts[edge] += 1

    _prev_filename = filename
    _prev_line = line
    return trace


def get_result():
    """Return ``(edges, counts, total_hits)`` for the current input."""
    counts = dict(_current_counts)
    return tuple(counts.keys()), counts, int(sum(counts.values()))
