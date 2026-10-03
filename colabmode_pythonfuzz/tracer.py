"""Per-input Python line-transition tracer.

The parent fuzzer owns cumulative coverage.  The worker only returns the
transitions observed for the current input, so coverage is not lost when a
worker process is restarted after a timeout.
"""

_current_edges = set()
_current_hits = 0
_prev_filename = None
_prev_line = None


def reset_input():
    """Reset trace state before executing one testcase."""
    global _current_edges, _current_hits, _prev_filename, _prev_line
    _current_edges = set()
    _current_hits = 0
    _prev_filename = None
    _prev_line = None


def trace(frame, event, _arg):
    """Record consecutive executed Python source lines for one testcase."""
    if event != "line":
        return trace

    global _current_hits, _prev_filename, _prev_line

    filename = frame.f_code.co_filename
    line = frame.f_lineno

    if _prev_filename is not None:
        edge = (_prev_filename, _prev_line, filename, line)
        _current_edges.add(edge)
        _current_hits += 1

    _prev_filename = filename
    _prev_line = line
    return trace


def get_result():
    """Return a serializable snapshot for the current testcase."""
    return tuple(_current_edges), _current_hits
