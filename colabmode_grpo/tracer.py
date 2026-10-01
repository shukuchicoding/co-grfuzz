import collections
import threading
import json

prev_line = 0
prev_filename = ''
data = collections.defaultdict(collections.Counter)
lock = threading.Lock()

def trace(frame, event, _):
    if event != 'line':
        return trace

    global prev_line
    global prev_filename

    func_filename = frame.f_code.co_filename
    func_line_no = frame.f_lineno

    key = func_filename if func_filename == prev_filename else func_filename + prev_filename
    data[key][(prev_line, func_line_no)] += 1

    prev_line = func_line_no
    prev_filename = func_filename

    return trace

def get_coverage():
    with lock:
        unique_transitions = sum(len(counter) for counter in data.values())
        total_executions = sum(sum(counter.values()) for counter in data.values())

        return unique_transitions, total_executions