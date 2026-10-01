# import collections

# prev_line = 0
# prev_filename = ''
# data = collections.defaultdict(set)

# def trace(frame, event, arg):
#     if event != 'line':
#         return trace

#     global prev_line
#     global prev_filename

#     func_filename = frame.f_code.co_filename
#     func_line_no = frame.f_lineno

#     if func_filename != prev_filename:
#         # We need a way to keep track of inter-files transferts,
#         # and since we don't really care about the details of the coverage,
#         # concatenating the two filenames in enough.
#         data[func_filename + prev_filename].add((prev_line, func_line_no))
#     else:
#         data[func_filename].add((prev_line, func_line_no))

#     prev_line = func_line_no
#     prev_filename = func_filename

#     return trace


# def get_coverage():
#     return sum(map(len, data.values()))

import collections
import threading

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
        # return sum(sum(counter.values()) for counter in data.values())
        return sum(len(counter) for counter in data.values())
