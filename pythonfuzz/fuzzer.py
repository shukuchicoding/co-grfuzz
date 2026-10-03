import hashlib
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

import psutil

import corpus
import tracer


if sys.platform != "win32":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

SAMPLING_WINDOW = 5


def worker(target, child_conn, close_fd_mask):
    class DummyFile:
        def write(self, _):
            pass

        def flush(self):
            pass

    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.ERROR)
    if close_fd_mask & 1:
        sys.stdout = DummyFile()
    if close_fd_mask & 2:
        sys.stderr = DummyFile()

    while True:
        buf = child_conn.recv_bytes()
        tracer.reset_input()
        status = "ok"
        error_text = None
        try:
            sys.settrace(tracer.trace)
            target(buf)
        except Exception:
            status = "exception"
            error_text = traceback.format_exc()
        finally:
            sys.settrace(None)
        edges, edge_hits = tracer.get_result()
        child_conn.send((status, edges, edge_hits, error_text))


class Fuzzer:
    def __init__(
        self,
        target,
        dirs=None,
        exact_artifact_path=None,
        rss_limit_mb=2048,
        timeout=120,
        regression=False,
        max_input_size=4096,
        close_fd_mask=0,
        runs=-1,
        dict_path=None,
        log_path="coverage_log.txt",
    ):
        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._exact_artifact_path = exact_artifact_path
        self._rss_limit_mb = rss_limit_mb
        self._timeout = timeout
        self._regression = regression
        self._close_fd_mask = close_fd_mask
        self._corpus = corpus.Corpus(self._dirs, max_input_size, dict_path)
        self._total_executions = 0
        self._executions_in_sample = 0
        self._last_sample_time = time.time()
        self._coverage_edges = set()
        self._total_coverage = 0
        self._p = None
        self._parent_conn = None
        self.runs = runs
        self._log_path = log_path

    def _start_worker(self):
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
        process.start()
        self._parent_conn = parent_conn
        self._p = process

    def _restart_worker(self):
        if self._p is not None:
            if self._p.is_alive():
                self._p.kill()
            self._p.join(timeout=1)
        if self._parent_conn is not None:
            self._parent_conn.close()
        self._start_worker()

    def log_stats(self, log_type):
        worker_rss = 0
        if self._p is not None and self._p.is_alive():
            worker_rss = psutil.Process(self._p.pid).memory_info().rss
        rss = (worker_rss + psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
        elapsed_time = max(time.time() - self._last_sample_time, 1e-6)
        execs_per_second = int(self._executions_in_sample / elapsed_time)
        self._last_sample_time = time.time()
        self._executions_in_sample = 0
        with open(self._log_path, "a") as log_file:
            log_file.write(
                "{},{},{},{},{}\n".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self._total_executions,
                    self._total_coverage,
                    execs_per_second,
                    rss,
                )
            )
        logging.info(
            "#%s %s     cov: %s corp: %s exec/s: %s rss: %s MB",
            self._total_executions,
            log_type,
            self._total_coverage,
            self._corpus.length,
            execs_per_second,
            rss,
        )
        return rss

    def write_sample(self, buf, prefix="crash-"):
        data = bytes(buf)
        digest = hashlib.sha256(data).hexdigest()
        crash_path = self._exact_artifact_path or os.path.join("crashes", prefix + digest)
        os.makedirs(os.path.dirname(crash_path) or ".", exist_ok=True)
        with open(crash_path, "wb") as f:
            f.write(data)
        logging.info("sample written to %s", crash_path)

    def start(self):
        logging.info("#0 READ units: %s", self._corpus.length)
        exit_code = 0
        self._start_worker()
        try:
            while True:
                if self.runs != -1 and self._total_executions >= self.runs:
                    logging.info("did %d runs, stopping now.", self.runs)
                    break
                buf = self._corpus.generate_input()
                self._parent_conn.send_bytes(buf)
                if not self._parent_conn.poll(self._timeout):
                    logging.info("timeout reached. testcase took: %s", self._timeout)
                    self.write_sample(buf, prefix="timeout-")
                    self._restart_worker()
                    continue
                try:
                    status, input_edges, _input_edge_hits, error_text = self._parent_conn.recv()
                except (EOFError, OSError, ValueError) as exc:
                    logging.error("Error receiving worker result: %s", exc)
                    self.write_sample(buf)
                    self._restart_worker()
                    continue

                self._total_executions += 1
                self._executions_in_sample += 1
                input_edges = set(input_edges)
                new_edges = input_edges.difference(self._coverage_edges)
                self._coverage_edges.update(input_edges)
                self._total_coverage = len(self._coverage_edges)

                if status == "exception":
                    if error_text:
                        logging.error("Target exception:\n%s", error_text)
                    self.write_sample(buf)
                if new_edges and status == "ok":
                    self._corpus.put(buf)

                rss = 0
                if new_edges:
                    rss = self.log_stats("NEW")
                elif (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                    rss = self.log_stats("PULSE")
                if rss > self._rss_limit_mb:
                    logging.info("MEMORY OOM: exceeded %s MB. Killing worker", self._rss_limit_mb)
                    self.write_sample(buf, prefix="oom-")
                    break
        finally:
            if self._p is not None:
                if self._p.is_alive():
                    self._p.terminate()
                self._p.join(timeout=1)
            if self._parent_conn is not None:
                self._parent_conn.close()
        sys.exit(exit_code)
