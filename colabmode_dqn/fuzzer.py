import hashlib
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

import psutil
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import corpus
import coverage_events
import dqn_core
import tracer
from dqn import DQN


if sys.platform != "win32":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

SAMPLING_WINDOW = 5


class CorpusWatcher(FileSystemEventHandler):
    def __init__(self, corpus_obj):
        self.corpus = corpus_obj

    def on_created(self, event):
        if not event.is_directory:
            self.corpus._add_file(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.corpus._add_file(event.dest_path)


def start_watcher(corpus_obj):
    observer = Observer()
    event_handler = CorpusWatcher(corpus_obj)
    scheduled = False
    for directory in corpus_obj._dirs:
        if os.path.isdir(directory):
            observer.schedule(event_handler, directory, recursive=False)
            scheduled = True
    if scheduled:
        observer.start()
        corpus_obj.rescan()
        return observer
    return None


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
        try:
            buf = child_conn.recv_bytes()
        except (EOFError, OSError):
            break

        tracer.reset_input(exclude_files=(__file__, tracer.__file__))
        status = "ok"
        error_text = ""
        try:
            sys.settrace(tracer.trace)
            target(buf)
        except BaseException:  # target crashes must be reported, not kill IPC silently
            status = "exception"
            error_text = traceback.format_exc()
        finally:
            sys.settrace(None)

        edges, counts, total_hits = tracer.get_result()
        try:
            child_conn.send((status, edges, counts, total_hits, error_text))
        except (BrokenPipeError, EOFError, OSError):
            break


class Fuzzer:
    def __init__(
        self,
        target,
        dirs=None,
        exact_artifact_path=None,
        rss_limit_mb=4096,
        timeout=120,
        regression=False,
        max_input_size=4096,
        close_fd_mask=0,
        runs=-1,
        dict_path=None,
        agent=None,
        state_width=32,
        log_path="coverage_log.txt",
        coverage_event_path=None,
        worker_id=None,
    ):
        if int(max_input_size) < 1:
            raise ValueError("max_input_size must be >= 1")
        if int(state_width) < 1:
            raise ValueError("state_width must be >= 1")
        if float(timeout) <= 0:
            raise ValueError("timeout must be > 0")

        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._exact_artifact_path = exact_artifact_path
        self._rss_limit_mb = max(float(rss_limit_mb), 0.0)
        self._timeout = float(timeout)
        self._regression = regression
        self._close_fd_mask = int(close_fd_mask)
        self._corpus = corpus.Corpus(self._dirs, max_input_size, dict_path)
        self._state_width = int(state_width)
        # Construct the TensorFlow model lazily after the worker is forked.
        # This reduces fork-after-TensorFlow-thread-initialization risk.
        self._agent = agent
        self._log_path = log_path
        self._coverage_event_path = (
            f"{log_path}.edges.jsonl" if coverage_event_path is None else coverage_event_path
        )
        self._coverage_worker_id = worker_id
        self._coverage_event_logger = None
        self._observer = None

        self._total_executions = 0
        self._executions_in_sample = 0
        self._last_sample_time = time.time()
        self._coverage_edges = set()
        self._p = None
        self._parent_conn = None
        self.runs = int(runs)

    def _init_coverage_logger(self):
        if self._coverage_event_logger is not None or not self._coverage_event_path:
            return
        self._coverage_event_logger = coverage_events.CoverageEventLogger(
            self._coverage_event_path,
            worker_id=self._coverage_worker_id,
            method="dqn",
        )

    @property
    def coverage(self):
        return len(self._coverage_edges)

    def _start_worker(self):
        parent_conn, child_conn = mp.Pipe()
        self._parent_conn = parent_conn
        self._p = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
        self._p.start()
        child_conn.close()

    def _stop_worker(self, kill=False):
        if self._p is not None:
            try:
                if self._p.is_alive():
                    self._p.kill() if kill else self._p.terminate()
            except (psutil.NoSuchProcess, OSError):
                pass
            self._p.join(timeout=1)
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
        self._p = None
        self._parent_conn = None

    def _restart_worker(self):
        self._stop_worker(kill=True)
        self._start_worker()

    def _safe_rss_mb(self):
        total = 0
        try:
            total += psutil.Process(os.getpid()).memory_info().rss
        except (psutil.Error, OSError):
            pass
        if self._p is not None and self._p.is_alive():
            try:
                total += psutil.Process(self._p.pid).memory_info().rss
            except (psutil.Error, OSError):
                pass
        return total / 1024.0 / 1024.0

    def log_stats(self, log_type):
        rss = self._safe_rss_mb()
        elapsed = max(time.time() - self._last_sample_time, 1e-6)
        execs_per_second = int(self._executions_in_sample / elapsed)
        self._last_sample_time = time.time()
        self._executions_in_sample = 0
        with open(self._log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                "{},{},{},{},{}\n".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self._total_executions,
                    self.coverage,
                    execs_per_second,
                    rss,
                )
            )
        logging.info(
            "#%s %s cov: %s corp: %s exec/s: %s rss: %.2f MB",
            self._total_executions,
            log_type,
            self.coverage,
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
        return crash_path

    def _apply_coverage(self, edges):
        edges = set(edges or ())
        new_edges = edges.difference(self._coverage_edges)
        self._coverage_edges.update(edges)
        if new_edges and self._coverage_event_logger is not None:
            self._coverage_event_logger.record_discovery(
                new_edges,
                execution=self._total_executions,
                local_coverage=self.coverage,
            )
        return new_edges

    def _execute_input(self, buf):
        """Execute one target input and always return a structured result.

        Attempts, including timeouts/worker failures, count toward ``runs`` so a
        permanently hanging target cannot cause an infinite finite-run campaign.
        """
        if not buf:
            buf = b"\x00"
        if self._p is None or not self._p.is_alive():
            self._restart_worker() if self._p is not None else self._start_worker()

        sent = False
        for attempt in range(2):
            try:
                self._parent_conn.send_bytes(bytes(buf))
                sent = True
                break
            except (BrokenPipeError, EOFError, OSError):
                if attempt == 0:
                    self._restart_worker()
        self._total_executions += 1
        self._executions_in_sample += 1

        if not sent:
            return "worker_error", (), {}, 0, "failed to send input to worker"

        if not self._parent_conn.poll(self._timeout):
            self.write_sample(buf, prefix="timeout-")
            self._restart_worker()
            return "timeout", (), {}, 0, f"timeout after {self._timeout}s"

        try:
            status, edges, counts, total_hits, error_text = self._parent_conn.recv()
            if status not in {"ok", "exception"}:
                status = "worker_error"
            counts = dict(counts or {})
            edges = tuple(edges or ())
            total_hits = max(int(total_hits), 0)
            return status, edges, counts, total_hits, str(error_text or "")
        except (EOFError, OSError, ValueError, TypeError) as exc:
            self._restart_worker()
            return "worker_error", (), {}, 0, str(exc)

    def start(self):
        logging.info("#0 READ units: %s", self._corpus.length)
        self._start_worker()
        exit_code = 0

        try:
            self._init_coverage_logger()
            if self._agent is None:
                self._agent = DQN(input_dim=self._state_width)
            self._observer = start_watcher(self._corpus)
            while self.runs == -1 or self._total_executions < self.runs:
                seed = self._corpus.get_input()
                if not seed:
                    seed = bytearray(b"\x00")

                window, offset, _valid_len = self._corpus.extract_substring(seed, self._state_width)
                state = dqn_core.make_state(window, self._state_width)

                try:
                    if hasattr(self._agent, "select_action"):
                        action = int(self._agent.select_action(state))
                    else:
                        action = int(self._agent.choose_action(state))
                except Exception as exc:
                    logging.error("DQN action selection failed: %s", exc)
                    action = 0
                if not 0 <= action < 5:
                    action = 0

                mutant = self._corpus.mutate(seed, window, offset, action)
                if not mutant:
                    mutant = bytearray(b"\x00")

                status, input_edges, _counts, _total_hits, error_text = self._execute_input(mutant)
                if status in {"timeout", "worker_error"}:
                    if error_text:
                        logging.error("%s", error_text)
                    continue

                input_edges = set(input_edges)
                new_edges = self._apply_coverage(input_edges)

                # Coverage-only reward; resource usage is deliberately excluded.
                reward = dqn_core.coverage_reward(input_edges)

                # Observe the post-action state at the same approximate offset.
                next_offset = min(offset, max(len(mutant) - 1, 0))
                next_window = bytes(mutant[next_offset : next_offset + self._state_width])
                next_state = dqn_core.make_state(next_window, self._state_width)
                try:
                    self._agent.update(state, action, reward, next_state, False)
                except Exception as exc:
                    # A learning failure must not destroy the fuzzing campaign.
                    logging.error("DQN update failed: %s", exc)

                if status == "exception":
                    if error_text:
                        logging.error("Target exception:\n%s", error_text)
                    self.write_sample(mutant)

                if new_edges and status == "ok":
                    self._corpus.put(mutant)
                    self.log_stats("NEW")
                elif (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                    self.log_stats("PULSE")

                if self._rss_limit_mb > 0 and self._safe_rss_mb() > self._rss_limit_mb:
                    logging.info("MEMORY OOM: exceeded %s MB", self._rss_limit_mb)
                    self.write_sample(mutant, prefix="oom-")
                    break
        finally:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join()
            self._stop_worker(kill=False)
            if self._coverage_event_logger is not None:
                self._coverage_event_logger.close()

        return exit_code
