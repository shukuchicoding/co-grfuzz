import hashlib
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import psutil
import tensorflow as tf
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import corpus
import coverage_events
import grpo
import grpo_core
import tracer


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
        # Close the race between the constructor's initial directory scan and
        # watcher activation. A peer may have published a seed in that window.
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
        buf = child_conn.recv_bytes()
        tracer.reset_input()
        status = "ok"
        error_text = None
        try:
            # Trace only the target execution.  Do not include IPC/fuzzer lines
            # in the target's coverage state.
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
        max_input_size=1024,
        close_fd_mask=0,
        runs=-1,
        dict_path=None,
        log_path="coverage_log.txt",
        coverage_event_path=None,
        worker_id=None,
    ):
        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._exact_artifact_path = exact_artifact_path
        self._rss_limit_mb = rss_limit_mb
        self._timeout = timeout
        self._regression = regression
        self._close_fd_mask = close_fd_mask
        self._corpus = corpus.Corpus(self._dirs, max_input_size, dict_path)
        self._max_input_size = max_input_size
        self._total_executions = 0
        self._executions_in_sample = 0
        self._last_sample_time = time.time()
        self._coverage_edges = set()
        self._total_edge_hits = 0
        self._arc_coverage = 0
        self._p = None
        self._parent_conn = None
        self.runs = runs
        self._mutation_count = {}
        self._observer = None
        self._log_path = log_path
        self._coverage_event_path = (
            f"{log_path}.edges.jsonl" if coverage_event_path is None else coverage_event_path
        )
        self._coverage_worker_id = worker_id
        self._coverage_event_logger = None
        self._policy_initialized = False

    def _init_coverage_logger(self):
        if self._coverage_event_logger is not None or not self._coverage_event_path:
            return
        self._coverage_event_logger = coverage_events.CoverageEventLogger(
            self._coverage_event_path,
            worker_id=self._coverage_worker_id,
            method="grpo",
        )

    def _init_policy(self):
        if self._policy_initialized:
            return
        # Start the target executor before TensorFlow creates background threads.
        # This reduces fork-after-TensorFlow risk on Linux.
        grpo.init(
            self._max_input_size,
            2,
            64,
            0.001,
            tf.keras.activations.swish,
            64,
            0.1,
            3.0,
        )
        self._policy_initialized = True

    def _start_worker(self):
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(
            target=worker,
            args=(self._target, child_conn, self._close_fd_mask),
        )
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
                    self._arc_coverage,
                    execs_per_second,
                    rss,
                )
            )

        logging.info(
            "#{} {}     cov: \033[92m{}\033[0m corp: \033[92m{}\033[0m exec/s: \033[92m{}\033[0m rss: \033[92m{}\033[0m MB".format(
                self._total_executions,
                log_type,
                self._arc_coverage,
                self._corpus.length,
                execs_per_second,
                rss,
            )
        )
        return rss

    def write_sample(self, buf, prefix="crash-"):
        data = bytes(buf)
        digest = hashlib.sha256(data).hexdigest()
        crash_path = (
            self._exact_artifact_path
            if self._exact_artifact_path
            else os.path.join("crashes", prefix + digest)
        )
        os.makedirs(os.path.dirname(crash_path) or ".", exist_ok=True)
        with open(crash_path, "wb") as f:
            f.write(data)
        logging.info(
            "[%s] sample written to %s",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            crash_path,
        )

    def _apply_coverage_result(self, input_edges, input_edge_hits):
        input_edges = set(input_edges)
        new_edges = input_edges.difference(self._coverage_edges)
        previous_coverage = self._arc_coverage
        previous_hits = self._total_edge_hits

        self._coverage_edges.update(input_edges)
        self._arc_coverage = len(self._coverage_edges)
        self._total_edge_hits += int(input_edge_hits)

        if new_edges and self._coverage_event_logger is not None:
            self._coverage_event_logger.record_discovery(
                new_edges,
                execution=self._total_executions,
                local_coverage=self._arc_coverage,
            )

        return new_edges, previous_coverage, previous_hits

    def start(self):
        logging.info("#0 READ units: %s", self._corpus.length)
        exit_code = 0
        stop_requested = False
        self._start_worker()

        try:
            self._init_coverage_logger()
            self._init_policy()
            self._observer = start_watcher(self._corpus)
            while not stop_requested:
                if self.runs != -1 and self._total_executions >= self.runs:
                    logging.info("did %d runs, stopping now.", self.runs)
                    break

                buf = self._corpus.get_input()
                # Corpus enforces non-empty seeds, but keep the fuzzer defensive
                # against a malformed external corpus implementation.
                if len(buf) <= 0:
                    continue

                # GRPO samples G independent actions from pi_old(position|seed).
                # Sampling is WITH replacement, so group_size stays 64 even for
                # seeds shorter than 64 bytes; duplicate positions are valid.
                sample_count = grpo.group_size
                if self.runs != -1:
                    remaining = self.runs - self._total_executions
                    if remaining <= 0:
                        break
                    sample_count = min(sample_count, remaining)

                actions, old_log_probs = grpo.sampling(buf, sample_count)
                if len(actions) == 0:
                    continue
                if len(actions) != len(old_log_probs):
                    logging.error("GRPO sampler returned mismatched action/log-prob lengths")
                    continue

                # All group members are evaluated relative to the same pre-group
                # state. This removes execution-order bias inside the GRPO group.
                pre_group_edges = set(self._coverage_edges)
                pre_group_coverage = len(pre_group_edges)
                pre_group_mutation_count = dict(self._mutation_count)

                # corpus.mutate() copies the original seed first, so these are
                # independent mutants of one common GRPO state.
                mutated_inputs = [
                    (self._corpus.mutate(buf, int(pos)), int(pos), float(old_lp))
                    for pos, old_lp in zip(actions, old_log_probs)
                ]

                train_actions = []
                train_old_log_probs = []
                train_rewards = []
                observed_positions = []

                for mutated_buf, pos, old_log_prob in mutated_inputs:
                    if self.runs != -1 and self._total_executions >= self.runs:
                        stop_requested = True
                        break

                    # Mutation postcondition: the target never receives an empty
                    # input even when deletion operators are selected.
                    if len(mutated_buf) <= 0:
                        logging.error("mutator produced an empty input; skipping candidate")
                        continue

                    # Count every testcase attempt, including timeout/worker
                    # failure, so a finite --runs budget always terminates.
                    self._total_executions += 1
                    self._executions_in_sample += 1

                    self._parent_conn.send_bytes(mutated_buf)
                    if not self._parent_conn.poll(self._timeout):
                        logging.info("timeout reached. testcase took: %s", self._timeout)
                        self.write_sample(mutated_buf, prefix="timeout-")
                        self._restart_worker()
                        # No coverage observation exists for this action, so it is
                        # excluded from the GRPO update.
                        continue

                    try:
                        status, input_edges, input_edge_hits, error_text = self._parent_conn.recv()
                    except (EOFError, OSError, ValueError) as exc:
                        logging.error("Error receiving worker result: %s", exc)
                        self.write_sample(mutated_buf)
                        self._restart_worker()
                        continue

                    candidate_edges = set(input_edges)
                    candidate_new_edges = candidate_edges.difference(pre_group_edges)
                    safe_input_edge_hits = max(int(input_edge_hits), 0)

                    # Preserve the manuscript's reward form, but evaluate every
                    # member against the SAME pre-group coverage state. The tested
                    # helper guards candidate_coverage==0 and malformed counters.
                    delta = grpo_core.coverage_delta(
                        pre_group_coverage,
                        len(candidate_new_edges),
                        safe_input_edge_hits,
                    )

                    # Snapshot the mutation-count term as well so duplicate sampled
                    # positions do not receive different rewards merely because of
                    # their order within this group.
                    prior_count = pre_group_mutation_count.get(pos, 0)
                    reward = grpo_core.mutation_reward(prior_count, delta)

                    train_actions.append(pos)
                    train_old_log_probs.append(old_log_prob)
                    train_rewards.append(reward)
                    observed_positions.append(pos)

                    if status == "exception":
                        if error_text:
                            logging.error("Target exception:\n%s", error_text)
                        self.write_sample(mutated_buf)

                    # Maintain the real cumulative local coverage used for COT
                    # logging. GRPO reward is based on the frozen pre-group state,
                    # while shared-corpus admission uses live local novelty.
                    live_new_edges, _, _ = self._apply_coverage_result(
                        candidate_edges, safe_input_edge_hits
                    )

                    # Publish only if this execution expands the worker's CURRENT
                    # local coverage. Two group members that discover the same edge
                    # may receive comparable GRPO rewards, but only the first live
                    # discovery should enlarge the shared corpus.
                    if live_new_edges and status == "ok":
                        self._corpus.put(mutated_buf)

                    rss = 0
                    if live_new_edges:
                        rss = self.log_stats("NEW")
                    elif (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                        rss = self.log_stats("PULSE")

                    if rss > self._rss_limit_mb:
                        logging.info(
                            "MEMORY OOM: exceeded %s MB. Killing worker",
                            self._rss_limit_mb,
                        )
                        self.write_sample(mutated_buf, prefix="oom-")
                        stop_requested = True
                        break

                # Mutation counts represent completed, observed actions. Update
                # only after the group so the reward baseline is order-independent.
                for pos in observed_positions:
                    self._mutation_count[pos] = self._mutation_count.get(pos, 0) + 1

                if train_rewards:
                    grpo.train_on_group(
                        buf,
                        np.asarray(train_actions, dtype=np.int32),
                        np.asarray(train_old_log_probs, dtype=np.float32),
                        np.asarray(train_rewards, dtype=np.float32),
                    )

        finally:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join()
            if self._p is not None:
                if self._p.is_alive():
                    self._p.terminate()
                self._p.join(timeout=1)
            if self._parent_conn is not None:
                self._parent_conn.close()
            if self._policy_initialized:
                grpo.finished_callback()
            if self._coverage_event_logger is not None:
                self._coverage_event_logger.close()

        sys.exit(exit_code)
