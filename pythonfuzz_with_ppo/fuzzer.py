import hashlib
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from datetime import datetime

import psutil

import corpus
import ppo
import ppo_core
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
        except BaseException:
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
        rss_limit_mb=2048,
        timeout=120,
        regression=False,
        max_input_size=1024,
        close_fd_mask=0,
        runs=-1,
        dict_path=None,
        policy_module=None,
        random_rate=0.75,
        log_path="coverage_log.txt",
    ):
        if int(max_input_size) < 1:
            raise ValueError("max_input_size must be >= 1")
        if float(timeout) <= 0:
            raise ValueError("timeout must be > 0")
        random_rate = float(random_rate)
        if not 0.0 <= random_rate <= 1.0:
            raise ValueError("random_rate must be in [0, 1]")

        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._exact_artifact_path = exact_artifact_path
        self._rss_limit_mb = max(float(rss_limit_mb), 0.0)
        self._timeout = float(timeout)
        self._regression = regression
        self._close_fd_mask = int(close_fd_mask)
        self._corpus = corpus.Corpus(self._dirs, max_input_size, dict_path)
        self._max_input_size = int(max_input_size)
        self._random_rate = random_rate
        self._policy = policy_module if policy_module is not None else ppo
        self._log_path = log_path

        self._policy_initialized = False

        self._total_executions = 0
        self._executions_in_sample = 0
        self._last_sample_time = time.time()
        self._coverage_edges = set()
        self._p = None
        self._parent_conn = None
        self.runs = int(runs)

        # Per-seed execution coverage used by RainFuzz R1. Newly admitted seeds
        # inherit the counts already observed when they were executed, avoiding
        # unnecessary baseline replays.
        self._seed_counts_cache = {}

    @property
    def coverage(self):
        return len(self._coverage_edges)

    @staticmethod
    def _seed_key(buf):
        return hashlib.sha256(bytes(buf)).digest()

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
            except (psutil.Error, OSError):
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

    def _execute_input(self, buf):
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
            return (
                status,
                tuple(edges or ()),
                dict(counts or {}),
                max(int(total_hits), 0),
                str(error_text or ""),
            )
        except (EOFError, OSError, ValueError, TypeError) as exc:
            self._restart_worker()
            return "worker_error", (), {}, 0, str(exc)

    def _apply_coverage(self, edges):
        edges = set(edges or ())
        new_edges = edges.difference(self._coverage_edges)
        self._coverage_edges.update(edges)
        return new_edges

    def _get_original_counts(self, seed):
        key = self._seed_key(seed)
        cached = self._seed_counts_cache.get(key)
        if cached is not None:
            return dict(cached), True

        if self.runs != -1 and self._total_executions >= self.runs:
            return None, False
        status, edges, counts, _hits, error_text = self._execute_input(seed)
        if status in {"timeout", "worker_error"}:
            if error_text:
                logging.error("seed baseline execution failed: %s", error_text)
            return None, False

        self._apply_coverage(edges)
        self._seed_counts_cache[key] = dict(counts)
        if status == "exception":
            if error_text:
                logging.error("Target exception while evaluating seed:\n%s", error_text)
            self.write_sample(seed)
        return dict(counts), True

    def start(self):
        logging.info("#0 READ units: %s", self._corpus.length)
        self._start_worker()
        exit_code = 0
        stop_requested = False

        try:
            # Initialize the TensorFlow policy only after the target worker exists,
            # avoiding fork of an already-initialized policy model.
            self._policy.init(self._max_input_size, 128, 0.0001, 0.5, 3.0, 1, 50, "tanh", 1)
            self._policy_initialized = True
            while not stop_requested and (self.runs == -1 or self._total_executions < self.runs):
                seed = self._corpus.get_input()
                if not seed:
                    seed = bytearray(b"\x00")
                seed_bytes = bytes(seed)
                valid_length = min(len(seed_bytes), self._max_input_size)
                if valid_length < 1:
                    continue

                original_counts, ready = self._get_original_counts(seed_bytes)
                if not ready:
                    continue
                if self.runs != -1 and self._total_executions >= self.runs:
                    break

                random_action = random.random() < self._random_rate
                if random_action:
                    pos = random.randrange(valid_length)
                    probability = 1.0 / valid_length
                else:
                    try:
                        pos, probability, _entropy = self._policy.pick_action(seed_bytes, valid_length)
                        pos = min(max(int(pos), 0), valid_length - 1)
                        probability = ppo_core.clamp_old_probability(probability)
                    except Exception as exc:
                        logging.error("PPO action selection failed; using random action: %s", exc)
                        random_action = True
                        pos = random.randrange(valid_length)
                        probability = 1.0 / valid_length

                mutation_types = self._corpus.valid_mutation_types(seed_bytes, pos)
                mutant_counts_for_reward = []

                for mutation_type in mutation_types:
                    if self.runs != -1 and self._total_executions >= self.runs:
                        stop_requested = True
                        break

                    mutant = self._corpus.mutate(seed_bytes, mutation_type, pos)
                    if not mutant:
                        logging.error("PPO mutator produced empty input; skipping")
                        continue

                    status, edges, counts, _hits, error_text = self._execute_input(mutant)
                    if status in {"timeout", "worker_error"}:
                        if error_text:
                            logging.error("%s", error_text)
                        continue

                    mutant_counts_for_reward.append(dict(counts))
                    new_edges = self._apply_coverage(edges)

                    if status == "exception":
                        if error_text:
                            logging.error("Target exception:\n%s", error_text)
                        self.write_sample(mutant)

                    if new_edges and status == "ok":
                        if self._corpus.put(mutant):
                            self._seed_counts_cache[self._seed_key(mutant)] = dict(counts)
                        self.log_stats("NEW")
                    elif (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                        self.log_stats("PULSE")

                    if self._rss_limit_mb > 0 and self._safe_rss_mb() > self._rss_limit_mb:
                        logging.info("MEMORY OOM: exceeded %s MB", self._rss_limit_mb)
                        self.write_sample(mutant, prefix="oom-")
                        stop_requested = True
                        break

                # RainFuzz R1: average positive hit-count increase of the mutation
                # set relative to the original seed coverage vector.
                reward = ppo_core.rainfuzz_r1(original_counts, mutant_counts_for_reward)

                # Official RainFuzz does not train the policy on the 75% random
                # actions; only policy-selected positions enter PPO memory.
                if not random_action and mutant_counts_for_reward:
                    try:
                        self._policy.add_experience(
                            seed_bytes,
                            pos,
                            probability,
                            reward,
                            valid_length=valid_length,
                        )
                    except TypeError:
                        # Compatibility with lightweight test doubles / legacy API.
                        self._policy.add_experience(seed_bytes, pos, probability, reward)
                    except Exception as exc:
                        logging.error("PPO update failed: %s", exc)
        finally:
            self._stop_worker(kill=False)
            if self._policy_initialized:
                try:
                    self._policy.finished_callback()
                except Exception as exc:
                    logging.error("PPO cleanup failed: %s", exc)

        return exit_code
