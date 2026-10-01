import os
import time
import sys
import psutil
import hashlib
import logging
from datetime import datetime
import multiprocessing as mp
import random
import math
import corpus, tracer
import tensorflow as tf
import grpo
grpo.init(1024, 2, 64, 0.001, tf.keras.activations.swish, 64, 0.1, 3.0)

if sys.platform != 'win32':
    mp.set_start_method('fork')

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

SAMPLING_WINDOW = 5  # IN SECONDS

def worker(target, child_conn, close_fd_mask):
    # Silence the fuzzee's noise
    class DummyFile:
        """No-op to trash stdout away."""
        def write(self, _):
            pass
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.ERROR)
    if close_fd_mask & 1:
        sys.stdout = DummyFile()
    if close_fd_mask & 2:
        sys.stderr = DummyFile()

    sys.settrace(tracer.trace)
    while True:
        buf = child_conn.recv_bytes()
        try:
            target(buf)
        except Exception as e:
            logging.exception(e)
            child_conn.send(e)
            continue
        else:
            child_conn.send(tracer.get_coverage())

class Fuzzer(object):
    def __init__(self,
                 target,
                 dirs=None,
                 exact_artifact_path=None,
                 rss_limit_mb=2048,
                 timeout=120,
                 regression=False,
                 max_input_size=1024,
                 close_fd_mask=0,
                 runs=-1,
                 dict_path=None):
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
        self._total_coverage = 0
        self._arc_coverage = 0
        self._p = None
        self.runs = runs
        self._mutation_count = {}  # Theo dõi số lần mutation tại mỗi vị trí

    def log_stats(self, log_type):
        rss = (psutil.Process(self._p.pid).memory_info().rss + psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
        endTime = time.time()
        elapsed_time = endTime - self._last_sample_time
        elapsed_time = max(elapsed_time, 1e-6)
        execs_per_second = int(self._executions_in_sample / elapsed_time)
        self._last_sample_time = time.time()
        self._executions_in_sample = 0
        with open("coverage_log.txt", "a") as log_file:
            log_file.write("{},{},{},{},{}\n".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                self._total_executions,
                self._arc_coverage,
                execs_per_second,
                rss
            ))
        logging.info('#{} {}     cov: \033[92m{}\033[0m corp: \033[92m{}\033[0m exec/s: \033[92m{}\033[0m rss: \033[92m{}\033[0m MB'.format(
            self._total_executions, log_type, self._arc_coverage, self._corpus.length, execs_per_second, rss))
        return rss

    def write_sample(self, buf, prefix='crash-'):
        self._corpus.put(buf)
        m = hashlib.sha256()
        m.update(buf)
        if self._exact_artifact_path:
            crash_path = self._exact_artifact_path
        else:
            dir_path = 'crashes'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info("The crashes directory is created")
            crash_path = os.path.join(dir_path, prefix + m.hexdigest())
        with open(crash_path, 'wb') as f:
            f.write(buf)
        logging.info('[{}] sample was written to {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), crash_path))
        if len(buf) < 200:
            logging.info('sample = {}'.format(buf.hex()))

    def start(self):
        print(self._corpus)
        logging.info("#0 READ units: {}".format(self._corpus.length))
        exit_code = 0
        parent_conn, child_conn = mp.Pipe()
        print(parent_conn, child_conn)
        self._p = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
        self._p.start()
        while True:
            if self.runs != -1 and self._total_executions >= self.runs:
                self._p.terminate()
                logging.info('did %d runs, stopping now.', self.runs)
                break

            buf = self._corpus.get_input()
            if len(buf) == 0:
                continue
            rewards = []
            if random.random() < 0.5 and len(buf) > grpo.group_size:
                mutated_inputs = [(self._corpus.mutate(buf, pos), pos) for pos in random.sample(range(0, len(buf)), grpo.group_size)]
            else:
                positions, _ = grpo.sampling(buf, grpo.group_size)
                mutated_inputs = [(self._corpus.mutate(buf, pos), pos) for pos in positions]

            for mutated_buf, pos in mutated_inputs:
                parent_conn.send_bytes(mutated_buf)
                if not parent_conn.poll(self._timeout):
                    self._p.kill()
                    logging.info("=================================================================")
                    logging.info("timeout reached. testcase took: {}".format(self._timeout))
                    self.write_sample(buf, prefix='timeout-')
                    parent_conn, child_conn = mp.Pipe()
                    self._p = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
                    self._p.start()
                    continue

                try:
                    coverage_info = parent_conn.recv()
                    arc_coverage, total_coverage = coverage_info
                except Exception as e:
                    logging.error("Error receiving coverage data: {}".format(e))
                    self.write_sample(buf)
                    continue

                self._total_executions += 1
                self._executions_in_sample += 1

                # Cập nhật số lần mutate cho vị trí pos
                self._mutation_count[pos] = self._mutation_count.get(pos, 0) + 1

                # Tính base reward (tránh chia cho 0)
                if arc_coverage > 0:
                    base_reward = (total_coverage - self._total_coverage) / arc_coverage
                else:
                    base_reward = 0

                # Tính reward cuối cùng bằng compute_reward
                final_reward = math.log(1 + self._mutation_count.get(pos, 0)) + base_reward
                rewards.append(final_reward)
                self._total_coverage = total_coverage

                rss = 0
                if arc_coverage > self._arc_coverage:
                    rss = self.log_stats("NEW")
                    self._arc_coverage = arc_coverage
                    self._corpus.put(buf)
                else:
                    if (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                        rss = self.log_stats('PULSE')
                if rss > self._rss_limit_mb:
                    logging.info('MEMORY OOM: exceeded {} MB. Killing worker'.format(self._rss_limit_mb))
                    self.write_sample(buf)
                    self._p.kill()
                    break

            grpo.train_on_group(buf, rewards)

        grpo.finished_callback()
        self._p.join()
        sys.exit(exit_code)
