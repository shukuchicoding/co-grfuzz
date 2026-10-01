import os
import time
import sys
import psutil
import hashlib
import logging
import multiprocessing as mp
import random

import corpus, tracer
from datetime import datetime
from dqn import DQN, preprocess_input

if sys.platform != 'win32':
    mp.set_start_method('fork')

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.DEBUG)

SAMPLING_WINDOW = 5 # IN SECONDS

dqn_agent = DQN()

def worker(target, child_conn, close_fd_mask):
    class DummyFile:
        def write(self, x):
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
            child_conn.send_bytes(b'%d' % tracer.get_coverage())

class Fuzzer(object):
    def __init__(self,
                 target,
                 dirs=None,
                 exact_artifact_path=None,
                 rss_limit_mb=4096,
                 timeout=120,
                 regression=False,
                 max_input_size=4096,
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
        self._p = None
        self.runs = runs

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
                self._total_coverage,
                execs_per_second,
                rss
            ))
        logging.info('#{} {}     cov: {} corp: {} exec/s: {} rss: {} MB'.format(
            self._total_executions, log_type, self._total_coverage, self._corpus.length, execs_per_second, rss))
        return rss

    def write_sample(self, buf, prefix='crash-'):
        self._corpus.put(buf)
        m = hashlib.sha256()
        m.update(buf)
        if self._exact_artifact_path:
            crash_path = self._exact_artifact_path
        else:
            dir_path = 'crashes'
            os.makedirs(dir_path, exist_ok=True)
            crash_path = os.path.join(dir_path, prefix + m.hexdigest())
        with open(crash_path, 'wb') as f:
            f.write(buf)
        logging.info('sample was written to {}'.format(crash_path))
        if len(buf) < 200:
            logging.info('sample = {}'.format(buf.hex()))

    def start(self):
        logging.info("#0 READ units: {}".format(self._corpus.length))
        exit_code = 0
        parent_conn, child_conn = mp.Pipe()
        self._p = mp.Process(target=worker, args=(self._target, child_conn, self._close_fd_mask))
        self._p.start()

        while True:
            if self.runs != -1 and self._total_executions >= self.runs:
                self._p.terminate()
                logging.info('did %d runs, stopping now.', self.runs)
                break

            origin_buf = self._corpus.get_input()
            window = self._corpus.extract_substring(origin_buf, 32)
            state = preprocess_input(window)
            if random.random() < 0.5:
                action = random.randint(0,4)
            else:
                action = dqn_agent.choose_action(state)
            new_buf = self._corpus.mutate(origin_buf, window, action + 1)
            parent_conn.send_bytes(new_buf)
            if not parent_conn.poll(self._timeout):
                self._p.kill()
                logging.info("=================================================================")
                logging.info("timeout reached. testcase took: {}".format(self._timeout))
                self.write_sample(new_buf, prefix='timeout-')
                continue

            try:
                total_coverage = int(parent_conn.recv_bytes())
                reward = total_coverage - self._total_coverage + (psutil.Process(self._p.pid).memory_info().rss + psutil.Process(os.getpid()).memory_info().rss) / 1024 / 1024
            except ValueError:
                self.write_sample(new_buf)

            self._total_executions += 1
            self._executions_in_sample += 1
            rss = 0
            next_state = preprocess_input(new_buf[:32])
            done = False
            dqn_agent.update(state, action, reward, next_state, done)

            if total_coverage > self._total_coverage:
                rss = self.log_stats("NEW")
                self._total_coverage = total_coverage
                self._corpus.put(new_buf)
            else:
                if (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                    rss = self.log_stats('PULSE')

            if rss > self._rss_limit_mb:
                logging.info('MEMORY OOM: exceeded {} MB. Killing worker'.format(self._rss_limit_mb))
                self.write_sample(new_buf)
                self._p.kill()
                break

        self._p.join()
        sys.exit(exit_code)
