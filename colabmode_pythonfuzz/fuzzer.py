import os
import time
import sys
import psutil
import hashlib
import logging
import multiprocessing as mp
import corpus, tracer
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

if sys.platform != 'win32':
    mp.set_start_method('fork')

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

SAMPLING_WINDOW = 5 # IN SECONDS

class CorpusWatcher(FileSystemEventHandler):
    def __init__(self, corpus):
        self.corpus = corpus
    
    def on_created(self, event):
        # logging.info(f"New file detected: {event.src_path}")
        self.corpus._add_file(event.src_path)

def start_watcher(corpus):
    observer = Observer()
    event_handler = CorpusWatcher(corpus)
    for directory in corpus._dirs:
        observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    return observer

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
    def __init__(self, target, dirs=None, exact_artifact_path=None, rss_limit_mb=2048,
                 timeout=120, regression=False, max_input_size=4096, close_fd_mask=0,
                 runs=-1, dict_path=None):
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
        self._observer = start_watcher(self._corpus)

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
        logging.info('#{} {}     cov: \033[92m{}\033[0m corp: \033[92m{}\033[0m exec/s: \033[92m{}\033[0m rss: \033[92m{}\033[0m MB'.format(
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
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info("The crashes directory is created")
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

        try:
            while True:
                if self.runs != -1 and self._total_executions >= self.runs:
                    self._p.terminate()
                    logging.info('did %d runs, stopping now.', self.runs)
                    break
                buf = self._corpus.generate_input()
                parent_conn.send_bytes(buf)
                if not parent_conn.poll(self._timeout):
                    self._p.kill()
                    logging.info("timeout reached. testcase took: {}".format(self._timeout))
                    self.write_sample(buf, prefix='timeout-')
                    continue
                try:
                    total_coverage = int(parent_conn.recv_bytes())
                except ValueError:
                    self.write_sample(buf)
                    continue
                self._total_executions += 1
                self._executions_in_sample += 1
                rss = 0
                if total_coverage > self._total_coverage:
                    rss = self.log_stats("NEW")
                    self._total_coverage = total_coverage
                    self._corpus.put(buf)
                else:
                    if (time.time() - self._last_sample_time) > SAMPLING_WINDOW:
                        rss = self.log_stats('PULSE')
                if rss > self._rss_limit_mb:
                    logging.info('MEMORY OOM: exceeded {} MB. Killing worker'.format(self._rss_limit_mb))
                    self.write_sample(buf)
                    self._p.kill()
                    break
        finally:
            self._observer.stop()
            self._observer.join()
        self._p.join()
        sys.exit(exit_code)
