import hashlib
import os
import random
import struct
import threading

import dictionnary

INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]


class Corpus:
    def __init__(self, dirs=None, max_input_size=4096, dict_path=None):
        if int(max_input_size) < 1:
            raise ValueError("max_input_size must be >= 1")

        self._inputs = []
        self._hashes = set()
        self._lock = threading.RLock()
        self._dict = dictionnary.Dictionary(dict_path)
        self._max_input_size = int(max_input_size)
        self._dirs = dirs or []

        if self._dirs and not os.path.exists(self._dirs[0]):
            os.makedirs(self._dirs[0], exist_ok=True)
        self.rescan()

        self._seed_run_finished = not self._inputs
        self._seed_idx = 0
        self._save_corpus = bool(self._dirs and os.path.isdir(self._dirs[0]))
        if not self._inputs:
            self._add_bytes(b"\x00")
            self._seed_run_finished = False

    def _add_bytes(self, data):
        data = bytes(data[: self._max_input_size])
        if len(data) == 0:
            return False
        digest = hashlib.sha256(data).digest()
        with self._lock:
            if digest in self._hashes:
                return False
            self._hashes.add(digest)
            self._inputs.append(bytearray(data))
        return True

    def _add_file(self, path):
        if not os.path.isfile(path) or ".tmp." in os.path.basename(path):
            return False
        try:
            with open(path, "rb") as f:
                data = f.read(self._max_input_size)
        except (FileNotFoundError, PermissionError, OSError):
            return False
        return self._add_bytes(data)

    def rescan(self):
        """Reconcile in-memory seeds with the shared corpus directory.

        Repeated scans are safe because corpus entries are SHA-256 deduplicated.
        Empty and temporary files are ignored.
        """
        added = 0
        for path in self._dirs:
            if os.path.isfile(path):
                added += int(self._add_file(path))
            elif os.path.isdir(path):
                try:
                    with os.scandir(path) as it:
                        for entry in it:
                            if entry.is_file() and ".tmp." not in entry.name:
                                added += int(self._add_file(entry.path))
                except (FileNotFoundError, PermissionError, OSError):
                    continue
        return added

    @property
    def length(self):
        with self._lock:
            return len(self._inputs)

    @staticmethod
    def _rand(n):
        return random.randrange(n) if n > 1 else 0

    @staticmethod
    def _rand_exp():
        value = random.getrandbits(32)
        count = 0
        while count < 32 and (value & (1 << (31 - count))) == 0:
            count += 1
        return count

    @staticmethod
    def _choose_len(n):
        if n <= 0:
            return 0
        x = Corpus._rand(100)
        if x < 90:
            return Corpus._rand(min(8, n)) + 1
        if x < 99:
            return Corpus._rand(min(32, n)) + 1
        return Corpus._rand(n) + 1

    @staticmethod
    def copy(src, dst, start_source, start_dst, end_source=None, end_dst=None):
        end_source = len(src) if end_source is None else end_source
        end_dst = len(dst) if end_dst is None else end_dst
        count = min(end_source - start_source, end_dst - start_dst)
        if count > 0:
            dst[start_dst : start_dst + count] = src[start_source : start_source + count]

    def put(self, buf):
        data = bytes(buf[: self._max_input_size])
        if len(data) == 0:
            return False
        if not self._add_bytes(data):
            return False
        if self._save_corpus:
            digest = hashlib.sha256(data).hexdigest()
            final_path = os.path.join(self._dirs[0], digest)
            if not os.path.exists(final_path):
                tmp_path = f"{final_path}.tmp.{os.getpid()}.{threading.get_ident()}"
                with open(tmp_path, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, final_path)
        return True

    @staticmethod
    def seed_id(buf):
        return hashlib.sha256(bytes(buf)).hexdigest()

    def generate_input_with_parent(self):
        """Return (candidate, direct_parent_seed_id, was_mutated).

        The selection and mutation path is intentionally identical to
        generate_input(); provenance adds hashing only and does not consume RNG
        state. Initial corpus replay is marked as unmutated so it is not counted
        as peer seed reuse in Phase 5.
        """
        with self._lock:
            if not self._seed_run_finished:
                if self._seed_idx >= len(self._inputs):
                    self._seed_idx = 0
                    self._seed_run_finished = True
                else:
                    out = bytearray(self._inputs[self._seed_idx])
                    self._seed_idx += 1
                    if self._seed_idx >= len(self._inputs):
                        self._seed_idx = 0
                        self._seed_run_finished = True
                    return out, self.seed_id(out), False
            parent = bytearray(self._inputs[self._rand(len(self._inputs))])
        return self.mutate(parent), self.seed_id(parent), True

    def generate_input(self):
        candidate, _parent_seed_id, _was_mutated = self.generate_input_with_parent()
        return candidate

    def mutate(self, buf):
        res = bytearray(buf)
        nm = self._rand_exp()
        for _ in range(nm):
            x = self._rand(15)
            if x == 0:
                if len(res) <= 1:
                    continue
                pos0 = self._rand(len(res))
                n = self._choose_len(len(res) - pos0)
                del res[pos0 : pos0 + n]
            elif x == 1:
                pos = self._rand(len(res) + 1)
                n = self._choose_len(10)
                res[pos:pos] = bytes(self._rand(256) for _ in range(n))
            elif x == 2:
                if len(res) <= 1:
                    continue
                src = self._rand(len(res))
                n = self._choose_len(len(res) - src)
                dst = self._rand(len(res) + 1)
                res[dst:dst] = res[src : src + n]
            elif x == 3:
                if len(res) <= 1:
                    continue
                src = self._rand(len(res))
                dst = self._rand(len(res))
                n = self._choose_len(len(res) - src)
                chunk = bytes(res[src : src + n])
                end = min(dst + len(chunk), len(res))
                res[dst:end] = chunk[: end - dst]
            elif x == 4:
                if res:
                    pos = self._rand(len(res))
                    res[pos] ^= 1 << self._rand(8)
            elif x == 5:
                if res:
                    pos = self._rand(len(res))
                    res[pos] ^= self._rand(255) + 1
            elif x == 6:
                if len(res) > 1:
                    src = self._rand(len(res))
                    dst = self._rand(len(res))
                    if src != dst:
                        res[src], res[dst] = res[dst], res[src]
            elif x == 7:
                if res:
                    pos = self._rand(len(res))
                    res[pos] = (res[pos] + self._rand(2**8)) % 256
            elif x == 8:
                if len(res) >= 2:
                    pos = self._rand(len(res) - 1)
                    packed = struct.pack(">H" if random.getrandbits(1) else "<H", self._rand(2**16))
                    res[pos] = (res[pos] + packed[0]) % 256
                    # Fix original typo: use res[pos + 1], not the already modified res[pos].
                    res[pos + 1] = (res[pos + 1] + packed[1]) % 256
            elif x == 9:
                if len(res) >= 4:
                    pos = self._rand(len(res) - 3)
                    packed = struct.pack(">I" if random.getrandbits(1) else "<I", self._rand(2**32))
                    for i in range(4):
                        res[pos + i] = (res[pos + i] + packed[i]) % 256
            elif x == 10:
                if len(res) >= 8:
                    pos = self._rand(len(res) - 7)
                    packed = struct.pack(">Q" if random.getrandbits(1) else "<Q", self._rand(2**64))
                    for i in range(8):
                        res[pos + i] = (res[pos + i] + packed[i]) % 256
            elif x == 11:
                if res:
                    res[self._rand(len(res))] = random.choice(INTERESTING8) % 256
            elif x == 12:
                if len(res) >= 2:
                    pos = self._rand(len(res) - 1)
                    packed = struct.pack(">H" if random.getrandbits(1) else "<H", random.choice(INTERESTING16))
                    res[pos : pos + 2] = packed
            elif x == 13:
                if len(res) >= 4:
                    pos = self._rand(len(res) - 3)
                    packed = struct.pack(">I" if random.getrandbits(1) else "<I", random.choice(INTERESTING32))
                    res[pos : pos + 4] = packed
            elif x == 14:
                digits = [i for i, value in enumerate(res) if ord("0") <= value <= ord("9")]
                if digits:
                    pos = random.choice(digits)
                    old = res[pos]
                    new = old
                    while new == old:
                        new = self._rand(10) + ord("0")
                    res[pos] = new

        return res[: self._max_input_size]
