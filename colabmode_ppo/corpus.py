import hashlib
import os
import random
import threading

import dictionnary


class Corpus:
    INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
    INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
    INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]

    def __init__(self, dirs=None, max_input_size=1024, dict_path=None):
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

        self._save_corpus = bool(self._dirs and os.path.isdir(self._dirs[0]))
        if not self._inputs:
            self._add_bytes(b"\x00")

    @staticmethod
    def _rand(n):
        return random.randrange(n) if n > 1 else 0

    def _add_bytes(self, data):
        data = bytes(data[: self._max_input_size])
        if not data:
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
        """Import seeds currently present in shared corpus paths.

        Watchdog is only a notification mechanism; this reconciliation closes
        the initial scan/watch race and recovers from missed filesystem events.
        SHA-256 corpus deduplication makes repeated rescans idempotent.
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

    def put(self, buf):
        data = bytes(buf[: self._max_input_size])
        if not data or not self._add_bytes(data):
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

    def get_input(self):
        with self._lock:
            return bytearray(self._inputs[self._rand(len(self._inputs))])

    def valid_mutation_types(self, buf, position):
        """Return mutation operators valid at ``position`` for ``buf``."""
        res = bytearray(buf[: self._max_input_size])
        if not res:
            res = bytearray(b"\x00")
        position = min(max(int(position), 0), len(res) - 1)
        n = len(res)

        # Operators 3..7 and 11 are always valid for non-empty seeds.
        ops = [3, 4, 5, 6, 7, 11]
        if n > 1:
            ops.append(0)
        if n < self._max_input_size:
            ops.extend([1, 2])
        if position + 2 <= n:
            ops.extend([8, 12])
        if position + 4 <= n:
            ops.extend([9, 13])
        if position + 8 <= n:
            ops.append(10)
        if ord("0") <= res[position] <= ord("9"):
            ops.append(14)
        if position + 2 <= n and n - 2 >= 1:
            ops.append(15)
        if position + 4 <= n and n - 4 >= 1:
            ops.append(16)
        return sorted(set(ops))

    def mutate(self, buf, mutation_type, position=0):
        """Return an independent mutant of ``buf``.

        The corpus-owned seed is never modified in place. Deletion operators are
        allowed only when the complete requested width fits and at least one byte
        remains afterward.
        """
        res = bytearray(buf[: self._max_input_size])
        if not res:
            res = bytearray(b"\x00")
        position = min(max(int(position), 0), len(res) - 1)
        mutation_type = int(mutation_type)
        if not 0 <= mutation_type <= 16:
            raise ValueError("mutation_type must be in [0, 16]")
        if mutation_type not in self.valid_mutation_types(res, position):
            return bytearray(res)

        if mutation_type == 0:
            del res[position]
        elif mutation_type == 1:
            res.insert(position, self._rand(256))
        elif mutation_type == 2:
            res.insert(position, res[position])
        elif mutation_type == 3:
            dst = self._rand(len(res))
            if len(res) > 1 and dst == position:
                dst = (dst + 1) % len(res)
            old = res[dst]
            res[dst] = res[position]
            if res[dst] == old:
                res[dst] ^= 0x01
        elif mutation_type == 4:
            res[position] ^= 1 << self._rand(8)
        elif mutation_type == 5:
            old = res[position]
            value = self._rand(256)
            if value == old:
                value = (value + 1) % 256
            res[position] = value
        elif mutation_type == 6:
            if len(res) == 1:
                res[0] ^= 0x01
            else:
                dst = self._rand(len(res))
                if dst == position:
                    dst = (dst + 1) % len(res)
                res[position], res[dst] = res[dst], res[position]
                if res[position] == res[dst]:
                    res[position] ^= 0x01
        elif mutation_type == 7:
            delta = self._rand(255) + 1
            res[position] = (res[position] + delta) % 256
        elif mutation_type == 8:
            v = self._rand((2**16) - 1) + 1
            res[position] = (res[position] + (v & 0xFF)) % 256
            res[position + 1] = (res[position + 1] + ((v >> 8) & 0xFF)) % 256
        elif mutation_type == 9:
            v = self._rand((2**32) - 1) + 1
            for k in range(4):
                res[position + k] = (res[position + k] + ((v >> (8 * k)) & 0xFF)) % 256
        elif mutation_type == 10:
            v = self._rand((2**64) - 1) + 1
            for k in range(8):
                res[position + k] = (res[position + k] + ((v >> (8 * k)) & 0xFF)) % 256
        elif mutation_type == 11:
            value = self.INTERESTING8[self._rand(len(self.INTERESTING8))] % 256
            if value == res[position]:
                value = (value + 1) % 256
            res[position] = value
        elif mutation_type == 12:
            current = int.from_bytes(res[position : position + 2], "little")
            choices = [v for v in self.INTERESTING16 if v != current]
            value = random.choice(choices or self.INTERESTING16)
            res[position : position + 2] = value.to_bytes(2, "little")
        elif mutation_type == 13:
            current = int.from_bytes(res[position : position + 4], "little")
            choices = [v for v in self.INTERESTING32 if v != current]
            value = random.choice(choices or self.INTERESTING32)
            res[position : position + 4] = value.to_bytes(4, "little")
        elif mutation_type == 14:
            old = res[position]
            value = self._rand(10) + ord("0")
            if value == old:
                value = ord("0") + ((value - ord("0") + 1) % 10)
            res[position] = value
        elif mutation_type == 15:
            del res[position : position + 2]
        elif mutation_type == 16:
            del res[position : position + 4]

        if len(res) > self._max_input_size:
            res = res[: self._max_input_size]
        if not res:
            res = bytearray(b"\x00")
        return bytearray(res)
