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

        for path in self._dirs:
            if os.path.isfile(path):
                self._add_file(path)
            elif os.path.isdir(path):
                with os.scandir(path) as it:
                    for entry in it:
                        if entry.is_file() and ".tmp." not in entry.name:
                            self._add_file(entry.path)

        self._save_corpus = bool(self._dirs and os.path.isdir(self._dirs[0]))

        # Keep the invariant 1 <= len(seed) <= max_input_size. Empty files are
        # ignored; when no usable seed exists, use one deterministic byte.
        if not self._inputs:
            self._add_bytes(b"\x00")

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

    @property
    def length(self):
        with self._lock:
            return len(self._inputs)

    @staticmethod
    def _rand(n):
        return random.randrange(n) if n > 1 else 0

    @staticmethod
    def _rand_exp():
        return (random.getrandbits(32) & -random.getrandbits(32)).bit_length()

    @staticmethod
    def _choose_len(n):
        if n <= 0:
            return 0
        x = Corpus._rand(100)
        return Corpus._rand(min((8, 32, n)[(x > 90) + (x > 99)], n)) + 1

    @staticmethod
    def copy(src, dst, start_source, start_dst, end_source=None, end_dst=None):
        dst[start_dst : start_dst + min((end_source or len(src)) - start_source, (end_dst or len(dst)) - start_dst)] = src[
            start_source:end_source
        ]

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

    def get_input(self):
        with self._lock:
            # __init__ guarantees at least one non-empty seed.
            return bytearray(self._inputs[self._rand(len(self._inputs))])

    def _valid_mutation_ops(self, res, position):
        """Return operators that are valid for this seed/position.

        In particular, every deletion preserves at least one input byte and
        multi-byte operations are enabled only when their entire width fits from
        the selected position. This avoids both empty inputs and accidental
        short deletes near the end of a seed.
        """
        length = len(res)
        ops = [3, 4, 5, 6, 7, 11]  # Always safe for a non-empty seed.

        if length > 1:
            ops.append(0)  # delete one byte, leaves at least one byte
        if length < self._max_input_size:
            ops.extend([1, 2])  # insert / duplicate

        if position + 2 <= length:
            ops.extend([8, 12])  # uint16 add / interesting16
        if position + 4 <= length:
            ops.extend([9, 13])  # uint32 add / interesting32
        if position + 8 <= length:
            ops.append(10)  # uint64 add
        if ord("0") <= res[position] <= ord("9"):
            ops.append(14)

        # Exact-width bulk deletes. The resulting seed must remain non-empty.
        if position + 2 <= length and length - 2 >= 1:
            ops.append(15)
        if position + 4 <= length and length - 4 >= 1:
            ops.append(16)

        return ops

    def mutate(self, buf, position):
        # Every candidate in a GRPO group is derived independently from the same
        # original seed. Never mutate a corpus-owned bytearray in place.
        res = bytearray(buf[: self._max_input_size])
        if not res:
            # Defensive fallback; normal corpus paths reject empty seeds.
            res = bytearray(b"\x00")

        position = min(max(int(position), 0), len(res) - 1)
        valid_ops = self._valid_mutation_ops(res, position)
        # There are always several safe operators for any non-empty seed, so no
        # retry loop is needed and mutation cannot spin forever.
        x = random.choice(valid_ops)

        if x == 0:
            del res[position]
        elif x == 1:
            res.insert(position, self._rand(256))
        elif x == 2:
            res.insert(position, res[position])
        elif x == 3:
            res[self._rand(len(res))] = res[position]
        elif x == 4:
            res[position] ^= 1 << self._rand(8)
        elif x == 5:
            res[position] = self._rand(256)
        elif x == 6:
            dst = self._rand(len(res))
            res[position], res[dst] = res[dst], res[position]
        elif x == 7:
            res[position] = (res[position] + self._rand(256)) % 256
        elif x == 8:
            v = self._rand(2**16)
            res[position] = (res[position] + (v & 0xFF)) % 256
            res[position + 1] = (res[position + 1] + ((v >> 8) & 0xFF)) % 256
        elif x == 9:
            v = self._rand(2**32)
            for k in range(4):
                res[position + k] = (res[position + k] + ((v >> (8 * k)) & 0xFF)) % 256
        elif x == 10:
            v = self._rand(2**64)
            for k in range(8):
                res[position + k] = (res[position + k] + ((v >> (8 * k)) & 0xFF)) % 256
        elif x == 11:
            res[position] = self.INTERESTING8[self._rand(len(self.INTERESTING8))] % 256
        elif x == 12:
            v = random.choice(self.INTERESTING16)
            res[position : position + 2] = v.to_bytes(2, "little")
        elif x == 13:
            v = random.choice(self.INTERESTING32)
            res[position : position + 4] = v.to_bytes(4, "little")
        elif x == 14:
            res[position] = self._rand(10) + ord("0") if res[position] != ord("9") else ord("8")
        elif x == 15:
            del res[position : position + 2]
        elif x == 16:
            del res[position : position + 4]
        else:
            raise RuntimeError("unreachable mutation operator")

        # Strong postcondition for every mutation used by the fuzzer.
        if len(res) == 0:
            # This should be unreachable because deletion operators are guarded,
            # but keep the runtime invariant even if an operator changes later.
            res = bytearray(b"\x00")
        return res[: self._max_input_size]
