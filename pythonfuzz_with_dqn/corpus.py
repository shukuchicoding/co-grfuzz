import hashlib
import os
import random
import threading

import dictionnary


class Corpus:
    """Corpus and mutation engine for the DQN baseline.

    DQN observes a fixed-width substring (32 bytes in the paper configuration)
    together with its exact offset in the seed. Mutation actions are applied to
    that exact region rather than locating a duplicate byte pattern with find().
    """

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

        for path in self._dirs:
            if os.path.isfile(path):
                self._add_file(path)
            elif os.path.isdir(path):
                with os.scandir(path) as it:
                    for entry in it:
                        if entry.is_file() and ".tmp." not in entry.name:
                            self._add_file(entry.path)

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

    def extract_substring(self, buf, width):
        """Return ``(window, offset, valid_len)`` from a non-empty seed.

        The returned window contains only real seed bytes; the neural-state
        padding is handled by dqn_core.make_state(). Keeping the offset avoids
        the ambiguous ``res.find(window)`` behavior of the old implementation.
        """
        width = int(width)
        if width < 1:
            raise ValueError("substring width must be >= 1")
        seed = bytes(buf[: self._max_input_size])
        if not seed:
            seed = b"\x00"
        valid_len = min(width, len(seed))
        max_start = len(seed) - valid_len
        offset = self._rand(max_start + 1)
        return bytearray(seed[offset : offset + valid_len]), int(offset), int(valid_len)

    @staticmethod
    def _replace_slice(res, offset, old_len, replacement):
        offset = min(max(int(offset), 0), len(res))
        old_len = max(int(old_len), 0)
        end = min(offset + old_len, len(res))
        return res[:offset] + bytearray(replacement) + res[end:]

    def mutate(self, origin_buf, window, offset, action):
        """Apply one of five DQN mutation actions to the observed window.

        Actions are 0-based:
          0 random bit flips in the observed window
          1 move the observed window to another seed offset
          2 shuffle bytes inside the observed window
          3 copy the observed window (insert or overwrite)
          4 delete the observed window while preserving at least one byte
        """
        res = bytearray(origin_buf[: self._max_input_size])
        if not res:
            res = bytearray(b"\x00")

        action = int(action)
        if not 0 <= action < 5:
            raise ValueError("DQN action must be in [0, 4]")

        offset = min(max(int(offset), 0), len(res) - 1)
        valid_len = min(max(len(window), 1), len(res) - offset)
        selected = bytearray(res[offset : offset + valid_len])
        if not selected:
            selected = bytearray([res[offset]])
            valid_len = 1

        if action == 0:
            # Random bit flips (~1% of bits), written back to the exact window.
            num_bits = max(1, len(selected) * 8 // 100)
            mutated = bytearray(selected)
            for _ in range(num_bits):
                byte_idx = self._rand(len(mutated))
                bit_idx = self._rand(8)
                mutated[byte_idx] ^= 1 << bit_idx
            res[offset : offset + valid_len] = mutated

        elif action == 1:
            # Move the exact selected window. If the seed is exactly one window,
            # there is no distinct location; rotate one byte so the action still
            # changes the input without altering length.
            remainder = res[:offset] + res[offset + valid_len :]
            if remainder:
                insert_pos = self._rand(len(remainder) + 1)
                # Avoid reconstructing the identical layout when alternatives exist.
                original_pos = min(offset, len(remainder))
                if len(remainder) + 1 > 1 and insert_pos == original_pos:
                    insert_pos = (insert_pos + 1) % (len(remainder) + 1)
                res = remainder[:insert_pos] + selected + remainder[insert_pos:]
            elif len(selected) > 1:
                res = selected[1:] + selected[:1]
            else:
                res[0] ^= 0x01

        elif action == 2:
            mutated = bytearray(selected)
            if len(mutated) > 1:
                before = bytes(mutated)
                random.shuffle(mutated)
                if bytes(mutated) == before:
                    # Deterministic fallback avoids accidental no-op for distinct
                    # adjacent bytes while remaining a window shuffle.
                    swap_idx = next((i for i in range(1, len(mutated)) if mutated[i] != mutated[0]), None)
                    if swap_idx is not None:
                        mutated[0], mutated[swap_idx] = mutated[swap_idx], mutated[0]
            else:
                mutated[0] ^= 0x01
            res[offset : offset + valid_len] = mutated

        elif action == 3:
            # Copy window: insert or overwrite. Bound output length afterwards.
            insert_pos = self._rand(len(res) + 1)
            if self._rand(2) == 0 and len(res) < self._max_input_size:
                res = res[:insert_pos] + selected + res[insert_pos:]
            else:
                if insert_pos >= len(res):
                    insert_pos = max(0, len(res) - len(selected))
                end = min(insert_pos + len(selected), len(res))
                res[insert_pos:end] = selected[: end - insert_pos]
                # If overwrite was a no-op at the same location, force a small
                # semantically local change rather than returning an unchanged seed.
                if res == bytearray(origin_buf[: self._max_input_size]):
                    res[insert_pos] ^= 0x01

        elif action == 4:
            # Delete at most len(seed)-1 bytes so target input is never empty.
            if len(res) <= 1:
                res[0] ^= 0x01
            else:
                delete_len = min(valid_len, len(res) - 1)
                # Clamp offset so the full intended delete fits.
                delete_offset = min(offset, len(res) - delete_len)
                del res[delete_offset : delete_offset + delete_len]

        if len(res) > self._max_input_size:
            res = res[: self._max_input_size]
        if not res:
            res = bytearray(b"\x00")
        return bytearray(res)
