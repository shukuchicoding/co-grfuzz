import os
import random
import hashlib
import dictionnary

class Corpus:
    INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
    INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
    INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]

    def __init__(self, dirs=None, max_input_size=1024, dict_path=None):
        self._inputs = []
        self._dict = dictionnary.Dictionary(dict_path)
        self._max_input_size = max_input_size
        self._dirs = dirs or []
        
        if self._dirs:
            os.makedirs(self._dirs[0], exist_ok=True)

        for path in self._dirs:
            if os.path.isfile(path):
                self._add_file(path)
            else:
                with os.scandir(path) as it:
                    for entry in it:
                        if entry.is_file():
                            self._add_file(entry.path)

        self._seed_run_finished = not self._inputs
        self._seed_idx = 0
        self._save_corpus = bool(dirs and os.path.isdir(dirs[0]))
        # self._inputs.append(bytearray(0))
        self._inputs.append(bytearray(os.urandom(1000)))

    def _add_file(self, path):
        with open(path, 'rb') as f:
            self._inputs.append(bytearray(f.read()))

    @property
    def length(self):
        return len(self._inputs)

    @staticmethod
    def _rand(n):
        return random.randrange(n) if n > 1 else 0

    @staticmethod
    def _rand_exp():
        return (random.getrandbits(32) & -random.getrandbits(32)).bit_length()

    @staticmethod
    def _choose_len(n):
        x = Corpus._rand(100)
        return Corpus._rand(min((8, 32, n)[(x > 90) + (x > 99)], n)) + 1

    @staticmethod
    def copy(src, dst, start_source, start_dst, end_source=None, end_dst=None):
        dst[start_dst:start_dst + min((end_source or len(src)) - start_source, (end_dst or len(dst)) - start_dst)] = \
            src[start_source:end_source]

    def put(self, buf):
        self._inputs.append(buf)
        # self._seed_run_finished = False
        if self._save_corpus:
            fname = os.path.join(self._dirs[0], hashlib.sha256(buf).hexdigest())
            with open(fname, 'wb') as f:
                f.write(buf)

    def get_input(self):
        # return self._inputs[self._rand(len(self._inputs))]
        return self._inputs[len(self._inputs) - 1]

    def mutate(self, buf, position):
        position = min(position, len(buf) - 1) if len(buf) > 0 else 0
        mutated = False
        while not mutated:
            x = self._rand(16)
            if x == 0 and len(buf) > 1:
                del buf[position]
                mutated = True
            elif x == 1 and len(buf) < self._max_input_size:
                buf.insert(position, self._rand(256))
                mutated = True
            elif x == 2 and len(buf) < self._max_input_size:
                buf.insert(position, buf[position])
                mutated = True
            elif x == 3:
                buf[self._rand(len(buf))] = buf[position]
                mutated = True
            elif x == 4:
                buf[position] ^= 1 << self._rand(8)
                mutated = True
            elif x == 5:
                buf[position] = self._rand(256)
                mutated = True
            elif x == 6:
                dst = self._rand(len(buf))
                buf[position], buf[dst] = buf[dst], buf[position]
                mutated = True
            elif x == 7:
                buf[position] = (buf[position] + self._rand(256)) % 256
                mutated = True
            elif x == 8 and position < len(buf) - 1:
                v = self._rand(2**16)
                buf[position] = (buf[position] + (v & 0xFF)) % 256
                buf[position + 1] = (buf[position + 1] + ((v >> 8) & 0xFF)) % 256
                mutated = True
            elif x == 9 and position < len(buf) - 3:
                v = self._rand(2**32)
                for k in range(4):
                    buf[position + k] = (buf[position + k] + ((v >> (8 * k)) & 0xFF)) % 256
                mutated = True
            elif x == 10 and position < len(buf) - 7:
                v = self._rand(2**64)
                for k in range(8):
                    buf[position + k] = (buf[position + k] + ((v >> (8 * k)) & 0xFF)) % 256
                mutated = True
            elif x == 11:
                buf[position] = self.INTERESTING8[self._rand(len(self.INTERESTING8))] % 256
                mutated = True
            elif x == 12 and position < len(buf) - 1:
                v = random.choice(self.INTERESTING16)
                buf[position:position + 2] = v.to_bytes(2, 'little')
                mutated = True
            elif x == 13 and position < len(buf) - 3:
                v = random.choice(self.INTERESTING32)
                buf[position:position + 4] = v.to_bytes(4, 'little')
                mutated = True
            elif x == 14 and ord('0') <= buf[position] <= ord('9'):
                buf[position] = self._rand(10) + ord('0') if buf[position] != ord('9') else ord('8')
                mutated = True
            elif x == 15 and len(buf) > 2:
                del buf[position:position + 2]
                mutated = True
            elif x == 16 and len(buf) > 4:
                del buf[position:position + 4]
                mutated = True

        return buf[:self._max_input_size]
