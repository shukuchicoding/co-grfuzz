import os
import math
import random
import struct
import hashlib

import dictionnary

try:
    from random import _randbelow
except ImportError:
    from random import _inst
    _randbelow = _inst._randbelow

INTERESTING8 = [-128, -1, 0, 1, 16, 32, 64, 100, 127]
INTERESTING16 = [0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 65535]
INTERESTING32 = [0, 1, 32768, 65535, 65536, 100663045, 2147483647, 4294967295]


class Corpus(object):
    def __init__(self, dirs=None, max_input_size=4096, dict_path=None):
        self._inputs = []
        self._dict = dictionnary.Dictionary(dict_path)
        self._max_input_size = max_input_size
        self._dirs = dirs if dirs else []
        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                os.mkdir(path)

            if os.path.isfile(path):
                self._add_file(path)
            else:
                for i in os.listdir(path):
                    fname = os.path.join(path, i)
                    if os.path.isfile(fname):
                        self._add_file(fname)
        self._seed_run_finished = not self._inputs
        self._seed_idx = 0
        self._save_corpus = dirs and os.path.isdir(dirs[0])
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
        if n < 2:
            return 0
        return _randbelow(n)

    # Exp2 generates n with probability 1/2^(n+1).
    @staticmethod
    def _rand_exp():
        rand_bin = bin(random.randint(0, 2**32-1))[2:]
        rand_bin = '0'*(32 - len(rand_bin)) + rand_bin
        count = 0
        for i in rand_bin:
            if i == '0':
                count +=1
            else:
                break
        return count

    @staticmethod
    def _choose_len(n):
        x = Corpus._rand(100)
        if x < 90:
            return Corpus._rand(min(8, n)) + 1
        elif x < 99:
            return Corpus._rand(min(32, n)) + 1
        else:
            return Corpus._rand(n) + 1

    @staticmethod
    def copy(src, dst, start_source, start_dst, end_source=None, end_dst=None):
        end_source = len(src) if end_source is None else end_source
        end_dst = len(dst) if end_dst is None else end_dst
        byte_to_copy = min(end_source-start_source, end_dst-start_dst)
        dst[start_dst:start_dst+byte_to_copy] = src[start_source:start_source+byte_to_copy]

    def put(self, buf):
        self._inputs.append(buf)
        if self._save_corpus:
            m = hashlib.sha256()
            m.update(buf)
            fname = os.path.join(self._dirs[0], m.hexdigest())
            with open(fname, 'wb') as f:
                f.write(buf)

    def generate_input(self):
        if not self._seed_run_finished:
            next_input = self._inputs[self._seed_idx]
            self._seed_idx += 1
            if self._seed_idx >= len(self._inputs):
                self._seed_run_finished = True
            return next_input

        buf = self._inputs[self._rand(len(self._inputs))]
        return self.mutate(buf)

    def get_input(self):
        return self._inputs[self._rand(len(self._inputs))]    

    def extract_substring(self, buf, k):

        if not buf:
            return bytearray([0] * k)

        if len(buf) >= k:
            start = Corpus._rand(len(buf) - k + 1)
            return buf[start:start + k]
        else:
            padded = bytearray(buf)
            padded.extend([0] * (k - len(buf)))
            return padded

    def mutate(self, origin_buf, window, type):
        res = origin_buf[:]
        x = window

        if type == 1:
            # Lật bit ngẫu nhiên với tỉ lệ ~1%
            num_bits = max(1, len(x) * 8 // 100)
            for _ in range(num_bits):
                byte_idx = self._rand(len(x))
                bit_idx = self._rand(8)
                x[byte_idx] ^= 1 << bit_idx

        # elif type == 2:
        #     # Chèn token từ từ điển vào vị trí ngẫu nhiên trong x
        #     if self._dict.tokens:
        #         token = random.choice(self._dict.tokens)
        #         insert_pos = self._rand(len(x) + 1)
        #         x = x[:insert_pos] + bytearray(token) + x[insert_pos:]

        elif type == 2:
            # Dịch đoạn x trong origin_buf sang vị trí mới
            insert_pos = self._rand(len(res) + 1)
            x_len = len(x)
            # Xóa đoạn x khỏi gốc nếu có mặt
            idx = res.find(x)
            if idx != -1:
                res = res[:idx] + res[idx + x_len:]
            # Chèn lại tại vị trí mới
            res = res[:insert_pos] + x + res[insert_pos:]

        elif type == 3:
            # Xáo trộn các bytes trong x
            tmp = list(x)
            random.shuffle(tmp)
            x = bytearray(tmp)

        elif type == 4:
            # Sao chép x vào một vị trí mới trong res (chèn hoặc ghi đè)
            insert_pos = self._rand(len(res) + 1)
            if self._rand(2):
                # Ghi đè
                end_pos = min(insert_pos + len(x), len(res))
                self.copy(x, res, 0, insert_pos, len(x), end_pos)
            else:
                # Chèn thêm
                res = res[:insert_pos] + x + res[insert_pos:]

        elif type == 5:
            # Xóa đoạn x khỏi res (chỉ xóa đoạn đầu tiên khớp x)
            idx = res.find(x)
            if idx != -1:
                res = res[:idx] + res[idx + len(x):]

        if len(res) > self._max_input_size:
            res = res[:self._max_input_size]
        return res