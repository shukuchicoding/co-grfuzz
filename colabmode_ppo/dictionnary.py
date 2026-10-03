import os
import random
import re


class Dictionary:
    line_re = re.compile(r'"(.+)"$')

    def __init__(self, dict_path=None):
        if not dict_path or not os.path.exists(dict_path):
            self._dict = []
            return
        values = set()
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                line = line.lstrip()
                if line.startswith("#"):
                    continue
                word = self.line_re.search(line)
                if word:
                    values.add(word.group(1))
        self._dict = list(values)

    def get_word(self):
        return random.choice(self._dict) if self._dict else None
