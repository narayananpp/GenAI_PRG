"""
Logging utilities, matching the interface of guided_diffusion/logger.py.
"""

import os
import sys
import datetime

_kvs     = {}
_writers = []


def configure(dir=None, format_strs=None):
    if dir is None:
        dir = os.path.join(
            "logs",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    os.makedirs(dir, exist_ok=True)

    if format_strs is None:
        format_strs = ["stdout", "log"]

    global _writers
    _writers = []
    for fmt in format_strs:
        if fmt == "stdout":
            _writers.append(HumanOutputFormat(sys.stdout))
        elif fmt == "log":
            _writers.append(HumanOutputFormat(open(os.path.join(dir, "log.txt"), "a")))
        elif fmt == "csv":
            _writers.append(CSVOutputFormat(os.path.join(dir, "progress.csv")))


def log(*args):
    msg = " ".join(str(a) for a in args)
    for w in _writers:
        w.write_message(msg)
    if not _writers:
        print(msg)


def logkv(key, val):
    _kvs[key] = val


def dumpkvs():
    for w in _writers:
        w.write_kvs(dict(_kvs))
    _kvs.clear()


def get_dir():
    for w in _writers:
        if hasattr(w, "dir"):
            return w.dir
    return None


class HumanOutputFormat:
    def __init__(self, file_or_path):
        if isinstance(file_or_path, str):
            self.file = open(file_or_path, "a")
        else:
            self.file = file_or_path

    def write_message(self, msg):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.file.write(f"{now}  {msg}\n")
        self.file.flush()

    def write_kvs(self, kvs):
        parts = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in sorted(kvs.items()))
        self.write_message(parts)


class CSVOutputFormat:
    def __init__(self, path):
        self.path = path
        self._keys = []
        self.file  = open(path, "a")

    def write_message(self, msg):
        pass  # CSV only logs kvs

    def write_kvs(self, kvs):
        extra = [k for k in kvs if k not in self._keys]
        if extra:
            self._keys.extend(extra)
            self.file.seek(0)
            self.file.truncate()
            self.file.write(",".join(self._keys) + "\n")
        self.file.write(",".join(str(kvs.get(k, "")) for k in self._keys) + "\n")
        self.file.flush()
