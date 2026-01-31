# a/src/libimmortal/samples/PPO/utils/ppolog.py

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, TextIO


class PPOFileLogger:
    """
    Append-only file logger (rank0 only recommended).
    - Writes line-buffered.
    - Prefixes each line with timestamp.
    """

    def __init__(self, path: str, also_stdout: bool = False):
        self.path = str(path)
        self.also_stdout = bool(also_stdout)

        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

        self._f: TextIO = open(self.path, "a", buffering=1, encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = "" if msg is None else str(msg)
        for line in text.splitlines() or [""]:
            out = f"[{ts}] {line}"
            self._f.write(out + "\n")
            if self.also_stdout:
                print(out, flush=True)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
