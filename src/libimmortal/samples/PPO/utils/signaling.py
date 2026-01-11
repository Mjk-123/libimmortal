# src/libimmortal/samples/PPO/utils/signaling.py
from __future__ import annotations

import os
import contextlib
import signal
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass
class GracefulStop:
    """
    Installs SIGINT/SIGTERM handlers that DO NOT raise KeyboardInterrupt.
    They only set flags so the training loop can exit cleanly.

    - First Ctrl+C: stop_requested=True
    - Second Ctrl+C: force_requested=True (you can choose to bail out faster)
    """
    stop_requested: bool = False
    force_requested: bool = False
    _installed: bool = False
    _sig_count: int = 0
    _old_handlers: Optional[Dict[int, Callable]] = None

    def install(self) -> None:
        if self._installed:
            return

        self._old_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }

        def _handler(signum, frame):
            self._sig_count += 1
            if self._sig_count <= 1:
                self.stop_requested = True
            else:
                # Second signal: request a faster exit.
                self.force_requested = True
                self.stop_requested = True
                os._exit(130)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

        # Avoid some syscalls raising InterruptedError due to signals (best-effort).
        try:
            signal.siginterrupt(signal.SIGINT, False)
            signal.siginterrupt(signal.SIGTERM, False)
        except Exception:
            pass

        self._installed = True

    def restore(self) -> None:
        if not self._installed or not self._old_handlers:
            return
        for sig, h in self._old_handlers.items():
            try:
                signal.signal(sig, h)
            except Exception:
                pass
        self._installed = False

    @contextlib.contextmanager
    def ignore_signals(self):
        """
        Temporarily ignore SIGINT/SIGTERM (critical sections like checkpoint save).
        """
        old_int = signal.getsignal(signal.SIGINT)
        old_term = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            yield
        finally:
            try:
                signal.signal(signal.SIGINT, old_int)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGTERM, old_term)
            except Exception:
                pass

    def should_stop(self) -> bool:
        return bool(self.stop_requested)

    def sleep_poll(self, seconds: float, poll_hz: float = 20.0) -> None:
        """
        Sleep while still responsive to stop flags (useful for backoff loops).
        """
        if seconds <= 0:
            return
        dt = 1.0 / max(1.0, float(poll_hz))
        t_end = time.time() + float(seconds)
        while time.time() < t_end:
            if self.should_stop():
                return
            time.sleep(min(dt, t_end - time.time()))
