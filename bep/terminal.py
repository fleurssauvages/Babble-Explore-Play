"""Progress on interactive terminals, with room for ordinary log lines."""
from __future__ import annotations

import sys
import time


def run_main(main) -> None:
    """Run a command's ``main`` and exit quietly when its output stream has gone away (a pipe whose reader closed,
    some terminal wrappers): Python would otherwise report a BrokenPipeError while flushing at exit.  Exit code 0
    when ``main`` finished, 1 when the broken stream interrupted it."""
    import os
    finished = False
    try:
        main()
        finished = True
        sys.stdout.flush()
        return
    except BrokenPipeError:
        pass
    try:                                   # what is left to flush goes nowhere: no error at interpreter exit
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        pass
    sys.exit(0 if finished else 1)


def ascii_bar(fraction: float, width: int = 16) -> str:
    filled = int(width * min(1.0, max(0.0, fraction)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class ProgressBar:
    def __init__(self, label: str, total: float, unit: str) -> None:
        self.stream = sys.stderr
        self.enabled = self.stream.isatty()
        self.label, self.total, self.unit = label, total, unit
        self.value, self.phase = 0.0, None
        self.line, self.last_draw = "", 0.0

    def __enter__(self) -> ProgressBar:
        return self

    def update(self, value: float, status: str) -> None:
        if not self.enabled:
            return
        self.value = max(0.0, min(value, self.total))
        self.update_phase(self.value, self.total, self.unit, status, decimals=1)

    def update_phase(self, value: float, total: float, unit: str, status: str, *, decimals: int = 0) -> None:
        """Display a phase with its own budget while retaining the main progress value."""
        if not self.enabled:
            return
        value = max(0.0, min(value, total))
        now = time.monotonic()
        phase = (status, total, unit)
        if self.line and phase == self.phase and 0 < value < total and now - self.last_draw < 0.2:
            return
        self.phase = phase
        bar = ascii_bar(value / total if total > 0 else 1.0)
        line = f"{self.label} {bar} {value:.{decimals}f}/{total:g} {unit} {status}"
        if line != self.line:
            print(f"\r\x1b[2K{line}", end="", file=self.stream, flush=True)
            self.line, self.last_draw = line, now

    def clear(self) -> None:
        if self.line:
            print("\r\x1b[2K", end="", file=self.stream, flush=True)
            self.line = ""

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.line:
            print(file=self.stream, flush=True)
