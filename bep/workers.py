"""One persistent process pool for every activity of the loop.

Workers are spawned once per run; each task names the function to run (``"module:function"``) and its
argument.  Workers keep caches between tasks (``CACHE``): primitives, libraries, worlds, keyed by what they
were built from, so an explore block of a schema does not rebuild its primitive at every proposal.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys

CACHE: dict = {}


def _init(scene: dict) -> None:
    from .world.objects import use_body
    use_body("float")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # torch is not imported here (~180 MB per worker): workers evaluate the map with numpy (qmap.FrozenNet); if a
    # task ever imports it, it reads OMP_NUM_THREADS above and runs on one thread
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    CACHE.clear()
    CACHE["scene"] = scene


def trim() -> None:
    """Hand freed memory back to the system (glibc keeps it otherwise: a worker that rebuilt large fields stays
    at its peak).  No-op where unavailable."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def call(task):
    """Run ``task = (\"module:function\", argument)`` in a worker."""
    name, arg = task
    mod, fn = name.split(":")
    return getattr(importlib.import_module(mod), fn)(arg)


class Pool:
    """A spawn pool whose workers know the scene (kinds, pairs)."""

    def __init__(self, scene: dict, workers: int | None = None):
        self.n = workers or os.cpu_count() or 1
        self.scene = scene
        self.pool = mp.get_context("spawn").Pool(self.n, initializer=_init, initargs=(scene,))

    def restart(self) -> None:
        """Fresh workers (their caches and whatever their heaps accumulated are released).  Called at sleep, after
        which the workers rebuild their primitives and libraries anyway (the relations are rewritten)."""
        self.pool.terminate()
        self.pool.join()
        self.pool = mp.get_context("spawn").Pool(self.n, initializer=_init, initargs=(self.scene,))

    def map(self, name: str, args: list) -> list:
        return self.pool.map(call, [(name, a) for a in args], chunksize=1)

    def map_async(self, name: str, args: list):
        """``map`` without waiting: ``.get()`` the results."""
        return self.pool.map_async(call, [(name, a) for a in args], chunksize=1)

    def imap_unordered(self, name: str, args: list):
        return self.pool.imap_unordered(call, [(name, a) for a in args], chunksize=1)

    def apply_async(self, name: str, arg, callback, error_callback):
        return self.pool.apply_async(call, ((name, arg),), callback=callback, error_callback=error_callback)

    def close(self) -> None:
        self.pool.terminate()
