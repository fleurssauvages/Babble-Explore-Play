"""The arbiter: which activity gets the next block of the epoch, by learning progress.

Every activity reports a competence in [0, 1] after each of its blocks (babble: 1 - the share of new event
keys; explore k: the schema's executor success; play: 1 - prequential log-loss / log 2; imagine: agreement of
imagined and real outcomes).  Its learning progress is the absolute change of competence between its last
ARBITER['window'] blocks and the window before (Oudeyer, Kaplan & Hafner 2007: progress, not error, drives
curiosity; absolute, so that a competence that drops also attracts attention).  An activity tried fewer than
2 x window times has an optimistic progress.  The choice is a softmax over progress mixed with a floor, so
that every activity is revisited.

An activity that cannot run for now (``na``: e.g. nothing to
imagine yet) is set aside until the next sleep instead of being scored 0.
"""
from __future__ import annotations

import numpy as np

from .config import ARBITER


class Arbiter:
    def __init__(self, rng):
        self.rng = rng
        self.hist: dict[str, list[float]] = {}
        self.blocked: set = set()

    def set_activities(self, names) -> None:
        names = [n for n in names if n not in self.blocked] or list(names)
        for n in names:
            self.hist.setdefault(n, [])
        for n in [n for n in self.hist if n not in names]:
            del self.hist[n]

    def progress(self, name: str) -> float:
        h, w = self.hist[name], ARBITER["window"]
        if len(h) < 2 * w:
            return ARBITER["optimism"]
        return abs(float(np.mean(h[-w:])) - float(np.mean(h[-2 * w:-w])))

    def probs(self) -> dict[str, float]:
        names = list(self.hist)
        lp = np.array([self.progress(n) for n in names])
        z = np.exp((lp - lp.max()) / ARBITER["temperature"])
        f = min(ARBITER["floor"], 1.0 / len(names))
        p = f + (1 - f * len(names)) * z / z.sum()
        return dict(zip(names, p))

    def pick(self) -> str:
        p = self.probs()
        names = list(p)
        return names[int(self.rng.choice(len(names), p=np.array([p[n] for n in names])))]

    def update(self, name: str, competence: float | None) -> None:
        if competence is None:                         # not applicable for now: set aside until the next sleep
            self.blocked.add(name)
            return
        h = self.hist.setdefault(name, [])
        h.append(float(competence))
        del h[:-2 * ARBITER["window"]]                 # progress reads only the last two windows

    def wake(self) -> None:
        """After a sleep every activity may apply again."""
        self.blocked.clear()

    def summary(self) -> dict:
        return {n: {"progress": round(self.progress(n), 3), "p": round(p, 3),
                    "competence": round(self.hist[n][-1], 3) if self.hist[n] else None}
                for n, p in self.probs().items()}
