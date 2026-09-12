"""Progress / ETA tracking for a job: a plan of steps with expected durations, adaptively rescaled."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Step:
    key: str
    label: str
    expected: float  # seconds
    done: bool = False
    elapsed: float = 0.0


@dataclass
class Tracker:
    callback: object = None  # fn(fraction 0..1, stage label, eta seconds)
    steps: list[Step] = field(default_factory=list)
    t0: float = field(default_factory=time.time)
    current: Step | None = None
    started: float = 0.0
    sub: float = 0.0

    def add(self, key: str, label: str, expected: float) -> None:
        self.steps.append(Step(key, label, max(0.5, expected)))

    def start(self, key: str) -> None:
        self._finish_current()
        self.current = next((s for s in self.steps if s.key == key and not s.done), None)
        if self.current is None:  # unplanned step: append a small one
            self.current = Step(key, key, 2.0)
            self.steps.append(self.current)
        self.started = time.time()
        self.sub = 0.0
        self._emit()

    def progress(self, fraction: float) -> None:
        """Progress inside the current step (0..1)."""
        self.sub = max(0.0, min(1.0, fraction))
        self._emit()

    def finish(self) -> None:
        self._finish_current()
        if self.callback:
            self.callback(1.0, "完了", 0.0)

    def _finish_current(self) -> None:
        if self.current is not None:
            self.current.done = True
            self.current.elapsed = time.time() - self.started
            self.current = None

    def _emit(self) -> None:
        if not self.callback:
            return
        done = [s for s in self.steps if s.done]
        # Rescale the plan by measured speed, but only from real compute steps (the download may be a
        # cache hit and finish instantly) and once enough planned work has completed to be meaningful.
        calib = [s for s in done if s.key != "download"]
        exp_done_c = sum(s.expected for s in calib)
        el_done_c = sum(s.elapsed for s in calib)
        scale = min(3.0, max(0.5, el_done_c / exp_done_c)) if exp_done_c >= 15 and el_done_c > 0 else 1.0
        exp_done = sum(s.expected for s in done)
        cur = self.current
        cur_exp = cur.expected * scale if cur else 0.0
        cur_el = time.time() - self.started if cur else 0.0
        # progress inside the current step: sub-progress if reported, else elapsed vs expected (capped at 90 %)
        cur_frac = self.sub if self.sub > 0 else min(0.9, cur_el / cur_exp) if cur_exp else 0.0
        remaining_exp = sum(s.expected for s in self.steps if not s.done and s is not cur) * scale
        total = sum(s.expected for s in self.steps) * scale
        completed = (exp_done * scale) + cur_exp * cur_frac
        frac = min(0.99, completed / total) if total else 0.0
        eta = remaining_exp + max(0.0, cur_exp * (1 - cur_frac) if self.sub > 0 else cur_exp - cur_el)
        self.callback(frac, cur.label if cur else "", eta)
