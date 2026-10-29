"""Synthetic CSI generator for development, demos and training.

Produces realistic-looking CSI frame streams for several scenarios so the
whole pipeline (detect -> classify) can run and be trained without any WiFi
hardware. The synthetic model is deliberately simple but captures the physical
distinctions the classifier relies on.

Scenarios
---------
  empty        : static channel + thermal noise (no mover)
  human_still  : breathing modulation, broadband, coherent across subcarriers
  human_moving : gait energy in 0.8-2.5 Hz + breathing, strong & coherent
  pet          : faster, weaker, irregular, low respiration coherence
  fan          : strong narrowband ~6 Hz, highly periodic, low-dim
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .csi import CSIFrame

SCENARIOS = ("empty", "human_still", "human_moving", "pet", "fan")


class SimulatedSource:
    def __init__(
        self,
        scenario: str = "human_moving",
        n_subcarriers: int = 52,
        sample_rate: float = 20.0,
        duration: float | None = None,
        realtime: bool = False,
        seed: int | None = None,
        base_seed: int | None = None,
        warmup: float = 0.0,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; pick from {SCENARIOS}")
        self.scenario = scenario
        # Seconds of empty room emitted before the mover appears, so a live
        # detector can self-calibrate its noise floor first (as in the field).
        self.warmup = warmup
        self.n_sc = n_subcarriers
        self.fs = sample_rate
        self.duration = duration
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        # The static channel models the physical room+link geometry. It is held
        # fixed by `base_seed` so that an 'empty' priming run and a later
        # scenario run share the same environment (only the mover changes).
        self._base = self._static_channel(np.random.default_rng(base_seed))

    def _static_channel(self, rng: np.random.Generator) -> np.ndarray:
        """A stable multipath channel the movement will perturb."""
        mag = 1.0 + 0.3 * rng.standard_normal(self.n_sc)
        mag = np.clip(mag, 0.2, None)
        phase = rng.uniform(-np.pi, np.pi, self.n_sc)
        return (mag * np.exp(1j * phase)).astype(np.complex64)

    def __iter__(self) -> Iterator[CSIFrame]:
        t0 = time.time()
        k = 0
        dt = 1.0 / self.fs
        while True:
            t = k * dt
            if self.duration is not None and t > self.duration:
                return
            frame = self._frame(t)
            yield CSIFrame(timestamp=t0 + t, csi=frame)
            k += 1
            if self.realtime:
                time.sleep(dt)

    def _frame(self, t: float) -> np.ndarray:
        base = self._base.copy()
        amp = np.abs(base)
        sc = np.arange(self.n_sc)

        # Thermal noise, present in every scenario.
        noise = 0.01 * (self.rng.standard_normal(self.n_sc) + 1j * self.rng.standard_normal(self.n_sc))

        mod = np.zeros(self.n_sc)

        # During warmup the room is empty regardless of scenario.
        scenario = "empty" if t < self.warmup else self.scenario

        if scenario == "empty":
            pass

        elif scenario == "human_still":
            breath = 0.05 * np.sin(2 * np.pi * 0.28 * t)          # ~17 breaths/min
            broad = 0.012 * np.sin(2 * np.pi * 0.9 * t + sc * 0.1)
            coh = np.cos(sc * 0.05)                                # coherent across sc
            mod = (breath + broad) * coh

        elif scenario == "human_moving":
            gait = 0.12 * np.sin(2 * np.pi * 1.8 * t)             # walking cadence
            sway = 0.06 * np.sin(2 * np.pi * 0.7 * t)
            breath = 0.03 * np.sin(2 * np.pi * 0.25 * t)
            coh = np.cos(sc * 0.04)                                # broadband body
            mod = (gait + sway + breath) * coh
            mod += 0.02 * self.rng.standard_normal(self.n_sc)     # body is messy

        elif scenario == "pet":
            # Faster, weaker, irregular; low coherence across subcarriers.
            jitter = 2.5 + 1.5 * np.sin(2 * np.pi * 0.3 * t)
            motion = 0.05 * np.sin(2 * np.pi * jitter * t + self.rng.uniform(0, 1))
            incoh = np.cos(sc * 0.4 + self.rng.uniform(0, 2 * np.pi))  # decorrelated
            mod = motion * incoh
            mod += 0.03 * self.rng.standard_normal(self.n_sc)

        elif scenario == "fan":
            # Strong, highly periodic, narrowband high frequency; low-dim.
            spin = 0.15 * np.sin(2 * np.pi * 6.0 * t)
            harmonic = 0.04 * np.sin(2 * np.pi * 12.0 * t)
            localized = np.exp(-((sc - self.n_sc * 0.4) ** 2) / (2 * (self.n_sc * 0.15) ** 2))
            mod = (spin + harmonic) * localized

        new_amp = amp * (1.0 + mod)
        frame = (new_amp * np.exp(1j * np.angle(base))).astype(np.complex64) + noise
        return frame.astype(np.complex64)


def labeled_dataset(
    windows_per_scenario: int = 60,
    seed: int = 0,
):
    """Generate a labeled feature dataset across scenarios for training.

    Returns (X, y, groups) where y is 1 for human, 0 for non-human, and groups
    is the originating scenario name (useful for inspection).
    """
    from .csi import FrameBuffer
    from .processing import clean_window
    from .features import extract_features

    human_scenarios = {"human_still", "human_moving"}
    # 'empty' has no mover to classify; we train on scenarios that contain motion.
    train_scenarios = ("human_still", "human_moving", "pet", "fan")

    X, y, groups = [], [], []
    rng = np.random.default_rng(seed)
    for scen in train_scenarios:
        made = 0
        attempt = 0
        while made < windows_per_scenario:
            src = SimulatedSource(
                scenario=scen,
                duration=5.0,
                sample_rate=20.0,
                seed=int(rng.integers(0, 1_000_000)),
            )
            buf = FrameBuffer(window_seconds=4.0, sample_rate=20.0)
            got = None
            for frame in src:
                buf.push(frame)
                if buf.ready():
                    got = buf.window()
            attempt += 1
            if got is None:
                if attempt > windows_per_scenario * 3:
                    break
                continue
            feats = extract_features(clean_window(got))
            X.append(feats)
            y.append(1 if scen in human_scenarios else 0)
            groups.append(scen)
            made += 1
    return np.array(X), np.array(y), groups
