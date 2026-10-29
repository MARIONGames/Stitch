"""Turn a CSI window into a clean motion signal and a motion decision.

Pipeline per window:
  1. Drop dead / saturated subcarriers.
  2. Per-subcarrier: remove the static (mean) component -> keep the dynamic
     part caused by movement in the channel.
  3. Band-pass filter to the frequencies human/pet/mechanical motion lives in
     (removes DC drift and out-of-band noise).
  4. Collapse subcarriers into one aggregate motion waveform.
  5. Compare the window's motion energy against a running noise floor to decide
     present / absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .csi import CSIWindow


class MotionState(str, Enum):
    ABSENT = "absent"
    MOTION = "motion"


@dataclass
class ProcessedWindow:
    """Result of cleaning a CSIWindow."""

    motion_signal: np.ndarray  # (n_frames,) aggregated dynamic signal
    per_subcarrier: np.ndarray  # (n_frames, n_sc) filtered, mean-removed
    sample_rate: float
    energy: float  # motion energy metric for this window


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase band-pass via FFT masking (no SciPy dependency)."""
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    out = np.empty_like(x)
    for col in range(x.shape[1]):
        spec = np.fft.rfft(x[:, col])
        spec[~mask] = 0.0
        out[:, col] = np.fft.irfft(spec, n=n)
    return out


def clean_window(window: CSIWindow, lo: float = 0.1, hi: float = 10.0) -> ProcessedWindow:
    amps = window.amplitudes.astype(np.float64)  # (n, sc)

    # 1. Drop subcarriers that are dead or pathologically noisy.
    sc_std = amps.std(axis=0)
    sc_mean = amps.mean(axis=0)
    good = (sc_std > 1e-6) & (sc_mean > 1e-6)
    if good.sum() >= 4:
        amps = amps[:, good]

    # 2. Normalise each subcarrier and remove its static component.
    amps = amps / (amps.mean(axis=0, keepdims=True) + 1e-9)
    dynamic = amps - amps.mean(axis=0, keepdims=True)

    # 3. Band-pass to the motion band.
    nyq = window.sample_rate / 2.0
    filtered = _bandpass(dynamic, window.sample_rate, lo, min(hi, nyq * 0.95))

    # 4. Aggregate across subcarriers. Use the first principal component so a
    #    coherent, broadband disturbance (a body) dominates over independent
    #    per-subcarrier noise.
    agg = _principal_component(filtered)

    # 5. Motion energy = variance of the aggregate motion signal.
    energy = float(np.var(agg))

    return ProcessedWindow(
        motion_signal=agg,
        per_subcarrier=filtered,
        sample_rate=window.sample_rate,
        energy=energy,
    )


def _principal_component(x: np.ndarray) -> np.ndarray:
    """First PCA component of (n_frames, n_sc), sign-stabilised."""
    if x.shape[1] == 1:
        return x[:, 0]
    xc = x - x.mean(axis=0, keepdims=True)
    # SVD is more stable than forming the covariance matrix.
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    pc = u[:, 0] * s[0]
    # Stabilise sign so downstream features are deterministic.
    if np.sum(pc) < 0:
        pc = -pc
    return pc


class MotionDetector:
    """Adaptive presence detector with a self-calibrating noise floor.

    While no motion is seen, the noise floor tracks the ambient energy. A window
    is flagged as motion when its energy exceeds the floor by `k` times the
    floor's own variability.
    """

    def __init__(self, k: float = 6.0, floor_alpha: float = 0.05):
        self.k = k
        self.floor_alpha = floor_alpha
        self._floor_mean: float | None = None
        self._floor_var: float = 0.0

    def update(self, processed: ProcessedWindow) -> tuple[MotionState, float]:
        """Return (state, confidence 0..1) for this window."""
        e = processed.energy
        if self._floor_mean is None:
            self._floor_mean = e
            self._floor_var = (0.1 * e) ** 2 + 1e-12
            return MotionState.ABSENT, 0.0

        std = np.sqrt(self._floor_var)
        threshold = self._floor_mean + self.k * std
        if e > threshold:
            # Motion: don't let it pollute the noise floor.
            excess = (e - threshold) / (threshold + 1e-12)
            confidence = float(1.0 - np.exp(-excess))
            return MotionState.MOTION, confidence

        # Quiet window: slowly adapt the floor.
        a = self.floor_alpha
        self._floor_mean = (1 - a) * self._floor_mean + a * e
        self._floor_var = (1 - a) * self._floor_var + a * (e - self._floor_mean) ** 2
        return MotionState.ABSENT, 0.0

    @property
    def noise_floor(self) -> float:
        return self._floor_mean or 0.0
