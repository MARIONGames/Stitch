"""Feature extraction for human vs. non-human classification.

The idea: different movers leave different fingerprints on the WiFi channel.

  * A human body is large and broadband. At rest it still produces a slow
    quasi-periodic signal from breathing (~0.15-0.4 Hz). Walking adds energy
    in ~0.5-3 Hz (limb/gait) and the disturbance is coherent across many
    subcarriers (a big reflector).
  * A pet is smaller and faster, with weaker/absent respiration coherence and
    more irregular (less periodic) motion.
  * A fan / mechanical mover is strongly periodic in a narrow high band
    (~4-12 Hz), with almost no energy in the respiration band.

We summarise each motion window with features that separate these cases.
"""

from __future__ import annotations

import numpy as np

from .processing import ProcessedWindow

FEATURE_NAMES = [
    "resp_band_power",     # 0.1-0.5 Hz  (breathing)
    "activity_band_power", # 0.5-3 Hz    (human limb / gait)
    "mech_band_power",     # 4-12 Hz     (fans, machinery)
    "dominant_freq",       # Hz of spectral peak
    "spectral_entropy",    # flat (irregular) vs peaky (periodic), 0..1
    "periodicity",         # autocorrelation peak strength, 0..1
    "band_ratio_human",    # (resp+activity) / (mech + eps)
    "sc_coherence",        # how coherent the motion is across subcarriers
    "crest_factor",        # peakiness of the waveform (bursty gait vs steady)
    "energy_log",          # log overall motion energy
]


def _power_spectrum(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    sig = sig - sig.mean()
    n = sig.shape[0]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def _band_power(freqs: np.ndarray, spec: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(spec[mask].sum())


def extract_features(processed: ProcessedWindow) -> np.ndarray:
    sig = processed.motion_signal
    fs = processed.sample_rate
    freqs, spec = _power_spectrum(sig, fs)
    total = spec.sum() + 1e-12

    resp = _band_power(freqs, spec, 0.1, 0.5)
    activity = _band_power(freqs, spec, 0.5, 3.0)
    mech = _band_power(freqs, spec, 4.0, 12.0)

    # Dominant frequency (ignore the DC bin).
    if len(spec) > 1:
        dom_idx = 1 + int(np.argmax(spec[1:]))
        dominant_freq = float(freqs[dom_idx])
    else:
        dominant_freq = 0.0

    # Spectral entropy, normalised to 0..1 (1 = white/irregular).
    p = spec / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum() / np.log(len(spec) + 1e-12))

    periodicity = _periodicity(sig)
    band_ratio = (resp + activity) / (mech + 1e-9)

    # Subcarrier coherence: a large body perturbs many subcarriers in step.
    coherence = _sc_coherence(processed.per_subcarrier)

    # Crest factor of the waveform.
    rms = np.sqrt(np.mean(sig**2)) + 1e-12
    crest = float(np.max(np.abs(sig)) / rms)

    energy_log = float(np.log1p(processed.energy * 1e3))

    feats = np.array(
        [
            resp / total,
            activity / total,
            mech / total,
            dominant_freq,
            entropy,
            periodicity,
            np.log1p(band_ratio),
            coherence,
            crest,
            energy_log,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(feats)


def _periodicity(sig: np.ndarray) -> float:
    """Strength of the strongest non-trivial autocorrelation peak, 0..1."""
    x = sig - sig.mean()
    if np.allclose(x, 0):
        return 0.0
    ac = np.correlate(x, x, mode="full")[x.size - 1 :]
    ac = ac / (ac[0] + 1e-12)
    # Skip the first few lags (trivial self-correlation).
    tail = ac[3:]
    if tail.size == 0:
        return 0.0
    return float(np.clip(np.max(tail), 0.0, 1.0))


def _sc_coherence(per_sc: np.ndarray) -> float:
    """Mean absolute pairwise correlation across subcarriers, 0..1."""
    if per_sc.shape[1] < 2:
        return 0.0
    # Subsample subcarriers for speed on wide channels.
    cols = per_sc
    if cols.shape[1] > 24:
        idx = np.linspace(0, cols.shape[1] - 1, 24).astype(int)
        cols = cols[:, idx]
    c = np.corrcoef(cols.T)
    iu = np.triu_indices_from(c, k=1)
    vals = np.abs(c[iu])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0
