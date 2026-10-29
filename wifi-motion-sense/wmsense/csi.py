"""CSI data model and capture sources.

A CSI frame is one measurement of the WiFi channel: a complex value per
OFDM subcarrier (amplitude + phase). A stream of frames over time is what we
analyse for motion.

Capture backends
----------------
Real CSI comes from hardware that exposes the channel estimates the WiFi chip
already computes to decode packets. The common ones:

  * ESP32 with the ESP-CSI firmware  -> emits "CSI_DATA,..." CSV lines over
    USB serial. Supported here by `ESP32SerialSource`.
  * Nexmon CSI (Broadcom, e.g. Raspberry Pi / some routers) and the Intel 5300
    CSI Tool dump UDP/pcap. Supported here by `PcapCSVSource` once you have
    exported them to CSV (one row per frame: timestamp + I/Q or amp/phase).

No hardware? Use `wmsense.simulator.SimulatedSource` for a synthetic stream.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Iterator, Deque

import numpy as np


@dataclass
class CSIFrame:
    """One CSI measurement across all subcarriers at a point in time."""

    timestamp: float
    csi: np.ndarray  # complex64 array, shape (n_subcarriers,)

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.csi)

    @property
    def n_subcarriers(self) -> int:
        return self.csi.shape[0]


@dataclass
class CSIWindow:
    """A sliding window of recent frames, resampled to a uniform rate.

    `amplitudes` has shape (n_frames, n_subcarriers).
    """

    timestamps: np.ndarray
    amplitudes: np.ndarray
    sample_rate: float

    @property
    def n_frames(self) -> int:
        return self.amplitudes.shape[0]

    @property
    def duration(self) -> float:
        if self.n_frames < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


class FrameBuffer:
    """Ring buffer that turns a frame stream into fixed-length CSIWindows."""

    def __init__(self, window_seconds: float = 4.0, sample_rate: float = 20.0):
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.maxlen = int(window_seconds * sample_rate) + 1
        self._frames: Deque[CSIFrame] = deque(maxlen=self.maxlen * 3)

    def push(self, frame: CSIFrame) -> None:
        self._frames.append(frame)

    def ready(self) -> bool:
        if len(self._frames) < 2:
            return False
        span = self._frames[-1].timestamp - self._frames[0].timestamp
        return span >= self.window_seconds * 0.9

    def window(self) -> CSIWindow | None:
        """Resample buffered frames onto a uniform grid for analysis."""
        if not self.ready():
            return None
        frames = list(self._frames)
        t0 = frames[-1].timestamp - self.window_seconds
        src_t = np.array([f.timestamp for f in frames])
        keep = src_t >= t0
        frames = [f for f, k in zip(frames, keep) if k]
        if len(frames) < 2:
            return None

        src_t = np.array([f.timestamp for f in frames])
        amps = np.stack([f.amplitude for f in frames])  # (n, sc)
        n_out = int(self.window_seconds * self.sample_rate)
        grid = np.linspace(src_t[0], src_t[-1], n_out)
        out = np.empty((n_out, amps.shape[1]), dtype=np.float64)
        for sc in range(amps.shape[1]):
            out[:, sc] = np.interp(grid, src_t, amps[:, sc])
        return CSIWindow(timestamps=grid, amplitudes=out, sample_rate=self.sample_rate)


# --------------------------------------------------------------------------
# Capture sources. Each is an iterable of CSIFrame.
# --------------------------------------------------------------------------


class ESP32SerialSource:
    """Read CSI from an ESP32 running ESP-CSI firmware over USB serial.

    The firmware prints lines like:
        CSI_DATA,STA,<mac>,<rssi>,...,[<i0> <q0> <i1> <q1> ...]
    where the bracketed list is interleaved I/Q per subcarrier.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 921600):
        self.port = port
        self.baud = baud

    def __iter__(self) -> Iterator[CSIFrame]:
        try:
            import serial  # pyserial, optional dependency
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "pyserial is required for ESP32SerialSource: pip install pyserial"
            ) from e

        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            while True:
                raw = ser.readline().decode("utf-8", "ignore").strip()
                if not raw.startswith("CSI_DATA"):
                    continue
                frame = self._parse(raw)
                if frame is not None:
                    yield frame

    @staticmethod
    def _parse(line: str) -> CSIFrame | None:
        if "[" not in line or "]" not in line:
            return None
        body = line[line.index("[") + 1 : line.index("]")]
        try:
            nums = [int(x) for x in body.split()]
        except ValueError:
            return None
        if len(nums) < 2:
            return None
        iq = np.array(nums[: len(nums) // 2 * 2], dtype=np.float64).reshape(-1, 2)
        csi = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
        return CSIFrame(timestamp=time.time(), csi=csi)


class PcapCSVSource:
    """Read CSI frames from a CSV exported from Nexmon / Intel 5300 tooling.

    Expected columns: `timestamp` then either
      * `amp_0,phase_0,amp_1,phase_1,...`  (amplitude/phase pairs), or
      * `i_0,q_0,i_1,q_1,...`              (I/Q pairs).
    Column order is auto-detected from the header prefix.
    """

    def __init__(self, path: str, realtime: bool = False):
        self.path = path
        self.realtime = realtime

    def __iter__(self) -> Iterator[CSIFrame]:
        with open(self.path, newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            mode = "amp" if any(c.startswith("amp_") for c in cols) else "iq"
            data_cols = [c for c in cols if c != "timestamp"]
            prev_t = None
            for row in reader:
                vals = np.array([float(row[c]) for c in data_cols], dtype=np.float64)
                vals = vals[: len(vals) // 2 * 2].reshape(-1, 2)
                if mode == "amp":
                    csi = (vals[:, 0] * np.exp(1j * vals[:, 1])).astype(np.complex64)
                else:
                    csi = (vals[:, 0] + 1j * vals[:, 1]).astype(np.complex64)
                t = float(row["timestamp"])
                if self.realtime and prev_t is not None:
                    time.sleep(max(0.0, t - prev_t))
                prev_t = t
                yield CSIFrame(timestamp=t, csi=csi)
