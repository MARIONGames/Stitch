"""Real-time pipeline: capture -> detect motion -> classify mover.

Ties the modules together and renders a live terminal dashboard. Works with any
frame source (ESP32 serial, pcap CSV replay, or the simulator).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque, Counter

from .csi import FrameBuffer, ESP32SerialSource, PcapCSVSource
from .processing import clean_window, MotionDetector, MotionState
from .features import extract_features
from .classifier import PresenceClassifier


def build_source(args):
    if args.source == "esp32":
        return ESP32SerialSource(port=args.port, baud=args.baud)
    if args.source == "pcap":
        return PcapCSVSource(args.file, realtime=args.realtime)
    if args.source == "sim":
        from .simulator import SimulatedSource

        return SimulatedSource(
            scenario=args.scenario,
            sample_rate=args.rate,
            realtime=args.realtime,
            duration=args.duration,
            warmup=args.warmup,
        )
    raise ValueError(args.source)


class LiveApp:
    def __init__(self, classifier: PresenceClassifier, args):
        self.clf = classifier
        self.args = args
        self.detector = MotionDetector(k=args.sensitivity)
        self.buffer = FrameBuffer(window_seconds=args.window, sample_rate=args.rate)
        self.vote = deque(maxlen=args.smoothing)  # smooth classification over windows

    def run(self, source) -> None:
        last_render = 0.0
        last_analysis = 0.0
        step = 1.0 / max(self.args.rate, 1.0)
        for frame in source:
            self.buffer.push(frame)
            now = frame.timestamp
            if now - last_analysis >= self.args.hop and self.buffer.ready():
                last_analysis = now
                self._analyse(now)
            if time.time() - last_render >= 0.2:
                last_render = time.time()
                self._render()
        self._render(final=True)

    def _analyse(self, now: float) -> None:
        window = self.buffer.window()
        if window is None:
            return
        processed = clean_window(window)
        state, motion_conf = self.detector.update(processed)
        self._last_state = state
        self._last_motion_conf = motion_conf
        self._last_energy = processed.energy
        if state == MotionState.MOTION:
            feats = extract_features(processed)
            pred = self.clf.predict(feats)
            self.vote.append(pred.label)
            self._last_pred = pred
        else:
            self.vote.clear()
            self._last_pred = None

    _last_state = MotionState.ABSENT
    _last_motion_conf = 0.0
    _last_energy = 0.0
    _last_pred = None

    def _smoothed_label(self):
        if not self.vote:
            return None, 0.0
        counts = Counter(self.vote)
        label, n = counts.most_common(1)[0]
        return label, n / len(self.vote)

    def _render(self, final: bool = False) -> None:
        state = self._last_state
        if state == MotionState.MOTION:
            label, agree = self._smoothed_label()
            if label == "human":
                who = "\033[1;32mHUMAN\033[0m"
            elif label == "non_human":
                who = "\033[1;33mNON-HUMAN\033[0m"
            else:
                who = "classifying..."
            conf = self._last_pred.confidence if self._last_pred else 0.0
            line = (
                f"\r[\033[1;31mMOTION\033[0m] {who}  "
                f"cls-conf={conf:4.0%} agree={agree:4.0%} "
                f"motion-conf={self._last_motion_conf:4.0%} "
                f"energy={self._last_energy:.2e}   "
            )
        else:
            line = (
                f"\r[\033[1;34m quiet\033[0m] no motion  "
                f"floor={self.detector.noise_floor:.2e}            "
                f"                                        "
            )
        sys.stdout.write(line)
        sys.stdout.flush()
        if final:
            sys.stdout.write("\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="wmsense",
        description="WiFi CSI motion detection + human/non-human classification",
    )
    p.add_argument("--source", choices=["sim", "esp32", "pcap"], default="sim")
    p.add_argument("--scenario", default="human_moving",
                   help="sim only: empty|human_still|human_moving|pet|fan")
    p.add_argument("--port", default="/dev/ttyUSB0", help="esp32 serial port")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--file", help="pcap-exported CSV path (source=pcap)")
    p.add_argument("--model", help="trained classifier path (optional)")
    p.add_argument("--rate", type=float, default=20.0, help="analysis sample rate Hz")
    p.add_argument("--window", type=float, default=4.0, help="analysis window seconds")
    p.add_argument("--hop", type=float, default=0.5, help="seconds between analyses")
    p.add_argument("--sensitivity", type=float, default=6.0,
                   help="detector threshold in sigmas above noise floor")
    p.add_argument("--smoothing", type=int, default=5,
                   help="windows of majority vote for the label")
    p.add_argument("--realtime", action="store_true",
                   help="pace sim/pcap at wall-clock speed")
    p.add_argument("--duration", type=float, default=None, help="sim: stop after N s")
    p.add_argument("--warmup", type=float, default=5.0,
                   help="sim: seconds of empty room before the mover enters")
    args = p.parse_args(argv)

    if args.source == "pcap" and not args.file:
        p.error("--file is required when --source=pcap")

    clf = PresenceClassifier()
    if args.model:
        clf.load(args.model)
    else:
        clf.ensure_model()
    print(f"classifier: {clf.kind}", file=sys.stderr)

    source = build_source(args)
    app = LiveApp(clf, args)
    try:
        app.run(source)
    except KeyboardInterrupt:
        app._render(final=True)
        print("stopped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
