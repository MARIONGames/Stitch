#!/usr/bin/env python3
"""End-to-end demo with no hardware required.

Runs every simulated scenario through the full pipeline and prints what the
system detects and classifies, so you can sanity-check the whole chain.
"""

from __future__ import annotations

import numpy as np

from wmsense.csi import FrameBuffer
from wmsense.processing import clean_window, MotionDetector, MotionState
from wmsense.features import extract_features, FEATURE_NAMES
from wmsense.classifier import PresenceClassifier
from wmsense.simulator import SimulatedSource, SCENARIOS


def run_scenario(scenario: str, clf: PresenceClassifier):
    # Model one fixed room/link (base_seed) observed first empty, then with the
    # mover present -- exactly how the detector self-calibrates in the field.
    ROOM = 7
    detector = MotionDetector(k=6.0)
    buf = FrameBuffer(window_seconds=4.0, sample_rate=20.0)

    for frame in SimulatedSource("empty", duration=8.0, seed=1, base_seed=ROOM):
        buf.push(frame)
        if buf.ready():
            detector.update(clean_window(buf.window()))

    # New observation period in the same room; start the window buffer fresh so
    # it does not straddle the empty->mover boundary.
    buf = FrameBuffer(window_seconds=4.0, sample_rate=20.0)
    results = []
    for frame in SimulatedSource(scenario, duration=8.0, seed=2, base_seed=ROOM):
        buf.push(frame)
        if buf.ready():
            processed = clean_window(buf.window())
            state, mconf = detector.update(processed)
            if state == MotionState.MOTION:
                pred = clf.predict(extract_features(processed))
                results.append((pred.label, pred.confidence))

    motion_frac = len(results)
    if results:
        from collections import Counter
        labels = Counter(l for l, _ in results)
        top_label, n = labels.most_common(1)[0]
        avg_conf = np.mean([c for l, c in results if l == top_label])
        verdict = f"{top_label.upper():10s} ({n}/{len(results)} windows, conf {avg_conf:.0%})"
    else:
        verdict = "no motion detected"
    return motion_frac, verdict


def main() -> int:
    print("Training classifier on simulated scenarios...\n")
    from wmsense.simulator import labeled_dataset

    X, y, _ = labeled_dataset(windows_per_scenario=60)
    clf = PresenceClassifier().fit(X, y)
    print(f"classifier backend: {clf.kind}\n")
    print(f"{'scenario':14s} | {'motion?':9s} | verdict")
    print("-" * 60)
    for scen in SCENARIOS:
        motion, verdict = run_scenario(scen, clf)
        detected = "yes" if motion else "no"
        print(f"{scen:14s} | {detected:9s} | {verdict}")
    print("\nExpected: empty->no motion; human_* -> HUMAN; pet/fan -> NON_HUMAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
