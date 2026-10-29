#!/usr/bin/env python3
"""Train the human/non-human classifier.

By default it trains on the built-in simulator across all motion scenarios and
reports held-out accuracy and a per-scenario breakdown. To train on real data,
point `--csv-dir` at a folder of pcap-exported CSVs named <label>_*.csv where
<label> is `human` or `non_human` (see README).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from wmsense.classifier import PresenceClassifier, LABELS
from wmsense.simulator import labeled_dataset


def _from_csv_dir(path: str):
    from wmsense.csi import PcapCSVSource, FrameBuffer
    from wmsense.processing import clean_window
    from wmsense.features import extract_features

    X, y, groups = [], [], []
    for csv_path in sorted(glob.glob(os.path.join(path, "*.csv"))):
        name = os.path.basename(csv_path).lower()
        label = 1 if name.startswith("human") else 0
        buf = FrameBuffer(window_seconds=4.0, sample_rate=20.0)
        last_emit = None
        for frame in PcapCSVSource(csv_path):
            buf.push(frame)
            if buf.ready() and (last_emit is None or frame.timestamp - last_emit >= 0.5):
                last_emit = frame.timestamp
                w = buf.window()
                if w is not None:
                    X.append(extract_features(clean_window(w)))
                    y.append(label)
                    groups.append(os.path.basename(csv_path))
    return np.array(X), np.array(y), groups


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train human/non-human classifier")
    p.add_argument("--out", default="model.pkl", help="output model path")
    p.add_argument("--csv-dir", help="train on real pcap CSVs instead of simulator")
    p.add_argument("--windows", type=int, default=80,
                   help="simulator windows per scenario")
    args = p.parse_args(argv)

    if args.csv_dir:
        X, y, groups = _from_csv_dir(args.csv_dir)
        print(f"loaded {len(y)} windows from {args.csv_dir}")
    else:
        print("generating simulated training data...")
        X, y, groups = labeled_dataset(windows_per_scenario=args.windows)

    if len(y) == 0:
        print("no training data", file=sys.stderr)
        return 1

    # Simple shuffled train/test split.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    split = int(len(y) * 0.75)
    tr, te = idx[:split], idx[split:]

    clf = PresenceClassifier().fit(X[tr], y[tr])
    print(f"model type: {clf.kind}")

    # Evaluate.
    preds = np.array([clf.predict(x).label for x in X[te]])
    truth = np.array([LABELS[v] for v in y[te]])
    acc = float((preds == truth).mean())
    print(f"held-out accuracy: {acc:.1%}  (n={len(te)})")

    # Per-scenario / per-file breakdown.
    print("\nbreakdown:")
    by_group = {}
    for i in te:
        g = groups[i]
        by_group.setdefault(g, [0, 0])
        pred = clf.predict(X[i]).label
        by_group[g][1] += 1
        if pred == LABELS[y[i]]:
            by_group[g][0] += 1
    for g, (ok, tot) in sorted(by_group.items()):
        print(f"  {g:14s} {ok}/{tot} correct")

    clf.save(args.out)
    print(f"\nsaved model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
