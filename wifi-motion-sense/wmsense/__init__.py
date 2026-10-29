"""wmsense - WiFi motion sensing & human/non-human classification.

Uses Channel State Information (CSI) from commodity WiFi links between a
router/AP and a receiver. Movement in the environment perturbs the multipath
channel; those perturbations are measured, turned into a motion signal, and
classified as human vs. non-human.
"""

__version__ = "0.1.0"

from .csi import CSIFrame, CSIWindow
from .processing import MotionDetector, MotionState
from .features import extract_features, FEATURE_NAMES
from .classifier import PresenceClassifier

__all__ = [
    "CSIFrame",
    "CSIWindow",
    "MotionDetector",
    "MotionState",
    "extract_features",
    "FEATURE_NAMES",
    "PresenceClassifier",
]
