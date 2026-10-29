"""Human vs. non-human classifier over motion feature vectors.

If scikit-learn is installed, a RandomForest is trained/loaded. If not, a
transparent heuristic classifier (hand-tuned on the feature definitions) is
used so the app still works out of the box. Both expose the same interface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES

LABELS = ["non_human", "human"]

try:  # optional dependency
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    import joblib  # type: ignore

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


@dataclass
class Prediction:
    label: str          # "human" | "non_human"
    confidence: float   # 0..1
    probabilities: dict  # {"human": .., "non_human": ..}


class _HeuristicModel:
    """Rule-of-thumb scorer used when scikit-learn is unavailable.

    Produces a 'humanness' score in 0..1 from the feature vector, following the
    physical reasoning documented in features.py.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        f = {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}
        score = np.zeros(X.shape[0])
        # Primary discriminator: a human body is a large reflector, so its
        # motion is coherent across many subcarriers. A small mover (pet) is
        # not. Centre the decision on ~0.45 coherence.
        score += 3.0 * (f["sc_coherence"] - 0.45)
        # Breathing energy is a strong human tell.
        score += 2.5 * f["resp_band_power"]
        # Motion concentrated in the human (resp+limb) bands vs. mechanical.
        score += 1.0 * np.clip(f["band_ratio_human"] / 3.0, 0, 1)
        # Mechanical / narrowband high-freq, very periodic -> non-human.
        score -= 2.5 * f["mech_band_power"]
        score -= 1.2 * np.clip(f["periodicity"] - 0.5, 0, 1)
        score -= 1.0 * np.clip((f["dominant_freq"] - 3.5) / 6.0, 0, 1)
        p_human = 1.0 / (1.0 + np.exp(-2.2 * score))
        return np.column_stack([1 - p_human, p_human])


class PresenceClassifier:
    def __init__(self):
        self._model = None
        self._kind = "none"

    # -- training -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PresenceClassifier":
        """Train on feature matrix X and labels y (0=non_human, 1=human)."""
        if _HAVE_SKLEARN:
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=0, class_weight="balanced"
            )
            clf.fit(X, y)
            self._model = clf
            self._kind = "random_forest"
        else:
            self._model = _HeuristicModel()
            self._kind = "heuristic"
        return self

    def ensure_model(self) -> None:
        if self._model is None:
            self._model = _HeuristicModel()
            self._kind = "heuristic"

    # -- inference ----------------------------------------------------------
    def predict(self, feats: np.ndarray) -> Prediction:
        self.ensure_model()
        X = feats.reshape(1, -1)
        proba = self._model.predict_proba(X)[0]
        # Align to LABELS order regardless of sklearn's class ordering.
        classes = getattr(self._model, "classes_", np.array([0, 1]))
        prob_map = {LABELS[int(c)]: float(proba[i]) for i, c in enumerate(classes)}
        label = max(prob_map, key=prob_map.get)
        return Prediction(
            label=label,
            confidence=prob_map[label],
            probabilities=prob_map,
        )

    @property
    def kind(self) -> str:
        return self._kind

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        self.ensure_model()
        meta = {"kind": self._kind, "features": FEATURE_NAMES}
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        if self._kind == "random_forest" and _HAVE_SKLEARN:
            joblib.dump(self._model, path)

    def load(self, path: str) -> "PresenceClassifier":
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            self._kind = meta.get("kind", "heuristic")
        if self._kind == "random_forest" and _HAVE_SKLEARN and os.path.exists(path):
            self._model = joblib.load(path)
        else:
            self._model = _HeuristicModel()
            self._kind = "heuristic"
        return self
