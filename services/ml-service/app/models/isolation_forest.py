import os
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
from sklearn.ensemble import IsolationForest
from app.config import settings

class IsolationForestModel:
    """
    Primary tabular unsupervised anomaly detection model wrapper.
    Uses sklearn IsolationForest (n_estimators=200, contamination=0.02).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[IsolationForest] = None
        self.is_loaded = False
        self._load_or_initialize()

    def _load_or_initialize(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                print(f"[ML Model] Successfully loaded Isolation Forest from {self.model_path}")
            except Exception as e:
                print(f"[ML Model] Failed to load model from {self.model_path}: {e}. Initializing fresh default.")
                self._initialize_default()
        else:
            self._initialize_default()

    def _initialize_default(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
            n_jobs=-1
        )
        self.is_loaded = False

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        """
        Fits the Isolation Forest on feature matrix X (shape [N, D]).
        """
        self.model.fit(X)
        self.is_loaded = True
        return self

    def save(self, target_path: Optional[Path] = None) -> Path:
        """
        Saves fitted model to disk.
        """
        save_path = target_path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"[ML Model] Saved model to {save_path}")
        return save_path

    def score_event(self, feature_vector: np.ndarray) -> float:
        """
        Computes normalized anomaly score in range [0.0, 1.0].
        Higher score indicates higher anomaly likelihood.
        """
        if not self.is_loaded or self.model is None:
            # Fallback heuristic score if model is unfitted
            return 0.15

        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        # score_samples returns opposite of anomaly score (lower is more anomalous, typically [-0.8, -0.3])
        # We invert so higher means more anomalous
        raw_score = -float(self.model.score_samples(X)[0])
        
        # Typical score_samples range is between -0.7 (normal) and -0.3 (anomalous)
        # We calibrate and sigmoid/clip to [0.0, 1.0]
        # Calibrated baseline mapping: ~0.4 -> 0.0, ~0.65 -> 1.0
        normalized = (raw_score - 0.38) / (0.70 - 0.38)
        return float(np.clip(normalized, 0.0, 1.0))

    def predict_anomaly(self, feature_vector: np.ndarray) -> bool:
        """
        Returns True if the model predicts the sample as an outlier (-1).
        """
        if not self.is_loaded or self.model is None:
            return False

        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        pred = self.model.predict(X)[0]
        return bool(pred == -1)

# Global singleton model instance
model_instance = IsolationForestModel()
