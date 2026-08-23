import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import settings

def compute_psi(
    baseline_scores: List[float],
    current_scores: List[float],
    n_bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI) between baseline and current normalized score distributions.
    Scores should be in [0.0, 1.0].
    """
    if not baseline_scores or not current_scores:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Compute frequencies per bin
    baseline_counts, _ = np.histogram(baseline_scores, bins=bins)
    current_counts, _ = np.histogram(current_scores, bins=bins)

    epsilon = 1e-6
    baseline_frac = (baseline_counts + epsilon) / (len(baseline_scores) + epsilon * n_bins)
    current_frac = (current_counts + epsilon) / (len(current_scores) + epsilon * n_bins)

    psi = np.sum((current_frac - baseline_frac) * np.log(current_frac / baseline_frac))
    return float(max(0.0, psi))

def get_drift_status(psi: float) -> str:
    if psi < settings.PSI_STABLE_MAX:
        return "STABLE"
    elif psi < settings.PSI_WARNING_MAX:
        return "WARNING"
    else:
        return "DRIFT"

class ModelDriftMonitor:
    """
    Tracks inference score distributions against the training baseline to detect distribution shift.
    """

    def __init__(self, baseline_dist_path: Optional[Path] = None):
        self.baseline_path = baseline_dist_path or settings.BASELINE_DIST_PATH
        self.baseline_scores: List[float] = []
        self._load_baseline()

    def _load_baseline(self):
        if self.baseline_path and os.path.exists(self.baseline_path):
            try:
                arr = np.load(self.baseline_path)
                self.baseline_scores = arr.tolist()
                print(f"[ML Drift Monitor] Loaded baseline distribution of {len(self.baseline_scores)} samples.")
            except Exception as e:
                print(f"[ML Drift Monitor] Could not load baseline from {self.baseline_path}: {e}")
                self._generate_synthetic_baseline()
        else:
            self._generate_synthetic_baseline()

    def _generate_synthetic_baseline(self):
        # Generate calibrated gamma/beta distribution mimicking normal low-risk baseline
        # Mean around 0.15 with tail up to 0.6
        np.random.seed(42)
        base = np.random.beta(2, 8, 2000)
        self.baseline_scores = base.tolist()

    def save_baseline(self, scores: np.ndarray, target_path: Optional[Path] = None):
        save_path = target_path or self.baseline_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, scores)
        self.baseline_scores = scores.tolist()
        print(f"[ML Drift Monitor] Saved baseline score distribution ({len(scores)} samples) to {save_path}")

    def evaluate_drift(self, recent_scores: List[float]) -> Dict[str, Any]:
        """
        Calculates PSI and generates a health diagnostic summary.
        """
        if len(recent_scores) < 10:
            return {
                "status": "INSUFFICIENT_DATA",
                "psiScore": 0.0,
                "thresholds": {
                    "stable": settings.PSI_STABLE_MAX,
                    "warning": settings.PSI_WARNING_MAX
                },
                "baselineEventCount": len(self.baseline_scores),
                "currentEventCount": len(recent_scores),
                "recommendation": f"Collect at least 10 recent events to evaluate drift. Current: {len(recent_scores)}."
            }

        psi = compute_psi(self.baseline_scores, recent_scores)
        status = get_drift_status(psi)

        recommendations = {
            "STABLE": "Model inference distribution is healthy and closely aligns with baseline.",
            "WARNING": "Mild distribution shift observed. Monitor incoming traffic patterns closely.",
            "DRIFT": "Significant model drift detected! Retraining recommended with recent verified records."
        }

        # Histogram summary for UI charting
        bins = np.linspace(0.0, 1.0, 11)
        b_counts, _ = np.histogram(self.baseline_scores, bins=bins)
        c_counts, _ = np.histogram(recent_scores, bins=bins)

        return {
            "status": status,
            "psiScore": round(psi, 4),
            "thresholds": {
                "stable": settings.PSI_STABLE_MAX,
                "warning": settings.PSI_WARNING_MAX
            },
            "baselineEventCount": len(self.baseline_scores),
            "currentEventCount": len(recent_scores),
            "recommendation": recommendations[status],
            "binAnalysis": {
                "binEdges": [round(float(b), 2) for b in bins],
                "baselineFrequencies": [int(x) for x in b_counts],
                "currentFrequencies": [int(x) for x in c_counts]
            }
        }

drift_monitor = ModelDriftMonitor()
