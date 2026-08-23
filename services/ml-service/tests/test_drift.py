import pytest
import numpy as np
from app.scoring.drift_monitor import compute_psi, get_drift_status, ModelDriftMonitor

def test_psi_identical_distributions():
    np.random.seed(42)
    baseline = np.random.beta(2, 5, 1000).tolist()
    current = np.random.beta(2, 5, 1000).tolist()

    psi = compute_psi(baseline, current)
    assert psi < 0.10
    assert get_drift_status(psi) == "STABLE"

def test_psi_drifted_distributions():
    np.random.seed(42)
    baseline = np.random.beta(2, 8, 1000).tolist()  # heavily skewed left (low scores)
    current = np.random.beta(8, 2, 1000).tolist()   # heavily skewed right (high scores)

    psi = compute_psi(baseline, current)
    assert psi > 0.25
    assert get_drift_status(psi) == "DRIFT"

def test_drift_monitor_diagnostics():
    monitor = ModelDriftMonitor()
    np.random.seed(42)
    recent = np.random.beta(2, 8, 100).tolist()
    report = monitor.evaluate_drift(recent)
    assert "status" in report
    assert "psiScore" in report
    assert "recommendation" in report
