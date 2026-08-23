import pytest
import numpy as np
from app.models.isolation_forest import IsolationForestModel
from app.models.dbscan import ComplaintClusterer
from training.generate_synthetic_data import generate_synthetic_dataset

def test_isolation_forest_training_and_scoring():
    df, X, y = generate_synthetic_dataset(n_normal_packs=500, n_anomalous_packs=50, seed=42)
    model_wrapper = IsolationForestModel()
    model_wrapper.fit(X[y == 1])

    # Normal sample score should be lower
    normal_sample = X[y == 1][0]
    score_normal = model_wrapper.score_event(normal_sample)

    # Injected anomaly score should be higher
    anomaly_sample = X[y == -1][0]
    score_anomaly = model_wrapper.score_event(anomaly_sample)

    assert 0.0 <= score_normal <= 1.0
    assert 0.0 <= score_anomaly <= 1.0
    assert score_anomaly >= score_normal

def test_dbscan_clustering_hotspot():
    # Create 5 reports concentrated around Delhi CP (28.6315, 77.2167)
    reports = [
        {"report_id": f"R-{i}", "latitude": 28.6315 + (i * 0.002), "longitude": 77.2167 + (i * 0.002), "batch_id": "B-1", "shop_id": "S-1", "report_type": "COUNTERFEIT", "timestamp": "2026-08-22T10:00:00Z"}
        for i in range(5)
    ]
    # Add 2 far away noise reports (Mumbai)
    reports.append({"report_id": "R-NOISE-1", "latitude": 19.0760, "longitude": 72.8777, "batch_id": "B-2", "shop_id": "S-2", "report_type": "COUNTERFEIT", "timestamp": "2026-08-22T10:00:00Z"})

    clusterer = ComplaintClusterer(eps_km=3.0, min_samples=3)
    result = clusterer.cluster_reports(reports)

    assert result["clusters_count"] == 1
    assert result["clustered_reports_count"] == 5
    assert result["noise_count"] == 1
    assert len(result["clusters"]) == 1
    assert result["clusters"][0]["size"] == 5
    assert result["clusters"][0]["risk_level"] in ("HIGH", "CRITICAL")
