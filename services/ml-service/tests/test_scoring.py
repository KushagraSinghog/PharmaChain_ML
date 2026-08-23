import pytest
from app.scoring.risk import compute_composite_risk

def test_composite_risk_normal():
    features = {
        "is_impossible_travel": False,
        "implied_travel_speed_kmh": 45.0,
        "duplicate_scan_count": 0,
        "scan_count": 1,
        "reports_count_7d": 0,
        "is_inventory_anomaly": False,
        "inventory_ratio": 1.0,
        "is_shop_spike": False,
        "is_batch_spike": False,
        "is_backward_jump": False,
        "is_tier_skip": False,
        "is_lateral_move": False
    }
    # Low ML score
    result = compute_composite_risk(ml_anomaly_score=0.15, features=features)
    assert 0 <= result.risk_score <= 30
    assert result.risk_level == "LOW"
    assert result.requires_investigation is False

def test_composite_risk_impossible_travel():
    features = {
        "is_impossible_travel": True,
        "implied_travel_speed_kmh": 2200.0,
        "duplicate_scan_count": 1,
        "scan_count": 2,
        "reports_count_7d": 0,
        "is_inventory_anomaly": False,
        "inventory_ratio": 1.0,
        "is_shop_spike": False,
        "is_batch_spike": False,
        "is_backward_jump": False,
        "is_tier_skip": False,
        "is_lateral_move": False
    }
    result = compute_composite_risk(ml_anomaly_score=0.85, features=features)
    assert result.risk_score >= 81
    assert result.risk_level == "CRITICAL"
    assert "IMPOSSIBLE_TRAVEL" in result.anomalies
    assert result.requires_investigation is True

def test_composite_risk_backward_route_jump():
    features = {
        "is_impossible_travel": False,
        "implied_travel_speed_kmh": 20.0,
        "duplicate_scan_count": 0,
        "scan_count": 2,
        "reports_count_7d": 0,
        "is_inventory_anomaly": False,
        "inventory_ratio": 1.0,
        "is_shop_spike": False,
        "is_batch_spike": False,
        "is_backward_jump": True,
        "is_tier_skip": False,
        "is_lateral_move": False
    }
    result = compute_composite_risk(ml_anomaly_score=0.30, features=features)
    assert result.risk_score >= 85
    assert result.risk_level == "CRITICAL"
    assert "ROUTE_BACKWARD_JUMP" in result.anomalies
