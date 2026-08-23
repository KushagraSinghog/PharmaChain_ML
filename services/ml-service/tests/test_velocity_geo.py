import pytest
from datetime import datetime, timezone, timedelta
from app.features.geo import haversine_distance_km, calculate_implied_speed
from app.features.velocity import compute_velocity_features

def test_haversine_known_cities():
    # Delhi (28.6139, 77.2090) to Jaipur (26.9124, 75.7873)
    dist = haversine_distance_km(28.6139, 77.2090, 26.9124, 75.7873)
    assert 230.0 <= dist <= 270.0  # Approx 240-260 km

def test_calculate_implied_speed():
    # 240 km in 3 hours -> 80 km/h
    speed = calculate_implied_speed(240.0, 3 * 3600.0)
    assert round(speed, 1) == 80.0

    # 240 km in 8 minutes (480s) -> 1800 km/h
    speed_fast = calculate_implied_speed(240.0, 480.0)
    assert speed_fast >= 1700.0

def test_impossible_travel_flag():
    now = datetime.now(timezone.utc)
    history = [
        {
            "event_id": "EV-1",
            "pack_hash": "PACK-001",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timestamp": now - timedelta(minutes=8),
            "event_type": "INTAKE"
        }
    ]
    curr_event = {
        "event_id": "EV-2",
        "pack_hash": "PACK-001",
        "latitude": 26.9124,
        "longitude": 75.7873,
        "timestamp": now,
        "event_type": "SALE"
    }
    features = compute_velocity_features(curr_event, history)
    assert features.is_impossible_travel is True
    assert features.implied_travel_speed_kmh > 1500.0
    assert features.distance_between_scans_km > 200.0

def test_normal_travel_flag():
    now = datetime.now(timezone.utc)
    history = [
        {
            "event_id": "EV-1",
            "pack_hash": "PACK-002",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timestamp": now - timedelta(hours=6),
            "event_type": "INTAKE"
        }
    ]
    curr_event = {
        "event_id": "EV-2",
        "pack_hash": "PACK-002",
        "latitude": 26.9124,
        "longitude": 75.7873,
        "timestamp": now,
        "event_type": "SALE"
    }
    features = compute_velocity_features(curr_event, history)
    assert features.is_impossible_travel is False
    assert features.implied_travel_speed_kmh < 100.0
