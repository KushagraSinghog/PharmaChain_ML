import pytest
from datetime import datetime, timezone, timedelta
from app.features.spike_detector import compute_online_spike

def test_spike_detector_normal_volume():
    now = datetime.now(timezone.utc)
    events = []
    # 5 scans per hour consistently over trailing 5 days
    for day in range(5):
        for hour in range(8, 18):
            for scan in range(5):
                events.append({
                    "timestamp": now - timedelta(days=day, hours=hour, minutes=scan * 10)
                })

    # Current hour has 6 scans (normal)
    for scan in range(6):
        events.append({
            "timestamp": now - timedelta(minutes=scan * 5)
        })

    res = compute_online_spike("SHOP_001", "shop", events, current_time=now)
    assert res.is_spike_event is False
    assert res.risk_contribution == "NONE"

def test_spike_detector_massive_spike():
    now = datetime.now(timezone.utc)
    events = []
    # Baseline: 2 scans per hour for past 4 days
    for day in range(1, 5):
        for hour in range(8, 18):
            for scan in range(2):
                events.append({
                    "timestamp": now - timedelta(days=day, hours=hour, minutes=scan * 20)
                })

    # Current hour suddenly has 80 scans! (40x surge)
    for scan in range(80):
        events.append({
            "timestamp": now - timedelta(seconds=scan * 30)
        })

    res = compute_online_spike("SHOP_SURGE", "shop", events, current_time=now)
    assert res.is_spike_event is True
    assert res.scan_spike_zscore > 3.0
    assert res.spike_magnitude > 5.0
    assert res.risk_contribution in ("HIGH", "CRITICAL")
