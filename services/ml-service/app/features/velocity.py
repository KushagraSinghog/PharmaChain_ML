from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.features.geo import haversine_distance_km, calculate_implied_speed
from app.config import settings

@dataclass
class VelocityFeatures:
    scan_count: int
    time_between_scans_sec: float
    distance_between_scans_km: float
    implied_travel_speed_kmh: float
    unique_locations_count: int
    is_impossible_travel: bool
    is_instant_teleport: bool
    duplicate_scan_count: int

def compute_velocity_features(
    current_event: Dict[str, Any],
    history_events: List[Dict[str, Any]]
) -> VelocityFeatures:
    """
    Computes scan velocity, spatial displacement, and travel plausibility
    between current scan and previous pack history.
    """
    curr_lat = current_event.get("latitude")
    curr_lon = current_event.get("longitude")
    curr_ts = current_event.get("timestamp")

    if isinstance(curr_ts, str):
        curr_ts = datetime.fromisoformat(curr_ts.replace("Z", "+00:00"))
    elif not isinstance(curr_ts, datetime):
        curr_ts = datetime.now(timezone.utc)
    if curr_ts.tzinfo is None:
        curr_ts = curr_ts.replace(tzinfo=timezone.utc)

    # Filter previous events strictly before current event or existing history
    prev_events = [
        e for e in history_events 
        if e.get("event_id") != current_event.get("event_id") and e.get("timestamp") <= curr_ts
    ]

    scan_count = len(prev_events) + 1

    if not prev_events:
        return VelocityFeatures(
            scan_count=1,
            time_between_scans_sec=86400.0,  # Default 24h baseline for 1st scan
            distance_between_scans_km=0.0,
            implied_travel_speed_kmh=0.0,
            unique_locations_count=1 if curr_lat is not None else 0,
            is_impossible_travel=False,
            is_instant_teleport=False,
            duplicate_scan_count=0
        )

    # Get most recent previous scan
    last_event = max(prev_events, key=lambda x: x["timestamp"])
    last_ts = last_event["timestamp"]
    if isinstance(last_ts, str):
        last_ts = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    time_diff_sec = max(0.1, (curr_ts - last_ts).total_seconds())

    last_lat = last_event.get("latitude")
    last_lon = last_event.get("longitude")

    dist_km = haversine_distance_km(last_lat, last_lon, curr_lat, curr_lon)
    speed_kmh = calculate_implied_speed(dist_km, time_diff_sec)

    # Count distinct (lat, lon) coordinates
    all_locs = set()
    for e in prev_events + [current_event]:
        lt, ln = e.get("latitude"), e.get("longitude")
        if lt is not None and ln is not None:
            all_locs.add((round(lt, 3), round(ln, 3)))
    
    unique_locations = max(1, len(all_locs))

    is_impossible = speed_kmh > settings.MAX_PLAUSIBLE_SPEED_KMH
    is_teleport = (dist_km > (settings.MAX_PLAUSIBLE_DISTANCE_METERS_INSTANT / 1000.0)) and (time_diff_sec < 5.0)

    # Count identical event types (e.g. repeated INTAKE or repeated SALE for the same pack)
    curr_type = current_event.get("event_type", "").upper()
    matching_types = sum(1 for e in prev_events if e.get("event_type", "").upper() == curr_type)

    return VelocityFeatures(
        scan_count=scan_count,
        time_between_scans_sec=round(time_diff_sec, 2),
        distance_between_scans_km=round(dist_km, 3),
        implied_travel_speed_kmh=round(speed_kmh, 2) if speed_kmh != float("inf") else 99999.0,
        unique_locations_count=unique_locations,
        is_impossible_travel=is_impossible or is_teleport,
        is_instant_teleport=is_teleport,
        duplicate_scan_count=matching_types
    )
