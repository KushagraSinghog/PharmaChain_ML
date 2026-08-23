import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta

from app.features.velocity import compute_velocity_features, VelocityFeatures
from app.features.inventory import compute_shop_inventory_features, ShopInventoryFeatures
from app.features.spike_detector import compute_online_spike, SpikeResult
from app.features.route_anomaly import check_route_anomaly, RouteAnomalyResult
from app.features.geo import haversine_distance_km
from app.storage.event_store import event_store

FEATURE_COLUMNS = [
    "scan_count",
    "time_between_scans_sec",
    "distance_between_scans_km",
    "implied_travel_speed_kmh",
    "unique_locations_count",
    "duplicate_scan_count",
    "daily_intake",
    "daily_sales",
    "inventory_ratio",
    "shop_scan_spike_zscore",
    "batch_scan_spike_zscore",
    "route_tier_delta",
    "reports_count_7d"
]

async def build_feature_vector(event: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Builds both a human-readable feature dictionary and a standardized 1D NumPy array
    for Isolation Forest model inference.
    """
    pack_hash = str(event.get("pack_hash") or event.get("packHash") or "")
    batch_id = str(event.get("batch_id") or event.get("batchId") or "")
    shop_id = str(event.get("shop_id") or event.get("shopId") or "")
    event_type = str(event.get("event_type") or event.get("eventType") or "SALE").upper()
    lat = event.get("latitude")
    lon = event.get("longitude")

    # Fetch context from storage
    pack_history = await event_store.get_pack_events(pack_hash) if pack_hash else []
    shop_history = await event_store.get_shop_events(shop_id) if shop_id else []
    batch_history = await event_store.get_batch_events(batch_id) if batch_id else []
    
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    reports = await event_store.get_reports(since=seven_days_ago)

    # 1. Velocity & Geography (A1 & A2)
    velocity_res: VelocityFeatures = compute_velocity_features(event, pack_history)

    # 2. Shop Inventory (A3)
    inventory_res: ShopInventoryFeatures = compute_shop_inventory_features(shop_id, shop_history)

    # 3. B2: Temporal Spikes (Shop & Batch)
    shop_spike: SpikeResult = compute_online_spike(shop_id, "shop", shop_history)
    batch_spike: SpikeResult = compute_online_spike(batch_id, "batch", batch_history)

    # 4. B3: Route Anomaly
    route_res: RouteAnomalyResult = check_route_anomaly(pack_hash, event_type, shop_id, pack_history)

    # 5. Nearby / Batch Reports (A5 connection)
    relevant_reports = 0
    for r in reports:
        if r.get("batch_id") == batch_id:
            relevant_reports += 1
        elif lat is not None and lon is not None:
            r_lat, r_lon = r.get("latitude"), r.get("longitude")
            if r_lat is not None and r_lon is not None:
                if haversine_distance_km(lat, lon, r_lat, r_lon) <= 10.0:
                    relevant_reports += 1

    feature_dict = {
        "scan_count": velocity_res.scan_count,
        "time_between_scans_sec": velocity_res.time_between_scans_sec,
        "distance_between_scans_km": velocity_res.distance_between_scans_km,
        "implied_travel_speed_kmh": min(5000.0, velocity_res.implied_travel_speed_kmh),
        "unique_locations_count": velocity_res.unique_locations_count,
        "duplicate_scan_count": velocity_res.duplicate_scan_count,
        "is_impossible_travel": velocity_res.is_impossible_travel,
        "is_instant_teleport": velocity_res.is_instant_teleport,
        
        "daily_intake": inventory_res.daily_intake,
        "daily_sales": inventory_res.daily_sales,
        "inventory_ratio": inventory_res.inventory_ratio,
        "is_inventory_anomaly": inventory_res.is_inventory_anomaly,
        
        "shop_scan_spike_zscore": max(-5.0, min(20.0, shop_spike.scan_spike_zscore)),
        "is_shop_spike": shop_spike.is_spike_event,
        "shop_spike_magnitude": shop_spike.spike_magnitude,
        
        "batch_scan_spike_zscore": max(-5.0, min(20.0, batch_spike.scan_spike_zscore)),
        "is_batch_spike": batch_spike.is_spike_event,
        "batch_spike_magnitude": batch_spike.spike_magnitude,
        
        "route_current_tier": route_res.current_tier,
        "route_prev_tier": route_res.prev_tier if route_res.prev_tier is not None else -1,
        "route_tier_delta": route_res.tier_delta if route_res.tier_delta is not None else 0,
        "is_backward_jump": route_res.is_backward_jump,
        "is_tier_skip": route_res.is_tier_skip,
        "is_lateral_move": route_res.is_lateral_move,
        "route_risk_contribution": route_res.risk_contribution,
        "route_anomaly_reason": route_res.reason,
        
        "reports_count_7d": relevant_reports
    }

    # Construct tabular 1D array matching FEATURE_COLUMNS
    vector = np.array([
        float(feature_dict["scan_count"]),
        float(feature_dict["time_between_scans_sec"]),
        float(feature_dict["distance_between_scans_km"]),
        float(feature_dict["implied_travel_speed_kmh"]),
        float(feature_dict["unique_locations_count"]),
        float(feature_dict["duplicate_scan_count"]),
        float(feature_dict["daily_intake"]),
        float(feature_dict["daily_sales"]),
        float(feature_dict["inventory_ratio"]),
        float(feature_dict["shop_scan_spike_zscore"]),
        float(feature_dict["batch_scan_spike_zscore"]),
        float(feature_dict["route_tier_delta"]),
        float(feature_dict["reports_count_7d"])
    ], dtype=np.float32)

    return feature_dict, vector
