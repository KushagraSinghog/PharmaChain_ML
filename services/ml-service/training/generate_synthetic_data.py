import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple

from app.features.geo import haversine_distance_km, calculate_implied_speed
from app.features.builder import FEATURE_COLUMNS

CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Jaipur": (26.9124, 75.7873),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
    "Indore": (22.7196, 75.8577),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Surat": (21.1702, 72.8311)
}

def generate_synthetic_dataset(
    n_normal_packs: int = 2000,
    n_anomalous_packs: int = 150,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates tabular feature dataset X (and binary label y: 1=normal, -1=anomaly).
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    labels = []  # 1 for normal, -1 for anomaly

    city_names = list(CITIES.keys())
    base_time = datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)

    # 1. GENERATE NORMAL SUPPLY-CHAIN EVENTS
    for i in range(n_normal_packs):
        origin_city = random.choice(city_names)
        dest_city = random.choice(city_names)
        
        orig_lat, orig_lon = CITIES[origin_city]
        dest_lat, dest_lon = CITIES[dest_city]

        # Normal transit duration: proportional to distance (avg 40-70 km/h) + warehouse stops
        dist_km = haversine_distance_km(orig_lat, orig_lon, dest_lat, dest_lon)
        transit_hours = max(4.0, dist_km / random.uniform(35.0, 65.0) + random.uniform(2.0, 12.0))
        time_sec = transit_hours * 3600.0
        speed_kmh = calculate_implied_speed(dist_km, time_sec)

        scan_count = random.randint(1, 4)
        unique_locs = min(scan_count, random.randint(1, 3))
        dup_count = 0

        # Normal shop inventory features
        daily_intake = random.randint(20, 150)
        daily_sales = int(daily_intake * random.uniform(0.75, 1.05)) + random.randint(0, 10)
        inv_ratio = (daily_intake + 1.0) / (daily_sales + 1.0)

        shop_zscore = random.gauss(0.0, 0.8)
        batch_zscore = random.gauss(0.0, 0.8)

        # Normal forward progression tier delta: 1 (or 0 for repeated sale query)
        tier_delta = random.choice([1, 1, 1, 0])
        reports_count = 0 if random.random() > 0.05 else 1

        feature_row = [
            scan_count,
            time_sec,
            dist_km,
            speed_kmh,
            unique_locs,
            dup_count,
            daily_intake,
            daily_sales,
            inv_ratio,
            shop_zscore,
            batch_zscore,
            tier_delta,
            reports_count
        ]
        rows.append(feature_row)
        labels.append(1)

    # 2. GENERATE ANOMALOUS FRAUD VECTORS
    for i in range(n_anomalous_packs):
        anomaly_type = random.choice([
            "impossible_travel",
            "clone_burst",
            "route_backward",
            "inventory_hoarding",
            "spike_burst",
            "complaint_hotspot"
        ])

        if anomaly_type == "impossible_travel":
            # Distance 200-1500 km within 10 to 300 seconds -> speed 2000 - 50000 km/h
            dist_km = random.uniform(200.0, 1200.0)
            time_sec = random.uniform(10.0, 300.0)
            speed_kmh = calculate_implied_speed(dist_km, time_sec)
            scan_count = random.randint(2, 6)
            unique_locs = random.randint(2, 4)
            dup_count = random.randint(1, 3)
            daily_intake = random.randint(30, 100)
            daily_sales = random.randint(20, 90)
            inv_ratio = (daily_intake + 1.0) / (daily_sales + 1.0)
            shop_zscore = random.gauss(0.5, 1.0)
            batch_zscore = random.gauss(1.0, 1.2)
            tier_delta = 0
            reports_count = random.randint(1, 4)

        elif anomaly_type == "clone_burst":
            # High scan count, high duplicate replays, multiple cities
            dist_km = random.uniform(500.0, 2000.0)
            time_sec = random.uniform(5.0, 60.0)
            speed_kmh = calculate_implied_speed(dist_km, time_sec)
            scan_count = random.randint(8, 30)
            unique_locs = random.randint(3, 8)
            dup_count = random.randint(4, 15)
            daily_intake = random.randint(40, 120)
            daily_sales = random.randint(50, 150)
            inv_ratio = 1.0
            shop_zscore = random.uniform(3.5, 8.0)
            batch_zscore = random.uniform(4.0, 10.0)
            tier_delta = 0
            reports_count = random.randint(2, 6)

        elif anomaly_type == "route_backward":
            # Backward jump: tier delta < 0 (e.g. -2, -1)
            dist_km = random.uniform(50.0, 500.0)
            time_sec = random.uniform(3600.0, 86400.0)
            speed_kmh = calculate_implied_speed(dist_km, time_sec)
            scan_count = random.randint(2, 4)
            unique_locs = random.randint(2, 3)
            dup_count = 1
            daily_intake = random.randint(20, 80)
            daily_sales = random.randint(20, 80)
            inv_ratio = 1.0
            shop_zscore = random.gauss(0.0, 1.0)
            batch_zscore = random.gauss(0.5, 1.0)
            tier_delta = random.choice([-1, -2, -3])
            reports_count = random.randint(0, 2)

        elif anomaly_type == "inventory_hoarding":
            # Very high intake, near zero sales, high ratio
            dist_km = random.uniform(0.0, 50.0)
            time_sec = random.uniform(3600.0, 86400.0)
            speed_kmh = 10.0
            scan_count = random.randint(1, 2)
            unique_locs = 1
            dup_count = 0
            daily_intake = random.randint(300, 1000)
            daily_sales = random.randint(0, 3)
            inv_ratio = (daily_intake + 1.0) / (daily_sales + 1.0)
            shop_zscore = random.uniform(3.2, 7.0)
            batch_zscore = random.gauss(0.0, 1.0)
            tier_delta = 1
            reports_count = random.randint(1, 3)

        elif anomaly_type == "spike_burst":
            # Massive scan spike on a shop or batch
            dist_km = random.uniform(10.0, 100.0)
            time_sec = random.uniform(60.0, 600.0)
            speed_kmh = calculate_implied_speed(dist_km, time_sec)
            scan_count = random.randint(10, 50)
            unique_locs = random.randint(1, 3)
            dup_count = random.randint(2, 10)
            daily_intake = random.randint(100, 300)
            daily_sales = random.randint(100, 400)
            inv_ratio = 1.0
            shop_zscore = random.uniform(5.0, 15.0)
            batch_zscore = random.uniform(6.0, 18.0)
            tier_delta = 0
            reports_count = random.randint(1, 5)

        else:  # complaint_hotspot
            dist_km = random.uniform(0.0, 30.0)
            time_sec = random.uniform(3600.0, 43200.0)
            speed_kmh = 25.0
            scan_count = random.randint(2, 5)
            unique_locs = random.randint(1, 2)
            dup_count = random.randint(1, 3)
            daily_intake = random.randint(50, 150)
            daily_sales = random.randint(40, 120)
            inv_ratio = 1.2
            shop_zscore = random.gauss(1.0, 1.0)
            batch_zscore = random.uniform(2.0, 5.0)
            tier_delta = 1
            reports_count = random.randint(5, 12)

        feature_row = [
            scan_count,
            time_sec,
            dist_km,
            speed_kmh,
            unique_locs,
            dup_count,
            daily_intake,
            daily_sales,
            inv_ratio,
            shop_zscore,
            batch_zscore,
            tier_delta,
            reports_count
        ]
        rows.append(feature_row)
        labels.append(-1)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    X = df.values.astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    return df, X, y
