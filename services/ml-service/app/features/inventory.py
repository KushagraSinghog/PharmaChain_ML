from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ShopInventoryFeatures:
    daily_intake: int
    daily_sales: int
    weekly_intake: int
    weekly_sales: int
    inventory_ratio: float
    unique_batches_count: int
    scan_frequency_per_hour: float
    avg_order_size: float
    is_inventory_anomaly: bool

def compute_shop_inventory_features(
    shop_id: str,
    shop_events: List[Dict[str, Any]],
    current_time: Optional[datetime] = None
) -> ShopInventoryFeatures:
    """
    Computes rolling intake vs. sales metrics and behavioral ratios for a pharmacy.
    """
    if not current_time:
        current_time = datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    one_day_ago = current_time - timedelta(days=1)
    seven_days_ago = current_time - timedelta(days=7)

    daily_intake = 0
    daily_sales = 0
    weekly_intake = 0
    weekly_sales = 0
    total_intake = 0
    total_sales = 0
    batches = set()
    intake_batches_count = 0

    for e in shop_events:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        etype = (e.get("event_type") or "").upper()
        b_id = e.get("batch_id")
        if b_id:
            batches.add(b_id)

        if "INTAKE" in etype:
            total_intake += 1
            intake_batches_count += 1
            if ts >= seven_days_ago:
                weekly_intake += 1
            if ts >= one_day_ago:
                daily_intake += 1
        elif "SALE" in etype or "SELL" in etype or "SOLD" in etype:
            total_sales += 1
            if ts >= seven_days_ago:
                weekly_sales += 1
            if ts >= one_day_ago:
                daily_sales += 1

    # Ratio calculation with Laplace smoothing epsilon
    inventory_ratio = (weekly_intake + 1.0) / (weekly_sales + 1.0)
    
    # Estimate scan frequency over last 24h
    scans_24h = daily_intake + daily_sales
    scan_freq_per_hour = scans_24h / 24.0

    avg_order_size = total_intake / max(1, len(batches))

    # Flag abnormal hoarding or diversion
    # Red flags:
    # 1) High weekly intake (>= 100) but negligible sales (<= 5)
    # 2) Inventory ratio > 10.0 with significant intake
    # 3) Extreme sudden daily intake
    is_anomaly = False
    if weekly_intake >= 50 and weekly_sales <= 2:
        is_anomaly = True
    elif weekly_intake >= 100 and inventory_ratio > 10.0:
        is_anomaly = True
    elif daily_intake >= 200 and daily_sales == 0:
        is_anomaly = True

    return ShopInventoryFeatures(
        daily_intake=daily_intake,
        daily_sales=daily_sales,
        weekly_intake=weekly_intake,
        weekly_sales=weekly_sales,
        inventory_ratio=round(inventory_ratio, 2),
        unique_batches_count=len(batches),
        scan_frequency_per_hour=round(scan_freq_per_hour, 2),
        avg_order_size=round(avg_order_size, 2),
        is_inventory_anomaly=is_anomaly
    )
