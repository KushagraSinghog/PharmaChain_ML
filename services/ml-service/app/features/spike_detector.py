import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.config import settings

@dataclass
class SpikeResult:
    entity_id: str
    entity_type: str  # "shop" or "batch"
    scans_last_1h: int
    scans_mean_7d: float
    scans_std_7d: float
    scan_spike_zscore: float
    is_spike_event: bool
    spike_magnitude: float
    risk_contribution: str  # "NONE", "MEDIUM", "HIGH", "CRITICAL"

def compute_online_spike(
    entity_id: str,
    entity_type: str,
    events: List[Dict[str, Any]],
    current_time: Optional[datetime] = None
) -> SpikeResult:
    """
    Computes real-time rolling 1-hour scan spike vs trailing 7-day historical hourly baseline.
    """
    if not current_time:
        current_time = datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    # 1-hour window
    one_hour_ago = current_time - timedelta(hours=1)
    seven_days_ago = current_time - timedelta(days=7)

    # Filter events within last 7 days
    valid_events = []
    scans_last_1h = 0

    for e in events:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if ts >= seven_days_ago:
            valid_events.append(ts)
            if ts >= one_hour_ago:
                scans_last_1h += 1

    if len(valid_events) <= 1:
        return SpikeResult(
            entity_id=entity_id,
            entity_type=entity_type,
            scans_last_1h=scans_last_1h,
            scans_mean_7d=float(scans_last_1h),
            scans_std_7d=0.0,
            scan_spike_zscore=0.0,
            is_spike_event=False,
            spike_magnitude=1.0,
            risk_contribution="NONE"
        )

    # Group into hourly buckets across 7 days (168 hours)
    # Exclude current 1 hour from historical baseline to avoid leakage
    history_events = [ts for ts in valid_events if ts < one_hour_ago]

    if not history_events:
        # Entity only has activity in the current hour
        if scans_last_1h >= 20:
            # Immediate burst from a brand new entity
            return SpikeResult(
                entity_id=entity_id,
                entity_type=entity_type,
                scans_last_1h=scans_last_1h,
                scans_mean_7d=0.0,
                scans_std_7d=0.0,
                scan_spike_zscore=float(scans_last_1h),
                is_spike_event=True,
                spike_magnitude=float(scans_last_1h),
                risk_contribution="HIGH" if scans_last_1h < 50 else "CRITICAL"
            )
        return SpikeResult(
            entity_id=entity_id,
            entity_type=entity_type,
            scans_last_1h=scans_last_1h,
            scans_mean_7d=1.0,
            scans_std_7d=0.0,
            scan_spike_zscore=0.0,
            is_spike_event=False,
            spike_magnitude=1.0,
            risk_contribution="NONE"
        )

    # Calculate hourly counts
    bucket_counts = {}
    for ts in history_events:
        bucket = ts.replace(minute=0, second=0, microsecond=0)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    counts = list(bucket_counts.values())
    # Fill in zero-scan active hours across trailing 7 days
    total_hours_span = max(1, int((current_time - min(history_events)).total_seconds() / 3600))
    zero_buckets = max(0, total_hours_span - len(counts))
    all_counts = counts + [0] * zero_buckets

    mean_7d = float(np.mean(all_counts)) if all_counts else 0.0
    std_7d = float(np.std(all_counts)) if all_counts else 0.0
    epsilon = settings.SPIKE_EPSILON

    zscore = (scans_last_1h - mean_7d) / (std_7d + epsilon)
    magnitude = scans_last_1h / (mean_7d + epsilon)

    is_spike = zscore > settings.SPIKE_ZSCORE_THRESHOLD or magnitude > 5.0

    if zscore > 8.0 or magnitude > 10.0:
        risk = "CRITICAL"
    elif zscore > 5.0 or magnitude > 5.0:
        risk = "HIGH"
    elif zscore > 3.0 or magnitude > 3.0:
        risk = "MEDIUM"
    else:
        risk = "NONE"

    return SpikeResult(
        entity_id=entity_id,
        entity_type=entity_type,
        scans_last_1h=scans_last_1h,
        scans_mean_7d=round(mean_7d, 2),
        scans_std_7d=round(std_7d, 2),
        scan_spike_zscore=round(float(zscore), 2),
        is_spike_event=is_spike,
        spike_magnitude=round(float(magnitude), 2),
        risk_contribution=risk
    )

def compute_spike_features_dataframe(events_df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """
    Batch feature engineering for training datasets.
    """
    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    hourly = (
        df.groupby([entity_col, "hour_bucket"])
        .size()
        .reset_index(name="scans_last_1h")
    )

    hourly = hourly.sort_values([entity_col, "hour_bucket"])
    hourly["scans_mean_7d"] = (
        hourly.groupby(entity_col)["scans_last_1h"]
        .transform(lambda x: x.shift(1).rolling(168, min_periods=3).mean().fillna(x.mean()))
    )
    hourly["scans_std_7d"] = (
        hourly.groupby(entity_col)["scans_last_1h"]
        .transform(lambda x: x.shift(1).rolling(168, min_periods=3).std().fillna(0.0))
    )

    epsilon = 1.0
    hourly["scan_spike_zscore"] = (
        (hourly["scans_last_1h"] - hourly["scans_mean_7d"])
        / (hourly["scans_std_7d"] + epsilon)
    )
    hourly["is_spike_event"] = (hourly["scan_spike_zscore"] > 3.0).astype(int)
    hourly["spike_magnitude"] = (
        hourly["scans_last_1h"] / (hourly["scans_mean_7d"] + epsilon)
    )

    return hourly
