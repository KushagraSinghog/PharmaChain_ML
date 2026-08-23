from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from app.config import settings

@dataclass
class RiskBreakdown:
    ml_anomaly_contrib: float
    impossible_travel_contrib: float
    duplicate_clone_contrib: float
    complaint_cluster_contrib: float
    inventory_anomaly_contrib: float
    spike_penalty: float
    route_penalty: float
    raw_total: float

@dataclass
class CompositeRiskResult:
    risk_score: int
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    anomalies: List[str]
    requires_investigation: bool
    breakdown: Dict[str, Any]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "riskScore": self.risk_score,
            "riskLevel": self.risk_level,
            "anomalies": self.anomalies,
            "requiresInvestigation": self.requires_investigation,
            "breakdown": self.breakdown,
            "details": self.details
        }

def compute_composite_risk(
    ml_anomaly_score: float,  # Normalized 0.0 - 1.0 from Isolation Forest
    features: Dict[str, Any]
) -> CompositeRiskResult:
    """
    Computes weighted composite risk score (0-100), detects individual anomaly tags,
    and assigns actionable risk levels.
    """
    anomalies = []
    
    # 1. Base ML tabular contribution (40%)
    ml_contrib = float(ml_anomaly_score) * 100.0 * settings.WEIGHT_ML_ANOMALY
    if ml_anomaly_score > 0.70:
        anomalies.append("ML_BEHAVIORAL_OUTLIER")

    # 2. Travel & Velocity contribution (25%)
    is_impossible = features.get("is_impossible_travel", False)
    speed = features.get("implied_travel_speed_kmh", 0.0)
    travel_signal = 1.0 if is_impossible else min(1.0, speed / 500.0)
    travel_contrib = travel_signal * 100.0 * settings.WEIGHT_IMPOSSIBLE_TRAVEL

    if is_impossible:
        anomalies.append("IMPOSSIBLE_TRAVEL")
        if features.get("is_instant_teleport"):
            anomalies.append("INSTANT_TELEPORTATION")
    elif speed > 200.0:
        anomalies.append("HIGH_SCAN_VELOCITY")

    # 3. Duplicate / Clone QR Signal (15%)
    dup_count = features.get("duplicate_scan_count", 0)
    scan_count = features.get("scan_count", 1)
    clone_signal = 1.0 if dup_count > 0 else (0.5 if scan_count > 3 else 0.0)
    clone_contrib = clone_signal * 100.0 * settings.WEIGHT_DUPLICATE_CLONE

    if dup_count > 0:
        anomalies.append("DUPLICATE_SCAN_REPLAY")

    # 4. Complaint Cluster Signal (10%)
    reports_count = features.get("reports_count_7d", 0)
    cluster_signal = min(1.0, reports_count / 3.0)
    cluster_contrib = cluster_signal * 100.0 * settings.WEIGHT_COMPLAINT_CLUSTER

    if reports_count >= 3:
        anomalies.append("COMPLAINT_CLUSTER_HOTSPOT")

    # 5. Inventory Anomaly Signal (10%)
    is_inv_anomaly = features.get("is_inventory_anomaly", False)
    inv_ratio = features.get("inventory_ratio", 1.0)
    inv_signal = 1.0 if is_inv_anomaly else min(1.0, max(0.0, (inv_ratio - 3.0) / 7.0))
    inv_contrib = inv_signal * 100.0 * settings.WEIGHT_INVENTORY_ANOMALY

    if is_inv_anomaly:
        anomalies.append("SHOP_INVENTORY_ANOMALY")

    # 6. B2 Temporal Spike Modifiers
    spike_penalty = 0.0
    if features.get("is_shop_spike") or features.get("is_batch_spike"):
        anomalies.append("SCAN_SPIKE")
        max_mag = max(features.get("shop_spike_magnitude", 1.0), features.get("batch_spike_magnitude", 1.0))
        spike_penalty = min(25.0, max_mag * 2.5)

    # 7. B3 Route Anomaly Modifiers
    route_penalty = 0.0
    if features.get("is_backward_jump"):
        anomalies.append("ROUTE_BACKWARD_JUMP")
        route_penalty = 35.0  # Near-deterministic counterfeit signal
    elif features.get("is_tier_skip"):
        anomalies.append("ROUTE_TIER_SKIP")
        route_penalty = 15.0
    elif features.get("is_lateral_move"):
        anomalies.append("ROUTE_LATERAL_MOVE")
        route_penalty = 10.0

    raw_score = (
        ml_contrib
        + travel_contrib
        + clone_contrib
        + cluster_contrib
        + inv_contrib
        + spike_penalty
        + route_penalty
    )

    # Hard overrides for critical safety invariants
    if features.get("is_backward_jump") or features.get("is_impossible_travel"):
        # Instant escalation to high/critical
        raw_score = max(raw_score, 85.0)

    final_score = int(round(max(0.0, min(100.0, raw_score))))

    # Risk level classification
    if final_score <= settings.RISK_LEVEL_LOW_MAX:
        risk_level = "LOW"
    elif final_score <= settings.RISK_LEVEL_MED_MAX:
        risk_level = "MEDIUM"
    elif final_score <= settings.RISK_LEVEL_HIGH_MAX:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    requires_investigation = (risk_level in ("HIGH", "CRITICAL")) or len(anomalies) > 0

    breakdown = {
        "mlAnomalyContrib": round(ml_contrib, 2),
        "impossibleTravelContrib": round(travel_contrib, 2),
        "duplicateCloneContrib": round(clone_contrib, 2),
        "complaintClusterContrib": round(cluster_contrib, 2),
        "inventoryAnomalyContrib": round(inv_contrib, 2),
        "spikePenalty": round(spike_penalty, 2),
        "routePenalty": round(route_penalty, 2),
        "rawTotal": round(raw_score, 2)
    }

    details = {
        "impliedTravelSpeedKmh": features.get("implied_travel_speed_kmh", 0.0),
        "distanceBetweenScansKm": features.get("distance_between_scans_km", 0.0),
        "timeBetweenScansSec": features.get("time_between_scans_sec", 0.0),
        "inventoryRatio": features.get("inventory_ratio", 1.0),
        "routeTierDelta": features.get("route_tier_delta", 0),
        "routeAnomalyReason": features.get("route_anomaly_reason")
    }

    return CompositeRiskResult(
        risk_score=final_score,
        risk_level=risk_level,
        anomalies=anomalies,
        requires_investigation=requires_investigation,
        breakdown=breakdown,
        details=details
    )
