from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

class CustodyTier(IntEnum):
    MANUFACTURER = 0
    WAREHOUSE    = 1
    DISTRIBUTOR  = 2
    SHOP         = 3
    CONSUMER     = 4

# Mapping various system event types and ledger states to custody tiers
EVENT_TYPE_TO_TIER: Dict[str, CustodyTier] = {
    "MINT": CustodyTier.MANUFACTURER,
    "MINTED": CustodyTier.MANUFACTURER,
    "MANUFACTURED": CustodyTier.MANUFACTURER,
    "PACKAGED": CustodyTier.MANUFACTURER,
    
    "WAREHOUSE_INTAKE": CustodyTier.WAREHOUSE,
    "WAREHOUSE_DISPATCH": CustodyTier.WAREHOUSE,
    "IN_TRANSIT": CustodyTier.WAREHOUSE,
    "WAREHOUSE": CustodyTier.WAREHOUSE,
    
    "DISTRIBUTOR_INTAKE": CustodyTier.DISTRIBUTOR,
    "DISTRIBUTOR_DISPATCH": CustodyTier.DISTRIBUTOR,
    "DISTRIBUTOR": CustodyTier.DISTRIBUTOR,
    
    "INTAKE": CustodyTier.SHOP,
    "AT_SHOP": CustodyTier.SHOP,
    "RECEIVE": CustodyTier.SHOP,
    "STOCK": CustodyTier.SHOP,
    
    "SALE": CustodyTier.CONSUMER,
    "SELL": CustodyTier.CONSUMER,
    "SOLD": CustodyTier.CONSUMER,
    "ADMINISTERED": CustodyTier.CONSUMER,
    "CONSUMER_VERIFY": CustodyTier.CONSUMER,
    "VERIFY": CustodyTier.CONSUMER,
    "PATIENT_SCAN": CustodyTier.CONSUMER,
}

@dataclass
class RouteAnomalyResult:
    pack_hash: str
    prev_tier: Optional[int]
    current_tier: int
    tier_delta: Optional[int]
    is_backward_jump: bool
    is_tier_skip: bool
    is_lateral_move: bool
    risk_contribution: str  # "NONE", "MEDIUM", "HIGH", "CRITICAL"
    reason: Optional[str] = None

def get_tier_for_event_type(event_type: str) -> CustodyTier:
    normalized = event_type.strip().upper()
    if normalized in EVENT_TYPE_TO_TIER:
        return EVENT_TYPE_TO_TIER[normalized]
    # Default heuristics if custom string
    if "MINT" in normalized:
        return CustodyTier.MANUFACTURER
    if "WAREHOUSE" in normalized or "TRANSIT" in normalized:
        return CustodyTier.WAREHOUSE
    if "DISTRIBUTOR" in normalized:
        return CustodyTier.DISTRIBUTOR
    if "INTAKE" in normalized or "SHOP" in normalized:
        return CustodyTier.SHOP
    if "SALE" in normalized or "SOLD" in normalized or "CONSUMER" in normalized or "VERIFY" in normalized:
        return CustodyTier.CONSUMER
    return CustodyTier.SHOP

def check_route_anomaly(
    pack_hash: str,
    current_event_type: str,
    current_entity_id: str,
    previous_events: List[Dict[str, Any]]
) -> RouteAnomalyResult:
    """
    Validates supply chain custody progression.
    Flags backward moves (Shop -> Warehouse, Sold -> Intake), large stage skips,
    or unauthorized lateral shop transfers.
    """
    current_tier = get_tier_for_event_type(current_event_type)

    if not previous_events:
        return RouteAnomalyResult(
            pack_hash=pack_hash,
            prev_tier=None,
            current_tier=int(current_tier),
            tier_delta=None,
            is_backward_jump=False,
            is_tier_skip=False,
            is_lateral_move=False,
            risk_contribution="NONE",
            reason="INITIAL_CUSTODY_EVENT"
        )

    # Get the last recorded custody event
    last_event = previous_events[-1]
    last_event_type = last_event.get("event_type", "INTAKE")
    prev_entity = last_event.get("shop_id") or last_event.get("entity_id", "")
    prev_tier = int(get_tier_for_event_type(last_event_type))

    tier_delta = int(current_tier) - prev_tier
    is_backward = tier_delta < 0
    is_skip = tier_delta > 1
    is_lateral = (tier_delta == 0) and bool(current_entity_id and prev_entity and current_entity_id != prev_entity)

    # Note: If consumer verifies multiple times after SALE, that's tier 4 -> tier 4 (same tier, normal)
    if tier_delta == 0 and current_tier == CustodyTier.CONSUMER:
        is_lateral = False

    risk = "NONE"
    reason = None

    if is_backward:
        risk = "CRITICAL"
        reason = f"Illegal backward custody jump from tier {prev_tier} to tier {int(current_tier)} ({last_event_type} -> {current_event_type})"
    elif is_skip and tier_delta >= 3:
        risk = "HIGH"
        reason = f"Large tier bypass ({tier_delta} tiers jumped directly to {current_event_type})"
    elif is_skip and tier_delta == 2:
        # e.g. Manufacturer direct to Shop
        risk = "MEDIUM"
        reason = f"Tier skip: direct transfer bypassing distributor ({tier_delta} tiers)"
    elif is_lateral:
        risk = "MEDIUM"
        reason = f"Lateral transfer between entities ({prev_entity} -> {current_entity_id})"

    return RouteAnomalyResult(
        pack_hash=pack_hash,
        prev_tier=prev_tier,
        current_tier=int(current_tier),
        tier_delta=tier_delta,
        is_backward_jump=is_backward,
        is_tier_skip=is_skip,
        is_lateral_move=is_lateral,
        risk_contribution=risk,
        reason=reason
    )
