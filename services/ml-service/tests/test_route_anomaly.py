import pytest
from app.features.route_anomaly import check_route_anomaly, CustodyTier

def test_route_normal_progression():
    # Manufacturer (Tier 0) -> Warehouse (Tier 1) -> Shop (Tier 3) -> Consumer (Tier 4)
    history = [
        {"event_type": "MINTED", "shop_id": "MFR_01"},
        {"event_type": "WAREHOUSE_INTAKE", "shop_id": "WH_01"}
    ]
    res = check_route_anomaly("PACK-100", "INTAKE", "SHOP_01", history)
    assert res.is_backward_jump is False
    assert res.current_tier == CustodyTier.SHOP
    assert res.risk_contribution in ("NONE", "MEDIUM")

def test_route_backward_jump():
    # Pack is at Shop (Tier 3), but suddenly an event claims Warehouse Intake (Tier 1)
    history = [
        {"event_type": "MINTED", "shop_id": "MFR_01"},
        {"event_type": "INTAKE", "shop_id": "SHOP_01"}
    ]
    res = check_route_anomaly("PACK-100", "WAREHOUSE_INTAKE", "WH_02", history)
    assert res.is_backward_jump is True
    assert res.tier_delta == (CustodyTier.WAREHOUSE - CustodyTier.SHOP) # 1 - 3 = -2
    assert res.risk_contribution == "CRITICAL"

def test_route_lateral_move():
    # Pack moves between two shops at same tier (Tier 3 -> Tier 3)
    history = [
        {"event_type": "INTAKE", "shop_id": "SHOP_A"}
    ]
    res = check_route_anomaly("PACK-100", "INTAKE", "SHOP_B", history)
    assert res.is_lateral_move is True
    assert res.is_backward_jump is False
    assert res.risk_contribution == "MEDIUM"
