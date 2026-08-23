import asyncio
import uuid
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter

from app.storage.event_store import event_store
from app.features.builder import build_feature_vector
from app.models.isolation_forest import model_instance
from app.scoring.risk import compute_composite_risk
from app.models.dbscan import clusterer

router = APIRouter(prefix="/ml", tags=["Dashboard & Attack Simulation API"])

# Indian city coordinates reference
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
    "Lucknow": (26.8467, 80.9462)
}

class SimulateRequest(BaseModel):
    scenario: str  # "impossible_travel", "clone_burst", "route_backward_jump", "inventory_hoarding", "complaint_burst", "normal_flow"

@router.get("/stats")
async def get_dashboard_stats():
    """
    Real-time statistics for the ML Risk Monitoring Dashboard.
    """
    stats = await event_store.get_system_stats()
    reports = await event_store.get_reports()
    clusters_data = clusterer.cluster_reports(reports)
    
    recent_events = await event_store.get_all_events(limit=20)
    anomalies_data = await event_store.get_anomalies(limit=10)

    return {
        "stats": {
            **stats,
            "complaintClustersCount": clusters_data.get("clusters_count", 0),
            "clusteredComplaints": clusters_data.get("clustered_reports_count", 0)
        },
        "recentEvents": recent_events,
        "recentAnomalies": anomalies_data.get("anomalies", []),
        "clusters": clusters_data.get("clusters", [])
    }

@router.post("/clear")
async def clear_data():
    """
    Reset data store (useful for clean demo resets).
    """
    await event_store.clear_all()
    return {"status": "success", "message": "Event store reset"}

@router.post("/simulate")
async def simulate_scenario(payload: SimulateRequest):
    """
    Live Interactive Demonstration Engine: Triggers realistic fraud or normal scenarios.
    """
    scenario = payload.scenario.lower()
    now = datetime.now(timezone.utc)
    results = []

    if scenario == "impossible_travel":
        # Scenario: Delhi at 10:00 -> Jaipur at 10:08 (260km in 8 mins = 1950 km/h)
        pack_hash = f"PACK-SIM-TRAVEL-{uuid.uuid4().hex[:8]}"
        batch_id = "PC-BATCH-CIPLA0-20260822-7D3A1F"

        # Scan 1: Delhi (Intake at pharmacy)
        ev1 = {
            "pack_hash": pack_hash,
            "batch_id": batch_id,
            "shop_id": "SHOP_DELHI_01",
            "event_type": "INTAKE",
            "latitude": CITIES["Delhi"][0],
            "longitude": CITIES["Delhi"][1],
            "timestamp": (now - timedelta(minutes=8)).isoformat()
        }
        res1 = await _process_and_score(ev1)
        results.append({"step": "1. Normal Intake in Delhi", "event": ev1, "result": res1})

        # Scan 2: Jaipur (Sale attempt 8 mins later)
        ev2 = {
            "pack_hash": pack_hash,
            "batch_id": batch_id,
            "shop_id": "SHOP_JAIPUR_02",
            "event_type": "SALE",
            "latitude": CITIES["Jaipur"][0],
            "longitude": CITIES["Jaipur"][1],
            "timestamp": now.isoformat()
        }
        res2 = await _process_and_score(ev2)
        results.append({"step": "2. Cloned QR Scan in Jaipur (8 mins later)", "event": ev2, "result": res2})

    elif scenario == "clone_burst":
        # Scenario: 20 rapid scans across different cities within seconds
        pack_hash = f"PACK-SIM-CLONE-{uuid.uuid4().hex[:8]}"
        batch_id = "PC-BATCH-LUPIN-20260823-9F2B8C"
        city_list = list(CITIES.keys())

        for i in range(15):
            c_name = city_list[i % len(city_list)]
            lat, lon = CITIES[c_name]
            # Add slight jitter
            lat += random.uniform(-0.05, 0.05)
            lon += random.uniform(-0.05, 0.05)
            
            ev = {
                "pack_hash": pack_hash,
                "batch_id": batch_id,
                "shop_id": f"SHOP_{c_name.upper()}_{i+1:02d}",
                "event_type": "SALE" if i > 0 else "INTAKE",
                "latitude": lat,
                "longitude": lon,
                "timestamp": (now - timedelta(seconds=(15 - i) * 2)).isoformat()
            }
            res = await _process_and_score(ev)
            results.append({"step": f"Burst Scan #{i+1} in {c_name}", "event": ev, "result": res})

    elif scenario == "route_backward_jump":
        # Scenario: Pack at Chemist (Tier 3) -> jumps backward to Warehouse (Tier 1)
        pack_hash = f"PACK-SIM-ROUTE-{uuid.uuid4().hex[:8]}"
        batch_id = "PC-BATCH-SUNPH-20260821-4E1A9D"

        # Step 1: Mint
        ev1 = {
            "pack_hash": pack_hash,
            "batch_id": batch_id,
            "shop_id": "MFR_SUNPHARMA_01",
            "event_type": "MINTED",
            "latitude": CITIES["Ahmedabad"][0],
            "longitude": CITIES["Ahmedabad"][1],
            "timestamp": (now - timedelta(days=2)).isoformat()
        }
        res1 = await _process_and_score(ev1)
        results.append({"step": "1. Minted at Factory", "event": ev1, "result": res1})

        # Step 2: Shop Intake
        ev2 = {
            "pack_hash": pack_hash,
            "batch_id": batch_id,
            "shop_id": "SHOP_PUNE_01",
            "event_type": "INTAKE",
            "latitude": CITIES["Pune"][0],
            "longitude": CITIES["Pune"][1],
            "timestamp": (now - timedelta(hours=5)).isoformat()
        }
        res2 = await _process_and_score(ev2)
        results.append({"step": "2. Intake at Retail Chemist", "event": ev2, "result": res2})

        # Step 3: Warehouse Intake (Illegal backward move)
        ev3 = {
            "pack_hash": pack_hash,
            "batch_id": batch_id,
            "shop_id": "WH_MUMBAI_CENTRAL",
            "event_type": "WAREHOUSE_INTAKE",
            "latitude": CITIES["Mumbai"][0],
            "longitude": CITIES["Mumbai"][1],
            "timestamp": now.isoformat()
        }
        res3 = await _process_and_score(ev3)
        results.append({"step": "3. Infiltration: Ingested back into Warehouse", "event": ev3, "result": res3})

    elif scenario == "inventory_hoarding":
        # Scenario: Rogue pharmacy intakes 200 packs in a few hours with zero sales
        rogue_shop = "SHOP_ROGUE_DISPENSARY_77"
        batch_id = "PC-BATCH-HOARD-20260823-11AA22"

        for i in range(25):
            p_hash = f"PACK-HOARD-{uuid.uuid4().hex[:6]}"
            ev = {
                "pack_hash": p_hash,
                "batch_id": batch_id,
                "shop_id": rogue_shop,
                "event_type": "INTAKE",
                "latitude": CITIES["Lucknow"][0] + random.uniform(-0.01, 0.01),
                "longitude": CITIES["Lucknow"][1] + random.uniform(-0.01, 0.01),
                "timestamp": (now - timedelta(minutes=(30 - i))).isoformat()
            }
            res = await _process_and_score(ev)
            if i % 5 == 0 or i == 24:
                results.append({"step": f"Bulk Intake Pack #{i+1}", "event": ev, "result": res})

    elif scenario == "complaint_burst":
        # Scenario: 6 consumer complaints filed within 2km in Jaipur
        batch_id = "PC-BATCH-COUNTERFEIT-ALERT-99"
        base_lat, base_lon = CITIES["Jaipur"]

        for i in range(6):
            r_lat = base_lat + random.uniform(-0.015, 0.015)
            r_lon = base_lon + random.uniform(-0.015, 0.015)
            report = {
                "batch_id": batch_id,
                "shop_id": f"SHOP_JAIPUR_SUSPECT_{i%2 + 1}",
                "latitude": r_lat,
                "longitude": r_lon,
                "report_type": "COUNTERFEIT" if i % 2 == 0 else "PACKAGING_TAMPERED",
                "notes": f"Medicine blister pack color differed and barcode had blur. Report #{i+1}",
                "timestamp": (now - timedelta(hours=i*3)).isoformat()
            }
            saved = await event_store.add_report(report)
            results.append({"step": f"Complaint Report #{i+1}", "report": saved})

        # Run clustering
        all_reps = await event_store.get_reports()
        clusters = clusterer.cluster_reports(all_reps)
        results.append({"step": "DBSCAN Cluster Formed", "clusters": clusters})

    else:  # normal_flow
        batch_id = "PC-BATCH-GENUINE-20260823-OK"
        for i in range(5):
            p_hash = f"PACK-GENUINE-{uuid.uuid4().hex[:6]}"
            # 1. Mint
            await _process_and_score({
                "pack_hash": p_hash,
                "batch_id": batch_id,
                "shop_id": "MFR_CIPLA_GOA",
                "event_type": "MINTED",
                "latitude": 15.2993,
                "longitude": 74.1240,
                "timestamp": (now - timedelta(days=3)).isoformat()
            })
            # 2. Intake
            await _process_and_score({
                "pack_hash": p_hash,
                "batch_id": batch_id,
                "shop_id": "SHOP_BENGALURU_01",
                "event_type": "INTAKE",
                "latitude": CITIES["Bengaluru"][0],
                "longitude": CITIES["Bengaluru"][1],
                "timestamp": (now - timedelta(days=1)).isoformat()
            })
            # 3. Sale
            res = await _process_and_score({
                "pack_hash": p_hash,
                "batch_id": batch_id,
                "shop_id": "SHOP_BENGALURU_01",
                "event_type": "SALE",
                "latitude": CITIES["Bengaluru"][0],
                "longitude": CITIES["Bengaluru"][1],
                "timestamp": (now - timedelta(minutes=random.randint(10, 60))).isoformat()
            })
            results.append({"step": f"Normal Pack {i+1} Flow", "result": res})

    return {
        "scenario": scenario,
        "resultsCount": len(results),
        "timeline": results
    }

async def _process_and_score(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    saved = await event_store.add_event(event_dict)
    f_dict, f_vec = await build_feature_vector(saved)
    ml_score = model_instance.score_event(f_vec)
    risk_res = compute_composite_risk(ml_score, f_dict)

    res_payload = {
        "eventId": saved["event_id"],
        "packHash": saved["pack_hash"],
        "batchId": saved["batch_id"],
        "shopId": saved["shop_id"],
        "eventType": saved["event_type"],
        "riskScore": risk_res.risk_score,
        "riskLevel": risk_res.risk_level,
        "anomalies": risk_res.anomalies,
        "requiresInvestigation": risk_res.requires_investigation,
        "breakdown": risk_res.breakdown,
        "details": risk_res.details,
        "timestamp": saved["timestamp"].isoformat()
    }
    await event_store.save_risk_result(res_payload)
    return res_payload
