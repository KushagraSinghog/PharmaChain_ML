from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from app.storage.event_store import event_store
from app.features.builder import build_feature_vector
from app.models.isolation_forest import model_instance
from app.scoring.risk import compute_composite_risk
from app.features.inventory import compute_shop_inventory_features

router = APIRouter(prefix="/ml", tags=["ML Anomaly & Risk"])

class ScanEventRequest(BaseModel):
    packHash: Optional[str] = Field(None, description="SHA-256 pack identity hash")
    pack_hash: Optional[str] = None
    batchId: Optional[str] = Field(None, description="System batch ID")
    batch_id: Optional[str] = None
    shopId: Optional[str] = Field(None, description="Retail chemist or hospital dispensary ID")
    shop_id: Optional[str] = None
    latitude: Optional[float] = Field(None, description="Latitude of scan location")
    longitude: Optional[float] = Field(None, description="Longitude of scan location")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of event")
    eventType: Optional[str] = Field("SALE", description="INTAKE | SALE | WAREHOUSE | CONSUMER_VERIFY")
    event_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RiskAnalyzeResponse(BaseModel):
    riskScore: int = Field(..., description="Calculated composite risk score (0-100)")
    riskLevel: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    anomalies: List[str] = Field(..., description="List of detected anomaly codes")
    requiresInvestigation: bool = Field(..., description="Flag indicating if investigator action is advised")
    breakdown: Dict[str, Any]
    details: Dict[str, Any]
    eventId: Optional[str] = None
    packHash: Optional[str] = None
    batchId: Optional[str] = None

@router.post("/analyze", response_model=RiskAnalyzeResponse)
async def analyze_event(payload: ScanEventRequest):
    """
    Synchronous Real-Time Anomaly & Risk Analysis endpoint for a scan or supply chain event.
    """
    event_dict = {
        "pack_hash": payload.packHash or payload.pack_hash or "",
        "batch_id": payload.batchId or payload.batch_id or "",
        "shop_id": payload.shopId or payload.shop_id or "",
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "timestamp": payload.timestamp or datetime.now(timezone.utc).isoformat(),
        "event_type": payload.eventType or payload.event_type or "SALE",
        "metadata": payload.metadata or {}
    }

    # 1. Store event
    saved_event = await event_store.add_event(event_dict)

    # 2. Extract tabular feature vector
    features_dict, feature_vec = await build_feature_vector(saved_event)

    # 3. Model score from Isolation Forest
    ml_score = model_instance.score_event(feature_vec)

    # 4. Composite risk calculation
    risk_result = compute_composite_risk(ml_score, features_dict)

    response_data = {
        "riskScore": risk_result.risk_score,
        "riskLevel": risk_result.risk_level,
        "anomalies": risk_result.anomalies,
        "requiresInvestigation": risk_result.requires_investigation,
        "breakdown": risk_result.breakdown,
        "details": risk_result.details,
        "eventId": saved_event["event_id"],
        "packHash": saved_event["pack_hash"],
        "batchId": saved_event["batch_id"]
    }

    # Persist risk result in store
    await event_store.save_risk_result({
        **response_data,
        "timestamp": saved_event["timestamp"].isoformat()
    })

    return RiskAnalyzeResponse(**response_data)

@router.get("/anomalies")
async def get_anomalies(
    riskLevel: Optional[str] = Query(None, description="Filter by risk level: LOW | MEDIUM | HIGH | CRITICAL"),
    limit: int = Query(50, ge=1, le=500),
    page: int = Query(1, ge=1)
):
    """
    Query list of flagged anomalies for manufacturer and regulator queues.
    """
    return await event_store.get_anomalies(risk_level=riskLevel, limit=limit, page=page)

@router.get("/batches/{batch_id}/risk")
async def get_batch_risk(batch_id: str):
    """
    Batch-level risk summary aggregating scans, geo-spread, reports, and risk flags.
    """
    return await event_store.get_batch_risk_record(batch_id)

@router.get("/shops/{shop_id}/risk")
async def get_shop_risk(shop_id: str):
    """
    Shop-level behavioral baseline and inventory metrics.
    """
    events = await event_store.get_shop_events(shop_id)
    features = compute_shop_inventory_features(shop_id, events)
    return {
        "shopId": shop_id,
        "totalEvents": len(events),
        "dailyIntake": features.daily_intake,
        "dailySales": features.daily_sales,
        "weeklyIntake": features.weekly_intake,
        "weeklySales": features.weekly_sales,
        "inventoryRatio": features.inventory_ratio,
        "uniqueBatchesCount": features.unique_batches_count,
        "isInventoryAnomaly": features.is_inventory_anomaly,
        "status": "SUSPICIOUS" if features.is_inventory_anomaly else "NORMAL"
    }

@router.get("/packs/{pack_hash}/history")
async def get_pack_history(pack_hash: str):
    """
    Get full scan history and risk progression for a single pack hash.
    """
    events = await event_store.get_pack_events(pack_hash)
    return {
        "packHash": pack_hash,
        "totalScans": len(events),
        "timeline": [
            {
                "eventId": e["event_id"],
                "eventType": e["event_type"],
                "shopId": e["shop_id"],
                "batchId": e["batch_id"],
                "latitude": e["latitude"],
                "longitude": e["longitude"],
                "timestamp": e["timestamp"].isoformat()
            }
            for e in events
        ]
    }
