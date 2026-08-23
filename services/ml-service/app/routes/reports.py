from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from app.storage.event_store import event_store
from app.models.dbscan import clusterer

router = APIRouter(prefix="/ml", tags=["Consumer Complaints & Spatial Clustering"])

class IncidentReportRequest(BaseModel):
    batchId: Optional[str] = Field(None, description="Batch identifier")
    batch_id: Optional[str] = None
    shopId: Optional[str] = Field(None, description="Pharmacy ID")
    shop_id: Optional[str] = None
    packHash: Optional[str] = Field(None, description="Pack hash if scanned")
    pack_hash: Optional[str] = None
    latitude: float = Field(..., description="GPS latitude of reporting device")
    longitude: float = Field(..., description="GPS longitude of reporting device")
    reportType: Optional[str] = Field("SUSPECT_COUNTERFEIT", description="COUNTERFEIT | PACKAGING_TAMPERED | ADVERSE_REACTION | OTHER")
    report_type: Optional[str] = None
    notes: Optional[str] = Field(None, description="Inspector or patient notes")
    timestamp: Optional[str] = None

@router.post("/reports", status_code=201)
async def submit_report(payload: IncidentReportRequest):
    """
    Submits a consumer/inspector suspicious medicine incident report.
    """
    report_dict = {
        "batch_id": payload.batchId or payload.batch_id or "",
        "shop_id": payload.shopId or payload.shop_id or "",
        "pack_hash": payload.packHash or payload.pack_hash or "",
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "report_type": payload.reportType or payload.report_type or "SUSPECT_COUNTERFEIT",
        "notes": payload.notes or "",
        "timestamp": payload.timestamp or datetime.now(timezone.utc).isoformat()
    }

    saved = await event_store.add_report(report_dict)
    return {
        "status": "success",
        "message": "Incident report registered for ML clustering",
        "reportId": saved["report_id"],
        "report": {
            **saved,
            "timestamp": saved["timestamp"].isoformat()
        }
    }

@router.get("/reports")
async def list_reports(
    batchId: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500)
):
    """
    List recorded incident reports.
    """
    reports = await event_store.get_reports(batch_id=batchId)
    return {
        "total": len(reports),
        "reports": [
            {
                **r,
                "timestamp": r["timestamp"].isoformat()
            }
            for r in reports[:limit]
        ]
    }

@router.get("/clusters")
async def get_complaint_clusters(
    epsKm: Optional[float] = Query(None, description="Radius in km for DBSCAN clustering (default: 3.0km)"),
    minSamples: Optional[int] = Query(None, description="Minimum reports required to form a cluster (default: 3)")
):
    """
    Executes DBSCAN Spatial Clustering across incident reports to identify counterfeit hotspots.
    """
    reports = await event_store.get_reports()
    
    if epsKm or minSamples:
        from app.models.dbscan import ComplaintClusterer
        c = ComplaintClusterer(
            eps_km=epsKm or clusterer.eps_km,
            min_samples=minSamples or clusterer.min_samples
        )
        return c.cluster_reports(reports)

    return clusterer.cluster_reports(reports)
