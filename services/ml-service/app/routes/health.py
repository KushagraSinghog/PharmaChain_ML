from fastapi import APIRouter
from app.config import settings
from app.storage.event_store import event_store
from app.scoring.drift_monitor import drift_monitor
from app.models.isolation_forest import model_instance

router = APIRouter(prefix="/ml", tags=["Health & Drift Monitoring"])

@router.get("/health")
@router.get("/healthz")
async def health_check():
    """
    Service liveness and readiness probe.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "modelLoaded": model_instance.is_loaded,
        "environment": settings.ENVIRONMENT
    }

@router.get("/model/health")
async def model_health():
    """
    Feature B9: Model Drift Monitoring.
    Computes Population Stability Index (PSI) comparing recent production scores against baseline.
    """
    recent_scores = await event_store.get_recent_scores(limit=1000)
    health_report = drift_monitor.evaluate_drift(recent_scores)
    return {
        "service": settings.APP_NAME,
        "modelType": "Isolation Forest (n_estimators=200, contamination=0.02)",
        "isFitted": model_instance.is_loaded,
        **health_report
    }
