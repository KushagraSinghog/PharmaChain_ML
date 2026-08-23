import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.models.isolation_forest import model_instance
from app.stream.consumer import stream_consumer
from app.routes.anomaly import router as anomaly_router
from app.routes.reports import router as reports_router
from app.routes.health import router as health_router
from app.routes.dashboard_api import router as dashboard_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure model is loaded or trained baseline exists
    print(f"[{settings.APP_NAME}] Initializing ML Service...")
    if not model_instance.is_loaded:
        print(f"[{settings.APP_NAME}] Model artifact not found at {settings.MODEL_PATH}. Checking training pipeline...")
        # Auto-train baseline if missing
        try:
            from training.train import run_training
            run_training()
            model_instance._load_or_initialize()
        except Exception as e:
            print(f"[{settings.APP_NAME}] Auto-train fallback failed: {e}. Running with initialized defaults.")

    # Start background Redis streaming consumer worker (B8)
    await stream_consumer.start()
    print(f"[{settings.APP_NAME}] ML Service initialized and operational on port {settings.PORT}.")
    
    yield

    # Shutdown: Stop stream worker cleanly
    print(f"[{settings.APP_NAME}] Shutting down ML Service...")
    await stream_consumer.stop()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Secondary Fraud & Supply Chain Anomaly Detection Layer for PharmaChain",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for manufacturer dashboard, shopkeeper mobile, and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API route modules
app.include_router(anomaly_router)
app.include_router(reports_router)
app.include_router(health_router)
app.include_router(dashboard_router)

# Mount static files for dashboard UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/ml/health",
        "modelHealth": "/ml/model/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
