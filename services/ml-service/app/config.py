import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "PharmaChain ML Anomaly & Risk Service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    # Model artifact paths
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = BASE_DIR / "models" / "isolation_forest.joblib"
    BASELINE_DIST_PATH: Path = BASE_DIR / "models" / "baseline_score_dist.npy"
    
    # Anomaly detection hyperparameters
    MAX_PLAUSIBLE_SPEED_KMH: float = 500.0  # Speed above which travel is physically impossible
    MAX_PLAUSIBLE_DISTANCE_METERS_INSTANT: float = 50.0 # Distance within < 5 seconds
    
    # B2: Spike detector parameters
    SPIKE_ZSCORE_THRESHOLD: float = 3.0
    SPIKE_EPSILON: float = 1.0
    
    # A5: DBSCAN cluster parameters
    DBSCAN_EPS_KM: float = 3.0
    DBSCAN_MIN_SAMPLES: int = 3
    DBSCAN_TIME_WINDOW_DAYS: int = 7
    EARTH_RADIUS_KM: float = 6371.0088

    # A6: Composite Risk Scoring Weights
    WEIGHT_ML_ANOMALY: float = 0.40
    WEIGHT_IMPOSSIBLE_TRAVEL: float = 0.25
    WEIGHT_DUPLICATE_CLONE: float = 0.15
    WEIGHT_COMPLAINT_CLUSTER: float = 0.10
    WEIGHT_INVENTORY_ANOMALY: float = 0.10
    
    # Risk Level Cutoffs
    RISK_LEVEL_LOW_MAX: int = 30
    RISK_LEVEL_MED_MAX: int = 60
    RISK_LEVEL_HIGH_MAX: int = 80
    # 81 - 100 = CRITICAL

    # B8: Redis Streams Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_STREAM_KEY: str = "pharmachain:events"
    REDIS_GROUP_NAME: str = "ml-processors"
    REDIS_CONSUMER_ID: str = "ml-worker-1"
    REDIS_ENABLED: bool = True
    REDIS_FALLBACK_IF_UNAVAILABLE: bool = True

    # B9: Model Drift PSI Thresholds
    PSI_STABLE_MAX: float = 0.10
    PSI_WARNING_MAX: float = 0.25

    # Storage
    STORAGE_TYPE: str = "memory" # "memory", "sqlite", "mongodb"
    SQLITE_DB_PATH: Path = BASE_DIR / "storage" / "ml_events.db"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()

# Ensure model directory exists
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
