# PharmaChain ML Layer — Implementation Walkthrough

We have built and validated the **PharmaChain ML Service (`services/ml-service`)**, an autonomous, explainable secondary fraud and behavioral anomaly detection layer built with **Python 3.11**, **FastAPI**, and **scikit-learn**.

---

## 1. Architectural Overview

```
                      Scan / Supply Chain Event
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Cryptographic Guard   │  (ES256 + Hyperledger Fabric)
                     │ (Authenticity Ground) │
                     └───────────┬───────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │        PharmaChain ML Service (:5000)           │
        │                                                 │
        │  ┌───────────────────────────────────────────┐  │
        │  │ 1. Feature Extraction & Engineering       │  │
        │  │    • A1/A2: QR Velocity & Haversine Geo   │  │
        │  │    • A3: Shop Intake/Sales Baselines      │  │
        │  │    • B2: Rolling Z-Score Spike Detector   │  │
        │  │    • B3: Monotonic Custody State Machine  │  │
        │  └─────────────────────┬─────────────────────┘  │
        │                        │                        │
        │  ┌─────────────────────┴─────────────────────┐  │
        │  │ 2. Unsupervised Machine Learning Models   │  │
        │  │    • Isolation Forest (n=200, c=0.02)     │  │
        │  │    • DBSCAN Spatial Clustering (eps=3km)  │  │
        │  └─────────────────────┬─────────────────────┘  │
        │                        │                        │
        │  ┌─────────────────────┴─────────────────────┐  │
        │  │ 3. Composite Risk Engine (0-100)          │  │
        │  │    • LOW / MEDIUM / HIGH / CRITICAL       │  │
        │  │    • Explainable Anomaly Tags             │  │
        │  │    • B9: PSI Model Drift Health Monitor   │  │
        │  └─────────────────────┬─────────────────────┘  │
        │                        │                        │
        │  ┌─────────────────────┴─────────────────────┐  │
        │  │ 4. Streaming & Storage                    │  │
        │  │    • B8: Redis Stream Consumer Worker     │  │
        │  │    • Async Event & Report Repository      │  │
        │  └───────────────────────────────────────────┘  │
        └────────────────────────┬────────────────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │ Interactive ML Risk Monitoring Dashboard (UI)   │
        │ Live Telemetry, Simulator & Investigation Queue │
        └─────────────────────────────────────────────────┘
```

---

## 2. Implemented Features & Modules

### Core Features (Part A)
| Feature ID | Module | Implementation Description |
|---|---|---|
| **A1 & A2** | [velocity.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/features/velocity.py) · [geo.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/features/geo.py) | Haversine great-circle distance & implied travel speed calculation ($v = d / \Delta t$). Speeds $> 500\text{ km/h}$ trigger `IMPOSSIBLE_TRAVEL`. |
| **A3** | [inventory.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/features/inventory.py) | Historical pharmacy behavioral profiling (rolling 24h & 7d intake/sales, inventory ratio, hoarding detection). |
| **A4** | [event_store.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/storage/event_store.py) · [anomaly.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/routes/anomaly.py) | Batch-level risk aggregation across pack scan volume, geographic dispersion, and consumer complaints. |
| **A5** | [dbscan.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/models/dbscan.py) · [reports.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/routes/reports.py) | scikit-learn DBSCAN with Haversine ball-tree metric ($\text{eps}=3.0\text{ km}, \text{min\_samples}=3$) clustering consumer incident reports into counterfeit hotspots. |
| **A6** | [risk.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/scoring/risk.py) | Formula: $\text{Score} = 0.40 \times \text{ML} + 0.25 \times \text{Travel} + 0.15 \times \text{Clone} + 0.10 \times \text{Cluster} + 0.10 \times \text{Inventory} + \text{Penalties}$. Outputs `LOW` (0-30), `MEDIUM` (31-60), `HIGH` (61-80), and `CRITICAL` (81-100). |
| **A7** | [isolation_forest.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/models/isolation_forest.py) | Sklearn `IsolationForest(n_estimators=200, contamination=0.02, random_state=42)` for tabular outlier scoring without requiring historical labeled fraud data. |

### Advanced Features (Part B)
| Feature ID | Module | Implementation Description |
|---|---|---|
| **B2** | [spike_detector.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/features/spike_detector.py) | **Temporal Spike Detection (Rolling Z-Score):** Maintains 1-hour scan buckets vs. 7-day hourly baseline ($Z = \frac{X - \mu}{\sigma + 1}$). Flags $Z > 3.0$ and quantifies spike magnitude multiplier. |
| **B3** | [route_anomaly.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/features/route_anomaly.py) | **Supply Chain Route Anomaly State Machine:** Monotonic custody tiers: `MANUFACTURER (0) → WAREHOUSE (1) → DISTRIBUTOR (2) → SHOP (3) → CONSUMER (4)`. Detects illegal backward jumps (e.g. Shop $\to$ Warehouse $\to$ `CRITICAL`), large tier bypasses, and unauthorized lateral transfers. |
| **B8** | [consumer.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/stream/consumer.py) | **Real-Time Redis Streams Streaming:** Asynchronous background worker listening on `pharmachain:events` with consumer group `ml-processors`. Real-time scan-time scoring with millisecond latency, caching `risk:{packHash}` in Redis. Resilient fallback for standalone execution. |
| **B9** | [drift_monitor.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/scoring/drift_monitor.py) · [health.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/app/routes/health.py) | **Model Drift Monitoring:** Population Stability Index (PSI) comparing active inference scores against training baseline: $\text{PSI} = \sum (A_i - E_i) \ln(A_i / E_i)$. Returns `STABLE` ($<0.10$), `WARNING` ($0.10-0.25$), or `DRIFT` ($>0.25$) on `GET /ml/model/health`. |

---

## 3. Training Results & Metrics

Executed [training/train.py](file:///d:/Honey/Fun/PharmaChain/services/ml-service/training/train.py) on a synthetic logistics dataset of 4,200 records:
- **ROC-AUC Score:** `0.9935`
- **Anomaly Detection Recall:** `95.00%`
- **Normal Sample Precision:** `99.75%`
- **Overall Accuracy:** `97.86%`
- **Serialized Artifacts:**
  - Model: [models/isolation_forest.joblib](file:///d:/Honey/Fun/PharmaChain/services/ml-service/models/isolation_forest.joblib)
  - Baseline distribution: [models/baseline_score_dist.npy](file:///d:/Honey/Fun/PharmaChain/services/ml-service/models/baseline_score_dist.npy)

---

## 4. Test Suite Verification

Ran complete pytest test suite covering all modules:
```powershell
python -m pytest tests -v
```
**Results:** **24 passed out of 24 tests** in 1.85s:
- `test_api_health` ✅
- `test_api_analyze_normal_event` ✅
- `test_api_analyze_impossible_travel` ✅
- `test_api_get_anomalies` ✅
- `test_api_model_drift_health` ✅
- `test_api_reports_and_clustering` ✅
- `test_api_simulate_scenario` ✅
- `test_psi_identical_distributions` ✅
- `test_psi_drifted_distributions` ✅
- `test_drift_monitor_diagnostics` ✅
- `test_isolation_forest_training_and_scoring` ✅
- `test_dbscan_clustering_hotspot` ✅
- `test_route_normal_progression` ✅
- `test_route_backward_jump` ✅
- `test_route_lateral_move` ✅
- `test_composite_risk_normal` ✅
- `test_composite_risk_impossible_travel` ✅
- `test_composite_risk_backward_route_jump` ✅
- `test_spike_detector_normal_volume` ✅
- `test_spike_detector_massive_spike` ✅
- `test_haversine_known_cities` ✅
- `test_calculate_implied_speed` ✅
- `test_impossible_travel_flag` ✅
- `test_normal_travel_flag` ✅

---

## 5. Live Interactive Dashboard & Simulator

Access the dashboard at `http://localhost:5000/` or `http://localhost:5000/dashboard`:
- **Real-Time KPI Cards:** Critical Alerts, High Risk Flagged, Monitored Batches, Processed Events, B9 PSI Model Drift Status.
- **One-Click Live Attack Simulator:**
  1. *Impossible Travel Attack* (Delhi $\to$ Jaipur in 8 mins)
  2. *Cloned QR Burst Replay* (15 rapid scans across 5 cities)
  3. *Route Infiltration Jump* (Chemist $\to$ Warehouse backward move)
  4. *Pharmacy Inventory Hoarding* (Bulk intake with 0 sales)
  5. *DBSCAN Hotspot Formation* (6 consumer reports in Jaipur)
  6. *Normal Supply Flow* (Smooth genuine lifecycle)
- **Live Anomaly & Investigation Table:** Formatted risk pills, pack hashes, and flagged anomaly tags.
- **DBSCAN Complaint Hotspot Inspector:** Centroids, radii, and participating batches.
- **Telemetry Console:** Live streaming log of event intake and inference results.

---

## 6. How to Run the ML Service

```powershell
cd services/ml-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```
Open [http://localhost:5000/](http://localhost:5000/) for the interactive dashboard or [http://localhost:5000/docs](http://localhost:5000/docs) for the interactive Swagger/OpenAPI documentation.
