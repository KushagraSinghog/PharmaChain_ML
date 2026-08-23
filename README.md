# 💊 PharmaChain — National Drug Track & Trace Infrastructure

> **Smart India Hackathon (SIH) 2026** · National Anti-Counterfeit Drug Security Platform  
> Combating the **₹40,000+ Crore annual counterfeit drug hazard** in developing economies.

---

## 🏛️ System Microservices Architecture

| Service | Port | Language / Framework | Purpose |
|---|---|---|---|
| **`pharma-core`** | `4000` | Node.js, ECDSA P-256 | Cryptographic Trust Root & High-Speed ES256 Pack Minting (5,000 packs in 518ms) |
| **`manufacturer-service`** | `3001` | Node.js / Express | Factory batch management, CDSCO regulatory schemas & laser QR generation |
| **`shopkeeper-service`** | `3002` | Node.js / Express | Point-of-Sale (POS) & retail pharmacy inventory custody intake/sale |
| **`consumer-service`** | `3003` | Node.js / Express | Zero-authentication public medicine verification & incident reporting |
| **`ml-service`** | `5000` | Python 3.11, FastAPI, scikit-learn | **Autonomous Secondary ML Anomaly, Behavioral Velocity & Risk Scoring Layer** |
| **`pharma-backend`** | `8080` | Java / Spring Boot | Hyperledger Fabric Consortium Blockchain Bridge & Ledger World State |

---

## 🤖 Machine Learning Layer (`services/ml-service`)

Complete behavioral anomaly detection and counterfeit risk scoring microservice:
- **Core Detectors:** QR Scan Velocity (A1/A2), Pharmacy Inventory Baselines (A3), Batch Risk Aggregation (A4), DBSCAN Spatial Hotspots (A5), Composite Risk Scoring (A6), Isolation Forest (A7).
- **Advanced Features:** Temporal Z-Score Spike Detector (B2), Supply Chain Monotonic Custody State Machine (B3), Real-Time Redis Streams Worker (B8), Population Stability Index (PSI) Model Drift Monitoring (B9).
- **Interactive UI:** Real-time visual telemetry dashboard and live attack simulator.

👉 **See Full Documentation:** [services/ml-service/README.md](file:///d:/Honey/Fun/PharmaChain/services/ml-service/README.md)

---

## 🚀 Quickstart

```powershell
# Run the ML service locally
cd services/ml-service
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

- **Live ML Telemetry Dashboard:** `http://localhost:5000/`
- **Interactive Swagger/OpenAPI Documentation:** `http://localhost:5000/docs`
- **Model Drift Diagnostics:** `http://localhost:5000/ml/model/health`
- **Test Suite:** `python -m pytest tests -v` (24/24 passing)
