# PharmaChain — ML Anomaly Detection Module

## 1. Purpose

This document defines a separate Machine Learning layer for PharmaChain.

The existing PharmaChain architecture already provides strong cryptographic verification and immutable supply-chain events. The ML layer should **not replace those checks**.

Instead, ML is used as a **quality-of-life and early-warning system** that learns normal medicine movement patterns and identifies unusual activity for manufacturers/regulators.

The feature is:

> **ML-based Supply Chain Anomaly Detection & Risk Scoring**

Examples:
- The same medicine QR is scanned from two distant locations within an impossible time window.
- A shop receives unusually large quantities compared with its historical sales.
- A shop has a sustained high `AtShop` / low `Sold` ratio.
- A sudden spike of suspicious reports occurs around one batch or location.
- A newly registered seller shows an unusual order/scan pattern.

---

## 2. What ML Does vs What the Existing System Does

| Responsibility | Existing PharmaChain | ML Layer |
|---|---|---|
| QR authenticity | ES256 signature verification | ❌ |
| Pack identity | SHA-256 pack hash | ❌ |
| Supply-chain state | Hyperledger Fabric | ❌ |
| Recall enforcement | Fabric state | ❌ |
| Expiry verification | Database/token | ❌ |
| Detect unusual behaviour | Basic rules | **✅ ML** |
| Risk score | ❌ | **✅ ML** |
| Detect geographic clusters | ❌ | **✅ ML** |
| Prioritize regulator investigations | ❌ | **✅ ML** |

**Important:** A high ML risk score must never be presented as proof that a medicine is counterfeit. It is an investigation signal.

---

## 3. Proposed Architecture

Add one independent service:

```text
                    ┌──────────────────────────┐
                    │ Manufacturer / Regulator │
                    │       Dashboard          │
                    └────────────┬─────────────┘
                                 │
                           GET /anomalies
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     ML Anomaly Service   │
                    │        Python            │
                    │       FastAPI            │
                    │                          │
                    │ Isolation Forest         │
                    │ DBSCAN                   │
                    │ Risk Scoring             │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 │               │                │
                 ▼               ▼                ▼
          Shopkeeper DB    Manufacturer DB   Incident Store
                 │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                         PharmaChain Events
                    Intake → Sale → Reports
```

The ML service should be **internal** and should not be exposed directly through the public NGINX routes.

---

# 4. ML Features

## 4.1 QR Scan Velocity

Measure how frequently the same token/pack is appearing in the system.

Example:

```text
Pack A
10:00 → Delhi
10:08 → Jaipur
```

If the physical travel time is impossible, the event receives a high anomaly score.

Features:

```text
scan_count
time_between_scans
distance_between_scans
implied_travel_speed
unique_locations
```

---

## 4.2 Geographic Anomaly

For every scan:

```text
previous_lat
previous_lon
current_lat
current_lon
time_difference
distance
```

Calculate:

```text
implied_speed = distance / time_difference
```

An unusually high implied speed becomes a strong fraud signal.

This is especially useful for detecting **cloned QR codes**.

---

## 4.3 Shop Inventory Anomaly

Build historical behaviour for every registered pharmacy.

Example features:

```text
daily_intake
daily_sales
weekly_intake
weekly_sales
inventory_ratio
unique_batches
average_order_size
scan_frequency
```

Example:

```text
Normal shop:
100 packs received
85–100 packs sold

Suspicious:
1000 packs received
20 packs sold
```

The model can identify this as an unusual operational pattern.

---

## 4.4 Batch-Level Anomaly

Aggregate activity by batch:

```text
batch_id
total_scans
unique_shops
unique_locations
sales_per_day
suspicious_reports
ocr_mismatches
recall_events
```

A sudden increase in suspicious activity around one batch can increase its risk score.

---

## 4.5 Complaint / Report Clustering

Consumer reports can contain:

```text
batch_id
shop_id
location
report_type
timestamp
```

Use **DBSCAN** to detect geographical or behavioural clusters.

Example:

```text
Shop A ──┐
Shop B ──┼── 8 suspicious reports
Shop C ──┘
       within 3 km
       within 7 days
```

This can generate a regulator alert such as:

```text
"Unusual concentration of suspicious reports detected."
```

---

# 5. ML Algorithms

## Primary Model — Isolation Forest

Use **Isolation Forest** for unsupervised anomaly detection.

Why:

- We will initially have very few labelled counterfeit examples.
- It works well with tabular behavioural data.
- It can detect unusual combinations of otherwise normal values.
- It does not require a large labelled fraud dataset.

Example:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(training_features)
```

The model produces an anomaly score for new events.

---

## Secondary Model — DBSCAN

Use DBSCAN for spatial/behavioural clustering.

Useful for:

- suspicious report clusters
- geographically concentrated scans
- unusual shop activity
- batch complaint hotspots

DBSCAN is useful because the number of clusters does not need to be known beforehand.

---

# 6. Risk Scoring

Do not expose the raw ML score directly to users.

Convert multiple signals into a simple risk score:

```text
Risk Score =

    40% ML anomaly score
  + 25% impossible travel signal
  + 15% duplicate/cloned QR signal
  + 10% complaint cluster signal
  + 10% inventory anomaly signal
```

Example:

```text
Pack: SHA256(...)

ML anomaly score:       0.81
Travel anomaly:         HIGH
Duplicate scan:         HIGH
Complaint cluster:      LOW
Inventory anomaly:      NORMAL

Final Risk Score:       87 / 100

Risk Level: CRITICAL
```

The exact weights should be configurable and should be tuned using real data.

---

# 7. Risk Levels

| Score | Level | Meaning |
|---:|---|---|
| 0–30 | LOW | Normal behaviour |
| 31–60 | MEDIUM | Unusual behaviour; monitor |
| 61–80 | HIGH | Investigation recommended |
| 81–100 | CRITICAL | Strong anomaly; immediate review |

These levels are **risk indicators**, not authenticity verdicts.

---

# 8. Data Pipeline

```text
PharmaChain Events
       │
       ▼
Event Collector
       │
       ▼
Feature Builder
       │
       ├── Scan velocity
       ├── Geography
       ├── Time patterns
       ├── Inventory behaviour
       ├── Batch activity
       └── Consumer reports
       │
       ▼
Feature Vector
       │
       ├───────────────┐
       ▼               ▼
Isolation Forest    DBSCAN
       │               │
       └───────┬───────┘
               ▼
          Risk Scoring
               │
               ▼
       Anomaly Database
               │
               ▼
      Manufacturer / Regulator
             Dashboard
```

---

# 9. Example Feature Vector

For one shop:

```json
{
  "shop_id": "SHOP_001",
  "daily_intake": 420,
  "daily_sales": 61,
  "inventory_ratio": 6.88,
  "unique_batches": 37,
  "unique_locations": 2,
  "avg_daily_scans": 380,
  "suspicious_reports_7d": 6,
  "ocr_mismatches_7d": 2,
  "duplicate_scan_count": 4
}
```

The ML model converts this into an anomaly score.

---

# 10. Suggested ML Service API

### Analyze an Event

```http
POST /ml/analyze
```

Request:

```json
{
  "packHash": "a8f4c...",
  "batchId": "BATCH-001",
  "shopId": "SHOP_001",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "timestamp": "2026-08-22T14:30:00Z",
  "eventType": "SALE"
}
```

Response:

```json
{
  "riskScore": 78,
  "riskLevel": "HIGH",
  "anomalies": [
    "UNUSUAL_SCAN_VELOCITY",
    "SHOP_ACTIVITY_ANOMALY"
  ],
  "requiresInvestigation": true
}
```

---

### Get Anomalies

```http
GET /ml/anomalies?riskLevel=HIGH
```

Response:

```json
{
  "count": 12,
  "anomalies": [
    {
      "packHash": "a8f4c...",
      "batchId": "BATCH-001",
      "riskScore": 91,
      "riskLevel": "CRITICAL",
      "reason": "IMPOSSIBLE_TRAVEL"
    }
  ]
}
```

---

### Batch Risk Analysis

```http
GET /ml/batches/:batchId/risk
```

Example:

```json
{
  "batchId": "BATCH-001",
  "riskScore": 68,
  "riskLevel": "HIGH",
  "suspiciousScans": 14,
  "suspiciousLocations": 4
}
```

---

# 11. Recommended Technology

Keep the ML service separate from the Node.js/Spring Boot services.

```text
Language:        Python
API:             FastAPI
ML:              scikit-learn
Data Processing: pandas
Numerical:       NumPy
Model Storage:   joblib
Database:        PostgreSQL / MongoDB
Deployment:      Docker + Kubernetes
```

No LLM is required for this feature.

This keeps the ML feature cheap and explainable.

---

# 12. Training Strategy

Initially, there may not be enough real counterfeit data.

Therefore use a two-stage approach.

### Stage 1 — Synthetic Training Data

Generate normal and anomalous supply-chain behaviour.

Examples:

```text
Normal:
Delhi → Delhi
100 intake → 92 sales
5–20 scans/hour

Anomalous:
Delhi → Mumbai in 5 minutes
100 intake → 2 sales
500 scans/hour
same QR sold multiple times
```

Train the first Isolation Forest model using these behavioural patterns.

### Stage 2 — Real Production Data

Once PharmaChain receives real events:

```text
Real scans
   ↓
Feature store
   ↓
Human/regulator feedback
   ↓
Confirmed fraud / normal
   ↓
Retraining
```

The model gradually becomes better at identifying real-world behaviour.

---

# 13. Important Design Rule

### ML must never replace cryptographic verification.

The verification pipeline remains:

```text
QR Scan
   │
   ▼
ES256 Signature Verification
   │
   ├── INVALID → Counterfeit warning
   │
   ▼
Blockchain State Verification
   │
   ▼
Expiry / Recall / Sale State
   │
   ▼
ML Risk Analysis
   │
   ▼
Additional Risk Indicator
```

ML is therefore a **secondary intelligence layer**, not the security foundation.

---

# 14. Example End-to-End Scenario

A cloned QR code is used on two physical medicine boxes.

### First scan

```text
QR → Valid ES256
Blockchain → AtShop
ML → Normal
```

Result:

```text
LOW RISK
```

### Second scan

The same pack appears in another city shortly afterwards.

```text
QR → Valid ES256
Blockchain → Existing Sold state
ML → Impossible travel detected
```

Result:

```text
CRITICAL RISK
```

The system can then:

```text
1. Flag the pack
2. Alert manufacturer
3. Add incident to regulator queue
4. Show warning to consumer
5. Preserve the evidence for investigation
```

The cryptographic layer proves the token is genuine.

The ML layer detects that the **behaviour around that genuine token is suspicious**.

---

# 15. Dashboard Features

Add an **"AI/ML Risk Monitoring"** section to the manufacturer/regulator dashboard.

Display:

```text
┌─────────────────────────────────────────────┐
│          ML RISK MONITORING                 │
├─────────────────────────────────────────────┤
│ Critical Alerts                    12       │
│ High Risk                          31       │
│ Medium Risk                        87       │
│                                             │
│ Top Suspicious Batches                      │
│ BATCH-102       ██████████  91             │
│ BATCH-087       ████████    78             │
│ BATCH-201       ██████      64             │
│                                             │
│ Suspicious Activity Map                    │
│        ● ●                                  │
│      ● ● ●                                  │
│                  ●                          │
└─────────────────────────────────────────────┘
```

This gives judges a visible ML component instead of keeping ML hidden inside the backend.

---

# 16. MVP Implementation Plan

### Step 1
Create `ml-service`.

```text
services/
├── manufacturer/
├── shopkeeper/
├── consumer/
├── pharma-core/
└── ml-service/
```

### Step 2
Create an event collection table.

```text
ml_events
```

Store:

```text
event_id
pack_hash
batch_id
shop_id
event_type
latitude
longitude
timestamp
```

### Step 3
Create feature engineering code.

```text
ml-service/
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── anomaly.py
│   ├── features/
│   │   └── builder.py
│   ├── models/
│   │   ├── isolation_forest.py
│   │   └── dbscan.py
│   └── scoring/
│       └── risk.py
├── training/
│   └── train.py
└── requirements.txt
```

### Step 4
Train Isolation Forest using synthetic data.

### Step 5
Add DBSCAN for suspicious location/report clusters.

### Step 6
Connect `shopkeeper-service` and `consumer-service` events to the ML service.

### Step 7
Display risk alerts on the manufacturer/regulator dashboard.

---

# 17. Hackathon Demo

For the SIH demo, show one clear scenario instead of trying to demonstrate every ML capability.

### Demo

```text
1. Create genuine medicine pack
2. Scan it normally
3. Clone/reuse the QR token
4. Generate a second scan from a different location
5. ML detects impossible travel
6. Risk score jumps to CRITICAL
7. Manufacturer dashboard receives alert
```

### Judge Explanation

> "Blockchain tells us whether the medicine identity and supply-chain event are legitimate. ML looks at how that identity is being used across the network and detects behaviour that is statistically abnormal."

This gives PharmaChain a genuine ML component without weakening the existing cryptographic architecture.

---

# 18. Scope

### Included

- Isolation Forest anomaly detection
- DBSCAN clustering
- QR scan behaviour analysis
- Geographic anomaly detection
- Shop inventory anomaly detection
- Batch-level risk scoring
- Manufacturer/regulator alerts
- ML monitoring dashboard

### Not Included Initially

- Deep learning
- LLM-based verification
- ML-based QR authenticity
- Automatic declaration of a medicine as counterfeit
- Facial recognition
- Image-based medicine classification

The first ML version should stay **small, explainable, and deployable**.
