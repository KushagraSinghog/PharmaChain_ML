# 🤖 PharmaChain — ML Pipeline Breakdown

> What the docs mandate you build, what you can add to stand out, and how it all wires together.

---

## Mental Model First

```
QR Scan / Supply Event
        │
        ▼
  [Crypto Layer]  ← Already built. Not your job.
  ES256 + Fabric
        │
        ▼
  [ML Layer]      ← YOUR JOB
  Feature Engineering → Anomaly Models → Risk Score → Dashboard Alert
```

ML is the **secondary intelligence layer** — it never overrides crypto. It answers the question:
> *"Even if this QR is cryptographically valid, is the behaviour around it suspicious?"*

---

## Part A — What the Docs Mandate You Build

### A1. QR Scan Velocity Detector

**The idea:** The same pack hash appears in the system multiple times. If the time gap and geographic distance imply an impossible travel speed, it's a cloned QR.

**Features to engineer per pack event:**

| Feature | How to compute |
|---|---|
| `scan_count` | Count of events for this `pack_hash` |
| `time_between_scans` | `current_ts - previous_ts` (seconds) |
| `distance_between_scans` | Haversine distance between consecutive scan lat/lon (km) |
| `implied_travel_speed` | `distance / time_diff` (km/h) |
| `unique_locations` | Count of distinct (lat, lon) clusters for this pack |

**Trigger condition:** `implied_travel_speed > ~500 km/h` (faster than any land transport, possibly faster than commercial flight) → HIGH/CRITICAL signal.

---

### A2. Geographic Anomaly Detector

**The idea:** Even for *different* packs of the same batch, if they're appearing in geographically scattered, implausible locations simultaneously, something is wrong.

**Features per scan event:**

| Feature | Source |
|---|---|
| `previous_lat`, `previous_lon` | Last scan location of this pack |
| `current_lat`, `current_lon` | Current scan location |
| `time_difference` | Seconds between events |
| `distance` | Haversine(prev, curr) |
| `implied_speed` | `distance / time_difference` |

**Use case:** Cloned QR codes — the original and clone both get scanned in different cities in the same hour.

---

### A3. Shop Inventory Anomaly Detector

**The idea:** Build a behavioural baseline for every registered pharmacy. Flag deviations.

**Features per shop, aggregated over time windows:**

| Feature | Window |
|---|---|
| `daily_intake` | Last 24h |
| `daily_sales` | Last 24h |
| `weekly_intake` | Last 7d |
| `weekly_sales` | Last 7d |
| `inventory_ratio` | `total_intake / total_sales` |
| `unique_batches` | Distinct batch IDs handled |
| `average_order_size` | Mean packs per intake event |
| `scan_frequency` | Scans per hour |

**Red flags:**
- High intake, near-zero sales → hoarding / diversion
- Sudden 10× spike in order size vs. historical average
- A newly registered shop immediately handling industrial-scale quantities

---

### A4. Batch-Level Risk Aggregator

**The idea:** Some batches attract disproportionate suspicious activity. Aggregate signals at the batch level.

**Features per batch:**

| Feature | Source |
|---|---|
| `total_scans` | All pack events for this batch |
| `unique_shops` | Distinct shop IDs touching this batch |
| `unique_locations` | Geographic spread of scans |
| `sales_per_day` | Rolling daily sale count |
| `suspicious_reports` | Consumer-reported incidents for this batch |
| `ocr_mismatches` | (Future) Physical label vs. digital record mismatches |
| `recall_events` | Whether this batch has been recalled |

**Output:** A per-batch risk score that manufacturers and regulators can sort/filter.

---

### A5. Complaint / Report Cluster Detector (DBSCAN)

**The idea:** Consumer reports have location + timestamp. Use DBSCAN to find geographic or temporal clusters of complaints — a sign of a counterfeit distribution ring operating in one area.

**Input per report:**

```json
{
  "batch_id": "PC-BATCH-...",
  "shop_id": "SHOP_001",
  "location": { "lat": 28.61, "lon": 77.20 },
  "report_type": "COUNTERFEIT | QUALITY | OTHER",
  "timestamp": "2026-08-22T14:30:00Z"
}
```

**DBSCAN parameters:**
- `eps` (neighbourhood radius): ~3 km for geographic clustering
- `min_samples`: 3–5 reports to form a cluster
- Time window: rolling 7-day window

**Output:** Cluster ID, cluster size, centroid location, participating shops → regulator alert.

---

### A6. Composite Risk Scoring Engine

Combine all signals into one actionable number, per the doc's exact formula:

$$\text{Risk Score} = 0.40 \times \text{ML\_anomaly} + 0.25 \times \text{impossible\_travel} + 0.15 \times \text{duplicate\_clone} + 0.10 \times \text{complaint\_cluster} + 0.10 \times \text{inventory\_anomaly}$$

**Risk levels:**

| Score | Level | Action |
|---|---|---|
| 0–30 | LOW | Normal |
| 31–60 | MEDIUM | Monitor |
| 61–80 | HIGH | Flag for investigation |
| 81–100 | CRITICAL | Immediate regulator review |

> [!IMPORTANT]
> The score is an **investigation signal**, never an authenticity verdict. Never display it as "this medicine is fake."

---

### A7. ML Models

| Model | Role | Library |
|---|---|---|
| **Isolation Forest** | Primary unsupervised anomaly detector on tabular features | `sklearn.ensemble.IsolationForest` |
| **DBSCAN** | Spatial/behavioural complaint clustering | `sklearn.cluster.DBSCAN` |

```python
# Isolation Forest — doc-specified params
IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
```

**Why Isolation Forest?** No labelled counterfeit data exists yet. It works on tabular behavioural features without a fraud dataset.

---

### A8. Service Structure (as specified in doc)

```
services/ml-service/
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── anomaly.py
│   ├── features/
│   │   └── builder.py          ← feature engineering
│   ├── models/
│   │   ├── isolation_forest.py
│   │   └── dbscan.py
│   └── scoring/
│       └── risk.py             ← composite risk formula
├── training/
│   └── train.py                ← synthetic data training
└── requirements.txt
```

---

### A9. API Endpoints (mandated)

#### `POST /ml/analyze` — Real-time event scoring
```json
// Request
{ "packHash": "a8f4c...", "batchId": "BATCH-001", "shopId": "SHOP_001",
  "latitude": 28.6139, "longitude": 77.2090,
  "timestamp": "2026-08-22T14:30:00Z", "eventType": "SALE" }

// Response
{ "riskScore": 78, "riskLevel": "HIGH",
  "anomalies": ["UNUSUAL_SCAN_VELOCITY", "SHOP_ACTIVITY_ANOMALY"],
  "requiresInvestigation": true }
```

#### `GET /ml/anomalies?riskLevel=HIGH` — List flagged anomalies
```json
{ "count": 12, "anomalies": [
  { "packHash": "a8f4c...", "batchId": "BATCH-001",
    "riskScore": 91, "riskLevel": "CRITICAL", "reason": "IMPOSSIBLE_TRAVEL" }
]}
```

#### `GET /ml/batches/:batchId/risk` — Batch risk summary
```json
{ "batchId": "BATCH-001", "riskScore": 68, "riskLevel": "HIGH",
  "suspiciousScans": 14, "suspiciousLocations": 4 }
```

---

### A10. Training Strategy

**Stage 1 — Synthetic data** (before production data exists):
- Generate normal patterns: same city, 100 intake → 90 sales, 5–20 scans/hr
- Generate anomalous patterns: cross-city in 5 minutes, 100 intake → 2 sales, same QR sold 3 times

**Stage 2 — Real production data:**
- Collect real events → human/regulator feedback → confirmed fraud labels → retrain

---

### A11. Event Store (ml_events table)

```text
event_id | pack_hash | batch_id | shop_id | event_type | latitude | longitude | timestamp
```

This is the raw log the feature builder reads from.

---

### A12. Dashboard Integration

Add a **"ML Risk Monitoring"** section to the manufacturer/regulator dashboard showing:
- Critical / High / Medium alert counts
- Top suspicious batches with risk bars
- Suspicious activity map (geographic heatmap)

---

## Part B — Features You Can Add (Not in Docs)

These are additions that would meaningfully strengthen the pipeline and impress judges/reviewers without contradicting the existing architecture.

---

### B1. 🕐 Time-of-Day / Day-of-Week Profiling

**What:** Pharmacies have natural operating hours. A shop scanning packs at 3 AM is anomalous.

**How:**
- Build per-shop `hour_of_day` and `day_of_week` histograms over their history
- Flag scans that fall in the bottom 5% of their historical activity window
- Feature: `is_off_hours_scan`, `off_hours_zscore`

**Value:** Catches automated scanning scripts that bots use to cycle through cloned QRs overnight.

---

### B2. 📈 Temporal Spike Detection (Rolling Z-Score)

**What:** A batch or shop suddenly shows a 10× spike in scan volume within a short time window — a hallmark of coordinated clone-scanning or bot attacks.

**Approach:**

Maintain a rolling hourly scan count per `(entity_id, entity_type)` — where `entity_type` is either `shop` or `batch`. Compute the Z-score of the current window count against the trailing 7-day history.

$$Z = \frac{X_{\text{current}} - \mu_{\text{7d}}}{\sigma_{\text{7d}} + \epsilon}$$

where $\epsilon = 1$ prevents division by zero for new entities.

**Features engineered:**

| Feature | Description |
|---|---|
| `scans_last_1h` | Raw scan count in the current 1-hour bucket |
| `scans_mean_7d` | Rolling 7-day hourly mean for this entity |
| `scans_std_7d` | Rolling 7-day hourly std for this entity |
| `scan_spike_zscore` | Z-score of current window vs. 7d history |
| `is_spike_event` | `1` if `zscore > 3.0`, else `0` |
| `spike_magnitude` | `scans_last_1h / scans_mean_7d` (ratio, e.g. 10× normal) |

**Implementation (`features/spike_detector.py`):**

```python
import pandas as pd
import numpy as np

def compute_spike_features(events_df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """
    events_df: DataFrame with columns [entity_id, timestamp]
    entity_col: 'shop_id' or 'batch_id'
    Returns spike features per entity per hour bucket.
    """
    events_df["hour_bucket"] = events_df["timestamp"].dt.floor("1H")

    # Count scans per entity per hour
    hourly = (
        events_df.groupby([entity_col, "hour_bucket"])
        .size()
        .reset_index(name="scans_last_1h")
    )

    # Rolling 7-day stats (168 hourly buckets)
    hourly = hourly.sort_values([entity_col, "hour_bucket"])
    hourly["scans_mean_7d"] = (
        hourly.groupby(entity_col)["scans_last_1h"]
        .transform(lambda x: x.shift(1).rolling(168, min_periods=5).mean())
    )
    hourly["scans_std_7d"] = (
        hourly.groupby(entity_col)["scans_last_1h"]
        .transform(lambda x: x.shift(1).rolling(168, min_periods=5).std())
    )

    # Z-score and spike flag
    epsilon = 1.0
    hourly["scan_spike_zscore"] = (
        (hourly["scans_last_1h"] - hourly["scans_mean_7d"])
        / (hourly["scans_std_7d"] + epsilon)
    )
    hourly["is_spike_event"] = (hourly["scan_spike_zscore"] > 3.0).astype(int)
    hourly["spike_magnitude"] = (
        hourly["scans_last_1h"] / (hourly["scans_mean_7d"] + epsilon)
    )

    return hourly
```

**Alert thresholds:**

| Z-Score | Spike Magnitude | Risk Contribution |
|---|---|---|
| > 3.0 | > 3× | MEDIUM |
| > 5.0 | > 5× | HIGH |
| > 8.0 | > 10× | CRITICAL |

**Integration point:** Run this check inside `POST /ml/analyze` for the `shopId` and `batchId` of every incoming event. Store the result in `ml_events` alongside the event. If `is_spike_event = 1`, add `"SCAN_SPIKE"` to the `anomalies[]` list in the response.

**Why it's demo-friendly:** You can demonstrate it live — mint a batch, scan the same QR 50 times in a loop from a script, and watch the dashboard alert go from LOW → CRITICAL in real time.

**Value:** Catches coordinated clone-scanning events and bot attacks. Simple, explainable, and the most visually impressive signal for a live demo.

---

### B3. 🗺️ Supply Chain Route Anomaly

**What:** Medicine should flow in a strict, predictable direction: `Manufacturer → Warehouse → Distributor → Retail Shop`. A pack appearing at a retail pharmacy *before* it was ever logged at a warehouse is a sign of mid-supply-chain counterfeit infiltration — exactly the attack scenario described in the master plan.

**Approach:**

Assign a monotonically increasing **custody tier** to every entity type, then validate that every new custody event for a pack has a tier ≥ the last recorded tier for that pack. Any backwards or tier-skipping transition is flagged.

```
Tier 0 — MANUFACTURER  (MINTED state)
Tier 1 — WAREHOUSE     (IN_TRANSIT state)
Tier 2 — DISTRIBUTOR   (DISTRIBUTOR state)
Tier 3 — SHOP          (AT_SHOP state)
Tier 4 — CONSUMER      (SOLD / ADMINISTERED state)
```

**Valid transitions (forward only):**

```
0 → 1 → 2 → 3 → 4   ✅ Normal
0 → 3               ⚠️ Tier skip — possible for direct factory-to-pharmacy delivery
                        (flag but not block — configurable)
3 → 1               ❌ BACKWARD — strong anomaly signal
2 → 0               ❌ BACKWARD — strong anomaly signal
3 → 3 (diff shop)   ⚠️ Lateral move — shop-to-shop transfer, needs review
```

**Features engineered per custody event:**

| Feature | Description |
|---|---|
| `prev_tier` | Tier of the last recorded entity for this `pack_hash` |
| `current_tier` | Tier of the entity triggering this event |
| `tier_delta` | `current_tier - prev_tier` |
| `is_backward_jump` | `1` if `tier_delta < 0` |
| `is_tier_skip` | `1` if `tier_delta > 1` (jumped over a stage) |
| `is_lateral_move` | `1` if `tier_delta == 0` and entity changed |
| `expected_next_tier` | Expected tier given valid supply chain logic |

**Implementation (`features/route_anomaly.py`):**

```python
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

class CustodyTier(IntEnum):
    MANUFACTURER = 0
    WAREHOUSE    = 1
    DISTRIBUTOR  = 2
    SHOP         = 3
    CONSUMER     = 4

# Map Fabric ledger states → tiers
LEDGER_STATE_TO_TIER = {
    "MINTED":       CustodyTier.MANUFACTURER,
    "IN_TRANSIT":   CustodyTier.WAREHOUSE,
    "DISTRIBUTOR":  CustodyTier.DISTRIBUTOR,
    "AT_SHOP":      CustodyTier.SHOP,
    "SOLD":         CustodyTier.CONSUMER,
    "ADMINISTERED": CustodyTier.CONSUMER,
}

@dataclass
class RouteAnomalyResult:
    pack_hash: str
    prev_tier: Optional[int]
    current_tier: int
    tier_delta: Optional[int]
    is_backward_jump: bool
    is_tier_skip: bool
    is_lateral_move: bool
    risk_contribution: str   # "NONE", "MEDIUM", "HIGH", "CRITICAL"

def check_route_anomaly(
    pack_hash: str,
    current_ledger_state: str,
    last_custody_record: Optional[dict]   # from ml_events or custody store
) -> RouteAnomalyResult:

    current_tier = LEDGER_STATE_TO_TIER.get(current_ledger_state)
    if current_tier is None:
        raise ValueError(f"Unknown ledger state: {current_ledger_state}")

    if last_custody_record is None:
        # First event for this pack — no anomaly possible yet
        return RouteAnomalyResult(
            pack_hash=pack_hash, prev_tier=None, current_tier=current_tier,
            tier_delta=None, is_backward_jump=False,
            is_tier_skip=False, is_lateral_move=False,
            risk_contribution="NONE"
        )

    prev_tier = last_custody_record["tier"]
    prev_entity = last_custody_record["entity_id"]
    current_entity = last_custody_record.get("current_entity_id", "")

    tier_delta = current_tier - prev_tier
    is_backward = tier_delta < 0
    is_skip = tier_delta > 1
    is_lateral = (tier_delta == 0) and (current_entity != prev_entity)

    # Assign risk contribution
    if is_backward:
        risk = "CRITICAL"   # e.g. Shop → Warehouse = impossible in legitimate flow
    elif is_skip and tier_delta > 2:
        risk = "HIGH"       # jumped 3+ tiers (Manufacturer → Consumer)
    elif is_skip:
        risk = "MEDIUM"     # jumped 2 tiers (e.g. Manufacturer → Shop)
    elif is_lateral:
        risk = "MEDIUM"     # pack moved between same-tier entities
    else:
        risk = "NONE"

    return RouteAnomalyResult(
        pack_hash=pack_hash, prev_tier=prev_tier, current_tier=current_tier,
        tier_delta=tier_delta, is_backward_jump=is_backward,
        is_tier_skip=is_skip, is_lateral_move=is_lateral,
        risk_contribution=risk
    )
```

**Integration point:** Call `check_route_anomaly()` inside `POST /ml/analyze`. The function requires the last custody record for the `pack_hash`, which you maintain in the `ml_events` table keyed by `pack_hash`. Store the result with the event. If `is_backward_jump = True`, add `"ROUTE_BACKWARD_JUMP"` to `anomalies[]` and boost the risk score directly — this is a near-deterministic fraud signal that warrants CRITICAL regardless of the Isolation Forest score.

**New anomaly codes added to the API response:**
- `"ROUTE_BACKWARD_JUMP"` — pack moving against the legitimate custody flow
- `"ROUTE_TIER_SKIP"` — pack bypassed one or more expected custody stages
- `"ROUTE_LATERAL_MOVE"` — pack transferred between same-tier entities (review needed)

**Value:** Catches mid-supply-chain infiltration of counterfeit stock — the exact attack described in the master plan's "Mid-Supply Infiltration" threat row.

---

### B4. 🏪 New Seller Behaviour Fingerprinting

**What:** Newly registered sellers (< 30 days old) handling industrial-scale quantities are disproportionately suspicious. Established pharmacies have predictable order sizes.

**How:**
- Feature: `days_since_registration`, `order_size_vs_peer_percentile`
- Compare the shop's first-week order volumes against the distribution of first-week volumes across all historical shops
- Flag if a new shop's intake in week 1 is in the top 1% of all new-shop intakes

**Value:** Catches fly-by-night fake pharmacies created solely to launder counterfeit stock.

---

### B5. 🔗 Graph-Based Collusion Detection

**What:** Two or more shops might be colluding — accepting the same pack into inventory simultaneously, or passing packs back and forth to game the system.

**How:**
- Build a bipartite graph: `Pack ↔ Shop`
- Detect cliques where multiple shops share an unusually high fraction of the same pack hashes
- Use `networkx` for graph construction; flag subgraphs with high edge density

**Feature:** `shared_pack_ratio_with_shop_X`, `clique_size`

**Value:** Detects organized distribution rings — the hardest class of fraud to catch with simple rules.

---

### B6. 📊 Expiry-Date Distribution Anomaly

**What:** A legitimate batch has expiry dates concentrated in a single window (e.g. all expire Aug 2028). Counterfeit batches may reuse the same JWT structure but with oddly distributed or mismatched expiry dates.

**How:**
- Compute the distribution of `expiryDate` values across packs claimed to belong to a batch
- Flag batches where `std(expiryDate_days)` is non-zero (it should be exactly 0 for a real batch)
- Feature: `expiry_date_std`, `expiry_date_unique_count`

**Value:** Catches a specific forgery technique where someone generates fake JWTs with randomised expiry dates.

---

### B7. 🧠 One-Class SVM as Ensemble Member

**What:** Add a One-Class SVM alongside Isolation Forest as an ensemble. The two models often catch different anomaly subspaces, and combining them (e.g. by averaging anomaly scores) is more robust.

**How:**
```python
from sklearn.svm import OneClassSVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.02)
# Final score = 0.6 * IF_score + 0.4 * OCSVM_score
```

**Value:** More robust anomaly detection; shows ML sophistication without adding data complexity.

---

### B8. ⏱️ Real-Time Streaming (Redis Streams)

**What:** Currently the doc implies the ML service polls databases periodically. This means fraud is detected *minutes after* it happens. Adding a streaming layer means ML scores are computed **at the moment of each scan** — transforming the ML layer from a batch report tool into an active, real-time safety system.

**Approach:**

Use **Redis Streams** (simpler than Kafka for a single-service demo, natively supported by Redis which is likely already in the stack). Each time `shopkeeper-service` or `consumer-service` processes a scan event, they publish to a Redis stream. The `ml-service` runs a **consumer group** that picks up each event, runs the full feature + scoring pipeline, and writes the risk result back — all within milliseconds.

**Full Data Flow:**

```
shopkeeper-service          consumer-service
      │                           │
      │  XADD pharmachain:events  │  XADD pharmachain:events
      └──────────┬────────────────┘
                 │
                 ▼
         Redis Stream
     "pharmachain:events"
                 │
                 │ XREADGROUP (Consumer Group: ml-processors)
                 ▼
          ml-service
     ┌─────────────────────┐
     │ 1. Parse event      │
     │ 2. Build features   │
     │    (B2 spike check, │
     │     B3 route check, │
     │     geo anomaly...) │
     │ 3. Run IF model     │
     │ 4. Compute risk     │
     │ 5. Write to result  │
     │    store            │
     └─────────────────────┘
                 │
          ┌──────┴──────────────────────┐
          ▼                             ▼
  MongoDB anomaly_results        Redis Key-Value
  (persistent record)          "risk:{pack_hash}" → 87
                                (for fast dashboard polling)
                 │
                 ▼
     Manufacturer/Regulator Dashboard
     GET /ml/anomalies  (polls Redis or MongoDB)
```

**Event Schema published to Redis Stream:**

```json
{
  "event_id":   "uuid",
  "pack_hash":  "a8f4c...",
  "batch_id":   "PC-BATCH-...",
  "shop_id":    "SHOP_001",
  "event_type": "INTAKE | SALE | CONSUMER_VERIFY",
  "latitude":   28.6139,
  "longitude":  77.2090,
  "timestamp":  "2026-08-22T14:30:00.000Z"
}
```

**Publisher — inside `shopkeeper-service` (Node.js):**

```js
// utils/mlStream.js
const redis = require("redis");
const client = redis.createClient({ url: process.env.REDIS_URL });

async function publishScanEvent(eventData) {
    await client.xAdd("pharmachain:events", "*", {
        payload: JSON.stringify(eventData)
    });
}

module.exports = { publishScanEvent };
```

```js
// Inside scan/intake controller — after successful intake
await publishScanEvent({
    event_id:   uuidv4(),
    pack_hash:  packHash,
    batch_id:   batchId,
    shop_id:    shopkeeper._id.toString(),
    event_type: "INTAKE",
    latitude:   req.body.latitude ?? null,
    longitude:  req.body.longitude ?? null,
    timestamp:  new Date().toISOString()
});
```

**Consumer — inside `ml-service` (Python):**

```python
# app/stream/consumer.py
import redis.asyncio as redis
import json
import asyncio
from app.features.builder import build_feature_vector
from app.models.isolation_forest import score_event
from app.features.spike_detector import compute_spike_score
from app.features.route_anomaly import check_route_anomaly
from app.scoring.risk import compute_composite_risk

STREAM_KEY  = "pharmachain:events"
GROUP_NAME  = "ml-processors"
CONSUMER_ID = "ml-worker-1"

async def start_consumer(redis_client):
    # Create consumer group if it doesn't exist
    try:
        await redis_client.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
    except Exception:
        pass  # Group already exists

    print(f"[ML Stream] Listening on '{STREAM_KEY}'...")

    while True:
        # Read up to 10 messages, block for 2 seconds if empty
        messages = await redis_client.xreadgroup(
            GROUP_NAME, CONSUMER_ID,
            {STREAM_KEY: ">"},
            count=10, block=2000
        )

        if not messages:
            continue

        for _, entries in messages:
            for entry_id, data in entries:
                try:
                    event = json.loads(data[b"payload"])
                    await process_event(redis_client, event)
                    # Acknowledge processed message
                    await redis_client.xack(STREAM_KEY, GROUP_NAME, entry_id)
                except Exception as e:
                    print(f"[ML Stream] Error processing {entry_id}: {e}")
                    # Do NOT ack — message stays for retry

async def process_event(redis_client, event: dict):
    pack_hash  = event["pack_hash"]
    batch_id   = event["batch_id"]
    shop_id    = event["shop_id"]

    # 1. Build feature vector
    features = await build_feature_vector(event)

    # 2. Run Isolation Forest scoring
    if_score = score_event(features)

    # 3. B2 Spike detection
    spike_result = await compute_spike_score(shop_id, batch_id)

    # 4. B3 Route anomaly
    route_result = await check_route_anomaly(pack_hash, event["event_type"])

    # 5. Composite risk score
    risk = compute_composite_risk(
        ml_score       = if_score,
        travel_signal  = features.get("impossible_travel_flag", 0),
        clone_signal   = features.get("duplicate_scan_flag", 0),
        cluster_signal = features.get("complaint_cluster_flag", 0),
        inventory_signal = features.get("inventory_anomaly_flag", 0),
        spike_signal   = spike_result.is_spike_event,
        route_signal   = route_result.risk_contribution
    )

    # 6. Write to MongoDB (persistent) + Redis (fast cache)
    await save_anomaly_result(risk, event)
    await redis_client.setex(
        f"risk:{pack_hash}",
        3600,                        # expire after 1 hour
        json.dumps(risk.to_dict())
    )

    if risk.level in ("HIGH", "CRITICAL"):
        print(f"[ML Alert] {risk.level} — packHash: {pack_hash}, score: {risk.score}")
```

**Starting the consumer (FastAPI lifespan):**

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import redis.asyncio as redis
import asyncio
from app.stream.consumer import start_consumer

@asynccontextmanager
async def lifespan(app: FastAPI):
    r = redis.from_url(settings.REDIS_URL, decode_responses=False)
    # Start consumer in background task
    task = asyncio.create_task(start_consumer(r))
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
```

**New environment variables needed:**

```bash
REDIS_URL=redis://redis-service:6379   # shared between Node.js services and ml-service
```

**Kubernetes addition:** Add a Redis deployment to `k8s/` if one doesn't exist. All four services + the ml-service share the same Redis instance.

**Why Redis Streams over Kafka:** Redis is simpler to deploy (single container), has built-in consumer groups, supports message acknowledgment and retry, and is fast enough for thousands of scans per second. Kafka is the right answer at national scale — mention this to judges as the upgrade path.

**Value:** Transforms the ML layer from a batch "report generator" into a real-time active safety system. Every scan instantly produces a risk score visible on the dashboard within milliseconds. This is the most impressive architectural upgrade for a live demo — show the risk score updating live on screen as you scan a QR code.

---

### B9. 📉 Model Drift Monitoring

**What:** Once real production data flows in, the Isolation Forest model's learned notion of "normal" behaviour will gradually drift from reality — either because the real world changed (legitimate business growth) or because bad actors adapted to evade the model. Most hackathon ML projects skip this entirely. Adding a monitoring endpoint shows production ML maturity.

**Approach:**

Use **Population Stability Index (PSI)** — the industry-standard metric for measuring how much a score distribution has shifted between a reference window (training period) and a monitoring window (recent production data).

$$\text{PSI} = \sum_{i=1}^{N} \left( A_i - E_i \right) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

where $A_i$ = actual (recent) fraction in bucket $i$, $E_i$ = expected (baseline) fraction in bucket $i$.

**PSI interpretation:**

| PSI Value | Status | Action |
|---|---|---|
| < 0.10 | **STABLE** | No action needed |
| 0.10 – 0.25 | **WARNING** | Monitor more closely |
| > 0.25 | **DRIFT** | Retrain the model |

**What to track:**

| Distribution | Meaning if drifted |
|---|---|
| `anomaly_score` distribution | Model is firing differently — either more/less sensitive |
| `risk_level` distribution | Business-level shift in fraud patterns |
| Feature distributions (`scan_spike_zscore`, `implied_speed`) | Input data has changed character |

**Implementation (`scoring/drift_monitor.py`):**

```python
import numpy as np
from typing import List

def compute_psi(
    baseline_scores: List[float],
    current_scores: List[float],
    n_bins: int = 10
) -> float:
    """
    Compute Population Stability Index between baseline and current score distributions.
    Scores should be in [0, 1] range (normalised anomaly scores).
    """
    bins = np.linspace(0, 1, n_bins + 1)

    # Compute fractions per bin
    baseline_counts, _ = np.histogram(baseline_scores, bins=bins)
    current_counts, _  = np.histogram(current_scores,  bins=bins)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    baseline_frac = (baseline_counts + epsilon) / (len(baseline_scores) + epsilon * n_bins)
    current_frac  = (current_counts  + epsilon) / (len(current_scores)  + epsilon * n_bins)

    psi = np.sum((current_frac - baseline_frac) * np.log(current_frac / baseline_frac))
    return float(psi)

def get_drift_status(psi: float) -> str:
    if psi < 0.10:
        return "STABLE"
    elif psi < 0.25:
        return "WARNING"
    else:
        return "DRIFT"
```

**Storing the baseline:**

When you train the Isolation Forest on synthetic (or early real) data, save the distribution of anomaly scores from the training set:

```python
# training/train.py — after fitting
import joblib, json, numpy as np

model.fit(X_train)
train_scores = -model.score_samples(X_train)   # higher = more anomalous, normalise to [0,1]
train_scores_norm = (train_scores - train_scores.min()) / (train_scores.max() - train_scores.min())

# Save model and baseline distribution
joblib.dump(model, "models/isolation_forest.pkl")
np.save("models/baseline_score_dist.npy", train_scores_norm)
```

**API endpoint (`GET /ml/model/health`):**

```python
# routes/anomaly.py
@router.get("/ml/model/health")
async def model_health():
    # Load baseline saved at training time
    baseline = np.load("models/baseline_score_dist.npy").tolist()

    # Load recent scores from MongoDB (last 1000 events)
    recent_scores = await get_recent_anomaly_scores(limit=1000)

    if len(recent_scores) < 50:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "Need at least 50 real events to compute drift.",
            "event_count": len(recent_scores)
        }

    psi = compute_psi(baseline, recent_scores)
    status = get_drift_status(psi)

    return {
        "status": status,
        "psi_score": round(psi, 4),
        "thresholds": { "warning": 0.10, "drift": 0.25 },
        "baseline_event_count": len(baseline),
        "current_event_count": len(recent_scores),
        "recommendation": (
            "Model is healthy."                 if status == "STABLE"  else
            "Monitor closely. Consider retraining soon." if status == "WARNING" else
            "Retrain the model with recent labelled data."
        )
    }
```

**Example response:**

```json
{
  "status": "WARNING",
  "psi_score": 0.18,
  "thresholds": { "warning": 0.10, "drift": 0.25 },
  "baseline_event_count": 5000,
  "current_event_count": 847,
  "recommendation": "Monitor closely. Consider retraining soon."
}
```

**Additional: per-feature drift** (stretch goal)

```python
# Check drift on individual input features too
feature_drift = {}
for feature_name in TRACKED_FEATURES:
    baseline_feat = baseline_feature_store[feature_name]
    current_feat  = [e[feature_name] for e in recent_events]
    feature_drift[feature_name] = {
        "psi": compute_psi(baseline_feat, current_feat),
        "status": get_drift_status(compute_psi(baseline_feat, current_feat))
    }
```

**Integration point:** Add `GET /ml/model/health` to the regulator dashboard as a small "Model Health" badge. It's 3 extra lines on the frontend and makes the demo look production-grade.

**Value:** Shows you understand the full ML lifecycle — not just training, but deployment, monitoring, and retraining. Most hackathon ML implementations stop at model training. This separates your pipeline from all of them.

---

## Part C — What NOT to Build (Doc Explicitly Excludes)

| Feature | Reason |
|---|---|
| Deep learning / neural networks | Overkill, not explainable |
| LLM-based verification | Unnecessary complexity |
| ML-based QR authenticity | That's crypto's job |
| Automatic counterfeit declaration | ML is investigation signal only |
| Facial recognition | Out of scope |
| Image-based medicine classification | Not in current architecture |

---

## Part D — Recommended Priority Order

### Must-Build (Core, Documented)
1. `ml_events` event store
2. Feature engineering pipeline (`builder.py`)
3. Isolation Forest model + training on synthetic data
4. DBSCAN complaint clustering
5. Composite risk scoring engine
6. `POST /ml/analyze`, `GET /ml/anomalies`, `GET /ml/batches/:id/risk`
7. Dashboard section with alert counts + suspicious batch list

### High-Value Additions (Pick 2–3)
8. **Temporal Spike Detection (B2)** — simple, explainable, very demo-friendly
9. **Time-of-Day Profiling (B1)** — easy to implement, great story
10. **Supply Chain Route Anomaly (B3)** — directly tied to a threat in the master plan
11. **New Seller Fingerprinting (B4)** — good narrative for judges

### Stretch Goals (If Time Permits)
12. Real-time streaming via Redis Streams (B8)
13. Ensemble with One-Class SVM (B7)
14. Graph-based collusion detection (B5)

---

## Part E — Tech Stack Summary

```
Language:        Python 3.11+
API:             FastAPI
ML:              scikit-learn  (IsolationForest, DBSCAN, OneClassSVM)
Feature Eng.:    pandas, NumPy
Geo Distance:    haversine (pip install haversine)
Graph (B5):      networkx
Model Storage:   joblib
Event Store:     PostgreSQL or MongoDB
Streaming (B8):  Redis Streams or Kafka
Deployment:      Docker + Kubernetes (internal service, no public NGINX route)
```

---

*Based on: [ML_README.md](file:///d:/Honey/Fun/PharmaChain/docs/ML_README.md) · [PHARMACHAIN_SUPPLY_CHAIN_MASTER_PLAN.md](file:///d:/Honey/Fun/PharmaChain/docs/PHARMACHAIN_SUPPLY_CHAIN_MASTER_PLAN.md) · [CURRENT_STATE_REPORT.md](file:///d:/Honey/Fun/PharmaChain/docs/CURRENT_STATE_REPORT.md)*
