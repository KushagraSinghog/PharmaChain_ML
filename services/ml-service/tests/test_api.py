import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from datetime import datetime, timezone, timedelta
from app.main import app
from app.storage.event_store import event_store

@pytest.mark.asyncio
async def test_api_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/ml/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_api_analyze_normal_event():
    await event_store.clear_all()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "packHash": "TEST-PACK-NORMAL-001",
            "batchId": "PC-BATCH-001",
            "shopId": "SHOP_DELHI_01",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "eventType": "INTAKE"
        }
        res = await client.post("/ml/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "riskScore" in data
        assert "riskLevel" in data
        assert data["riskLevel"] in ("LOW", "MEDIUM")

@pytest.mark.asyncio
async def test_api_analyze_impossible_travel():
    await event_store.clear_all()
    now = datetime.now(timezone.utc)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Step 1: Delhi intake 8 mins ago
        await client.post("/ml/analyze", json={
            "packHash": "TEST-PACK-TRAVEL-002",
            "batchId": "PC-BATCH-002",
            "shopId": "SHOP_DELHI_01",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "eventType": "INTAKE",
            "timestamp": (now - timedelta(minutes=8)).isoformat()
        })

        # Step 2: Jaipur sale attempt now
        res = await client.post("/ml/analyze", json={
            "packHash": "TEST-PACK-TRAVEL-002",
            "batchId": "PC-BATCH-002",
            "shopId": "SHOP_JAIPUR_02",
            "latitude": 26.9124,
            "longitude": 75.7873,
            "eventType": "SALE",
            "timestamp": now.isoformat()
        })
        assert res.status_code == 200
        data = res.json()
        assert data["riskScore"] >= 80
        assert data["riskLevel"] in ("HIGH", "CRITICAL")
        assert "IMPOSSIBLE_TRAVEL" in data["anomalies"]
        assert data["requiresInvestigation"] is True

@pytest.mark.asyncio
async def test_api_get_anomalies():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/ml/anomalies")
        assert res.status_code == 200
        data = res.json()
        assert "total" in data
        assert "anomalies" in data

@pytest.mark.asyncio
async def test_api_model_drift_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/ml/model/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "psiScore" in data

@pytest.mark.asyncio
async def test_api_reports_and_clustering():
    await event_store.clear_all()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for i in range(4):
            await client.post("/ml/reports", json={
                "batchId": "PC-BATCH-HOTSPOT",
                "shopId": f"SHOP_{i}",
                "latitude": 28.6139 + (i * 0.001),
                "longitude": 77.2090 + (i * 0.001),
                "reportType": "COUNTERFEIT"
            })
        
        cluster_res = await client.get("/ml/clusters")
        assert cluster_res.status_code == 200
        cluster_data = cluster_res.json()
        assert cluster_data["clusters_count"] >= 1
        assert cluster_data["clustered_reports_count"] == 4

@pytest.mark.asyncio
async def test_api_simulate_scenario():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/ml/simulate", json={"scenario": "impossible_travel"})
        assert res.status_code == 200
        data = res.json()
        assert data["scenario"] == "impossible_travel"
        assert len(data["timeline"]) >= 2
