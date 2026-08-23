import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

class EventStore:
    """
    High-performance in-memory and asynchronous event repository for ML inference,
    feature engineering, reporting, and anomaly inspection.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._events: List[Dict[str, Any]] = []
        self._pack_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._batch_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._shop_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._risk_results: Dict[str, Dict[str, Any]] = {} # keyed by pack_hash / event_id
        self._anomalies: List[Dict[str, Any]] = []
        self._reports: List[Dict[str, Any]] = []
        self._recent_scores: List[float] = []

    async def add_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a new scan/supply event into the ML store.
        """
        async with self._lock:
            # Ensure mandatory fields
            event_id = event.get("event_id") or str(uuid.uuid4())
            ts = event.get("timestamp")
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    dt = datetime.now(timezone.utc)
            elif isinstance(ts, datetime):
                dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)

            cleaned_event = {
                "event_id": event_id,
                "pack_hash": str(event.get("pack_hash") or event.get("packHash") or ""),
                "batch_id": str(event.get("batch_id") or event.get("batchId") or ""),
                "shop_id": str(event.get("shop_id") or event.get("shopId") or ""),
                "event_type": str(event.get("event_type") or event.get("eventType") or "SALE").upper(),
                "latitude": float(event["latitude"]) if event.get("latitude") is not None else None,
                "longitude": float(event["longitude"]) if event.get("longitude") is not None else None,
                "timestamp": dt,
                "metadata": event.get("metadata", {})
            }

            self._events.append(cleaned_event)
            if cleaned_event["pack_hash"]:
                self._pack_events[cleaned_event["pack_hash"]].append(cleaned_event)
            if cleaned_event["batch_id"]:
                self._batch_events[cleaned_event["batch_id"]].append(cleaned_event)
            if cleaned_event["shop_id"]:
                self._shop_events[cleaned_event["shop_id"]].append(cleaned_event)

            return cleaned_event

    async def save_risk_result(self, result: Dict[str, Any]) -> None:
        """
        Save the computed risk score & anomaly details.
        """
        async with self._lock:
            pack_hash = result.get("pack_hash") or result.get("packHash", "")
            event_id = result.get("event_id", str(uuid.uuid4()))
            
            self._risk_results[event_id] = result
            if pack_hash:
                self._risk_results[pack_hash] = result

            score = float(result.get("risk_score") or result.get("riskScore") or 0.0)
            norm_score = max(0.0, min(1.0, score / 100.0))
            self._recent_scores.append(norm_score)
            # Keep rolling window of last 5000 scores
            if len(self._recent_scores) > 5000:
                self._recent_scores.pop(0)

            level = result.get("risk_level") or result.get("riskLevel", "LOW")
            if level in ("MEDIUM", "HIGH", "CRITICAL") or (result.get("anomalies") and len(result["anomalies"]) > 0):
                self._anomalies.append(result)

    async def get_pack_events(self, pack_hash: str) -> List[Dict[str, Any]]:
        async with self._lock:
            events = self._pack_events.get(pack_hash, [])
            return sorted(events, key=lambda x: x["timestamp"])

    async def get_batch_events(self, batch_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            events = self._batch_events.get(batch_id, [])
            return sorted(events, key=lambda x: x["timestamp"])

    async def get_shop_events(self, shop_id: str, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            events = self._shop_events.get(shop_id, [])
            if since:
                events = [e for e in events if e["timestamp"] >= since]
            return sorted(events, key=lambda x: x["timestamp"])

    async def get_all_events(self, limit: int = 1000) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(reversed(self._events[-limit:]))

    async def add_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a consumer or inspector complaint.
        """
        async with self._lock:
            report_id = report.get("report_id") or f"RPT-{uuid.uuid4().hex[:8].upper()}"
            ts = report.get("timestamp")
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    dt = datetime.now(timezone.utc)
            elif isinstance(ts, datetime):
                dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)

            loc = report.get("location") or {}
            lat = float(report.get("latitude") if report.get("latitude") is not None else loc.get("lat", 0.0))
            lon = float(report.get("longitude") if report.get("longitude") is not None else loc.get("lon", 0.0))

            cleaned = {
                "report_id": report_id,
                "batch_id": str(report.get("batch_id") or report.get("batchId") or ""),
                "shop_id": str(report.get("shop_id") or report.get("shopId") or ""),
                "pack_hash": str(report.get("pack_hash") or report.get("packHash") or ""),
                "report_type": str(report.get("report_type") or report.get("reportType") or "SUSPECT_COUNTERFEIT").upper(),
                "latitude": lat,
                "longitude": lon,
                "notes": str(report.get("notes") or ""),
                "timestamp": dt
            }
            self._reports.append(cleaned)
            return cleaned

    async def get_reports(self, since: Optional[datetime] = None, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            res = self._reports
            if since:
                res = [r for r in res if r["timestamp"] >= since]
            if batch_id:
                res = [r for r in res if r["batch_id"] == batch_id]
            return list(res)

    async def get_anomalies(self, risk_level: Optional[str] = None, limit: int = 50, page: int = 1) -> Dict[str, Any]:
        async with self._lock:
            filtered = self._anomalies
            if risk_level:
                filtered = [a for a in filtered if (a.get("risk_level") or a.get("riskLevel", "")).upper() == risk_level.upper()]
            
            # Sort newest first
            sorted_items = list(reversed(filtered))
            total = len(sorted_items)
            start = (page - 1) * limit
            end = start + limit
            page_items = sorted_items[start:end]

            return {
                "total": total,
                "page": page,
                "limit": limit,
                "count": len(page_items),
                "anomalies": page_items
            }

    async def get_recent_scores(self, limit: int = 1000) -> List[float]:
        async with self._lock:
            return list(self._recent_scores[-limit:])

    async def get_batch_risk_record(self, batch_id: str) -> Dict[str, Any]:
        async with self._lock:
            events = self._batch_events.get(batch_id, [])
            reports = [r for r in self._reports if r["batch_id"] == batch_id]
            
            unique_shops = set(e["shop_id"] for e in events if e["shop_id"])
            locs = set((round(e["latitude"], 3), round(e["longitude"], 3)) for e in events if e["latitude"] is not None and e["longitude"] is not None)
            
            # Aggregate anomalous scans for this batch
            suspicious_scans = 0
            max_risk = 0
            for e in events:
                p_hash = e.get("pack_hash")
                if p_hash and p_hash in self._risk_results:
                    r = self._risk_results[p_hash]
                    score = int(r.get("risk_score") or r.get("riskScore", 0))
                    if score > max_risk:
                        max_risk = score
                    if score >= 61:
                        suspicious_scans += 1

            risk_level = "LOW"
            if max_risk > 80:
                risk_level = "CRITICAL"
            elif max_risk > 60:
                risk_level = "HIGH"
            elif max_risk > 30:
                risk_level = "MEDIUM"

            return {
                "batchId": batch_id,
                "totalScans": len(events),
                "uniqueShops": len(unique_shops),
                "uniqueLocations": len(locs),
                "suspiciousScans": suspicious_scans,
                "suspiciousReports": len(reports),
                "riskScore": max_risk,
                "riskLevel": risk_level
            }

    async def get_system_stats(self) -> Dict[str, Any]:
        async with self._lock:
            critical = sum(1 for a in self._anomalies if (a.get("risk_level") or a.get("riskLevel")) == "CRITICAL")
            high = sum(1 for a in self._anomalies if (a.get("risk_level") or a.get("riskLevel")) == "HIGH")
            medium = sum(1 for a in self._anomalies if (a.get("risk_level") or a.get("riskLevel")) == "MEDIUM")
            low = sum(1 for a in self._anomalies if (a.get("risk_level") or a.get("riskLevel")) == "LOW")

            return {
                "totalEvents": len(self._events),
                "totalReports": len(self._reports),
                "totalAnomalies": len(self._anomalies),
                "criticalAlerts": critical,
                "highRiskAlerts": high,
                "mediumRiskAlerts": medium,
                "lowRiskCount": low,
                "trackedPacks": len(self._pack_events),
                "trackedBatches": len(self._batch_events),
                "trackedShops": len(self._shop_events)
            }

    async def clear_all(self):
        """Reset store (useful for tests)"""
        async with self._lock:
            self._events.clear()
            self._pack_events.clear()
            self._batch_events.clear()
            self._shop_events.clear()
            self._risk_results.clear()
            self._anomalies.clear()
            self._reports.clear()
            self._recent_scores.clear()

# Global singleton instance
event_store = EventStore()
