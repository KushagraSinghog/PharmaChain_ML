import asyncio
import json
from typing import Optional
from app.config import settings
from app.storage.event_store import event_store
from app.features.builder import build_feature_vector
from app.models.isolation_forest import model_instance
from app.scoring.risk import compute_composite_risk

class RedisStreamConsumer:
    """
    Background asynchronous worker consuming supply-chain events from Redis Streams.
    Computes real-time risk scores at scan-time with millisecond latency.
    """

    def __init__(self):
        self.stream_key = settings.REDIS_STREAM_KEY
        self.group_name = settings.REDIS_GROUP_NAME
        self.consumer_id = settings.REDIS_CONSUMER_ID
        self.redis_url = settings.REDIS_URL
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._redis_client = None

    async def start(self):
        if not settings.REDIS_ENABLED:
            print("[Redis Stream] Redis streaming disabled by configuration.")
            return

        self.is_running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._redis_client:
            await self._redis_client.close()
            print("[Redis Stream] Redis consumer connection closed.")

    async def _run_loop(self):
        try:
            import asyncio as aioredis
        except ImportError:
            print("[Redis Stream] redis-py not installed. Streaming worker disabled.")
            return

        print(f"[Redis Stream] Attempting to connect to Redis at {self.redis_url}...")
        try:
            self._redis_client = aioredis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_connect_timeout=2.0
            )
            # Test connection
            await self._redis_client.ping()
            print("[Redis Stream] Connected to Redis successfully.")
        except Exception as e:
            print(f"[Redis Stream] Redis not reachable ({e}). Running in standalone mode.")
            return

        # Create consumer group if not existing
        try:
            await self._redis_client.xgroup_create(
                self.stream_key,
                self.group_name,
                id="0",
                mkstream=True
            )
            print(f"[Redis Stream] Consumer group '{self.group_name}' initialized on '{self.stream_key}'.")
        except Exception:
            pass  # Group already exists

        print(f"[Redis Stream] Listening for scan events on '{self.stream_key}'...")

        while self.is_running:
            try:
                # Read up to 10 messages, block for 2 seconds
                messages = await self._redis_client.xreadgroup(
                    self.group_name,
                    self.consumer_id,
                    {self.stream_key: ">"},
                    count=10,
                    block=2000
                )

                if not messages:
                    continue

                for stream_name, entries in messages:
                    for entry_id, data in entries:
                        try:
                            payload_raw = data.get(b"payload") or data.get(b"data")
                            if payload_raw:
                                event_data = json.loads(payload_raw.decode("utf-8"))
                            else:
                                # Data might be flattened in key-values
                                event_data = {
                                    k.decode("utf-8"): v.decode("utf-8")
                                    for k, v in data.items()
                                }

                            await self.process_event(event_data)
                            await self._redis_client.xack(self.stream_key, self.group_name, entry_id)
                        except Exception as entry_err:
                            print(f"[Redis Stream] Error processing message {entry_id}: {entry_err}")

            except asyncio.CancelledError:
                break
            except Exception as loop_err:
                print(f"[Redis Stream] Stream read error: {loop_err}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def process_event(self, event_data: dict):
        """
        Executes end-to-end real-time ML inference on a streaming event.
        """
        # 1. Record event in store
        saved_event = await event_store.add_event(event_data)

        # 2. Build feature vector
        features_dict, feature_vector = await build_feature_vector(saved_event)

        # 3. Model score
        ml_score = model_instance.score_event(feature_vector)

        # 4. Composite risk calculation
        risk_result = compute_composite_risk(ml_score, features_dict)

        result_payload = {
            "eventId": saved_event["event_id"],
            "packHash": saved_event["pack_hash"],
            "batchId": saved_event["batch_id"],
            "shopId": saved_event["shop_id"],
            "eventType": saved_event["event_type"],
            "riskScore": risk_result.risk_score,
            "riskLevel": risk_result.risk_level,
            "anomalies": risk_result.anomalies,
            "requiresInvestigation": risk_result.requires_investigation,
            "breakdown": risk_result.breakdown,
            "details": risk_result.details,
            "timestamp": saved_event["timestamp"].isoformat()
        }

        # 5. Persist risk result
        await event_store.save_risk_result(result_payload)

        # 6. Cache in Redis for instantaneous query
        if self._redis_client and saved_event["pack_hash"]:
            try:
                await self._redis_client.setex(
                    f"risk:{saved_event['pack_hash']}",
                    3600,
                    json.dumps(result_payload)
                )
            except Exception:
                pass

        if risk_result.risk_level in ("HIGH", "CRITICAL"):
            print(f"[ML Stream ALERT] {risk_result.risk_level} ({risk_result.risk_score}/100) on pack {saved_event['pack_hash']}: {risk_result.anomalies}")

stream_consumer = RedisStreamConsumer()
