"""Redis caching layer for real-time velocity tracking and account stats.

Uses Redis sorted sets (ZSET) for sliding-window transaction velocity
and Redis hashes for per-account running statistics. All operations are
async-native via redis.asyncio — no thread pool needed.

Runs 100% locally via Docker. Gracefully degrades if Redis is unavailable.
"""

import logging
from datetime import datetime
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis client for real-time caching and velocity tracking."""

    def __init__(self):
        self._client = None
        self._enabled = False

    async def initialize(self):
        """Connect to Redis. Sets self._enabled = True on success."""
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self._client.ping()
            self._enabled = True
            logger.info(f"Redis: Connected to {settings.redis_url}")

        except Exception as e:
            logger.warning(f"Redis: Initialization failed (caching disabled): {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()

    # --- Transaction Velocity (Sliding Window) ---

    async def record_transaction_velocity(
        self, account_id: str, timestamp: str, txn_id: str
    ):
        """Record a transaction in the sliding-window velocity tracker.

        Uses Redis ZADD with unix timestamp as score for efficient
        time-range queries. Automatically trims entries older than 10 minutes.
        """
        if not self._enabled:
            return
        try:
            key = f"velocity:{account_id}"
            ts = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            ).timestamp()

            pipe = self._client.pipeline()
            pipe.zadd(key, {txn_id: ts})
            # Trim entries older than 10 minutes
            cutoff = ts - 600
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.expire(key, 3600)  # TTL 1 hour
            await pipe.execute()

        except Exception:
            pass  # Best-effort, never block pipeline

    async def get_velocity(self, account_id: str) -> int:
        """Get transaction count in the last 10 minutes for an account."""
        if not self._enabled:
            return 0
        try:
            key = f"velocity:{account_id}"
            now = datetime.utcnow().timestamp()
            cutoff = now - 600
            return await self._client.zcount(key, cutoff, "+inf")
        except Exception:
            return 0

    # --- Per-Account Running Statistics ---

    async def update_account_risk_stats(self, account_id: str, txn: dict):
        """Update per-account running statistics in a Redis hash.

        Tracks: txn_count, total_amount, last_city, last_country, last_risk_score.
        Uses pipelining for atomic batch updates.
        """
        if not self._enabled:
            return
        try:
            key = f"account_stats:{account_id}"
            pipe = self._client.pipeline()
            pipe.hincrby(key, "txn_count", 1)
            pipe.hincrbyfloat(key, "total_amount", float(txn.get("amount", 0)))
            pipe.hset(key, mapping={
                "last_city": txn.get("city", ""),
                "last_country": txn.get("country", "US"),
                "last_category": txn.get("category", ""),
                "last_risk_score": str(round(float(txn.get("risk_score", 0)), 3)),
                "last_merchant": txn.get("merchant_name", ""),
            })
            pipe.expire(key, 86400)  # 24h TTL
            await pipe.execute()

        except Exception:
            pass  # Best-effort

    async def get_account_stats(self, account_id: str) -> dict:
        """Retrieve cached account statistics from Redis."""
        if not self._enabled:
            return {}
        try:
            key = f"account_stats:{account_id}"
            data = await self._client.hgetall(key)
            if data:
                count = int(data.get("txn_count", 0))
                total = float(data.get("total_amount", 0))
                return {
                    "txn_count": count,
                    "total_amount": round(total, 2),
                    "mean_amount": round(total / max(count, 1), 2),
                    "last_city": data.get("last_city", ""),
                    "last_country": data.get("last_country", ""),
                    "last_category": data.get("last_category", ""),
                    "last_merchant": data.get("last_merchant", ""),
                }
            return {}
        except Exception:
            return {}

    # --- Rate Limiting ---

    async def check_rate_limit(
        self, key: str, max_requests: int = 60, window: int = 60
    ) -> bool:
        """Simple sliding-window rate limiter. Returns True if allowed."""
        if not self._enabled:
            return True  # Degrade to no rate limiting
        try:
            rk = f"ratelimit:{key}"
            pipe = self._client.pipeline()
            pipe.incr(rk)
            pipe.expire(rk, window)
            results = await pipe.execute()
            return results[0] <= max_requests
        except Exception:
            return True


# Singleton instance
redis_cache = RedisCache()
