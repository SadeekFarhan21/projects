"""Pure functions for feature extraction from transaction data"""

import math
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lon points"""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def is_international(country: str) -> int:
    """Check if transaction is international (non-US)"""
    return 0 if country == "US" else 1


def hour_from_timestamp(timestamp: str) -> int:
    """Extract hour of day from ISO timestamp"""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.hour
    except (ValueError, AttributeError):
        return 12


def day_from_timestamp(timestamp: str) -> int:
    """Extract day of week (0=Monday) from ISO timestamp"""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.weekday()
    except (ValueError, AttributeError):
        return 3


class TransactionTracker:
    """Tracks per-account transaction history for velocity and location features"""

    def __init__(self):
        self.account_history: Dict[str, list] = defaultdict(list)
        self.account_stats: Dict[str, dict] = defaultdict(lambda: {"total": 0.0, "count": 0, "mean": 0.0})

    def update(self, account_id: str, txn: dict):
        self.account_history[account_id].append(txn)
        # Keep last 100 transactions per account
        if len(self.account_history[account_id]) > 100:
            self.account_history[account_id] = self.account_history[account_id][-100:]
        # Update running stats
        s = self.account_stats[account_id]
        s["count"] += 1
        s["total"] += txn.get("amount", 0)
        s["mean"] = s["total"] / s["count"]

    def get_velocity(self, account_id: str, timestamp: str, window_minutes: int = 10) -> int:
        """Count transactions in the last N minutes for this account"""
        history = self.account_history.get(account_id, [])
        if not history:
            return 0
        try:
            current = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return 0

        count = 0
        for txn in reversed(history):
            try:
                t = datetime.fromisoformat(txn["timestamp"].replace("Z", "+00:00"))
                diff = (current - t).total_seconds() / 60
                if diff <= window_minutes:
                    count += 1
                elif diff > window_minutes:
                    break
            except (ValueError, KeyError):
                continue
        return count

    def get_distance_from_last(self, account_id: str, lat: Optional[float], lon: Optional[float]) -> tuple:
        """Get distance from last known location and time gap in minutes"""
        if lat is None or lon is None:
            return 0.0, float("inf")

        history = self.account_history.get(account_id, [])
        for txn in reversed(history[:-1]):  # Skip the current one
            prev_lat = txn.get("latitude")
            prev_lon = txn.get("longitude")
            if prev_lat is not None and prev_lon is not None:
                dist = haversine(prev_lat, prev_lon, lat, lon)
                try:
                    t1 = datetime.fromisoformat(txn["timestamp"].replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(history[-1]["timestamp"].replace("Z", "+00:00"))
                    minutes = max((t2 - t1).total_seconds() / 60, 1)
                except (ValueError, KeyError):
                    minutes = float("inf")
                return dist, minutes
        return 0.0, float("inf")

    def get_amount_zscore(self, account_id: str, amount: float) -> float:
        """How many standard deviations from the account mean"""
        history = self.account_history.get(account_id, [])
        if len(history) < 3:
            return 0.0
        amounts = [t.get("amount", 0) for t in history[:-1]]
        mean = sum(amounts) / len(amounts)
        variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
        std = max(math.sqrt(variance), 0.01)
        return (amount - mean) / std
