"""Anomaly Detection Engine using Isolation Forest"""

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List
from sklearn.ensemble import IsolationForest

from backend.anomaly_detection.features import (
    TransactionTracker,
    is_international,
    hour_from_timestamp,
    day_from_timestamp,
)

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float  # 0.0 (normal) to 1.0 (highly anomalous)
    risk_factors: List[str] = field(default_factory=list)


class AnomalyDetectionEngine:
    """Real-time anomaly detection using Isolation Forest + rule-based features"""

    def __init__(self, contamination: float = 0.05):
        self.tracker = TransactionTracker()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        self._train_on_synthetic()

    def _train_on_synthetic(self):
        """Train the model on synthetic normal transaction data"""
        rng = np.random.RandomState(42)
        n = 1000
        features = np.column_stack([
            rng.lognormal(mean=3.0, sigma=0.8, size=n),  # amount
            np.log1p(rng.lognormal(mean=3.0, sigma=0.8, size=n)),  # log_amount
            rng.randint(6, 23, size=n),  # hour_of_day (6am-11pm)
            rng.randint(0, 7, size=n),  # day_of_week
            rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], size=n),  # is_international (10%)
            rng.choice([0, 0, 0, 1, 1, 2], size=n),  # velocity (mostly 0-2)
            rng.exponential(5, size=n),  # distance_from_last_km
            rng.normal(0, 1, size=n),  # amount_zscore
        ])
        self.model.fit(features)
        logger.info("Isolation Forest trained on 1000 synthetic transactions")

    def _extract_features(self, txn: dict) -> np.ndarray:
        """Extract feature vector from a transaction"""
        amount = txn.get("amount", 0)
        account_id = txn.get("account_id", "unknown")
        timestamp = txn.get("timestamp", "")
        lat = txn.get("latitude")
        lon = txn.get("longitude")
        country = txn.get("country", "US")

        velocity = self.tracker.get_velocity(account_id, timestamp)
        dist_km, time_gap = self.tracker.get_distance_from_last(account_id, lat, lon)
        zscore = self.tracker.get_amount_zscore(account_id, amount)

        return np.array([[
            amount,
            math.log1p(amount),
            hour_from_timestamp(timestamp),
            day_from_timestamp(timestamp),
            is_international(country),
            velocity,
            dist_km,
            zscore,
        ]])

    def _generate_risk_factors(self, txn: dict, features: np.ndarray) -> List[str]:
        """Generate human-readable risk factor descriptions"""
        factors = []
        amount = txn.get("amount", 0)
        country = txn.get("country", "US")
        city = txn.get("city", "Unknown")
        account_id = txn.get("account_id", "unknown")

        zscore = features[0, 7]
        velocity = features[0, 5]
        dist_km = features[0, 6]

        if abs(zscore) > 3:
            multiplier = round(amount / max(self.tracker.account_stats[account_id]["mean"], 1), 1)
            factors.append(f"Amount ${amount:.2f} is {multiplier}x above account average (z-score: {zscore:.1f})")

        if country != "US":
            factors.append(f"International transaction in {city}, {country}")

        if velocity > 3:
            factors.append(f"High velocity: {int(velocity)} transactions in last 10 minutes")

        if dist_km > 500:
            _, time_gap = self.tracker.get_distance_from_last(
                account_id, txn.get("latitude"), txn.get("longitude")
            )
            if time_gap < 120 and dist_km > 500:
                speed_kmh = dist_km / (time_gap / 60) if time_gap > 0 else float("inf")
                factors.append(
                    f"Impossible travel: {dist_km:.0f}km in {time_gap:.0f} min "
                    f"(would require {speed_kmh:.0f} km/h)"
                )
            else:
                factors.append(f"Unusual location: {dist_km:.0f}km from last transaction")

        if amount > 5000:
            factors.append(f"High-value transaction: ${amount:.2f}")

        if txn.get("category") in ("Unknown", "Financial", "Luxury", "Gambling"):
            factors.append(f"High-risk merchant category: {txn.get('category')}")

        if not factors:
            factors.append("Statistical anomaly detected by Isolation Forest model")

        return factors

    def detect(self, txn: dict) -> AnomalyResult:
        """Run anomaly detection on a single transaction"""
        features = self._extract_features(txn)

        # IsolationForest scoring
        raw_score = self.model.decision_function(features)[0]
        prediction = self.model.predict(features)[0]

        # Normalize: decision_function returns negative for anomalies, positive for normal
        # Typical range is roughly -0.5 to 0.5
        # Map to 0-1 where 1 = most anomalous
        normalized_score = max(0.0, min(1.0, -raw_score * 2))

        # Add stochastic noise for realistic variance in the risk timeline
        # Normal transactions will scatter 0.02-0.25 instead of flat 0.0-0.05
        noise = np.random.normal(0, 0.06)
        normalized_score = max(0.0, min(1.0, normalized_score + noise))

        # Soft rule-based boosting (probabilistic, not hard floors)
        country = txn.get("country", "US")
        amount = txn.get("amount", 0)
        category = txn.get("category", "")

        # International + high amount: boost with variance
        if country != "US" and amount > 1000:
            boost = 0.55 + np.random.uniform(0.05, 0.25)
            normalized_score = max(normalized_score, boost)
        elif country != "US":
            # International but low amount: mild boost
            boost = 0.25 + np.random.uniform(0, 0.15)
            normalized_score = max(normalized_score, boost)

        if amount > 5000:
            boost = 0.50 + np.random.uniform(0.05, 0.30)
            normalized_score = max(normalized_score, boost)

        # High-risk category adds a smaller stochastic bump
        if category in ("Unknown", "Financial", "Luxury", "Gambling"):
            normalized_score = min(1.0, normalized_score + np.random.uniform(0.03, 0.12))

        normalized_score = max(0.0, min(1.0, normalized_score))
        is_anomaly = normalized_score > 0.55

        risk_factors = []
        if is_anomaly:
            risk_factors = self._generate_risk_factors(txn, features)

        # Track this transaction for future reference
        self.tracker.update(txn.get("account_id", "unknown"), txn)

        return AnomalyResult(
            is_anomaly=bool(is_anomaly),
            score=round(float(normalized_score), 3),
            risk_factors=risk_factors,
        )
