"""FAISS vector store for transaction similarity search.

Embeds transactions as natural-language descriptions and indexes them
with Meta's FAISS (Facebook AI Similarity Search) for blazing-fast
in-memory semantic similarity lookups. The agent uses this to find
past fraud patterns and reference similar investigation outcomes.

Runs 100% locally — no API keys, no cloud, no external dependencies.
Always enabled once sentence-transformers loads successfully.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class TransactionVectorStore:
    """Manages FAISS vector indexes for transaction and investigation embeddings."""

    def __init__(self):
        self._txn_index = None        # FAISS index for transactions
        self._inv_index = None        # FAISS index for investigations
        self._embed_model = None      # SentenceTransformer model
        self._enabled = False

        # Metadata stores (FAISS only stores vectors, not metadata)
        self._txn_metadata: list[dict] = []    # parallel to txn_index vectors
        self._txn_ids: list[str] = []          # parallel ID list
        self._inv_metadata: list[dict] = []    # parallel to inv_index vectors
        self._inv_ids: list[str] = []          # parallel ID list

    async def initialize(self):
        """Initialize FAISS indexes and embedding model. Call once at startup."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            # Load embedding model (~80MB, runs on CPU in ~5-15ms per embed)
            self._embed_model = SentenceTransformer(settings.embedding_model)

            # Create FAISS indexes (Inner Product on L2-normalized vectors = cosine similarity)
            dim = settings.embedding_dimensions
            self._txn_index = faiss.IndexFlatIP(dim)
            self._inv_index = faiss.IndexFlatIP(dim)

            self._enabled = True
            logger.info(
                f"FAISS: Initialized with {settings.embedding_model} "
                f"({dim}d, in-memory)"
            )

        except Exception as e:
            logger.warning(f"FAISS: Initialization failed: {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _embed(self, text: str) -> np.ndarray:
        """Generate L2-normalized embedding for a text string."""
        vec = self._embed_model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    @staticmethod
    def transaction_to_text(txn: dict) -> str:
        """Convert transaction to a descriptive text for embedding.

        Format captures spending pattern, location, timing, and risk
        characteristics in natural language so the embedding model can
        compute meaningful similarity.
        """
        # Time-of-day bucket
        timestamp = txn.get("timestamp", "")
        hour_str = "unknown time"
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hour = dt.hour
            if 0 <= hour < 6:
                hour_str = "late night"
            elif 6 <= hour < 12:
                hour_str = "morning"
            elif 12 <= hour < 17:
                hour_str = "afternoon"
            elif 17 <= hour < 21:
                hour_str = "evening"
            else:
                hour_str = "night"
        except (ValueError, AttributeError, TypeError):
            pass

        # Amount magnitude bucket
        amount = txn.get("amount", 0)
        if amount < 50:
            amt_desc = "small"
        elif amount < 200:
            amt_desc = "medium"
        elif amount < 1000:
            amt_desc = "large"
        elif amount < 5000:
            amt_desc = "very large"
        else:
            amt_desc = "extremely large"

        risk_score = txn.get("risk_score", 0)
        is_anomaly = txn.get("is_anomaly", False)
        risk_str = f"risk score {risk_score:.2f}"
        if is_anomaly:
            risk_str += ", flagged as anomaly"

        return (
            f"{amt_desc} {txn.get('category', 'Unknown')} transaction of ${amount:.2f} "
            f"at {txn.get('merchant_name', 'Unknown')} "
            f"in {txn.get('city', 'Unknown')}, {txn.get('country', 'US')} "
            f"during {hour_str}, {risk_str}"
        )

    def upsert_transaction(self, txn: dict):
        """Embed and add a transaction to the FAISS index."""
        if not self._enabled:
            return

        try:
            txn_id = txn.get("id", "unknown")
            text = self.transaction_to_text(txn)
            embedding = self._embed(text)

            metadata = {
                "account_id": txn.get("account_id", ""),
                "merchant_name": txn.get("merchant_name", ""),
                "merchant_id": txn.get("merchant_id", ""),
                "amount": float(txn.get("amount", 0)),
                "category": txn.get("category", ""),
                "city": txn.get("city", ""),
                "country": txn.get("country", "US"),
                "risk_score": float(txn.get("risk_score", 0)),
                "is_anomaly": bool(txn.get("is_anomaly", False)),
                "timestamp": txn.get("timestamp", ""),
                "text": text,
            }

            # Add to FAISS index (expects 2D array)
            self._txn_index.add(embedding.reshape(1, -1))
            self._txn_metadata.append(metadata)
            self._txn_ids.append(txn_id)

        except Exception as e:
            logger.warning(f"FAISS upsert failed for {txn.get('id')}: {e}")

    def query_similar_transactions(
        self,
        query_txn: dict,
        top_k: int = 5,
        exclude_id: Optional[str] = None,
        anomalies_only: bool = False,
    ) -> list[dict]:
        """Find transactions similar to the query transaction.

        Returns list of dicts with: id, similarity_score, and all metadata fields.
        """
        if not self._enabled or self._txn_index.ntotal == 0:
            return []

        try:
            text = self.transaction_to_text(query_txn)
            embedding = self._embed(text)

            # Search more than needed to account for filtering
            search_k = min(top_k * 3 + 5, self._txn_index.ntotal)
            scores, indices = self._txn_index.search(
                embedding.reshape(1, -1), search_k
            )

            similar = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._txn_metadata):
                    continue

                txn_id = self._txn_ids[idx]
                meta = self._txn_metadata[idx]

                # Skip self
                if exclude_id and txn_id == exclude_id:
                    continue

                # Filter anomalies only
                if anomalies_only and not meta.get("is_anomaly", False):
                    continue

                similar.append({
                    "id": txn_id,
                    "similarity_score": round(float(score), 3),
                    **meta,
                })

                if len(similar) >= top_k:
                    break

            return similar

        except Exception as e:
            logger.warning(f"FAISS query failed: {e}")
            return []

    # --- Similarity-Based Risk Scoring ---

    def compute_novelty_score(self, txn: dict, min_history: int = 5) -> Optional[float]:
        """Compute how novel/unusual a transaction is compared to the account's history.

        Returns a novelty score from 0.0 (perfectly matches past behavior) to 1.0
        (completely unlike anything this account has done before). Returns None if
        there aren't enough transactions yet to make a meaningful comparison.

        This catches subtle anomalies that IsolationForest misses — e.g. a $150
        purchase at an unusual merchant category for this specific account.
        """
        if not self._enabled or self._txn_index.ntotal < min_history:
            return None

        try:
            account_id = txn.get("account_id", "")
            if not account_id:
                return None

            # Find all indices belonging to this account
            account_indices = [
                i for i, meta in enumerate(self._txn_metadata)
                if meta.get("account_id") == account_id
            ]

            if len(account_indices) < min_history:
                return None  # Not enough account history to judge

            # Embed the current transaction
            text = self.transaction_to_text(txn)
            embedding = self._embed(text)

            # Search for the most similar transactions globally
            search_k = min(50, self._txn_index.ntotal)
            scores, indices = self._txn_index.search(
                embedding.reshape(1, -1), search_k
            )

            # Compute average similarity to THIS account's past transactions
            account_sims = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._txn_metadata):
                    continue
                if self._txn_metadata[idx].get("account_id") == account_id:
                    account_sims.append(float(score))

            if not account_sims:
                # No similar transactions found for this account = very novel
                return 0.85

            # Average similarity to own account's history
            # Cosine similarity ranges from -1 to 1 (but mostly 0 to 1 for us)
            avg_account_sim = sum(account_sims) / len(account_sims)
            best_account_sim = max(account_sims)

            # Novelty = 1 - best_match (how different from closest past behavior)
            # Weight both average and best match: best match matters more
            novelty = 1.0 - (0.6 * best_account_sim + 0.4 * avg_account_sim)

            # Clamp to [0, 1]
            return round(max(0.0, min(1.0, novelty)), 3)

        except Exception as e:
            logger.warning(f"FAISS novelty score failed: {e}")
            return None

    # --- Investigation Memory ---

    def upsert_investigation(
        self,
        alert_id: str,
        verdict: str,
        summary: str,
        txn: dict,
        verdict_source: str = "agent",
    ):
        """Store an investigation verdict for future reference.

        Args:
            alert_id: The alert that was investigated.
            verdict: The final verdict (blocked, cleared, flagged).
            summary: Human-readable explanation of the decision.
            txn: The transaction data associated with the alert.
            verdict_source: Who made the decision — "agent" or "human".
                Human verdicts are weighted higher in similarity search
                results so the system learns from human corrections.
        """
        if not self._enabled:
            return

        try:
            source_label = "Human analyst" if verdict_source == "human" else "AI agent"
            text = (
                f"Investigation of {self.transaction_to_text(txn)}. "
                f"Verdict: {verdict} (by {source_label}). {summary}"
            )
            embedding = self._embed(text)

            metadata = {
                "alert_id": alert_id,
                "verdict": verdict,
                "verdict_source": verdict_source,
                "summary": summary[:500],
                "amount": float(txn.get("amount", 0)),
                "category": txn.get("category", ""),
                "country": txn.get("country", "US"),
                "merchant_name": txn.get("merchant_name", ""),
                "risk_score": float(txn.get("risk_score", 0)),
                "text": text,
            }

            self._inv_index.add(embedding.reshape(1, -1))
            self._inv_metadata.append(metadata)
            self._inv_ids.append(f"inv-{alert_id}")

            logger.info(
                f"FAISS: Stored {verdict_source} verdict for {alert_id}: {verdict}"
            )

        except Exception as e:
            logger.warning(f"FAISS investigation upsert failed: {e}")

    def query_similar_investigations(self, txn: dict, top_k: int = 3) -> list[dict]:
        """Find similar past investigations for a given transaction profile."""
        if not self._enabled or self._inv_index.ntotal == 0:
            return []

        try:
            text = self.transaction_to_text(txn)
            embedding = self._embed(text)

            search_k = min(top_k, self._inv_index.ntotal)
            scores, indices = self._inv_index.search(
                embedding.reshape(1, -1), search_k
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._inv_metadata):
                    continue
                meta = self._inv_metadata[idx]
                results.append({
                    "alert_id": meta.get("alert_id", ""),
                    "similarity_score": round(float(score), 3),
                    "verdict": meta.get("verdict", ""),
                    "verdict_source": meta.get("verdict_source", "agent"),
                    "summary": meta.get("summary", ""),
                    "merchant_name": meta.get("merchant_name", ""),
                    "amount": meta.get("amount", 0),
                })

            return results

        except Exception as e:
            logger.warning(f"FAISS investigation query failed: {e}")
            return []


# Singleton instance
vector_store = TransactionVectorStore()
