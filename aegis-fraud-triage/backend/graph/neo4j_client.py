"""Neo4j graph database client for fraud ring detection.

Models relationships between accounts and merchants as a property graph.
Detects fraud rings — groups of accounts sharing suspicious merchant
connections — using Cypher traversal queries.

Runs 100% locally via Docker (Neo4j Community Edition, no auth).
Gracefully degrades if Neo4j is unavailable.
"""

import logging
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class GraphClient:
    """Manages Neo4j graph database for account-merchant relationship analysis."""

    def __init__(self):
        self._driver = None
        self._enabled = False

    async def initialize(self):
        """Connect to Neo4j and create indexes. Sets self._enabled = True on success."""
        try:
            from neo4j import GraphDatabase

            auth = None
            if settings.neo4j_user:
                auth = (settings.neo4j_user, settings.neo4j_password)

            self._driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
            self._driver.verify_connectivity()

            # Create indexes for performance
            with self._driver.session() as session:
                session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (a:Account) ON (a.id)"
                )
                session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (m:Merchant) ON (m.id)"
                )

            self._enabled = True
            logger.info(f"Neo4j: Connected to {settings.neo4j_uri}")

        except Exception as e:
            logger.warning(
                f"Neo4j: Initialization failed (graph features disabled): {e}"
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def close(self):
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()

    # --- Transaction Graph Upserts ---

    def upsert_transaction(self, txn: dict):
        """Upsert Account and Merchant nodes, create TRANSACTED_AT relationship.

        Called from run_consumer() via loop.run_in_executor() (non-blocking).
        Uses MERGE to avoid duplicate nodes, CREATE for each relationship
        (since each transaction is unique).
        """
        if not self._enabled:
            return
        try:
            with self._driver.session() as session:
                session.run("""
                    MERGE (a:Account {id: $account_id})
                    MERGE (m:Merchant {id: $merchant_id})
                      ON CREATE SET m.name = $merchant_name,
                                    m.category = $category,
                                    m.city = $city,
                                    m.country = $country
                      ON MATCH SET m.name = $merchant_name
                    CREATE (a)-[:TRANSACTED_AT {
                        txn_id: $txn_id,
                        amount: $amount,
                        timestamp: $timestamp,
                        risk_score: $risk_score,
                        is_anomaly: $is_anomaly,
                        category: $category
                    }]->(m)
                """, {
                    "account_id": txn.get("account_id", ""),
                    "merchant_id": txn.get("merchant_id", ""),
                    "merchant_name": txn.get("merchant_name", ""),
                    "category": txn.get("category", ""),
                    "city": txn.get("city", ""),
                    "country": txn.get("country", "US"),
                    "txn_id": txn.get("id", ""),
                    "amount": float(txn.get("amount", 0)),
                    "timestamp": txn.get("timestamp", ""),
                    "risk_score": float(txn.get("risk_score", 0)),
                    "is_anomaly": bool(txn.get("is_anomaly", False)),
                })
        except Exception as e:
            logger.warning(f"Neo4j upsert failed: {e}")

    # --- Fraud Ring Detection ---

    def detect_fraud_ring(self, account_id: str) -> dict:
        """Detect potential fraud ring connections for an account.

        Finds other accounts that share 2+ merchants with the target account,
        weighted by anomaly counts. Returns ring structure with risk scoring.

        Cypher strategy: traverse Account→Merchant←Account paths to find
        accounts with overlapping merchant relationships.
        """
        if not self._enabled:
            return {"ring_detected": False, "reason": "Graph database not available"}

        try:
            with self._driver.session() as session:
                result = session.run("""
                    MATCH (target:Account {id: $account_id})-[r1:TRANSACTED_AT]->(m:Merchant)
                          <-[r2:TRANSACTED_AT]-(other:Account)
                    WHERE other.id <> $account_id
                    WITH other,
                         collect(DISTINCT m.name) AS shared_merchants,
                         count(r2) AS txn_count,
                         sum(CASE WHEN r2.is_anomaly THEN 1 ELSE 0 END) AS anomaly_count
                    WHERE size(shared_merchants) >= 2
                    RETURN other.id AS account_id,
                           shared_merchants,
                           txn_count,
                           anomaly_count
                    ORDER BY anomaly_count DESC, txn_count DESC
                    LIMIT 10
                """, {"account_id": account_id})

                ring_accounts = []
                for record in result:
                    ring_accounts.append({
                        "account_id": record["account_id"],
                        "shared_merchants": record["shared_merchants"],
                        "transaction_count": record["txn_count"],
                        "anomaly_count": record["anomaly_count"],
                    })

                ring_detected = any(
                    a["anomaly_count"] > 0 for a in ring_accounts
                )
                ring_risk = (
                    min(1.0, sum(a["anomaly_count"] for a in ring_accounts) * 0.15)
                    if ring_accounts
                    else 0
                )

                return {
                    "ring_detected": ring_detected,
                    "ring_risk_score": round(ring_risk, 2),
                    "connected_accounts": len(ring_accounts),
                    "accounts": ring_accounts[:5],
                }

        except Exception as e:
            logger.warning(f"Neo4j fraud ring query failed: {e}")
            return {
                "ring_detected": False,
                "reason": f"Query error: {str(e)[:100]}",
            }

    # --- Graph Visualization ---

    def get_account_neighborhood(self, account_id: str) -> dict:
        """Get the graph neighborhood around an account for frontend visualization.

        Returns nodes and edges compatible with React Flow (@xyflow/react).
        Includes the account's merchants and other accounts sharing those merchants.
        """
        if not self._enabled:
            return {"nodes": [], "edges": []}

        try:
            with self._driver.session() as session:
                # Get direct merchant connections and 1-hop neighbors
                result = session.run("""
                    MATCH (a:Account {id: $account_id})-[r:TRANSACTED_AT]->(m:Merchant)
                    WITH a, m, count(r) AS txn_count,
                         sum(CASE WHEN r.is_anomaly THEN 1 ELSE 0 END) AS anomalies,
                         sum(r.amount) AS total_amount
                    OPTIONAL MATCH (other:Account)-[r2:TRANSACTED_AT]->(m)
                    WHERE other.id <> $account_id
                    WITH a, m, txn_count, anomalies, total_amount,
                         collect(DISTINCT other.id)[..5] AS other_accounts
                    RETURN m.id AS merchant_id,
                           m.name AS merchant_name,
                           m.category AS category,
                           m.country AS country,
                           txn_count,
                           anomalies,
                           total_amount,
                           other_accounts
                    LIMIT 20
                """, {"account_id": account_id})

                nodes = [
                    {
                        "id": account_id,
                        "type": "Account",
                        "label": f"Account {account_id[:8]}...",
                    }
                ]
                edges = []
                seen_nodes = {account_id}

                for record in result:
                    mid = record["merchant_id"]
                    if mid not in seen_nodes:
                        nodes.append({
                            "id": mid,
                            "type": "Merchant",
                            "label": record["merchant_name"],
                            "category": record["category"],
                            "country": record["country"],
                        })
                        seen_nodes.add(mid)

                    edges.append({
                        "source": account_id,
                        "target": mid,
                        "txn_count": record["txn_count"],
                        "anomalies": record["anomalies"],
                        "total_amount": round(record["total_amount"], 2),
                    })

                    # Add connected accounts
                    for other_id in record["other_accounts"]:
                        if other_id and other_id not in seen_nodes:
                            nodes.append({
                                "id": other_id,
                                "type": "Account",
                                "label": f"Account {other_id[:8]}...",
                            })
                            seen_nodes.add(other_id)
                            edges.append({
                                "source": other_id,
                                "target": mid,
                                "txn_count": 0,
                                "anomalies": 0,
                                "total_amount": 0,
                            })

                return {"nodes": nodes, "edges": edges}

        except Exception as e:
            logger.warning(f"Neo4j neighborhood query failed: {e}")
            return {"nodes": [], "edges": []}


# Singleton instance
graph_client = GraphClient()
