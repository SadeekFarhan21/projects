"""FastAPI server - Central orchestration for Aegis"""

import asyncio
import random
import logging
import traceback
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.connection_manager import ConnectionManager
from backend.api.models import DashboardStats, TransactionEvent, AlertEvent, AlertReviewRequest
from backend.streaming.producer import TransactionProducer, generate_dispute
from backend.streaming.consumer import TransactionConsumer
from backend.anomaly_detection.engine import AnomalyDetectionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
manager = ConnectionManager()
transaction_buffer = deque(maxlen=200)
alert_buffer = deque(maxlen=50)
dispute_buffer = deque(maxlen=100)
stats = DashboardStats()
investigation_queue: asyncio.Queue = asyncio.Queue()
dispute_queue: asyncio.Queue = asyncio.Queue()
producer: TransactionProducer | None = None
anomaly_engine: AnomalyDetectionEngine | None = None
nessie_client = None
_demo_running = False


async def _send_block_webhook(
    alert_data: dict, verdict_summary: str, confidence: float | None,
):
    """Fire-and-forget Slack webhook to #fraud-alerts on AI BLOCK"""
    url = settings.webhook_url
    if not url:
        return
    try:
        import httpx
        txn = alert_data.get("transaction", alert_data)
        conf_str = f" ({confidence:.0f}% confidence)" if confidence else ""
        payload = {
            "text": (
                f":rotating_light: *AEGIS BLOCK*{conf_str}\n"
                f"*{txn.get('merchant_name', '?')}* — ${txn.get('amount', 0):,.2f}\n"
                f"Account: `{str(txn.get('account_id', '?'))[:12]}...`\n"
                f"Location: {txn.get('city', '?')}, {txn.get('country', '?')}\n"
                f"Risk: {txn.get('risk_score', 0):.0%}\n"
                f"_{verdict_summary[:200]}_"
            )
        }
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json=payload)
    except Exception as e:
        logger.warning(f"Webhook failed: {e}")


async def _send_review_webhook(alert_data: dict, verdict_summary: str, confidence: float | None):
    """Send a Slack notification to #human-reviews when an alert needs human review"""
    url = settings.webhook_url_human or settings.webhook_url
    if not url:
        return
    try:
        import httpx
        txn = alert_data.get("transaction", alert_data)
        alert_id = alert_data.get("id", "?")
        conf_str = f"{confidence:.0f}%" if confidence else "N/A"
        payload = {
            "text": (
                f":warning: *NEEDS HUMAN REVIEW*\n"
                f"*{txn.get('merchant_name', '?')}* — ${txn.get('amount', 0):,.2f}\n"
                f"Account: `{str(txn.get('account_id', '?'))[:12]}...`\n"
                f"Location: {txn.get('city', '?')}, {txn.get('country', '?')}\n"
                f"Risk: {txn.get('risk_score', 0):.0%} · Confidence: {conf_str}\n"
                f"Alert: `{alert_id}`\n"
                f"_AI Analysis: {verdict_summary[:200]}_\n"
                f"Review in the AEGIS dashboard to BLOCK or CLEAR."
            )
        }
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json=payload)
    except Exception as e:
        logger.warning(f"Review webhook failed: {e}")


async def _send_resolution_webhook(alert_data: dict, action: str, reason: str):
    """Send a Slack follow-up to #human-reviews when a reviewed alert is resolved"""
    url = settings.webhook_url_human or settings.webhook_url
    if not url:
        return
    try:
        import httpx
        txn = alert_data.get("transaction", alert_data)
        alert_id = alert_data.get("id", "?")
        emoji = ":no_entry:" if action == "blocked" else ":white_check_mark:"
        verb = "BLOCKED" if action == "blocked" else "CLEARED"
        payload = {
            "text": (
                f"{emoji} *RESOLVED: {verb}*\n"
                f"*{txn.get('merchant_name', '?')}* — ${txn.get('amount', 0):,.2f}\n"
                f"Alert: `{alert_id}`\n"
                f"_{reason[:200]}_"
            )
        }
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json=payload)
    except Exception as e:
        logger.warning(f"Resolution webhook failed: {e}")


async def run_producer(prod: TransactionProducer):
    """Background task: generate and publish mock transactions"""
    try:
        await prod.start()
        interval_min = settings.demo_txn_interval if settings.demo_mode else settings.txn_interval_min
        interval_max = settings.demo_txn_interval if settings.demo_mode else settings.txn_interval_max
        await prod.produce_loop(interval_min, interval_max)
    except asyncio.CancelledError:
        await prod.stop()
    except Exception as e:
        logger.error(f"Producer error: {e}\n{traceback.format_exc()}")
        await prod.stop()


async def run_consumer():
    """Background task: consume transactions, run anomaly detection, broadcast, generate disputes"""
    global stats
    consumer = TransactionConsumer(settings.kafka_bootstrap_servers, topic="transactions")
    try:
        await consumer.start()
        async for txn_data in consumer:
            txn = TransactionEvent(**txn_data)

            # Run anomaly detection
            if anomaly_engine:
                result = anomaly_engine.detect(txn_data)
                txn.risk_score = result.score
                txn.is_anomaly = result.is_anomaly

                # Similarity-based risk boosting via FAISS
                # Catches subtle anomalies IsolationForest misses — e.g. a $150
                # purchase at an unusual merchant type for this specific account
                from backend.vector_store.faiss_store import vector_store
                if vector_store.enabled and not txn.is_anomaly:
                    try:
                        novelty = vector_store.compute_novelty_score(txn_data)
                        if novelty is not None and novelty > 0.45:
                            # Blend novelty into risk score: high novelty boosts risk
                            # Scale: 0.45 novelty → +0.08, 0.7 novelty → +0.20, 1.0 → +0.35
                            boost = (novelty - 0.45) * 0.64
                            original_score = txn.risk_score
                            txn.risk_score = min(1.0, txn.risk_score + boost)

                            # Re-check anomaly threshold with boosted score
                            if txn.risk_score > 0.55:
                                txn.is_anomaly = True
                                result.is_anomaly = True
                                result.score = txn.risk_score
                                result.risk_factors.append(
                                    f"Unusual spending pattern for this account "
                                    f"(novelty: {novelty:.0%}, risk boosted {original_score:.2f} → {txn.risk_score:.2f})"
                                )
                                logger.info(
                                    f"FAISS novelty boost: {txn.account_id[:8]}... "
                                    f"novelty={novelty:.2f}, risk {original_score:.2f} → {txn.risk_score:.2f}"
                                )
                    except Exception as e:
                        pass  # Novelty scoring is best-effort, never block pipeline

            # Update stats
            stats.total_transactions += 1
            stats.total_amount += txn.amount
            stats.avg_risk_score = (
                (stats.avg_risk_score * (stats.total_transactions - 1) + txn.risk_score)
                / stats.total_transactions
            )

            # Store in buffer
            txn_dict = txn.model_dump()
            transaction_buffer.appendleft(txn_dict)

            # Update agent tool data store with this transaction
            from backend.agent.tools import _account_histories
            if txn.account_id not in _account_histories:
                _account_histories[txn.account_id] = []
            _account_histories[txn.account_id].append(txn_dict)
            # Keep only last 100 per account to prevent memory bloat
            if len(_account_histories[txn.account_id]) > 100:
                _account_histories[txn.account_id] = _account_histories[txn.account_id][-100:]

            # Upsert to vector store (non-blocking, in thread pool)
            from backend.vector_store.faiss_store import vector_store
            if vector_store.enabled:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, vector_store.upsert_transaction, txn_dict)

            # Upsert to Neo4j graph (non-blocking, in thread pool)
            from backend.graph.neo4j_client import graph_client
            if graph_client.enabled:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, graph_client.upsert_transaction, txn_dict)

            # Update Redis velocity and account stats (async, non-blocking)
            from backend.cache.redis_client import redis_cache
            if redis_cache.enabled:
                asyncio.create_task(redis_cache.record_transaction_velocity(
                    txn.account_id, txn.timestamp, txn.id
                ))
                asyncio.create_task(redis_cache.update_account_risk_stats(
                    txn.account_id, txn_dict
                ))

            # Broadcast transaction to all WebSocket clients
            await manager.broadcast("transaction", txn_dict)

            # If anomaly, create alert and broadcast
            if txn.is_anomaly:
                alert = AlertEvent(
                    id=f"alert-{txn.id}",
                    transaction=txn,
                    risk_score=txn.risk_score,
                    risk_factors=result.risk_factors if anomaly_engine else [],
                    status="pending",
                    timestamp=datetime.utcnow().isoformat(),
                )
                stats.flagged_count += 1
                alert_buffer.appendleft(alert.model_dump())
                await manager.broadcast("alert", alert.model_dump())

                # Enqueue for agent investigation
                await investigation_queue.put(alert)

            # Random dispute generation (only for non-anomaly transactions)
            elif random.random() < settings.dispute_probability:
                dispute = generate_dispute(txn_dict)
                dispute_buffer.appendleft(dispute)
                stats.disputes_filed += 1
                await manager.broadcast("dispute", dispute)
                await dispute_queue.put(dispute)
                logger.info(f"Dispute filed: {dispute['id']} - {dispute['reason']} for ${dispute['amount']:.2f}")

            # Broadcast updated stats
            await manager.broadcast("stats", stats.model_dump())

    except asyncio.CancelledError:
        await consumer.stop()
    except Exception as e:
        logger.error(f"Consumer error: {e}\n{traceback.format_exc()}")
        await consumer.stop()


async def run_agent_worker():
    """Background task: investigate anomalies with AI agent"""
    try:
        from backend.agent.investigator import FraudInvestigatorAgent
        agent = FraudInvestigatorAgent(
            ollama_base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            broadcast_fn=manager.broadcast,
        )

        while True:
            alert = await investigation_queue.get()
            try:
                logger.info(f"Agent investigating alert: {alert.id}")
                verdict = await agent.investigate(alert.model_dump())

                # Update alert with verdict and broadcast
                verdict_action = verdict.get("action", "flagged")
                for stored_alert in alert_buffer:
                    if stored_alert["id"] == alert.id:
                        stored_alert["agent_verdict"] = verdict.get("summary", "")
                        stored_alert["confidence_score"] = verdict.get("confidence")

                        # HITL: "flagged" verdicts go to human review instead of
                        # being finalized. "blocked" and "cleared" are high-confidence
                        # and applied immediately.
                        if verdict_action == "flagged":
                            stored_alert["status"] = "awaiting_review"
                            stored_alert["review_status"] = "awaiting_review"
                            # Notify #human-reviews that this alert needs attention
                            asyncio.create_task(_send_review_webhook(
                                stored_alert, verdict.get("summary", ""), verdict.get("confidence")))
                        else:
                            stored_alert["status"] = verdict_action

                        await manager.broadcast("alert_update", stored_alert)
                        break

                # Only update stats for high-confidence verdicts;
                # awaiting_review alerts will update stats when a human decides
                if verdict_action == "blocked":
                    stats.blocked_count += 1
                    stats.money_saved += alert.transaction.amount
                    asyncio.create_task(_send_block_webhook(
                        alert.model_dump(), verdict.get("summary", ""), verdict.get("confidence")))
                elif verdict_action == "cleared":
                    stats.cleared_count += 1

                # Store investigation verdict in vector memory
                from backend.vector_store.faiss_store import vector_store
                if vector_store.enabled:
                    txn_data = alert.model_dump().get("transaction", {})
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(
                        None,
                        vector_store.upsert_investigation,
                        alert.id,
                        verdict.get("action", "flagged"),
                        verdict.get("summary", ""),
                        txn_data,
                    )

                stats.investigations_completed += 1
                await manager.broadcast("stats", stats.model_dump())
            except Exception as e:
                logger.error(f"Agent investigation failed: {e}\n{traceback.format_exc()}")
                await manager.broadcast("agent_trace", {
                    "alert_id": alert.id,
                    "step_type": "verdict",
                    "content": f"Investigation error: {str(e)[:200]}. Alert flagged for manual review.",
                    "timestamp": datetime.utcnow().isoformat(),
                })
    except asyncio.CancelledError:
        pass
    except ImportError:
        logger.warning("Agent module not available, skipping agent worker")


async def run_account_grower():
    """Background task: periodically create new Nessie customers/accounts so the pool grows"""
    if not nessie_client:
        logger.info("Account grower: no Nessie client, skipping")
        return

    from backend.nessie_client.seeder import _create_customer, _create_account
    from backend.streaming import producer as prod_module
    from backend.kyc.engine import set_account_customer_mapping

    # Pool of names to create accounts for over time
    new_customers = [
        {"first_name": "James", "last_name": "Taylor"},
        {"first_name": "Sophia", "last_name": "Nguyen"},
        {"first_name": "Marcus", "last_name": "Brown"},
        {"first_name": "Priya", "last_name": "Sharma"},
        {"first_name": "Liam", "last_name": "O'Connor"},
        {"first_name": "Mei", "last_name": "Zhang"},
        {"first_name": "Carlos", "last_name": "Garcia"},
        {"first_name": "Aisha", "last_name": "Mohammed"},
        {"first_name": "Ryan", "last_name": "Murphy"},
        {"first_name": "Yuki", "last_name": "Tanaka"},
    ]

    idx = 0
    # Wait a bit before starting, let the system warm up
    await asyncio.sleep(30)

    while idx < len(new_customers):
        try:
            info = new_customers[idx]
            customer_id = await _create_customer(nessie_client, info)
            if customer_id:
                # Create a checking account
                account = await _create_account(nessie_client, customer_id, "Checking", round(random.uniform(2000, 15000), 2))
                if account:
                    prod_module.ACCOUNT_IDS.append(account.id)
                    set_account_customer_mapping(account.id, customer_id)
                    logger.info(f"Account grower: added {info['first_name']} {info['last_name']} -> {account.id}")

                # ~40% chance of also creating a savings account
                if random.random() < 0.4:
                    account2 = await _create_account(nessie_client, customer_id, "Savings", round(random.uniform(5000, 30000), 2))
                    if account2:
                        prod_module.ACCOUNT_IDS.append(account2.id)
                        set_account_customer_mapping(account2.id, customer_id)

            idx += 1
        except Exception as e:
            logger.warning(f"Account grower: failed to create customer: {e}")
            idx += 1

        # Wait 45-90 seconds between new customers for a natural growth pace
        await asyncio.sleep(random.uniform(45, 90))

    logger.info(f"Account grower: finished, {len(prod_module.ACCOUNT_IDS)} total accounts in pool")


async def run_dispute_worker():
    """Background task: investigate customer disputes with AI agent"""
    try:
        from backend.agent.dispute_agent import DisputeInvestigatorAgent
        from backend.agent.dispute_tools import set_dispute_buffers, record_dispute_outcome

        # Wire buffers into dispute tools for context lookups
        set_dispute_buffers(alert_buffer, transaction_buffer)

        agent = DisputeInvestigatorAgent(
            ollama_base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            broadcast_fn=manager.broadcast,
        )

        while True:
            dispute = await dispute_queue.get()
            try:
                dispute_id = dispute["id"]
                logger.info(f"Dispute agent investigating: {dispute_id}")

                # Update status to investigating
                for stored in dispute_buffer:
                    if stored["id"] == dispute_id:
                        stored["status"] = "investigating"
                        await manager.broadcast("dispute_update", stored)
                        break

                # Run investigation
                result = await agent.investigate(dispute)

                # Update dispute with resolution
                action = result.get("action", "escalated")
                for stored in dispute_buffer:
                    if stored["id"] == dispute_id:
                        stored["status"] = action
                        stored["agent_resolution"] = action
                        stored["resolution_summary"] = result.get("summary", "")[:300]
                        stored["resolved_at"] = datetime.utcnow().isoformat()
                        await manager.broadcast("dispute_update", stored)
                        break

                # Update stats
                if action == "approved":
                    stats.disputes_approved += 1
                elif action == "denied":
                    stats.disputes_denied += 1

                # Record for history tracking
                record_dispute_outcome(dispute["account_id"], {
                    "reason": dispute["reason"],
                    "amount": dispute["amount"],
                    "action": action.upper(),
                })

                # Adaptive Shield: upsert dispute resolution into FAISS
                from backend.vector_store.faiss_store import vector_store
                if vector_store.enabled:
                    dispute_txn = {
                        "id": dispute.get("transaction_id", dispute_id),
                        "account_id": dispute.get("account_id", ""),
                        "merchant_name": dispute.get("merchant_name", "Unknown"),
                        "amount": dispute.get("amount", 0),
                        "category": "Dispute",
                        "city": "",
                        "country": "US",
                        "risk_score": 0.5 if action == "approved" else 0.2,
                        "is_anomaly": action == "approved",
                        "timestamp": dispute.get("timestamp", ""),
                    }
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(
                        None,
                        vector_store.upsert_investigation,
                        f"dispute-{dispute_id}",
                        "blocked" if action == "approved" else "cleared",
                        result.get("summary", "")[:300],
                        dispute_txn,
                        "human",
                    )

                await manager.broadcast("stats", stats.model_dump())
                logger.info(f"Dispute {dispute_id} resolved: {action}")

            except Exception as e:
                logger.error(f"Dispute investigation failed: {e}\n{traceback.format_exc()}")
                # Mark as escalated on error
                for stored in dispute_buffer:
                    if stored["id"] == dispute.get("id"):
                        stored["status"] = "escalated"
                        stored["resolution_summary"] = f"Error: {str(e)[:200]}"
                        stored["resolved_at"] = datetime.utcnow().isoformat()
                        await manager.broadcast("dispute_update", stored)
                        break
    except asyncio.CancelledError:
        pass
    except ImportError as e:
        logger.warning(f"Dispute agent module not available: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer, anomaly_engine, nessie_client

    # Initialize anomaly detection
    anomaly_engine = AnomalyDetectionEngine()
    logger.info("Anomaly detection engine initialized")

    # Initialize FAISS vector store (local, in-memory)
    from backend.vector_store.faiss_store import vector_store
    await vector_store.initialize()
    if vector_store.enabled:
        logger.info("Vector store (FAISS) initialized")

    # Initialize Neo4j graph database
    from backend.graph.neo4j_client import graph_client
    await graph_client.initialize()

    # Initialize Redis cache
    from backend.cache.redis_client import redis_cache
    await redis_cache.initialize()

    # Initialize Nessie API if key is configured
    if settings.nessie_api_key and settings.nessie_api_key != "your_key_here":
        try:
            from backend.nessie_client.client import NessieClient
            from backend.nessie_client.seeder import seed_nessie_data
            nessie_client = NessieClient(api_key=settings.nessie_api_key)
            nessie_data = await seed_nessie_data(nessie_client)

            # Wire Nessie account IDs into the producer
            from backend.streaming import producer as prod_module
            if nessie_data.get("account_ids"):
                prod_module.ACCOUNT_IDS[:] = nessie_data["account_ids"]
                logger.info(f"Nessie API seeded: {len(nessie_data['account_ids'])} accounts loaded")

            # Wire Nessie merchants into the producer
            if nessie_data.get("merchants"):
                prod_module.load_nessie_merchants(nessie_data["merchants"])

            # Wire Nessie client into producer (for creating real purchases)
            prod_module.set_nessie_client(nessie_client)

            # Wire Nessie client into agent tools
            from backend.agent import tools as agent_tools
            agent_tools.set_nessie_client(nessie_client)
        except Exception as e:
            logger.warning(f"Nessie API initialization failed (using mock data): {e}")
            nessie_client = None
    else:
        logger.info("No Nessie API key configured, using mock data")

    # Start producer
    producer = TransactionProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        anomaly_probability=settings.anomaly_probability,
    )

    # Launch background tasks (3 parallel agent workers for concurrent investigations)
    tasks = [
        asyncio.create_task(run_producer(producer)),
        asyncio.create_task(run_consumer()),
        asyncio.create_task(run_agent_worker()),
        asyncio.create_task(run_agent_worker()),
        asyncio.create_task(run_agent_worker()),
        asyncio.create_task(run_dispute_worker()),
        asyncio.create_task(run_account_grower()),
    ]

    logger.info("Aegis backend started")
    yield

    # Shutdown
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if producer:
        await producer.stop()
    if nessie_client:
        await nessie_client.close()
    # Close Neo4j and Redis
    from backend.graph.neo4j_client import graph_client
    await graph_client.close()
    from backend.cache.redis_client import redis_cache
    await redis_cache.close()
    logger.info("Aegis backend stopped")


app = FastAPI(title="Aegis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Silence /metrics 404 noise (hit by browser extensions / monitoring tools)
@app.get("/metrics")
@app.get("/metrics/")
async def metrics_noop():
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    from backend.vector_store.faiss_store import vector_store
    from backend.graph.neo4j_client import graph_client
    from backend.cache.redis_client import redis_cache
    return {
        "status": "ok",
        "service": "aegis",
        "nessie_connected": nessie_client is not None,
        "anomaly_engine": anomaly_engine is not None,
        "producer_active": producer is not None,
        "vector_store_enabled": vector_store.enabled,
        "neo4j_connected": graph_client.enabled,
        "redis_connected": redis_cache.enabled,
    }


@app.get("/api/transactions")
async def get_transactions(limit: int = 50):
    return list(transaction_buffer)[:limit]


@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    return list(alert_buffer)[:limit]


@app.get("/api/disputes")
async def get_disputes(limit: int = 20):
    return list(dispute_buffer)[:limit]


@app.get("/api/stats")
async def get_stats():
    return stats.model_dump()


@app.get("/api/kyc/{account_id}")
async def get_kyc(account_id: str):
    """Run KYC risk assessment for an account"""
    from backend.kyc.engine import kyc_engine
    from backend.agent.tools import _account_histories
    history = _account_histories.get(account_id, [])
    result = kyc_engine.assess(account_id, history)
    return result


@app.post("/api/demo/inject-fraud")
async def inject_fraud(scenario: int = 0):
    """Inject a dramatic fraud scenario for demo purposes"""
    if producer:
        txn = await producer.inject_fraud(scenario)
        return {"status": "injected", "transaction": txn}
    return {"status": "error", "message": "Producer not running"}


@app.post("/api/demo/inject-dispute")
async def inject_dispute():
    """Manually trigger a dispute on a recent transaction for demo"""
    if not transaction_buffer:
        return {"status": "error", "message": "No transactions available"}

    # Pick a random recent non-anomaly transaction
    candidates = [t for t in list(transaction_buffer)[:20] if not t.get("is_anomaly")]
    if not candidates:
        candidates = list(transaction_buffer)[:5]

    txn = random.choice(candidates)
    dispute = generate_dispute(txn)
    dispute_buffer.appendleft(dispute)
    stats.disputes_filed += 1
    await manager.broadcast("dispute", dispute)
    await dispute_queue.put(dispute)
    await manager.broadcast("stats", stats.model_dump())

    return {"status": "injected", "dispute": dispute}


@app.post("/api/demo/start")
async def start_demo():
    """Start an automated demo sequence that injects diverse fraud scenarios"""
    global _demo_running
    if _demo_running:
        return {"status": "already_running"}
    if not producer:
        return {"status": "error", "message": "Producer not running"}
    _demo_running = True
    asyncio.create_task(_run_demo_sequence())
    return {"status": "started"}


@app.post("/api/demo/stop")
async def stop_demo():
    """Stop the running demo sequence"""
    global _demo_running
    _demo_running = False
    return {"status": "stopped"}


@app.get("/api/demo/status")
async def demo_status():
    """Check if a demo sequence is currently running"""
    return {"running": _demo_running}


async def _run_demo_sequence():
    """Run a curated demo loop: fraud scenarios, disputes, and mixed events.

    Loops continuously with varied pacing to keep the dashboard lively.
    Injects a mix of fraud, disputes, and back-to-back events.
    """
    global _demo_running
    try:
        # Curated rounds — each round is a themed burst of activity
        rounds = [
            # Round 1: International fraud blitz
            {"frauds": [0, 1, 7], "dispute": True, "pace": (10, 16)},
            # Round 2: Financial crimes
            {"frauds": [4, 8, 2], "dispute": False, "pace": (12, 18)},
            # Round 3: Luxury + dispute wave
            {"frauds": [11, 5], "dispute": True, "pace": (10, 15)},
            # Round 4: Steady stream
            {"frauds": [3, 10, 6, 12], "dispute": True, "pace": (10, 14)},
            # Round 5: High-value targets
            {"frauds": [9, 13, 8], "dispute": False, "pace": (12, 16)},
            # Round 6: Mixed chaos
            {"frauds": [0, 7, 4, 11], "dispute": True, "pace": (10, 14)},
        ]

        round_idx = 0
        while _demo_running:
            current_round = rounds[round_idx % len(rounds)]
            pace_min, pace_max = current_round["pace"]

            # Inject fraud scenarios for this round
            for fraud_idx in current_round["frauds"]:
                if not _demo_running:
                    break
                await asyncio.sleep(random.uniform(pace_min, pace_max))
                try:
                    await producer.inject_fraud(fraud_idx)
                except Exception as e:
                    logger.warning(f"Demo fraud inject failed: {e}")

                # Occasionally inject a dispute mid-round for variety
                if random.random() < 0.3 and transaction_buffer:
                    await asyncio.sleep(random.uniform(4, 7))
                    candidates = [t for t in list(transaction_buffer)[:20] if not t.get("is_anomaly")]
                    if candidates:
                        txn = random.choice(candidates)
                        dispute = generate_dispute(txn)
                        dispute_buffer.appendleft(dispute)
                        stats.disputes_filed += 1
                        await manager.broadcast("dispute", dispute)
                        await dispute_queue.put(dispute)
                        await manager.broadcast("stats", stats.model_dump())

            # End-of-round dispute injection if flagged
            if current_round["dispute"] and _demo_running and transaction_buffer:
                await asyncio.sleep(random.uniform(4, 7))
                candidates = [t for t in list(transaction_buffer)[:20] if not t.get("is_anomaly")]
                if candidates:
                    txn = random.choice(candidates)
                    dispute = generate_dispute(txn)
                    dispute_buffer.appendleft(dispute)
                    stats.disputes_filed += 1
                    await manager.broadcast("dispute", dispute)
                    await dispute_queue.put(dispute)
                    await manager.broadcast("stats", stats.model_dump())

            # Brief breather between rounds
            if _demo_running:
                await asyncio.sleep(random.uniform(10, 16))

            round_idx += 1

    except Exception as e:
        logger.error(f"Demo sequence error: {e}")
    finally:
        _demo_running = False


@app.post("/api/alerts/{alert_id}/review")
async def review_alert(alert_id: str, review: AlertReviewRequest):
    """Human-in-the-loop: confirm or override an AI agent's verdict.

    When the agent returns FLAG_FOR_REVIEW, the alert enters 'awaiting_review' status.
    A human analyst can then confirm the agent's recommendation or override it.
    The decision is fed back into the FAISS vector store so future investigations
    learn from human corrections (Adaptive Shield).
    """
    # Find the alert in the buffer
    target_alert = None
    for stored_alert in alert_buffer:
        if stored_alert["id"] == alert_id:
            target_alert = stored_alert
            break

    if target_alert is None:
        return {"status": "error", "message": f"Alert {alert_id} not found"}

    # Human must choose an explicit action: block or clear
    chosen_action = review.override_action
    if not chosen_action or chosen_action not in ("blocked", "cleared"):
        return {"status": "error", "message": "override_action must be 'blocked' or 'cleared'"}

    old_status = target_alert["status"]
    target_alert["status"] = chosen_action
    target_alert["review_status"] = "resolved"
    target_alert["human_override"] = chosen_action
    target_alert["human_reason"] = review.reason or f"Human decided to {chosen_action}"

    # Update stats
    if chosen_action == "blocked":
        stats.blocked_count += 1
        txn_amount = target_alert.get("transaction", {}).get("amount", 0)
        stats.money_saved += txn_amount
    elif chosen_action == "cleared":
        stats.cleared_count += 1

    # Notify #human-reviews with the resolution (follow-up to the review request)
    asyncio.create_task(_send_resolution_webhook(
        target_alert, chosen_action, target_alert.get("human_reason", "")))

    logger.info(f"HITL: Alert {alert_id} resolved by human: {old_status} -> {chosen_action}")

    # Adaptive Shield: upsert human decision into FAISS for future learning
    from backend.vector_store.faiss_store import vector_store
    if vector_store.enabled:
        txn_data = target_alert.get("transaction", {})
        final_verdict = target_alert["status"]
        summary = target_alert.get("human_reason", "")
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            vector_store.upsert_investigation,
            alert_id,
            final_verdict,
            summary,
            txn_data,
            "human",  # verdict_source
        )

    # Broadcast the updated alert and stats
    await manager.broadcast("alert_update", target_alert)
    await manager.broadcast("stats", stats.model_dump())

    return {
        "status": "ok",
        "alert_id": alert_id,
        "review_status": target_alert["review_status"],
        "final_verdict": target_alert["status"],
    }


@app.get("/api/graph/account/{account_id}")
async def get_account_graph(account_id: str):
    """Get the graph neighborhood around an account for visualization"""
    from backend.graph.neo4j_client import graph_client
    if not graph_client.enabled:
        return {"nodes": [], "edges": [], "error": "Graph database not available"}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, graph_client.get_account_neighborhood, account_id
    )
    return result


@app.get("/api/graph/fraud-ring/{account_id}")
async def get_fraud_ring(account_id: str):
    """Detect fraud ring connections for an account"""
    from backend.graph.neo4j_client import graph_client
    if not graph_client.enabled:
        return {"ring_detected": False, "reason": "Graph database not available"}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, graph_client.detect_fraud_ring, account_id
    )
    return result


@app.get("/api/graph/network")
async def get_fraud_network():
    """Build a fraud-only network graph from confirmed blocked/flagged alerts.

    Only includes accounts and merchants that have been involved in confirmed
    fraud (blocked by AI or flagged for human review). Shows how fraudulent
    accounts connect through shared merchants, with real merchant IDs, amounts,
    verdict status, and confidence scores.
    """
    from collections import defaultdict

    # Collect fraud-confirmed data from alert buffer
    # Only include alerts that have a verdict (blocked, flagged, awaiting_review)
    fraud_edges: list[dict] = []
    fraud_accounts: dict[str, dict] = {}
    fraud_merchants: dict[str, dict] = {}
    merchant_fraud_accounts: dict[str, set[str]] = defaultdict(set)

    for alert_dict in alert_buffer:
        status = alert_dict.get("status", "")
        # Only show confirmed fraud: blocked, flagged, or awaiting human review
        if status not in ("blocked", "flagged", "awaiting_review"):
            continue

        txn = alert_dict.get("transaction", {})
        aid = txn.get("account_id", "")
        mid = txn.get("merchant_id", "")
        if not aid or not mid:
            continue

        merchant_fraud_accounts[mid].add(aid)

        # Build/update account node info
        if aid not in fraud_accounts:
            fraud_accounts[aid] = {
                "id": aid,
                "type": "account",
                "label": f"...{aid[-6:]}" if len(aid) > 6 else aid,
                "status": status,
                "alert_count": 0,
                "total_amount": 0.0,
            }
        fraud_accounts[aid]["alert_count"] += 1
        fraud_accounts[aid]["total_amount"] += txn.get("amount", 0)
        # Escalate status: blocked > awaiting_review > flagged
        if status == "blocked":
            fraud_accounts[aid]["status"] = "blocked"
        elif status == "awaiting_review" and fraud_accounts[aid]["status"] != "blocked":
            fraud_accounts[aid]["status"] = "awaiting_review"

        # Build/update merchant node info
        if mid not in fraud_merchants:
            fraud_merchants[mid] = {
                "id": mid,
                "type": "merchant",
                "label": txn.get("merchant_name", mid),
                "merchant_id": mid,
                "category": txn.get("category", "Unknown"),
                "city": txn.get("city", ""),
                "country": txn.get("country", ""),
                "fraud_count": 0,
                "total_fraud_amount": 0.0,
            }
        fraud_merchants[mid]["fraud_count"] += 1
        fraud_merchants[mid]["total_fraud_amount"] += txn.get("amount", 0)

        # Build edge
        fraud_edges.append({
            "source": aid,
            "target": mid,
            "amount": round(txn.get("amount", 0), 2),
            "status": status,
            "confidence": alert_dict.get("confidence_score"),
            "risk_score": round(txn.get("risk_score", 0), 2),
            "alert_id": alert_dict.get("id", ""),
        })

    # Mark merchants involved with multiple fraudulent accounts (fraud ring indicator)
    for mid, info in fraud_merchants.items():
        info["shared_accounts"] = len(merchant_fraud_accounts[mid])
        info["is_ring_node"] = len(merchant_fraud_accounts[mid]) >= 2

    # Build node list
    nodes = list(fraud_accounts.values()) + list(fraud_merchants.values())

    return {
        "nodes": nodes,
        "edges": fraud_edges,
        "total_accounts": len(fraud_accounts),
        "total_merchants": len(fraud_merchants),
        "ring_merchants": sum(1 for m in fraud_merchants.values() if m.get("is_ring_node")),
    }


@app.get("/api/cache/velocity/{account_id}")
async def get_account_velocity(account_id: str):
    """Get transaction velocity (count in last 10 min) for an account"""
    from backend.cache.redis_client import redis_cache
    if not redis_cache.enabled:
        return {"velocity": 0, "error": "Redis cache not available"}
    velocity = await redis_cache.get_velocity(account_id)
    stats_data = await redis_cache.get_account_stats(account_id)
    return {"velocity": velocity, "stats": stats_data}


@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current state snapshot on connect (backfill)
        await websocket.send_json({"type": "stats", "data": stats.model_dump()})
        await websocket.send_json({
            "type": "backfill",
            "data": {
                "transactions": list(transaction_buffer)[:20],
                "alerts": list(alert_buffer)[:10],
                "disputes": list(dispute_buffer)[:10],
            },
        })

        # Keep connection alive - handle pings and any client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Client can send "ping" to keep alive
                if data == "ping":
                    await websocket.send_json({"type": "pong", "data": {}})
            except asyncio.TimeoutError:
                # Send keepalive ping from server side
                await websocket.send_json({"type": "ping", "data": {}})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
