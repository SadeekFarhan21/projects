# Aegis - System Architecture

This document describes the full architecture of Aegis, from data ingestion through AI investigation to the real-time dashboard.

---

## High-Level Overview

```
                        ┌──────────────────────────────────────────────────────┐
                        │                   DATA SOURCES                       │
                        │                                                      │
                        │  Capital One Nessie API    Mock Transaction Generator │
                        └──────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────────────────┐
                        │              EVENT STREAMING LAYER                    │
                        │                                                      │
                        │  Redpanda (Kafka-compatible)                         │
                        │  Topic: "transactions"                               │
                        └──────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │                    PROCESSING PIPELINE                            │
            │                                                                  │
            │  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
            │  │ Anomaly Engine  │  │ FAISS Novelty│  │ Redis Velocity     │  │
            │  │ (IsolationForest│──│ Scoring      │  │ Tracking           │  │
            │  │  8 features)    │  └──────────────┘  └────────────────────┘  │
            │  └────────┬────────┘                                             │
            │           │ is_anomaly=true                                      │
            │           ▼                                                      │
            │  ┌─────────────────────────────────────────────────────────┐     │
            │  │            AI AGENT LAYER (3 workers)                   │     │
            │  │                                                         │     │
            │  │  Llama 3 (Ollama) + ReAct Loop + 8 Tools               │     │
            │  │  ┌───────┐  ┌───────┐  ┌───────┐                      │     │
            │  │  │BLOCK  │  │CLEAR  │  │FLAG   │                      │     │
            │  │  └───┬───┘  └───┬───┘  └───┬───┘                      │     │
            │  └──────┼──────────┼──────────┼────────────────────────────┘     │
            │         │          │          │                                   │
            │         ▼          ▼          ▼                                   │
            │    Auto-apply   Auto-apply  Human-in-the-Loop                    │
            │                             (awaiting_review)                     │
            └──────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │                    PRESENTATION LAYER                             │
            │                                                                  │
            │  FastAPI  ──WebSocket──>  React Dashboard                        │
            │  REST API                 (real-time bento grid)                  │
            └──────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Transaction Producer

**File:** `backend/streaming/producer.py`

Generates a continuous stream of financial transactions. Two modes:

- **Mock mode** (`MOCK_MODE=true`): Generates synthetic transactions with realistic patterns -- merchants, amounts, locations, timestamps. Includes a configurable fraud injection probability (default 8%).
- **Nessie mode** (`MOCK_MODE=false`): Calls the Capital One Nessie API to create real purchases against sandbox accounts.

The producer also:
- Injects scripted fraud scenarios on demand (14 predefined patterns via `/api/demo/inject-fraud`)
- Generates customer disputes (3% probability on normal transactions)
- Publishes all transactions to the `transactions` Kafka topic

### 2. Transaction Consumer

**File:** `backend/streaming/consumer.py`

Consumes messages from the `transactions` Kafka topic and feeds them into the anomaly detection pipeline. Runs as a background task inside the FastAPI process.

### 3. Anomaly Detection Engine

**File:** `backend/anomaly_detection/engine.py`, `backend/anomaly_detection/features.py`

Uses scikit-learn's `IsolationForest` (100 estimators, 5% contamination) to score every transaction.

**8 Features extracted per transaction:**

| Feature | Description |
|---|---|
| `amount` | Raw transaction amount |
| `log_amount` | log(amount + 1) -- normalizes large values |
| `hour_of_day` | Hour extracted from timestamp (0-23) |
| `day_of_week` | Day of week (0-6) |
| `is_international` | 1 if merchant country differs from account country |
| `velocity` | Transaction count in the last 10 minutes (via Redis) |
| `distance_from_last` | Haversine distance from account's last known location |
| `amount_zscore` | Standard deviations from account's mean spending (via Redis) |

**Risk scoring pipeline:**
1. IsolationForest `decision_function()` produces a raw score
2. Score is inverted and clamped to [0, 1] range
3. FAISS novelty boost: if the transaction is unlike anything the account has done before, the score is boosted
4. Rule-based adjustments: international transactions, high-risk merchant categories, high velocity
5. If final score > threshold: transaction is flagged as anomalous

### 4. FAISS Vector Store

**File:** `backend/vector_store/faiss_store.py`

In-memory semantic similarity search using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).

**Two separate indices:**

1. **Transaction index:** Every transaction is embedded as a text description and stored. Used for:
   - Novelty scoring: "How different is this transaction from what this account usually does?"
   - Similar transaction lookup: Agent tool that finds historically similar transactions

2. **Investigation index:** Every agent/human verdict is embedded and stored. Used for:
   - Similar investigation lookup: Agent tool that checks "What did we decide last time we saw something like this?"
   - Adaptive Shield: Human verdicts tagged with `verdict_source: "human"` carry extra weight

### 5. AI Agent (Fraud Investigation)

**Files:** `backend/agent/investigator.py`, `backend/agent/tools.py`, `backend/agent/prompts.py`

An autonomous investigation agent using the ReAct (Reasoning + Acting) pattern:

```
THOUGHT -> ACTION -> OBSERVATION -> THOUGHT -> ... -> VERDICT
```

**LLM:** Ollama running Llama 3 locally (no cloud API keys needed)

**8 Investigation Tools:**

| Tool | Purpose |
|---|---|
| `check_account_history` | Retrieve recent transactions for the account |
| `verify_merchant` | Check if the merchant is registered and legitimate |
| `check_travel_feasibility` | Calculate if travel between locations is physically possible |
| `get_account_risk_profile` | Get aggregate risk stats for the account (via Redis) |
| `run_kyc_check` | Identity fraud screening with multi-factor risk scoring |
| `find_similar_transactions` | FAISS search for historically similar transactions |
| `find_similar_investigations` | FAISS search for past investigation verdicts on similar cases |
| `recommend_action` | Final tool -- produces BLOCK, CLEAR, or FLAG_FOR_REVIEW |

**Concurrency:** 3 agent workers process the investigation queue in parallel.

**Verdict outcomes:**
- **BLOCK** -- Account is immediately blocked, alert marked as "blocked"
- **CLEAR** -- Transaction is cleared, alert marked as "cleared"
- **FLAG_FOR_REVIEW** -- Agent is uncertain, alert enters Human-in-the-Loop queue

### 6. AI Agent (Dispute Investigation)

**Files:** `backend/agent/dispute_agent.py`, `backend/agent/dispute_tools.py`

Separate agent that handles customer disputes (e.g., "I didn't make this purchase"). Uses a similar ReAct loop with dispute-specific tools. Outcomes: `approved`, `denied`, `escalated`.

### 7. Human-in-the-Loop (HITL)

**Backend:** `POST /api/alerts/{id}/review` in `backend/api/server.py`
**Frontend:** `HumanReviewPanel` in `frontend/.../AlertPanel.tsx`

When the AI agent returns a FLAG_FOR_REVIEW verdict:

1. Alert status changes to `awaiting_review`
2. Slack notification sent to #human-reviews with transaction details and AI analysis
3. Dashboard shows the full evidence trail (agent reasoning steps, tool outputs)
4. Human analyst sees two action buttons: **Block Account** or **Clear Transaction**
5. On decision:
   - Alert status changes to the human's choice (`blocked` or `cleared`)
   - `review_status` is set to `resolved`
   - Verdict is embedded and stored in FAISS with `verdict_source: "human"`
   - Follow-up Slack notification sent to #human-reviews with the resolution
   - Dashboard shows the final status with a `HUMAN` badge

### 8. Adaptive Shield (Feedback Loop)

The Adaptive Shield is not a separate service -- it's the feedback loop created by FAISS:

```
Human/Agent Verdict ──> Embedded in FAISS ──> Future Agent Queries FAISS
                                                  │
                                                  └──> "Similar cases were BLOCKED by a human"
                                                       (influences agent's recommendation)
```

- Every verdict (agent or human) is stored as a vector embedding
- When a new alert is investigated, the agent's `find_similar_investigations` tool retrieves past cases
- Past human verdicts provide direct evidence for the agent's reasoning
- No model retraining required -- the system adapts through retrieval

### 9. Neo4j Graph Database

**File:** `backend/graph/neo4j_client.py`

Models the account-merchant transaction network:

- **Nodes:** Accounts, Merchants
- **Edges:** TRANSACTED_AT relationships with amount, timestamp, risk score
- Used by the agent's `detect_fraud_ring` tool to identify accounts sharing suspicious merchant connections
- **Optional:** System runs without it; graph features are silently disabled

### 10. Redis Cache

**File:** `backend/cache/redis_client.py`

Provides fast per-account statistics:

- Sliding-window transaction velocity (count in last 10 minutes)
- Running statistics: total count, total amount, mean amount, variance, last location
- Used by the anomaly engine for z-score calculation and velocity features
- **Optional:** System runs without it; features fall back to defaults

### 11. Slack Webhook Notifications

**File:** `backend/api/server.py` (three webhook functions)

Three separate Slack webhooks fire at different points in the pipeline:

| Function | Channel | Trigger |
|---|---|---|
| `_send_block_webhook` | #fraud-alerts | AI agent issues a BLOCK verdict |
| `_send_review_webhook` | #human-reviews | Alert enters awaiting_review (needs human attention) |
| `_send_resolution_webhook` | #human-reviews | Human analyst resolves a flagged alert |

Webhooks are fire-and-forget (`asyncio.create_task`), optional (disabled if URLs are empty), and use Slack Incoming Webhook format.

### 12. WebSocket Connection Manager

**File:** `backend/api/connection_manager.py`

Manages real-time streaming to all connected dashboard clients. Event types:

| Event | Data | When |
|---|---|---|
| `transaction` | Full transaction payload | Every new transaction |
| `alert` | Alert with risk details | When anomaly detected |
| `alert_update` | Updated alert status | Agent verdict or human review |
| `dispute` | Dispute details | When dispute created |
| `dispute_update` | Updated dispute status | Agent resolves dispute |
| `dispute_trace` | Thought/Action/Result steps | During dispute investigation |
| `stats` | Aggregated statistics | Periodically |
| `agent_trace` | Thought/Action/Result steps | During fraud investigation |

### 13. React Dashboard

**Directory:** `frontend/mosaic-command-center/`

Bento-grid layout with 7 tiles:

| Tile | Component | Purpose |
|---|---|---|
| Stats Bar | `StatsOverview` | Live counts: transactions, alerts, blocks, money saved |
| Transaction Feed | `TransactionFeed` | Scrolling real-time transaction list with risk indicators |
| Alert Panel | `AlertPanel` | Expandable alerts with evidence, confidence scores, agent verdict, HITL review |
| Agent Console | `AgentConsole` | Tab-based agent brain trace — one tab per parallel investigation |
| Dispute Panel | `DisputePanel` | Customer dispute tracking with AI resolution summaries |
| Risk Chart | `RiskChart` | Real-time risk score distribution (Recharts) |
| Account Activity | `AccountActivity` | Per-account transaction timeline and stats |

**Agent Console tab-based UI:** When 3 agent workers investigate in parallel, each investigation gets its own tab. Active tabs show a pulsing cyan dot; completed tabs show the verdict badge (BLOCK/CLEAR/FLAG). Clicking a tab shows only that investigation's trace steps — no interleaving. New investigations auto-select.

---

## Infrastructure (Docker Compose)

All infrastructure runs via `docker compose up -d`:

| Service | Image | Port | Purpose |
|---|---|---|---|
| Redpanda | `redpandadata/redpanda:v24.1.1` | 19092 | Kafka-compatible message broker |
| Redpanda Console | `redpandadata/console:v2.4.5` | 8080 | Web UI for topic monitoring |
| Neo4j | `neo4j:5-community` | 7474, 7687 | Graph database |
| Redis | `redis:7-alpine` | 6379 | Cache layer |

---

## Data Flow Summary

```
1. Producer generates transaction (mock or Nessie API)
2. Transaction published to Redpanda topic "transactions"
3. Consumer reads transaction from Redpanda
4. Anomaly engine extracts 8 features, runs IsolationForest
5. FAISS checks novelty against account history
6. Redis provides velocity and z-score data
7. If anomalous: create AlertEvent, queue for investigation
8. Transaction + alert broadcast to dashboard via WebSocket
9. Agent worker (1 of 3) picks up alert, runs ReAct investigation
10. Agent calls tools (account history, merchant check, KYC, FAISS similarity, etc.)
11. Agent produces verdict with confidence score: BLOCK, CLEAR, or FLAG_FOR_REVIEW
12. Verdict stored in FAISS, alert updated, broadcast to dashboard
13. If BLOCK: Slack webhook fires to #fraud-alerts
14. If FLAG_FOR_REVIEW: Slack webhook fires to #human-reviews
15. Human reviews in dashboard, decides Block/Clear
16. Resolution webhook fires to #human-reviews with final decision
17. Human decision stored in FAISS (Adaptive Shield), alert finalized
```

---

## Environment Configuration

All settings are in `backend/config.py` and read from `.env`:

| Variable | Default | Required | Description |
|---|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:19092` | Yes | Redpanda/Kafka broker |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Yes | Ollama LLM server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Yes | LLM model name |
| `MOCK_MODE` | `true` | No | Use mock data instead of Nessie API |
| `NESSIE_API_KEY` | `your_key_here` | No | Capital One Nessie API key |
| `NEO4J_URI` | `bolt://localhost:7687` | No | Neo4j connection (optional) |
| `REDIS_URL` | `redis://localhost:6379/0` | No | Redis connection (optional) |
| `TXN_INTERVAL_MIN` | `1.0` | No | Min seconds between transactions |
| `TXN_INTERVAL_MAX` | `3.0` | No | Max seconds between transactions |
| `ANOMALY_PROBABILITY` | `0.08` | No | Fraud injection probability (8%) |
| `DISPUTE_PROBABILITY` | `0.03` | No | Dispute probability (3%) |
| `DEMO_MODE` | `false` | No | Slower rates for live demos |
| `WEBHOOK_URL` | _(empty)_ | No | Slack webhook for #fraud-alerts (AI blocks) |
| `WEBHOOK_URL_HUMAN` | _(empty)_ | No | Slack webhook for #human-reviews (HITL notifications) |

---

## Ports Reference

| Port | Service |
|---|---|
| 3000 | React dashboard |
| 8000 | FastAPI backend |
| 8080 | Redpanda Console |
| 7474 | Neo4j HTTP |
| 7687 | Neo4j Bolt |
| 6379 | Redis |
| 11434 | Ollama |
| 19092 | Redpanda (Kafka) |
