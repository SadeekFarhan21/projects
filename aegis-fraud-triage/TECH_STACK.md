# Aegis - Technology Stack & Justifications

A breakdown of every technology used in this project, why it was chosen, and what it brings to the platform.

---

## Event Streaming

### Redpanda (Kafka-compatible)

**What it is:** A Kafka-compatible streaming platform written in C++, designed for low latency with zero JVM overhead.

**Why we use it:**
- Decouples transaction generation from processing -- the producer and consumer can evolve independently
- Provides message durability and replay capability; if the anomaly engine crashes, no transactions are lost
- Kafka is the industry standard for financial event streaming; Redpanda gives us full compatibility without the JVM memory overhead
- Single-node setup runs in Docker with 512MB RAM -- lightweight for development and demos

**Why not just pass data directly?** In production, the producer and consumer would be separate services on separate machines. The Kafka topic is the contract between them. For a hackathon, it also demonstrates we understand distributed systems architecture.

---

## Backend

### FastAPI

**What it is:** A modern, async-first Python web framework built on Starlette and Pydantic.

**Why we use it:**
- Native `async/await` support -- critical for our concurrent pipeline (WebSockets, Kafka consumption, agent workers all run as async tasks)
- Automatic OpenAPI documentation at `/docs` -- useful for demos and debugging
- Built-in WebSocket support for real-time dashboard streaming
- Pydantic integration for request/response validation with zero boilerplate
- Lifespan events for clean startup/shutdown of background services (Kafka, Neo4j, Redis connections)

**Why not Flask/Django?** Flask lacks native async support. Django is too heavyweight for a real-time streaming API. FastAPI gives us the async primitives we need without framework bloat.

### Pydantic

**What it is:** Data validation library using Python type annotations.

**Why we use it:**
- Defines strict schemas for transactions, alerts, disputes, and API requests
- Catches malformed data at the boundary before it enters the pipeline
- `pydantic-settings` loads and validates environment variables from `.env` with type coercion
- Serialization to JSON for WebSocket broadcasting is automatic

### aiokafka

**What it is:** Async Python client for Apache Kafka.

**Why we use it:**
- Integrates cleanly with FastAPI's async event loop -- no thread pool overhead
- Runs the Kafka producer and consumer as lightweight coroutines alongside the web server
- Handles backpressure, reconnection, and offset management out of the box

### WebSockets

**What it is:** Persistent bidirectional connections between the backend and dashboard.

**Why we use it:**
- Sub-second latency for real-time event delivery (transactions, alerts, agent traces)
- No polling overhead -- events push to the client the instant they occur
- Connection manager broadcasts to all connected dashboard instances simultaneously
- Automatic reconnection logic on the frontend for resilience

---

## Machine Learning

### scikit-learn (Isolation Forest)

**What it is:** A well-established ML library. Isolation Forest is an unsupervised anomaly detection algorithm that isolates outliers by random partitioning.

**Why we use it:**
- Unsupervised -- no labeled fraud data needed (critical for a hackathon where we generate synthetic data)
- Trains in milliseconds on small datasets, scores individual transactions in microseconds
- Naturally suited for fraud detection: fraudulent transactions are "few and different," which is exactly what Isolation Forest detects
- 8-feature extraction (amount, log-amount, hour, day, international, velocity, distance, z-score) captures multiple fraud signals

**Why not deep learning?** Overkill for this use case. Isolation Forest is interpretable, fast, and works well with tabular financial data. No GPU required.

### FAISS (Facebook AI Similarity Search)

**What it is:** In-memory vector similarity search library from Meta.

**Why we use it:**
- Enables the "Adaptive Shield" -- storing and retrieving past verdicts by semantic similarity
- Sub-millisecond nearest-neighbor search even with thousands of vectors
- No external service required; runs entirely in-process
- Two indices: one for transaction history (novelty detection), one for investigation verdicts (learning from past decisions)

**Why not Pinecone/Weaviate?** Those require cloud accounts or additional Docker services. FAISS is zero-config, in-memory, and fast enough for our scale.

### SentenceTransformers (all-MiniLM-L6-v2)

**What it is:** A lightweight transformer model for generating text embeddings (384 dimensions).

**Why we use it:**
- Converts transaction descriptions and investigation summaries into dense vectors for FAISS
- MiniLM-L6-v2 is only 80MB -- loads in seconds, embeds in milliseconds
- Produces semantically meaningful embeddings: similar transactions cluster together in vector space
- Runs on CPU with no GPU required

---

## AI Agent

### LangChain

**What it is:** Framework for building LLM-powered applications with tool use, chains, and agents.

**Why we use it:**
- Provides the ReAct (Reasoning + Acting) agent pattern out of the box
- Tool abstraction lets us define 8 investigation tools as simple Python functions
- Handles the thought-action-observation loop, parsing, and error recovery
- Streaming callbacks give us real-time agent trace events for the dashboard

**Why not raw API calls?** Writing a ReAct loop from scratch is error-prone. LangChain handles prompt formatting, tool dispatch, output parsing, and retry logic.

### Ollama + Llama 3

**What it is:** Ollama is a local LLM runtime. Llama 3 is Meta's open-source language model.

**Why we use it:**
- Runs entirely locally -- no API keys, no cloud costs, no rate limits, no data leaving the machine
- Llama 3 (8B) provides strong reasoning for tool-use and investigation tasks
- Llama 3.2 (3B) is available as a faster fallback for lower-spec hardware
- Privacy-first: financial transaction data never leaves the local network
- Demonstrates that sophisticated AI agents can run on commodity hardware

**Why not GPT-4/Claude API?** Cost, latency to external APIs, and data privacy concerns. For a financial fraud system, keeping data local is a strong architectural choice.

---

## Data Storage

### Neo4j (Graph Database)

**What it is:** Native graph database using the Cypher query language.

**Why we use it:**
- Models the natural graph structure of financial transactions: accounts transact at merchants
- Fraud ring detection via graph traversal -- "find accounts that share 2+ suspicious merchants" is a single Cypher query
- Relationship-first data model is fundamentally better than SQL joins for network analysis
- Community Edition runs in Docker with no license required

**Why not just SQL?** Detecting fraud rings requires multi-hop graph traversal. In SQL, this requires recursive CTEs or multiple self-joins that become unreadable and slow. In Cypher: `MATCH (a)-[:TRANSACTED_AT]->(m)<-[:TRANSACTED_AT]-(b)` -- one line.

### Redis

**What it is:** In-memory key-value store with data structures (sorted sets, hashes).

**Why we use it:**
- Sliding-window velocity tracking using sorted sets (ZADD + ZCOUNT by timestamp) -- O(log n) operations
- Per-account running statistics (transaction count, total amount, last location) via hashes
- Sub-millisecond reads for the anomaly engine's feature extraction -- no disk I/O
- Automatic expiration (TTL) prevents unbounded memory growth
- LRU eviction policy as a safety net

**Why not just in-memory Python dicts?** Redis persists across backend restarts, supports atomic pipelining, and could be shared across multiple backend instances in a scaled deployment.

---

## Frontend

### React 19 + TypeScript

**What it is:** UI library with static type checking.

**Why we use it:**
- Component-based architecture maps cleanly to our bento-grid dashboard tiles
- TypeScript catches type errors at compile time -- especially valuable for the complex event types flowing over WebSocket
- React's state management handles real-time data updates efficiently (new transactions, alert status changes, agent traces)
- Large ecosystem for charting, layout, and UI components

### Tailwind CSS

**What it is:** Utility-first CSS framework.

**Why we use it:**
- Rapid prototyping -- style directly in JSX without switching between files
- Consistent spacing, colors, and typography via design tokens
- Dark theme built with utility classes (our dashboard uses a dark command-center aesthetic)
- No CSS bloat -- only the utilities we use are included in the build

### Recharts

**What it is:** React charting library built on D3.

**Why we use it:**
- Declarative React components for charts (AreaChart, BarChart, PieChart)
- Real-time updates: just update the data array and the chart re-renders with smooth transitions
- Used for risk score distribution and account activity timeline
- Lightweight compared to full D3 -- no imperative DOM manipulation needed

---

## Notifications

### Slack Incoming Webhooks

**What it is:** HTTP webhooks that post messages to Slack channels.

**Why we use it:**
- Real-time alerts when the AI agent blocks a fraudulent transaction — posts to #fraud-alerts with transaction details and confidence score
- Notifies #human-reviews when an alert needs human attention (FLAG_FOR_REVIEW verdict)
- Sends a follow-up to #human-reviews when a human resolves a flagged alert
- Fire-and-forget via `asyncio.create_task` — never blocks the pipeline
- Optional: leave `WEBHOOK_URL` and `WEBHOOK_URL_HUMAN` empty in `.env` to disable

**Why not email/push?** Slack is the standard for ops teams. Webhooks are trivial to set up (api.slack.com/apps → Incoming Webhooks). Same format works for Discord, Teams, or any HTTP endpoint.

---

## Infrastructure

### Docker + Docker Compose

**What it is:** Container runtime and multi-container orchestration.

**Why we use it:**
- Single `docker compose up -d` starts all infrastructure (Redpanda, Neo4j, Redis)
- Reproducible environments -- works identically on any machine
- Named volumes for data persistence across restarts
- Service isolation: each component gets its own container with defined resource limits

### Capital One Nessie API (Optional)

**What it is:** Sandbox banking API that simulates real financial operations.

**Why we use it:**
- Provides realistic account, customer, and merchant data structures
- Creates actual purchase records against sandbox accounts
- Demonstrates integration with real banking infrastructure
- Optional: mock mode generates synthetic data when no API key is configured, so the system works out of the box

---

## Summary Table

| Technology | Category | Key Benefit |
|---|---|---|
| Redpanda | Streaming | Kafka compatibility, zero JVM, message durability |
| FastAPI | Backend | Async-native, WebSocket support, auto-docs |
| Pydantic | Validation | Type-safe schemas, environment config |
| aiokafka | Streaming | Async Kafka client, event loop integration |
| scikit-learn | ML | Unsupervised anomaly detection, no labeled data needed |
| FAISS | Vector Search | In-memory similarity search, adaptive learning |
| SentenceTransformers | Embeddings | Lightweight text-to-vector, CPU-only |
| LangChain | AI Agent | ReAct pattern, tool abstraction, streaming traces |
| Ollama + Llama 3 | LLM | Local inference, no API keys, data privacy |
| Neo4j | Graph DB | Fraud ring detection via graph traversal |
| Redis | Cache | Sub-ms velocity tracking, per-account stats |
| Slack Webhooks | Notifications | Real-time BLOCK/review/resolution alerts (optional) |
| React + TypeScript | Frontend | Component-based UI, type safety, real-time state |
| Tailwind CSS | Styling | Rapid prototyping, consistent design tokens |
| Recharts | Charts | Declarative React charts, smooth real-time updates |
| Docker Compose | Infrastructure | One-command setup, reproducible environments |
| Nessie API | Banking | Realistic sandbox data, optional integration |
