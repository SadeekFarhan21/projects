# Aegis - Autonomous Financial Fraud Detection Platform

Real-time fraud detection combining event streaming, ML anomaly detection, autonomous AI agent investigation, and human-in-the-loop review. Built for CMU Hackathon 2026.

## What It Does

Aegis processes a continuous stream of financial transactions and:

1. **Detects anomalies** using an Isolation Forest ML model (8 features including geographic velocity, spending z-scores, and transaction patterns)
2. **Investigates alerts** with autonomous AI agents (Llama 3 via Ollama) running 3 parallel workers, each using a ReAct reasoning loop with 8 tools
3. **Routes uncertain cases** to human analysts for review (Human-in-the-Loop) with Slack notifications
4. **Resolves customer disputes** with a separate AI dispute agent that evaluates claims against transaction evidence
5. **Learns from decisions** by storing human and agent verdicts in a FAISS vector store, so future investigations reference past outcomes (Adaptive Shield)
6. **Notifies via Slack** with real-time webhooks for blocks, review requests, and resolutions
7. **Streams everything** to a real-time dashboard via WebSockets

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| **Docker** + Docker Compose | Any recent | Runs Redpanda (Kafka), Neo4j, Redis |
| **Python** | 3.11+ | Backend server |
| **Node.js** + npm | 18+ | Frontend dashboard |
| **Ollama** | Any recent | Local LLM inference (install from [ollama.ai](https://ollama.ai)) |

**Hardware:** 8 GB+ RAM recommended (Ollama + Redpanda + the ML models run concurrently).

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd cmu-hackathon

# 2. Copy environment config
cp .env.example .env

# 3. Start infrastructure (Redpanda, Neo4j, Redis)
docker compose up -d

# 4. Pull the LLM model
ollama pull llama3.2:3b

# 5. Install and start the backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ..
python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000

# 6. Install and start the frontend (new terminal)
cd frontend
npm install
npm start

# 7. Open the app
open http://localhost:3000        # Landing page
open http://localhost:3000/demo   # Dashboard
```

Verify everything is running:
```bash
curl http://localhost:8000/api/health
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, [TECH_STACK.md](TECH_STACK.md) for a breakdown of every technology with justifications, and [PRESENTATION.md](PRESENTATION.md) for the Figma AI diagram prompt.

```
Nessie API / Mock ──> Redpanda ──> Anomaly Engine (IsolationForest + FAISS)
                                          │
                                   AI Agent (Llama 3, ReAct loop, 3 parallel workers)
                                          │
                                   ┌──────┴──────┐
                                   │  Verdict     │
                                   ├──────────────┤
                                   │ BLOCK/CLEAR  │──> Applied immediately ──> Slack #fraud-alerts
                                   │ FLAG_REVIEW  │──> Human-in-the-Loop  ──> Slack #human-reviews
                                   └──────────────┘
                                          │
                                   FastAPI + WebSocket ──> React Dashboard
                                          │
                                   Customer Disputes ──> Dispute Agent ──> APPROVE/DENY/ESCALATE
```

## Tech Stack

| Layer | Technology |
|---|---|
| **Streaming** | Redpanda (Kafka-compatible, zero-JVM) + aiokafka |
| **ML** | scikit-learn Isolation Forest, FAISS + SentenceTransformers |
| **AI Agent** | LangChain + Ollama / Llama 3 (ReAct pattern, 8 tools, 3 parallel workers) |
| **Backend** | FastAPI, WebSockets, Pydantic |
| **Frontend** | React 19, TypeScript, Tailwind CSS, Recharts, GSAP, Three.js, Framer Motion |
| **Graph** | Neo4j (fraud ring detection) |
| **Cache** | Redis (transaction velocity tracking) |
| **Banking API** | Capital One Nessie (optional) |
| **Notifications** | Slack Incoming Webhooks (optional) |

## Key Features

### Anomaly Detection
- Isolation Forest trained on 8 features: amount, log-amount, hour, day-of-week, international flag, velocity, distance from last transaction, per-account z-score
- FAISS novelty scoring catches account-specific anomalies the global model misses
- Rule-based boosting for high-risk categories and international transactions

### AI Agent Investigation
- ReAct reasoning loop with up to 6 steps per investigation
- 8 tools: account history, merchant verification, travel feasibility, risk profile, KYC check, similar transactions (FAISS), similar investigations (FAISS), recommend action
- 3 concurrent agent workers for parallel investigations
- Confidence scores on every verdict (percentage-based)
- Verdicts: BLOCK, CLEAR, or FLAG_FOR_REVIEW
- Tab-based Agent Console in the dashboard isolates parallel investigations (no interleaving)

### Customer Dispute Resolution
- Separate AI dispute agent handles customer claims (unauthorized charge, never received, duplicate, etc.)
- 6 dispute tools: dispute context, account history, merchant verification, customer dispute history, duplicate transaction search, resolve
- Dedicated duplicate charge detection: searches transaction buffer for same merchant + same amount on the account
- Outcomes: APPROVE (refund), DENY (reject), or ESCALATE (human review)
- Clean summary extraction from AI reasoning

### Human-in-the-Loop (HITL)
- When the agent is unsure, alerts enter "Awaiting Review" status
- Slack notification sent to #human-reviews when an alert needs attention
- Human analysts see the full evidence trail and choose to Block or Clear
- Follow-up Slack notification when the alert is resolved
- Decision is stored in FAISS so future agents learn from human corrections

### Slack Notifications
- AI BLOCK verdicts post to #fraud-alerts with transaction details and confidence score
- FLAG_FOR_REVIEW alerts post to #human-reviews requesting human attention
- Human resolutions post a follow-up to #human-reviews with the final decision
- Optional: leave webhook URLs empty to disable

### Adaptive Shield
- Every verdict (agent or human) is embedded and stored in FAISS
- Future investigations retrieve similar past cases and their outcomes
- Human verdicts are tagged with `verdict_source: "human"` for higher weight
- The system adapts without retraining any models

### Landing Page
- Animated pixel mosaic hero with GSAP scroll-triggered sections
- 3D interactive globe (Three.js / React Three Fiber) with city connections
- Feature cards, buzzword marquee, CTA section
- "Try the Demo" / "Schedule Demo" buttons navigate to the live dashboard

### Real-Time Dashboard
- Live transaction feed, alert panel with expandable evidence, dispute resolution panel
- Tab-based Agent Brain Trace console: each parallel investigation gets its own tab with live step-by-step reasoning
- WebSocket streaming with automatic reconnection
- Confidence score badges on agent verdicts
- KYC identity risk checks on demand
- START/STOP DEMO button for one-click demo mode
- Fullscreen mode and infrastructure status monitoring

## Remote Backend (ngrok)

The frontend and backend can run on separate machines. The frontend connects to whatever backend URL is set in `frontend/.env`.

**On the backend machine (Mac):**

```bash
# 1. Install ngrok (one-time)
brew install ngrok

# 2. Sign up at https://ngrok.com and add your auth token (one-time)
ngrok config add-authtoken <your-token>

# 3. Start the backend normally
cd backend && source .venv/bin/activate && cd ..
python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000

# 4. In another terminal, expose it via ngrok
ngrok http 8000
```

ngrok will print a public URL like `https://abc123.ngrok-free.app`.

**On the frontend machine:**

Edit `frontend/.env` and set both URLs to the ngrok address:

```env
REACT_APP_API_URL=https://abc123.ngrok-free.app
REACT_APP_WS_URL=wss://abc123.ngrok-free.app/ws/feed
```

Then restart the frontend (`npm start`). Note: use `wss://` (not `ws://`) since ngrok provides HTTPS.

## Demo Mode

Click **START DEMO** in the dashboard header to automatically generate transactions, fraud scenarios, and disputes at a steady pace. Click **STOP DEMO** to pause.

Or use the API directly:

```bash
# Start/stop continuous demo
curl -X POST "http://localhost:8000/api/demo/start"
curl -X POST "http://localhost:8000/api/demo/stop"
curl "http://localhost:8000/api/demo/status"

# Inject specific fraud scenarios
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=0"  # Luxury watches in Bucharest
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=1"  # Gold exchange in Lagos
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=2"  # Crypto ATM in Moscow
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=3"  # Online casino in Malta
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=4"  # Wire services in Cayman Islands
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=5"  # Shell corp in Panama
curl -X POST "http://localhost:8000/api/demo/inject-fraud?scenario=6"  # Unknown vendor in Oman

# Inject a customer dispute
curl -X POST "http://localhost:8000/api/demo/inject-dispute"
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service health and connectivity status |
| GET | `/api/transactions` | Recent transactions |
| GET | `/api/alerts` | Active fraud alerts |
| GET | `/api/disputes` | Customer disputes |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/kyc/{account_id}` | KYC identity risk assessment |
| POST | `/api/alerts/{id}/review` | Human review: block or clear an alert |
| POST | `/api/demo/start` | Start continuous demo mode |
| POST | `/api/demo/stop` | Stop demo mode |
| GET | `/api/demo/status` | Check if demo is running |
| POST | `/api/demo/inject-fraud` | Inject a specific fraud scenario |
| POST | `/api/demo/inject-dispute` | Inject a customer dispute |
| GET | `/api/graph/network` | Account-merchant network graph |
| WS | `/ws/feed` | Real-time WebSocket event stream |

## Troubleshooting

| Issue | Fix |
|---|---|
| Dashboard shows DISCONNECTED | Restart backend: `python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000` |
| No transactions flowing | Check Redpanda: `docker compose ps` |
| Agent shows "model not found" | Run `ollama pull llama3.2:3b` |
| Agent is slow on first call | Normal -- model loads on first invocation. Subsequent calls are faster. |
| Want higher quality agent | Use `OLLAMA_MODEL=llama3:8b` in `.env` (requires more RAM) |
| Neo4j/Redis not connecting | These are optional. System runs without them. Check `docker compose ps`. |

## Project Structure

```
cmu-hackathon/
├── backend/
│   ├── agent/                 # AI investigation agents (fraud + dispute)
│   │   ├── investigator.py    #   Fraud investigation ReAct agent
│   │   ├── dispute_agent.py   #   Customer dispute ReAct agent
│   │   ├── tools.py           #   8 fraud investigation tools
│   │   ├── dispute_tools.py   #   6 dispute investigation tools
│   │   └── prompts.py         #   Agent system prompts
│   ├── anomaly_detection/     # Isolation Forest ML engine
│   ├── api/                   # FastAPI server, WebSocket, models
│   ├── cache/                 # Redis velocity/stats cache
│   ├── config.py              # Central configuration (pydantic-settings)
│   ├── graph/                 # Neo4j fraud ring detection
│   ├── kyc/                   # KYC identity risk engine
│   ├── nessie_client/         # Capital One Nessie API client
│   ├── streaming/             # Kafka producer + consumer
│   ├── vector_store/          # FAISS similarity search
│   └── requirements.txt
├── frontend/                  # React 19 + TypeScript + Tailwind CSS
│   └── src/
│       ├── landing/               # Landing page (ported from aegis-tartanhacks)
│       │   ├── LandingPage.tsx    #   Main landing page component
│       │   ├── components/        #   Hero, features, globe, CTA, footer
│       │   ├── styles/            #   Scoped CSS variables
│       │   ├── ui/                #   shadcn button & separator
│       │   └── lib/               #   cn() utility
│       ├── components/tiles/      # 7 dashboard tiles
│       ├── hooks/                 # WebSocket hook
│       ├── types/                 # Event type definitions
│       ├── utils/                 # Formatters
│       ├── App.tsx                # Router: / = landing, /demo = dashboard
│       └── DashboardApp.tsx       # Dashboard app (original App)
├── docker-compose.yml         # Infrastructure (Redpanda, Neo4j, Redis)
├── ARCHITECTURE.md            # Detailed system architecture
├── TECH_STACK.md              # Technology justifications
├── PRESENTATION.md            # Figma AI diagram prompt for slides
└── .env.example               # Environment variable template
```
