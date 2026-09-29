# Aegis - Presentation Assets

## Figma AI Architecture Diagram Prompt

Copy-paste this into Figma's AI design generator (or FigJam AI):

```
Create a system architecture diagram for a financial fraud detection platform called "Aegis". 

Left to right data flow:

DATA SOURCES (left side):
- Two boxes: "Nessie Banking API" and "Mock Transaction Generator"
- Both feed into a central streaming layer

STREAMING LAYER:
- One box: "Redpanda (Kafka)" - event streaming, topic: transactions

PROCESSING PIPELINE (center):
- Box: "Consumer" receives from Redpanda
- Arrow down to "Anomaly Engine" (Isolation Forest ML, 8 features)
- Three parallel supporting boxes: "FAISS Vector Store", "Redis Cache", "Neo4j Graph DB"
- Arrows from Anomaly Engine to these three

ALERT & AI LAYER:
- If anomaly detected → "Alert Created"
- Arrow to "AI Agent" (Llama 3, ReAct loop, 8 tools, confidence scores)
- Three verdict paths from AI Agent: 
  - "BLOCK" → auto-apply → Slack #fraud-alerts (webhook)
  - "CLEAR" → auto-apply
  - "FLAG FOR REVIEW" → Slack #human-reviews (webhook)

HUMAN-IN-THE-LOOP:
- FLAG FOR REVIEW → "Human Review Panel" → "Block" or "Clear" decision
- Arrow to "Slack #human-reviews" (resolution webhook)
- Arrow back to FAISS (Adaptive Shield - learns from human decisions)

PRESENTATION (right side):
- Box: "FastAPI + WebSocket"
- Arrow to "React Dashboard" with 7 tiles: Stats, Transaction Feed, Alerts, Agent Console, Disputes, Risk Chart, Account Activity

Use boxes and arrows. Label each component clearly. Use a dark or professional color scheme. Title at top: "Aegis - Autonomous Fraud Detection Architecture"
```

---

### Shorter variant (for FigJam / diagram tools that prefer brevity)

```
System architecture: Nessie API and Mock Generator → Redpanda Kafka → Consumer → Anomaly Engine (Isolation Forest) with FAISS, Redis, Neo4j → Alert → AI Agent (Llama 3, confidence scores) → BLOCK (Slack #fraud-alerts) / CLEAR / FLAG (Slack #human-reviews) → Human Review → resolution webhook to Slack → FastAPI WebSocket → React Dashboard
```
