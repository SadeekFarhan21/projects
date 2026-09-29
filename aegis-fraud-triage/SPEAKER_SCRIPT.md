# Aegis - Speaker Script
### TartanHacks 2026 | 3 minutes total

---

## SLIDE 1: Title + Hook (10 seconds)

> "This is Aegis -- autonomous fraud detection powered by AI agents. Three parallel investigators, zero cloud API keys, everything runs locally. Let me show you how."

**[Arrow right]**

---

## SLIDE 2: How It Works (30 seconds)

> "Three stages. First -- ML flags suspicious transactions. Second -- an AI agent picks up the case and investigates autonomously. It checks travel feasibility, account history, merchant data, past cases.
>
> Look at this example -- four-thousand-dollar watch in Bucharest, account's in Ohio. Agent checks travel -- five thousand miles, last transaction two hours ago. Impossible. Finds three similar cases, all blocked by analysts. Verdict: block, ninety-four percent confidence.
>
> Third -- when uncertain, it asks a human via Slack, and every decision feeds back so the system learns."

**[Arrow right]**

---

## SLIDE 3: Architecture (20 seconds)

> "Here's the full stack. Transactions stream through Redpanda into an Isolation Forest for scoring. FAISS handles vector search for finding similar cases, Redis caches velocity stats, Neo4j maps fraud rings. The agents run Llama 3.2 locally through Ollama -- and the Adaptive Shield feeds every verdict back into the vector store so the system keeps learning. Slack webhooks notify analysts in real time, React dashboard ties it all together. One Docker Compose command to deploy."

**[Arrow right]**

---

## SLIDE 4: Demo Transition (5 seconds)

> "Let's see it live."

**[Switch to browser -- open the Aegis dashboard at localhost:3000]**

---

## LIVE DEMO (60 seconds)

### Setup (5 seconds)
> "This is the Aegis command center. You can see the live transaction feed, risk distribution, infrastructure health, and the alert panel."

### Trigger Demo Mode (10 seconds)
> "I'm going to start demo mode, which simulates realistic transaction traffic."
>
> **[Click "Start Demo" button in the dashboard]**
>
> "Watch the transaction feed -- these are streaming in through Redpanda in real time."

### Wait for Alert (15 seconds)
> "Now watch the alert panel on the right. When the ML model flags something suspicious..."
>
> **[Wait for a red anomaly to appear]**
>
> "There -- we just got a flagged transaction. You can see the risk score, the amount, the merchant."

### Show Agent Investigation (20 seconds)
> "Now look at the agent console at the bottom. The AI agent has picked up this case and is investigating it autonomously."
>
> **[Point to the agent console tabs]**
>
> "You can see its reasoning in real time -- it's checking travel feasibility, pulling account history, searching for similar past cases. Each step of the ReAct loop is visible. It's thinking, calling tools, observing results, and building toward a verdict."
>
> "And there's the verdict -- with a confidence score and the evidence it used."

### Show Human-in-the-Loop (10 seconds)
> "For cases where the agent is uncertain, it flags them for human review. An analyst can click here to approve or override the AI's decision, and that feedback loops right back into the system -- making the next investigation smarter."

### Wrap Up
> "That's Aegis. Fully local, fully autonomous, and always learning. Thank you."

---

## Q&A PREP (Common Questions)

**Why local instead of cloud LLMs?**
> "Financial transaction data is extremely sensitive. Running Llama 3 locally via Ollama means zero data leaves the network. No API keys, no third-party data processing agreements, no latency to external servers."

**How does the adaptive learning work?**
> "Every investigation verdict -- both AI and human -- gets converted to a vector embedding and stored in FAISS. When a new case comes in, the agent can search for similar past investigations to inform its decision. So if a human analyst blocks a pattern three times, the AI learns to block it automatically."

**Why Redpanda over Kafka?**
> "Redpanda is API-compatible with Kafka but written in C++. It's faster, uses less memory, and doesn't need a JVM or ZooKeeper. Perfect for a hackathon where we need to move fast."

**How do the three parallel workers help?**
> "Multiple suspicious transactions can fire simultaneously. Instead of queuing them, three independent agent workers investigate in parallel. Each has its own LLM context and tool access, so throughput scales linearly."

**What's the false positive rate?**
> "The Isolation Forest provides the initial statistical filter, but the real false-positive reduction comes from the AI agent layer. It has contextual reasoning -- it can check if travel is feasible, if the merchant is known, if similar cases were cleared before. That multi-step investigation dramatically reduces false positives compared to a threshold-based system."

**Why Neo4j?**
> "Fraud is inherently a graph problem. Neo4j lets us model relationships between accounts, merchants, devices, and locations. The agent can traverse these connections to find patterns that a flat database would miss."
