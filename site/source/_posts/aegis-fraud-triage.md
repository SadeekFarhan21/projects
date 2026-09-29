---
layout: post
title: "Fraud Triage That Spares the LLM 97% of Transactions"
code: https://github.com/SadeekFarhan21/projects/tree/main/aegis-fraud-triage
date: 2026-02-06 18:44:45
tags:
  - fraud-detection
  - anomaly-detection
  - llm-agents
  - kafka
description: >-
  A 20-hour TartanHacks pipeline that scores every card transaction with an Isolation Forest and rules, then hands only 2.9 to 3.2 percent of synthetic traffic to a local LLM agent, with precision 1.000 in all 6 runs.
---

A bank's fraud desk can't read every card transaction by hand, and running an LLM on every transaction is too slow and too expensive. Aegis is the funnel I built for that problem at TartanHacks, CMU's hackathon, in February 2026, with my teammate Jalen Francis. Transactions stream through a Redpanda (Kafka) topic. A cheap gate, a scikit-learn Isolation Forest<sup>[[1]](#ref-1)</sup> plus hand-written rules, scores each one. Anything above 0.55 becomes an alert, and a pool of three asyncio workers hands each alert to a ReAct-style agent<sup>[[2]](#ref-2)</sup> on a local Ollama model, which answers BLOCK, CLEAR or FLAG_FOR_REVIEW. Flagged alerts wait for a human, and every verdict goes into a FAISS index<sup>[[3]](#ref-3)</sup> that later investigations can search.

The whole design rests on the gate, so after the hackathon I measured it on 5,000 synthetic transactions per run over 6 runs. It sent **2.9 to 3.2 percent** of traffic to the agent with **precision 1.000 in every run**: not one normal transaction reached the LLM. Recall was 0.368 to 0.403, and it splits cleanly by amount: in a 3,000-transaction run **every fraud above $1,000 was caught**, so the gate's recall is the share of fraud above that line.

## Why It Matters

The hackathon theme was to make a bank's fraud desk faster. The shape that fits is a funnel: something fast and cheap looks at everything, and something slow and smart looks only at what the fast stage flags. The whole system lives or dies on that first stage. Here is what Jalen and I built in roughly 20 hours. I wrote the backend and the React dashboard, and Jalen did the design, the demo and the slides.

- A stream of card transactions arriving through Kafka, the way a bank's would.
- A fast first stage that scores every transaction and lets most of them through.
- A slow second stage, an LLM agent with tools, that investigates only what the first stage flags and returns a verdict with a written reason.
- A human review queue for the verdicts the agent isn't sure about, with Slack notifications.
- A memory, so that a human decision today informs a similar investigation tomorrow.
- A live dashboard over WebSockets.
- A separate dispute-resolution agent, a Neo4j fraud-ring graph and a Redis cache as extras.

## Technical Details

### Isolation Forest as a First Stage

An Isolation Forest<sup>[[1]](#ref-1)</sup> builds random trees that split a randomly chosen feature at a random value. A point far from the rest gets separated in only a few splits, so its average path length is short, and the model turns that into an anomaly score. It needs no labels, which suits fraud, where labels arrive late. In scikit-learn, `decision_function` returns a negative value for points the model considers outliers and a positive value for inliers<sup>[[4]](#ref-4)</sup>. Aegis maps that raw value onto a 0 to 1 risk score.

### Gating an Expensive Stage

If the second stage costs seconds and the first costs milliseconds, the first stage decides how much work the second sees. Two numbers matter. Precision at the gate sets how much of the LLM's time goes to innocent transactions. Recall at the gate caps the whole system, because a fraud the gate lets through is never investigated by anything downstream. Those are the two numbers I measured.

### ReAct Without Native Tool Calling

ReAct<sup>[[2]](#ref-2)</sup> interleaves reasoning text with tool calls. The model writes a thought and names an action, and the harness runs the action and appends the observation. The prompt-based form has the model emit `Thought`, `Action` and `Action Input` as plain text, which the harness parses. That form is easy to build and sensitive to how well a small model follows a format.

### Retrieval as Memory

A sentence embedding maps text to a vector where similar meanings sit close together. FAISS<sup>[[3]](#ref-3)</sup> stores those vectors and returns nearest neighbors. With L2-normalized vectors, inner product equals cosine similarity, so an `IndexFlatIP` gives exact cosine top-k. Storing past verdicts as text and retrieving the nearest ones into the agent's prompt gives the agent a memory of what was decided before. It's retrieval, not learning: nothing about the model or the gate changes.

### Architecture

The producer, consumer and workers all run as background asyncio tasks inside one FastAPI process. Redpanda v24.1.1, Neo4j 5 and Redis 7 come up through docker-compose, and Ollama runs on the host.

<figure class="excal" data-diagram="aegis-fraud-triage-architecture"><a href="/img/diagrams/aegis-fraud-triage-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/aegis-fraud-triage-architecture.webp" alt="Aegis pipeline: a mock producer publishes to a Redpanda topic, a consumer and gate scores each transaction, those above 0.55 wait in an asyncio queue read by three ReAct agent workers on Ollama llama3.2:3b (up to 6 steps, 9 tools), BLOCK or CLEAR is applied at once while FLAG_FOR_REVIEW goes to a human queue with a Slack webhook, and verdicts are stored in an in-memory FAISS index that is retrieved into the agent's prompt." width="2400" height="1870" loading="lazy" decoding="async"></a></figure>

A transaction goes producer, topic, consumer, gate. Most stop there. About 3 percent cross 0.55 and go onto a plain `asyncio.Queue`, where three identical workers pull them and run the agent. The verdict either applies immediately (BLOCK and CLEAR) or waits for a human (FLAG_FOR_REVIEW). Both agent and human verdicts are embedded into FAISS, and the agent has a tool that queries it.

## Implementation

### The Features and the Forest

The forest trains at startup on 1,000 synthetic "normal" rows. There are 8 features: amount, log of amount, hour, day of week, whether the country isn't `US`, velocity (transactions in the last 10 minutes), distance in kilometers from the account's last transaction, and the account's amount z-score. Velocity, distance and z-score come from an in-process per-account tracker that keeps the last 100 transactions per account.

```python
self.model = IsolationForest(
    n_estimators=100,
    contamination=contamination,   # 0.05
    random_state=42,
)
```

### From Raw Score to Flag

This is a simplified version of `detect()`, not verbatim. The forest's output is scaled to 0 to 1 with a small Gaussian jitter, and then rules for foreign merchants, large amounts and risky categories raise the score.

```python
raw_score = self.model.decision_function(features)[0]
normalized_score = max(0.0, min(1.0, -raw_score * 2))
noise = np.random.normal(0, 0.06)
normalized_score = max(0.0, min(1.0, normalized_score + noise))

if country != "US" and amount > 1000:
    normalized_score = max(normalized_score, 0.55 + np.random.uniform(0.05, 0.25))
elif country != "US":
    normalized_score = max(normalized_score, 0.25 + np.random.uniform(0, 0.15))
if amount > 5000:
    normalized_score = max(normalized_score, 0.50 + np.random.uniform(0.05, 0.30))
if category in ("Unknown", "Financial", "Luxury", "Gambling"):
    normalized_score = min(1.0, normalized_score + np.random.uniform(0.03, 0.12))

is_anomaly = normalized_score > 0.55
```

A non-US transaction above $1,000 always lands between 0.60 and 0.80, so it always becomes an alert. Anything above $5,000 gets a floor of 0.55 to 0.80. A non-US transaction under $1,000 gets a floor of at most 0.40.

### The FAISS Novelty Boost

The consumer runs a second check on transactions the gate didn't flag. It's aimed at a small but odd purchase for one specific account, the case the amount and geography rules can't see.

```python
novelty = vector_store.compute_novelty_score(txn_data)
if novelty is not None and novelty > 0.45:
    boost = (novelty - 0.45) * 0.64
    txn.risk_score = min(1.0, txn.risk_score + boost)
    if txn.risk_score > 0.55:
        txn.is_anomaly = True
```

Novelty is `1 - (0.6 * best_sim + 0.4 * avg_sim)` over the same account's past transactions, and it needs at least 5 prior transactions for that account. Each transaction is embedded as a sentence (amount bucket, category, merchant, city, time-of-day bucket, risk score) with all-MiniLM-L6-v2<sup>[[5]](#ref-5)</sup>, 384 dimensions, in an `IndexFlatIP` on normalized vectors.

### The Queue and the Workers

Flagged transactions go onto a bare `asyncio.Queue()`, and the app starts three identical worker tasks.

```python
tasks = [
    asyncio.create_task(run_producer(producer)),
    asyncio.create_task(run_consumer()),
    asyncio.create_task(run_agent_worker()),
    asyncio.create_task(run_agent_worker()),
    asyncio.create_task(run_agent_worker()),
    ...
]
```

The three workers cap how many LLM investigations run at once, and the queue absorbs bursts.

### The Agent Loop

The agent is a manual ReAct loop over `ChatOllama` at temperature 0.1, with up to 6 steps and a `TOOL_MAP` of 9 tools: `check_account_history`, `verify_merchant`, `check_travel_feasibility`, `get_account_risk_profile`, `run_kyc_check`, `find_similar_transactions`, `find_similar_investigations`, `detect_fraud_ring` and the final `recommend_action`. The default model is llama3.2:3b. Output is parsed with regexes such as `Action:\s*(\w+)` and `Action Input:\s*({[^}]+})`, with fallbacks for malformed output.

### Human Review and Memory

A `flagged` verdict sets the alert to `awaiting_review` and posts a Slack webhook. `blocked` and `cleared` apply immediately. A human decision goes through a review endpoint and is stored in FAISS with `verdict_source='human'`. Later, the `find_similar_investigations` tool retrieves the nearest past verdicts into the agent's prompt, so a reviewer's call on one case shows up as context on the next similar one.

## Problems

### 1. Tool Use on a Small Local Model

The agent runs on a local Ollama model, and llama3:8b, the model I started with, had no native tool calling through Ollama. **So I stopped asking for structured tool calls and parsed plain text instead.** The prompt asks for `Thought`, `Action` and `Action Input` lines, regexes pull out the tool name and its JSON arguments, and fallbacks handle output that doesn't match. A low temperature (0.1) and a hard limit of 6 steps bound how long any investigation runs, and `recommend_action` is itself a tool, so ending the investigation is just another action the model names.

### 2. Keeping the LLM Off Most Traffic

Transactions never stop arriving, and each investigation is up to 6 rounds of LLM reasoning and tool calls, so the agent can't look at everything. **The gate is what makes the agent affordable: one scikit-learn call per transaction, 149 to 153 transactions per second on one thread, and only about 3 percent passed on.** Three workers bound how many investigations run at once. The producer, consumer and workers all run as asyncio tasks in one FastAPI process, so investigations overlap without any extra infrastructure.

### 3. Memory Without Retraining

I wanted a reviewer's decision today to inform a similar investigation tomorrow. **I made memory a retrieval problem instead of a training one.** Every verdict, the agent's or a human's, is embedded as a sentence and stored in FAISS with its `verdict_source`, and `find_similar_investigations` pulls the nearest ones into the agent's prompt. Nothing about the model or the gate changes, so there is no retraining step between a reviewer's call and the next investigation that can use it.

## Experiments

I measured the first stage on its own, because it decides what everything downstream sees. The harness runs the gate's scoring engine on a stream that mirrors the producer's mix: about 8 percent fraud (foreign merchants, amounts from the producer's `_get_anomaly_amount`), 12 percent elevated normal (domestic, 2 to 5 times the category amount) and the rest normal. Each run is 5,000 transactions, over 3 seeds and 2 simulated clock start times (12:00Z and 03:00Z), with numpy and Python seeds set per seed.

I recorded precision, recall, the share of traffic flagged and throughput. A second script, run on a 3,000-transaction stream, breaks recall down by fraud amount. Everything ran on 2026-09-29 with Python 3.14.7, scikit-learn 1.9.1 and numpy 2.5.3, single-threaded on my Mac. The traffic and its labels are synthetic, so the numbers describe the gate against the pipeline's own model of fraud, not against real fraud.

## Results

### The Gate

**The full gate had precision 1.000 in all 6 runs**, with 0 false positives among 3,959 to 4,011 normal and 600 to 633 elevated normal transactions per run. Recall was 0.368, 0.375 and 0.398 for the noon runs and 0.373, 0.378 and 0.403 for the 03:00 runs. Between 146 and 161 of 5,000 transactions were flagged, which is **2.9 to 3.2 percent of traffic**. Hour of day made almost no difference, within about 0.005 recall.

### Recall Tracks the $1,000 Line

The gate behaves close to "non-US merchant and amount above $1,000, or amount above $5,000". In the 3,000-transaction run, **all 76 fraud transactions between $1,000 and $5,000 and all 8 above $5,000 were flagged**. The 78 under $500 and the 71 between $500 and $1,000 went unflagged. **The 36.1 percent of injected fraud above $1,000 matched the recall (0.361) in that run**, so the recall number is a direct read of where the fraud amounts sit.

<figure data-figure="chart:projects/aegis-fraud-triage/aegis-fraud-triage-recall-by-amount"></figure>

### Throughput and Load on the Agent

**The gate ran at about 149 to 153 transactions per second** on one thread, with single-row scikit-learn calls. At the producer's default of one transaction every 1 to 3 seconds, it's nowhere near a bottleneck. As arithmetic, not a measurement: at the default mean of 0.5 transactions per second and about 3 percent flagged, roughly 0.015 alerts arrive per second, about one every 67 seconds, against 3 workers. Under default settings the queue should rarely build, and a backlog would come from bursts such as the demo's fraud injection.

## What I Would Change

### Label a Small Set With Subtle Fraud

The measurements above use my own fraud generator, so they measure agreement with the pipeline's own model of fraud. A labeled set with small, subtle fraud would turn the sub-$1,000 band into a number someone is trying to raise, and it would give the gate's rules and the novelty check a real target to tune against.

## References

1. <span id="ref-1"></span>Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou. *Isolation Forest*. IEEE International Conference on Data Mining (ICDM), 2008. [doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
2. <span id="ref-2"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu et al. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR, 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
3. <span id="ref-3"></span>Jeff Johnson, Matthijs Douze and Hervé Jégou. *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data 7(3), 2021. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
4. <span id="ref-4"></span>Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research 12, 2011. [link](https://jmlr.org/papers/v12/pedregosa11a.html)
5. <span id="ref-5"></span>Wenhui Wang, Furu Wei, Li Dong et al. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS, 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)
