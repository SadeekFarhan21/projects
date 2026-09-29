#!/usr/bin/env python3
"""Embed Excalidraw renders in the posts.

For each diagram in ALT below:
  diagrams/png/<name>.png  (4-5k px wide render, kept out of source/ so it is not deployed)
  -> source/img/diagrams/<name>.webp  (2400 px wide, what the posts load)
and every <figure data-figure="diagram:<name>"></figure> in source/_posts is
replaced with an <img> figure that opens full size on click.

Re-render a diagram after editing site/diagrams/<name>.excalidraw with the
site-local renderer, which swaps in the site's fonts (see its template):
  <excalidraw skill venv>/bin/python tools/excalidraw/render_excalidraw.py \
    diagrams/<name>.excalidraw --output diagrams/png/<name>.png --scale 2
then run this script again. In the .excalidraw files, fontFamily 2 marks text
(rendered in Quicksand) and fontFamily 3 marks code (rendered in Fragment Mono).
"""
import pathlib
import re
import subprocess
import sys

SITE = pathlib.Path(__file__).resolve().parent.parent
PNG = SITE / "diagrams" / "png"
IMG = SITE / "source" / "img" / "diagrams"
POSTS = SITE / "source" / "_posts"
# Drafts get the same swap, so a post reads the same before and after publishing.
DRAFTS = SITE / "source" / "_drafts"
WIDTH = 2400

ALT = {
    "backtester-pipeline": "Flow of the qrp backtester from the Binance archive through a point-in-time store and wide arrays to a look-ahead audit that stops leaky runs, then purged cross-validation, ridge, three matching backtest kernels, metrics and the run tracker.",
    "crypto-pipeline": "Crypto reversal pipeline from the Binance archive to one forecast that splits into a rank IC path, where H1 is supported, and a dollar backtest path, where H2 is not supported after costs, above a timeline of the development period and the one-time holdout.",
    "peeking-pipeline": "Two independent pipelines that meet only at results: the A/B side reduces event tables to per-arm sufficient statistics for Welch, CUPED and mSPRT tests; the off-policy side turns bandit logs and a cross-fitted reward model into per-round terms for IPS, DR and bootstrap intervals.",
    "sv39-translation": "Sv39 address translation: a 39-bit virtual address split into three 9-bit table indices and a 12-bit page offset, walked through the L2, L1 and L0 page tables to a leaf entry with its physical page number and permission bits.",
    "riscv-memory-map": "Physical memory map after a 128 MiB boot: the test device, PLIC and UART, then OpenSBI, the kernel's text, rodata and data sections with their permissions, guarded boot stacks, the page bitmap, free frames and the device tree blob.",
    "riscv-trap-path": "The kernel's single trap path: kernelvec saves a frame on the interrupted thread's stack, kernel_trap dispatches timer and external interrupts, probe recoveries and fatal exceptions, and kernelvec restores and returns with sret.",
    "riscv-thread-states": "Kernel thread state machine: UNUSED to RUNNABLE on thread_create, RUNNABLE and RUNNING via the scheduler and yield or preemption, RUNNING to SLEEPING and back to RUNNABLE on wakeup, and RUNNING to ZOMBIE to UNUSED on exit and join.",
    "kvdb-write-and-recovery": "kvdb write and recovery paths as a sequence over three files, P-wal, P-journal and P, with the commit point after Wal::append and sync and the atomicity point after the journal header sync marked across all files.",
    "kvdb-slotted-page": "Layout of a 4096-byte kvdb slotted page: a 16-byte header, 2-byte slot offsets growing forward, free space, and cells of key length, value length, key and value growing backward from the end.",
    "redis-event-loop": "The single-threaded event loop of the Redis clone: commands fill an AOF buffer and per-client reply buffers, and before_sleep runs Aof::flush before try_write on every iteration, so no reply leaves before its write is durable.",
    "matching-engine": "Matching engine architecture: an input log feeds the engine's validate, match and rest or cancel steps over its id map, order pool and two book sides; output events tagged with the input sequence number fan out to the output log, market data feed and digest, while a reference matcher must produce identical outputs.",
    "order-book-levels": "Order book layout: bid and ask sides as vectors of price levels sorted worst to best with back() as the best price, and one level's FIFO of order pool slots 17, 4 and 52, doubly linked by slot index.",
    "market-sim": "Market simulator architecture: a scheduler event heap drives the fundamental and the trader agents, which trade through the market and its order book; the market maker receives fills and requotes; the hidden fundamental value is read only by informed traders and analysis stamps.",
    "smolgrad-layers": "smolgrad package layers: user code calls nn, optim and gradcheck; nn goes through functional into the Tensor class, whose forward ops record a closure and whose backward sweeps them in reverse, while optim reads grad and updates data in place outside the graph; numpy sits underneath.",
    "smolgrad-step": "One smolgrad training step: the input flows through Linear, relu, Linear and cross_entropy to a scalar loss, backward runs the recorded closures in reverse to fill the leaf gradients, step updates the weights in place, and zero_grad clears the gradients for the next batch.",
    "kafka-sparse-index": "A sparse index lookup: index entries point to byte positions in the log, and fetching offset 6 binary searches to the floor entry at record 4, skips the headers of records 4 and 5, and returns records from byte 6000.",
    "kafka-broker": "Kafka-style broker architecture: producer and consumer connections each get a broker thread that dispatches to the topics map, group coordinator, offset store and long-poll wakeups; data and committed offsets end up as segment .log and .index files on disk.",
    "gpt-architecture": "GPT forward pass as one residual stream: token ids through a 1,000 × 256 token embedding plus a learned 64 × 256 position embedding give x of shape (8, 64, 256), which runs down a spine through six pre-LN blocks, where LayerNorm then 8-head causal self-attention and LayerNorm then a 256 to 1,024 to 256 ReLU MLP each branch off and add back; the spine then goes through a final LayerNorm and a 256 to 1,000 LM head, and the (8, 64, 1,000) logits meet the shifted targets in cross-entropy to give a scalar loss.",
    "gpt-attention": "Inside one SelfAttention head, x passes through self.query, self.key and self.value; queries @ keys.transpose(-2, -1) is scaled by head_size ** -0.5, positions where the lower-triangular self.mask is 0 are set to -inf with masked_fill, F.softmax turns the scores into weights, and weights @ values gives the head's output; MultiHeadAttention runs x through heads[0] through heads[7] (the default 8 heads), joins the results with torch.cat(dim=-1), and passes them through self.proj.",
    "gpt-pipeline": "The pipeline runs top to bottom: TinyStories text trains a 1,000-token byte-level BPE tokenizer.json, and tokenize_file() streams the text into a uint16 token cache that is rebuilt only when it is older than the text or the tokenizer. LanguageModelDataset memory-maps that cache and feeds (x, y) batches to the train.py loop (GPT forward, cross-entropy, backward, AdamW step), which writes gpt-tinystories.pt every 1,000 steps for generate.py and chat.py to load. Separately, launch_modal.py spawns modal_app.py, which runs the same train.py on an L4 GPU with a prebuilt token cache and saves its own checkpoint to the Modal volume every 500 steps.",
    "induction-two-head-mechanism": "Token strip '... A B ... A' where a previous-token head (typically layer 0) writes 'previous token = A' into the residual stream at B's position, and an induction head uses QK (W_Q W_K^T) to match the final A against that position and OV (W_O W_V) to copy B into the output, raising its logit; QK is bracketed as what the induction score measures, OV as not measured here (experiment 05), and the composition edge as assumed and tested by path patching in experiment 04.",
    "induction-score-setup": "Diagram of the induction-score setup: a [BOS][50 random][same 50] token strip above a 101x101 causal attention matrix with the single offset -49 diagonal (query 1+N+j to key 2+j) highlighted, the two first-half cells excluded, and the pipeline from one cached forward pass to 144 head scores.",
    "rlvr-pipeline": "GRPO pipeline on one Modal L4 GPU with no vLLM: prompts from the data.py loaders and a frozen Qwen2.5-0.5B-Instruct with LoRA (r=32, alpha=64) feed a rollout of 8 completions per prompt; correctness_reward (last boxed answer checked by math_verify, which prints reward mean and std) and format_reward feed a group advantage that is zero when std is 0, then a GRPO update of the LoRA weights only (lr 1e-6, KL beta 0.04) loops back to the policy, save_model writes the adapter to the volume, a dashed arrow reloads it with is_trainable=True, and step4_eval.py compares base against adapted.",
    "rlvr-silent-failure-map": "Diagram of one GRPO training step from tokenizer through rollout, completion mask, reward and LoRA update, with four numbered red badges marking the seams where the eos/pad mismatch, missing stop criterion, no-KV-cache generate, and frozen adapter each failed silently.",
    "tiny-circuits-2-experiment-setup": "Diagram of the patching setup: a clean run [BOS][A][A] and a corrupted run [BOS][A'][A] that differ only in the first half, so second-half targets and the normalized score (clean = 1, corrupted = 0) are identical, above three panels showing residual patching, head hook_z patching and path patching from a sender head into the induction heads' k/q/v inputs.",
    "tiny-circuits-2-recovered-circuit": "A left-to-right circuit diagram of GPT-2 small in which the effective embedding feeds a layer 4 previous-token head, L4H11, that supplies the keys of five induction heads in layers 5 to 7 while MLP4 partly compensates, copy suppression heads L10H7 and L11H10 oppose and backup heads run in parallel into the logits, with a callout that the textbook layer 0 previous-token head is absent and a badge that patching all five recovers 0.81.",
    "divvy-bike-analysis-pipeline": "Pipeline of index.qmd. Twelve monthly Divvy CSVs, marked as not in the repository, are stacked by bind_rows with a month column, then given derived columns (hour, time_of_day, season, ride_length), then grouped and summarised. The result feeds 10 ggplot2 charts and 2 leaflet maps, which quarto render turns into index.html, published as a static Netlify site at divvy.farhansadeek.com. A code panel shows the difftime ride_length expression, and a red note says nothing filters negative or extreme ride lengths.",
    "divvy-bike-analysis-audit": "Five rows pairing a sentence from the write-up with what the chart shows. Ride length around 20 minutes for both rider types against casual 25.2 and member 12.8. Longest on Mondays against Sunday 21.4 and Monday 16. Monday fewest rides against Sunday 787,201 lowest. Casual slightly outnumbering members against members at 55.2 percent. A dataset of 5,667,986 trips against weekday labels summing to 5,860,568 and a donut reading 5.86M.",
    "pillifyai-architecture": "PillifyAI architecture: a Firebase auth box (email login, doctor or patient role) points to both the Expo and React Native mobile app, whose tabs are Home, History and Account plus doctor-only Patients and Prescribe and which keeps an offlineOperationsQueue in AsyncStorage, and to the Node Express server; the app and the server talk over REST (POST /prescribe, PUT /status/:id); the server exposes /api/auth, /api/patients, /api/medications and /api/devices and holds a Postgres database with the User, Doctor, Patient, Medication and MedicationTracking models; an ESP32 sketch that polls every 30 s, shows its state with LEDs on pins 2, 4 and 5, and blinks, dispenses and reports when a dose is due talks to the server both ways through GET esp32/medications and POST esp32/dispense/:id with an X-API-Key header; a separate box holds the Raspberry Pi scripts (hacked SG90 servo on GPIO 2 at 50 Hz, piezo buzzer on GPIO 18).",
    "pillifyai-dose-loop": "One dose across three lanes: the doctor prescribes in the app's Prescribe tab (name, dosage, frequency, time); the server writes a Medication row plus scheduled MedicationTracking rows in the pending state; the ESP32 polls GET esp32/medications every 30 s and finds a due dose; its dispensing LED on pin 4 blinks at 500 ms and the sketch auto-dispenses; the ESP32 POSTs esp32/dispense/:id with deviceId and dispensed: true so the server marks the dose dispensed; and the patient's History screen reads the tracking rows for that patient.",
    "photography-site-architecture": "Two lanes: the public Jekyll repo takes 52 curated photos through a gulp ImageMagick resize, a Jekyll build driven by _config.yml and Vercel static hosting, crediting rampatra's template and AJ's design, while the private photo-graphy repo takes Google Photos (2,952 files) through blur and pHash filters, gpt-4o-mini per-month scoring, shooting-session clustering that keeps the best frame and a second brutal ranking pass, then upload_photos.mjs writes to Postgres and object storage that a Next.js exif-photo-blog fork reads, with the local Ollama attempt off this path because its 3B pass cut only 13 of 2,419.",
    "photography-site-clustering": "An illustrative timeline of photo capture times split into three sessions wherever the gap exceeds 30 seconds, followed by a flow that dHashes every frame, merges at Hamming distance 6 or less, clusters near-duplicates across the month and keeps the highest score in each cluster, with settings of 30 s, Hamming <= 6 and threshold 70, the real 2025-07 effect of 330 of 488 images excluded as near-duplicates.",
    "listenting-architecture": "ListenTing architecture in two halves: an offline pipeline where qinkan.net text goes through prepare-chapters.ts into public/chapters/1..13.json (13 chapters, 290 sentences, 169 idiom tags), which feeds enrich-translations.ts (Haiku 4.5 fills en) and generate-audio.ts (ElevenLabs per sentence, rewrites timestamps from real durations), plus ch1-4.mp3 ElevenLabs narration uploaded by Mulong, feeding a browser PWA (Player, CheckinModal, IdiomCard with 28 curated entries, VaultSheet on device) that calls the Next.js API routes /api/question, /api/grade, /api/speak, /api/transcribe and /api/translate.",
    "listenting-checkin": "The spoken check-in as a vertical chain of phases in CheckinModal.tsx: the player pauses after 3 minutes of audio, loading calls /api/question (Haiku 4.5, last 12 sentences), asking calls /api/speak with a speechSynthesis fallback, listening uses browser SpeechRecognition or typed input, grading calls /api/grade (Sonnet 4.5) and the result phase shows a verdict of great, good, partial or off with feedback under 30 words.",
    "airport-network-analysis-architecture": "Pipeline of the airport network notebook: 20 monthly flight lists (March 2020 to October 2021, about 8 GB in Git LFS) feed a sampling step that produced sampled_flights.csv with 5,065 rows; drop_na() keeps 3,297, group_by(origin, destination) builds the edge list, simplify() gives a directed graph of 1,909 vertices and 2,574 edges that collapses to 2,429 undirected edges for the degree, strength, cohesion, transitivity and centrality chunks, and quarto render makes a self-contained page that Netlify builds with build.sh and serves at airport.farhansadeek.com.",
    "airport-network-analysis-row-funnel": "From 5,065 sampled flights to a 1,909-airport graph: drop_na() drops 1,768 rows (34.9%) and keeps 3,297, of which 526 with origin equal to destination become self-loops that simplify() removes, leaving 148 airports with degree 0, and the directed graph of 1,909 airports and 2,574 edges has 393 weakly connected components drawn as a bar with a largest component of 1,177, 192 pairs, 148 singletons and 52 others.",
    "cure-improve-architecture": "Layout of the CURE-Improve repo: concept prompts become CLIP text embeddings of shape n by 768 and feed cure/ (the base CURE reimplementation editing attn2 to_k and to_v), cure_seq/ (a SubspaceBank with orthogonal projectors that imports from cure/) and cure_dit/ (targeting SD3 MM-DiT), all of which feed evaluation/ with one results.json schema computing LPIPS_e, LPIPS_u and CLIP_u, with labels crediting the packages and protocol to Arses Prasai and the Figure-6 sweeps to Jeffrey Xie.",
    "cure-improve-subspace-bank": "Flow of one sequential erasure in CURE-Sequential: forget prompts become CLIP embeddings and an SVD gives right singular vectors, an orthogonalize step against the SubspaceBank B and QR feed an adaptive alpha and the projector P, the weights are edited as W minus W P on to_k and to_v, and new directions with spectral weight above 0.01 are registered back into the bank, which lives in the 768-dimensional CLIP space so its capacity is bounded.",
    "quantathon-leap-reserve-loan-states": "Three-state loan chain: Active moves to Inherited at 0.8% a year and to Repaid at 7.6% (sale 5.6%, equity 2.0%), Inherited moves to Repaid at 9%, Repaid is absorbing, and a side panel lists the fundamental matrix results of 12.96 expected years from Active, 11.11 from Inherited and a half-life of about 8.7 years.",
    "quantathon-leap-reserve-architecture": "Architecture of the LEAP reserve model: challenge data aggregates (34,269 eligible lines, 64 observed loan amounts, 62 active pilot loans) feed Aaditya's core cohort Monte Carlo of 3,000 paths, then per-year loan status and a per-path maximum cumulative deficit whose P95 is the reserve, $2.73M for loans and $4.01M for grants, with Farhan's cross-checks below and a dashed arrow to the PyTorch and agent simulation, P95 $3.60M, treated as an upper bound.",
    "quantifyai-market-regimes-architecture": "QuantifyAI pipeline: a competition workbook is merged and forward-filled, a 252-day drawdown rule labels Bear, Bull and Static, a gradient boosting classifier predicts the state 63 days ahead, an anomaly ensemble runs beside it, and both feed a backtester of about ten strategies with prior-day weights that reports return, Sharpe and maximum drawdown against buy-and-hold.",
    "quantifyai-market-regimes-leakage-map": "Training window 2007 to 2018 (2,769 days) and test window 2019 to 2022 (1,008 days) with four boxes pointing at them: a shuffled 80/20 split with 0.9408 logged accuracy that falls to 56.8% on 2019-2022, an anomaly detector fit on the whole test window, hand-set parameters tuned with no held-out window, and the causal parts, a trailing 252-day label and prior-day weights.",
    "aegis-fraud-triage-architecture": "Aegis pipeline: a mock producer publishes to a Redpanda topic, a consumer and gate scores each transaction, those above 0.55 wait in an asyncio queue read by three ReAct agent workers on Ollama llama3.2:3b (up to 6 steps, 9 tools), BLOCK or CLEAR is applied at once while FLAG_FOR_REVIEW goes to a human queue with a Slack webhook, and verdicts are stored in an in-memory FAISS index that is retrieved into the agent's prompt.",
    "aegis-fraud-triage-gate": "Inside the gate: 8 features feed an Isolation Forest whose scaled fraud score has a median of 0.30 and a maximum of 0.33, then Gaussian noise, rule floors and a hard-coded 0.55 cutoff, with a FAISS novelty check that can re-flag unflagged transactions, and measured on synthetic traffic the full gate had precision 1.000 and recall 0.37 to 0.40 while the forest alone flagged 0 of 5,000.",
    "clinova-trial-emulation-pipeline": "Clinova pipeline: a free-text clinical question flows through the Question, Design and Validator agents on gpt-5.2, with a dashed feedback arrow from the validator back to the design agent for at most 3 revise_spec iterations, then a local OMOP concept lookup and the Code agent on gemini-3-pro-preview, which produces analysis code for the All of Us enclave.",
    "clinova-trial-emulation-validator-loop": "Three design spec drafts and the validator's verdicts on the Barrett 2006 run: v1 (t0 = contrast administration time) gets FEEDBACK with 3 CRITICAL and 1 WARNING issues, v2 (t0 = max of procedure and drug start) gets 1 CRITICAL left-truncation issue, and v3 (t0 = drug_exposure_start_datetime) is VALID with 6 of 6 gates passed.",
    "clinova-trial-emulation-omop-tiers": "Tiered OMOP concept search: exact match on normalised text returns confidence 1.0 for iodixanol and iopamidol, RapidFuzz fuzzy match returns at a score of 95 or more, and the SapBERT and FAISS semantic tier is drawn dashed and grey because it is off by default and its index is not in the repo, with a noisy example where intra-arterial angiography matches a LOINC radiology panel at 0.4061.",
    "signifyai-architecture": "SignifyAI screen overlay: two threads grab a screen region with mss every 100 ms, the face thread runs an OpenCV DNN face detector and a 48 by 48 emotion CNN, the hand thread runs MediaPipe Hands, a 20 pixel crop, rembg and a 128 by 128 grayscale image into a 10-class gesture CNN and a 29-class ASL CNN, and a frameless, always-on-top, mouse-transparent Qt overlay shows the results.",
    "signifyai-models": "Layer stacks of the three SignifyAI CNNs: the gesture CNN has one 16-filter conv and a Dense 32 holding 2,097,184 of its 2,097,674 parameters, the ASL CNN has three conv stages and a Dense 128 holding 3,211,392 of its 3,307,805, and the emotion CNN has four batch-normalized conv blocks and global average pooling for 914,151 parameters.",
    "empirica-biomedical-graphs-architecture": "Empirica architecture in two rows: biomedical PDFs go through PyMuPDF, scispaCy NER, a co-occurrence and regex relationship extractor and GraphBuilder into one NetworkX graph per PDF stored in SQLite and shown in a React 2D and 3D force-directed view, while a chunker and RAG index with no embedding step yet feed Claude 3 Haiku and 3.5 Sonnet for the chat and hypothesis bar.",
    "empirica-biomedical-graphs-edge-extraction": "Two ways Empirica makes an edge: sentence-level co-occurrence adds weight 1 to every entity pair in a sentence and keeps up to three evidence sentences, and regex verb patterns with single-word captures turn 'the p38 MAPK inhibits NF-kB signaling' into an edge with source MAPK and target NF because the hyphen cuts the entity.",
    "diffsense-architecture": "DiffSense architecture: a React, Vite and Tailwind dashboard and a VS Code chat extension both call one FastAPI service (main.py, 3,913 lines, 36 routes) that uses seven backend modules by Jalen Francis: github_service, git_analyzer, embedding_engine, breaking_change_detector, rag_system, claude_analyzer and a SQLite database.",
    "diffsense-hybrid-drift": "The hybrid drift score blends CodeBERT on the diff lines (0.7) with MiniLM on the commit message (0.3, zero-padded to 768 dimensions) as 1 minus cosine similarity against a 0.3 threshold, and measured on 17 commits MiniLM alone gives median drift 0.679 with 16 of 16 over threshold while CodeBERT gives median 0.024 with 0 of 16 over.",
    "wakafree-architecture": "WakaFree architecture with two ways in: the VS Code extension (heartbeat on file switch, on edit once per file per 2 minutes, and on every save, queued and flushed every 30 s) posts to heartbeats.bulk on the Next.js 15 server, while syncRange pulls /summaries and /durations from the WakaTime API (every 20 minutes for 2 days, nightly for 365 days, or from the Refresh button) and upserts one waka_daily row per date into Supabase Postgres beside the heartbeats and waka_meta tables; the Recharts dashboard reads 365 slim rows plus 3 rows with full timelines, and a 15-tool FastMCP server calls the WakaTime API for an LLM client.",
    "wakafree-day-boundary": "Two panels built from the repo's date helpers: the first lines up local time five hours behind UTC against UTC for January 15 to 16 and shows that at 19:00 local UTC is already January 16, which is why offsetDate takes today in the local zone anchored at 12:00 UTC; the second draws bars measured from local midnight to the next local midnight, 24 hours on an ordinary day, 23 on the US spring-forward day 2026-03-08, 25 on the fall-back day 2026-11-01 and 26 on a day that ends in a zone 2 hours west.",
}


def main() -> int:
    missing = [n for n in ALT if not (PNG / f"{n}.png").exists()]
    if missing:
        print("renders missing:", ", ".join(missing))
        return 1

    for name in ALT:
        png, webp = PNG / f"{name}.png", IMG / f"{name}.webp"
        tmp = IMG / f".{name}.resized.png"
        subprocess.run(["sips", "--resampleWidth", str(WIDTH), str(png), "--out", str(tmp)], check=True, capture_output=True)
        subprocess.run(["cwebp", "-quiet", "-q", "88", "-m", "6", str(tmp), "-o", str(webp)], check=True)
        tmp.unlink()
        w, h = (int(x) for x in subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(webp)], capture_output=True, text=True
        ).stdout.split()[-3::2])
        print(f"{name:26} {png.stat().st_size // 1024:5} KB png -> {webp.stat().st_size // 1024:4} KB webp ({w}x{h})")
        ALT[name] = (ALT[name], w, h)

    swapped = 0
    for post in sorted([*POSTS.glob("*.md"), *DRAFTS.glob("*.md")]):
        text = post.read_text()

        def repl(m: re.Match) -> str:
            nonlocal swapped
            name = m.group(1) or m.group(2)
            if name not in ALT:  # another draft's diagram, not registered yet
                return m.group(0)
            alt, w, h = ALT[name]
            swapped += 1
            svg = IMG / f"{name}.svg"
            if svg.exists():  # clickable version from tools/interactive-diagrams.py
                inline = svg.read_text().replace('role="img" ', f'role="img" aria-label="{alt}" ', 1)
                return (
                    f'<figure class="excal excal-interactive" data-diagram="{name}">{inline}'
                    f'<figcaption class="excal-hint">Hover a box to see where it is explained; click to jump there.</figcaption>'
                    f"</figure>"
                )
            return (
                f'<figure class="excal" data-diagram="{name}">'
                f'<a href="/img/diagrams/{name}.webp" class="excal-link" aria-label="Open the diagram full size">'
                f'<img src="/img/diagrams/{name}.webp" alt="{alt}" width="{w}" height="{h}" loading="lazy" decoding="async">'
                f"</a></figure>"
            )

        # Old placeholders, and figures from an earlier run (their size may have changed).
        new = re.sub(
            r'<figure data-figure="diagram:([a-z0-9-]+)"></figure>'
            r'|<figure class="excal(?: excal-interactive)?" data-diagram="([a-z0-9-]+)">.*?</figure>',
            repl, text)
        if new != text:
            post.write_text(new)
    print(f"swapped {swapped} figures")
    left = [p.name for p in [*POSTS.glob("*.md"), *DRAFTS.glob("*.md")] if 'data-figure="diagram:' in p.read_text()]
    if left:
        print("still unswapped in:", ", ".join(left))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
