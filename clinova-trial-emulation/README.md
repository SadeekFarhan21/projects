# clinova-trial-emulation

Port of the **Clinova** hackathon-style pipeline that turns a free-text clinical question into a Target Trial Emulation (TTE) design, validates it with an LLM critic, maps terms to OMOP concept IDs, and generates analysis code for the All of Us / OMOP enclave.

- Upstream: https://github.com/SadeekFarhan21/Clinova (main at `304445f`). Originally named LepisAI, renamed Clinova in Feb 2026.
- Original README preserved as `UPSTREAM_README.md`. Its "GPT-4" wording is out of date; `backend/agent/config.py` sets `gpt-5.2` (agents) and `gemini-3-pro-preview` (code agent).
- License: the upstream repo has no root LICENSE file (its README says "See LICENSE file"). Treat as all rights reserved by the authors; this folder is a portfolio copy.

## Attribution (from git history)

- Team project. The 35k-line initial commit (2026-01-25) was pushed from Farhan Sadeek's account, but git cannot show who wrote which part, and the docs say "our team". Authorship of individual backend modules is therefore **not claimed here**.
- Praneel Patel pushed the Barrett run folder (`backend/run/benchmark-barrett-*`, commits f9bf8c0, e860863) and the `example-for-website` folders. Git does not show who ran the benchmark or with what script (the run logs differ from `agent/main.py`, so the runner is not committed).
- Farhan Sadeek (git): the backend `GET /api/examples` endpoint in `server.py` and the frontend integration and UI (not copied here).
- A Copilot bot did the LepisAI to Clinova rename.

## What is copied

`backend/` agents (`agent/`), OMOP lookup (`omop_lookup/`), `server.py`, `rubric.md`, `whatwearebuilding.md`, requirements, the committed Barrett run artifacts and the four `example-for-website` JSONs plus their scripts.

Excluded: the Lovable/shadcn frontend, the third-party `visualization/` template, vendor sample LLM clients, `papernotes.md` (third-party paper excerpts), `SOTAstack.md` (provenance unclear), PNG/SVG plots, logs, local `.claude` settings, deployment glue.

## Pipeline

Question agent (causal PICO) -> Design agent (TTE spec) -> Validator agent (rubric gates, max 3 revise loops, then proceeds with best design) -> free-text to OMOP concept lookup (exact, RapidFuzz fuzzy, SapBERT+FAISS semantic, vocabulary weighting) -> Code agent (Gemini).

## Measurements (offline, no API calls)

Nothing here re-runs the pipeline: it needs OpenAI and Gemini keys, licensed Athena vocabularies plus a built FAISS index, and the All of Us enclave. What was measured is the committed Barrett-2006 run, via:

```
python3 analysis/summarize_barrett_run.py   # writes results/barrett_run_summary.json
```

Findings (all from `results/barrett_run_summary.json`):

- Run log: 837 s end to end (2026-01-25 04:29:47 to 04:43:44). Step 1 took 87 s, design 173 s, validation 490 s (216 + 221 + 53 across 3 iterations), OMOP lookup 6 s, code 81 s. Outputs: 6,718 / 22,127 / 16,368 chars.
- Validator loop: iteration 1 FEEDBACK (3 CRITICAL + 1 WARNING; Gate 4 time-zero failed, Gate 3 warning), iteration 2 FEEDBACK (1 CRITICAL Gate 4, left truncation from `t0 = max(procedure, drug)`), iteration 3 VALID (6/6 gates). Design spec grew 341 -> 381 -> 385 lines.
- OMOP lookup: the term file has 47 terms, but `step2c_omop_results.csv` holds only the first 20 (74 rows), so 27 terms have no committed results. Of the 20 top-1 hits, 6 are exact name, 5 exact synonym, 9 fuzzy; 14 have confidence >= 0.95; 4 top-1 rows have no `analytics_concept_id`. Lowest top-1 is `angiography with contrast (intra-arterial)` at 0.4061 (a LOINC radiology concept). For `multidetector computed tomography`, top-1 is SNOMED "Computed tomography" (0.9) but ranks 2-10 are all 0.4061 LOINC radiology panels.
- Not measured: concept-mapping accuracy, agreement with published effect sizes, whether the generated code executes. There is no test suite upstream.

## Caution on `example-for-website`

The four result JSONs (PREDICT, VALOR, NEPHRIC, CONTRAST-AKI) are UI demo data of unverified provenance. Every one reports the same initial population (71,743), they contain scripted chat turns, and the `generate_graphs.py` scripts draw propensity scores and SMDs from `np.random` (seed 42) rather than model output. Do not treat them as real emulation results or replications. The CONTRAST-AKI conclusion ("contrast is safe") is likely a confounding artifact and should not be repeated. The Barrett folder is the only demonstrably real pipeline run. CARE has no result folder.
