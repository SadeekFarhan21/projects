# Aegis fraud triage (CMU hackathon, Feb 2026)

Snapshot of my fork, https://github.com/SadeekFarhan21/Aegis (branch `master`, commit `d3ad4b9`), which forks https://github.com/jalenfran/cmu-hackathon (owner Jalen Francis). No LICENSE file exists in the repo, so this is a reference copy with attribution.

## Who did what (from `git shortlog`, 70 commits)
- Farhan Sadeek (me): all of `backend/` (+7490/-1062 lines incl. docs and docker-compose, per `git log --numstat`) and most of the dashboard in `frontend/`. First commit 2026-02-06, last hackathon commit 2026-02-07 (~20 hours).
- Jalen Francis: upstream repo owner; one commit, "fixed eslint" (5 frontend files, +10/-25). Other contributions (design, demo, slides) are not visible in git.
- gpt-engineer-app[bot] / Lovable: the landing page (frontend only). Excluded from this copy.

## What was copied
Backend, dashboard components, docker-compose, docs. Excluded: `aegis-tartanhacks/` and `frontend/src/landing/` (Lovable-generated), lockfiles, world_map.jpg, `.claude/`. Because `landing/` is removed, `frontend/src/App.tsx` (which lazy-imports it) will not build as-is. `README.upstream.md` is the original README; its claims (e.g. "8 tools", "higher weight" for human verdicts, Redis-based velocity) do not all match the code.

## Measurements (added for the blog; not part of the original project)
Harness: `scripts/gate_bench.py` and `scripts/extra.py` import the repo's own `backend/anomaly_detection/engine.py` and the producer's amount helpers, on a synthetic stream mirroring the producer mix (8% fraud, 12% elevated normal, 80% normal). Domestic merchants are my stand-ins (the real ones need a Nessie API key). Labels come from my generator, so this measures the gate against the repo's own synthetic fraud model, not real fraud.

Run on 2026-09-29, Python 3.14.7, scikit-learn 1.9.1, numpy 2.5.3, single thread, Mac (model not recorded), 5,000 txns x 3 seeds x 2 clock starts:
```
pip install numpy scikit-learn httpx pydantic pydantic-settings
python scripts/gate_bench.py   # -> results/gate_bench.json, gate_bench_out.txt
python scripts/extra.py        # -> results/extra_out.txt
```
- Gate (IF + noise + rule boosts, no FAISS step): precision 1.000 in all 6 runs; recall 0.368-0.403; 2.9-3.2% of traffic flagged.
- Isolation Forest alone at the equivalent cutoff (-2*decision_function > 0.55): 0 of 5,000 flagged in every run.
- Fraud recall by amount (3,000-txn run): $0-1,000 -> 0.0 (149 txns); >$1,000 -> 1.0 (84 txns).
- Gate throughput about 149-153 txn/s.
Not measured: FAISS novelty boost, LLM latency/verdicts, Kafka/queue behaviour, dispute agent.
