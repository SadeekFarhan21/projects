# DiffSense (hackathon project, upstream by Jalen Francis)

**This is not primarily my project.** DiffSense is a two-day hackathon build
(its README says "AI Berkeley Hackathon"; nothing in the repo documents a
placement or prize). The canonical, authoritative source is
**https://github.com/jalenfran/DiffSense**, owned by Jalen Francis. This folder
is a snapshot of my fork (https://github.com/SadeekFarhan21/DiffSense) at commit
`da6f760` (2025-06-22), the snapshot analyzed here. Upstream has since gained two commits (`e9d5d64` and the merge `f2be630`) that I did not analyze; the 25/18/9/7/2 counts below are as of `da6f760`. The repo has no
license file, so this is a reference copy with attribution; if you want to reuse
the code, ask the authors.

## Who did what

From `git shortlog` on the history (25 commits: 18 non-merge, 7 merges; merges
inflate the plain shortlog to 12/11/2):

| Author | Non-merge commits | Work (from commit titles and files touched) |
|---|---|---|
| Jalen Francis | 9 | Backend: FastAPI app, embedding engine, breaking-change detector, RAG system, GitHub OAuth, SQLite persistence and caching |
| Jayson Clark | 7 | Most of the React frontend, API integration, VS Code extension setup |
| Farhan Sadeek (me) | 2 | `0019d66` mobile-responsive layout (Dashboard, MainContent, Sidebar; +166/-50) and `ce28f3a` markdown rendering in chat (`MarkdownRenderer.jsx`, 178 lines, react-markdown + GFM + highlight.js) |

Per-file attribution was not checked with `git blame`; the split is from commit
titles and diffs. The commit titled "farhan" (`3ff5e3f`) is authored by Jalen
and is a one-line frontend README change, not mine.

## What it is

A web app plus VS Code chat extension for analysing a git repository's history.

- `backend/` FastAPI service (`main.py` 3913 lines, 36 route decorators; `src/`
  modules total about 12.6k lines with `main.py`): `embedding_engine.py`
  (CodeBERT + MiniLM), `breaking_change_detector.py` (AST/regex, 2824 lines),
  `rag_system.py`, `claude_analyzer.py`, `database.py` (SQLite), `github_service.py`
- `frontend/` React + Vite + Tailwind dashboard
- `code-extension/` VS Code webview chat (TypeScript, esbuild)
- `scripts/` measurement scripts added in this blog repo (not in upstream)
- `results/` outputs of those scripts

`README.upstream.md` is the original README. It is aspirational and stale (it
references files that do not exist in the repo); do not treat it as evidence of
what works.

## What was changed relative to upstream

- Excluded: lockfiles, built extension bundle (`code-extension/media/index.js`),
  `.git`, and everything untracked in my local working copy (env file, sqlite db,
  venv, cloned repos, debug log).
- Replaced a hardcoded server IP that appeared in `vite.config.js`,
  `frontend/src/config/index.js`, `backend/src/github_service.py`,
  `frontend/.env.example` and two frontend docs with `localhost`.
- Original `README.md` renamed to `README.upstream.md`. The empty
  `pyproject.toml` was not copied.
- Secret scan of the copied tree (patterns for keys, tokens, client secrets,
  PEM blocks, long random-looking strings, `.env`/`.db` files): no credentials
  found. Hits were code identifiers such as `sessionToken`.

## Measurements

Everything below comes from `scripts/measure.py` and `scripts/diagnose.py`,
run against the code in `backend/src` with no Claude or other paid API.
Environment: Python 3.12.13, torch 2.14.0, transformers 4.57.6,
sentence-transformers 5.7.0, gitpython 3.1.62, scikit-learn, CPU. These are newer
than the versions pinned in `backend/requirements.txt` (torch 2.1.1,
transformers 4.36.0, gitpython 3.1.40), so numbers may differ under the pins.
The subject repo for the history runs is DiffSense itself (18 non-merge commits).
There is no ground-truth label set for real history, so no accuracy is claimed there.

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch sentence-transformers "transformers<5" scikit-learn numpy gitpython python-dotenv anthropic
python scripts/measure.py planted results/planted_breaking_changes.json
python scripts/measure.py history <path-to-DiffSense-clone> results/history_breaking_changes.json
python scripts/measure.py drift <path-to-DiffSense-clone> results/drift_minilm.json --model all-MiniLM-L6-v2
python scripts/measure.py drift <path-to-DiffSense-clone> results/drift_codebert.json --model microsoft/codebert-base
python scripts/diagnose.py
```

### 1. Breaking-change detector on planted cases (`results/planted_breaking_changes.json`)

14 tiny hand-written cases (one commit each, Python and JS) that I wrote and
labeled myself: 8 changes I consider breaking, 6 I consider not.
The detector (`analyze_commit_for_breaking_changes`, heuristic path, no AI)
gave TP 8, FN 0, FP 3, TN 3. It caught every removal, rename, added or removed
parameter, and the JS cases. The false positives:

- adding an optional parameter (`c=0`) was flagged (`parameter_change`,
  plus `function_removal` and `behavioral_change`)
- a docstring-only edit was flagged `behavioral_change`
- a body change with an unchanged signature was flagged `behavioral_change`

Whether the last two are "false" is a definition call: the detector deliberately
has a behavioural-change category. 14 hand-picked cases show it works on the
obvious cases and over-reports; they say nothing about real-world precision.

### 2. Detector over DiffSense's own history (`results/history_breaking_changes.json`)

17 non-root, non-merge commits: 14 had at least one finding, 717 findings in
total (function_removal, parameter_change, behavioral_change, dependency_change,
return_type_change, api_signature_change). One commit (`d395719`) accounts for
440 of them. These are unlabeled detector outputs, not validated breaking changes.

### 3. Embedding drift and the 0.3 threshold (`results/drift_*.json`)

Hybrid embedding = 0.7 x code-diff embedding + 0.3 x commit-message embedding
(`SemanticAnalyzer.generate_hybrid_embedding`); drift = 1 - cosine similarity
between consecutive commits' embeddings; default significance threshold 0.3.
17 commits with Python/JS/TS changes in this repo.

| Code model | Dim | Drift vs previous: min / median / max | Commits over 0.3 |
|---|---|---|---|
| all-MiniLM-L6-v2 (the code path's fallback) | 384 | 0.501 / 0.679 / 0.985 | 16 of 16 |
| microsoft/codebert-base (the default) | 768 | 0.0135 / 0.0242 / 0.0762 | 0 of 16 |

(16 comparisons from 17 commits.) With the default CodeBERT model the 0.3
threshold never fires; with MiniLM it always fires. `results/diagnostics.json`
shows why: mean-pooled CodeBERT vectors of five unrelated snippets (Python, SQL,
Java, C, numpy) have pairwise cosine 0.935 to 0.963, whereas MiniLM's range from
-0.078 to 0.284. So the threshold is not calibrated for either model on this data.

### Two code observations from running it

- `GitAnalyzer.extract_diff_info` calls `commit.diff(parent)` without
  `create_patch=True`, and in my run returned empty added/removed lists
  (`results/diagnostics.json`, `diff_direction`). Nothing in `main.py` calls it
  (only the module's `example_usage`). My drift script therefore builds the
  patch itself, as the detector and `main.py` do.
- In the API's commit analysis, `semantic_drift_score` is computed in
  `breaking_change_detector.py` as `risk_score * 0.8 + (breaking / total changes) * 0.2`,
  not from embeddings. `SemanticAnalyzer` is instantiated in `main.py` and passed
  to the RAG system, but I did not find it computing per-commit drift in the routes.
  The `severity_predictor` and `intent_classifier` attributes are `None` placeholders.

## Not measured

The Claude analysis and Claude-backed chat (need an Anthropic key), GitHub OAuth,
the RAG retrieval quality, the frontend and VS Code extension (never launched;
no tests exist in the repo), and any accuracy on real history (no labels).
