---
layout: post
title: "The Encoder Sets the Scale of Commit Drift"
tab_title: DiffSense
code: https://github.com/SadeekFarhan21/projects/tree/main/diffsense
date: 2025-06-21 13:25:13
tags:
  - git
  - embeddings
  - fastapi
  - hackathon
description: >-
  DiffSense scores how far each commit drifts semantically by embedding its diff
  and message. On its own 17 commits, CodeBERT gives a median drift of 0.0242
  and MiniLM gives 0.679, and the two ranges never overlap.
---

DiffSense is a web app and VS Code extension for asking questions about a git repository's history. Jalen Francis, Jayson Clark and I built it in two days (June 21–22, 2025) at the AI Berkeley Hackathon. It clones a repository, scores each commit for "semantic drift" by embedding its diff and message, flags likely breaking changes by comparing function and class signatures, and puts a chat on top that can call Claude<sup>[[1]](#ref-1)</sup>. Jalen built the backend, Jayson built most of the frontend and the extension, and I built the mobile layout and the markdown rendering in the chat.

Afterwards I ran the analysis on DiffSense's own history, and the main lesson is that **a drift score only means something next to the encoder that produced it**. Over the same 17 commits, the same formula gives a median drift of **0.0242** with CodeBERT<sup>[[2]](#ref-2)</sup> and **0.679** with MiniLM<sup>[[3]](#ref-3)</sup>, and the two ranges never overlap. The signature-based breaking-change detector caught all 8 real breaks in 14 small cases I wrote and labeled.

## Why It Matters

The pitch is to point DiffSense at a repository and see where it changed and which commits are likely to break callers, without reading every diff.

Drift was the part I wanted to understand. A commit is a diff plus a message. Embed both, compare each commit to a reference, and look at the distance, and you get a cheap timeline of where a codebase changed character. The alternative is asking an LLM about every commit, which costs a call per commit and isn't reproducible. An embedding distance is one number, computed locally, the same on every run.

A number like that is only useful if you know what a big value looks like. So on top of the build I set myself a second goal: run the analysis parts that need no paid key on real history and see what the numbers actually do.

## Technical Details

### Drift as One Minus Cosine Similarity

Given two commit vectors $u$ and $v$, the drift is

$$
d(u, v) = 1 - \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}.
$$

Identical directions give 0, orthogonal vectors give 1, and opposite vectors give 2. The score is only about direction. A commit that touches ten times as many lines but points the same way in embedding space scores the same. That makes it cheap and explainable, and it also means the score says nothing about how much changed.

DiffSense's `calculate_drift_score` compares each commit to the first one, so its score is cumulative distance from where the project started. In my measurements I also compare each commit to the previous one, which reads as a step size.

### The Hybrid Embedding

A commit has two views, code in the diff and prose in the message, so DiffSense embeds them separately and blends them. The code encoder is CodeBERT<sup>[[2]](#ref-2)</sup>, mean pooled over the last hidden state, and the text encoder is a sentence transformer, `all-MiniLM-L6-v2`<sup>[[3]](#ref-3)</sup>, which was trained so that cosine similarity between sentence vectors is meaningful<sup>[[4]](#ref-4)</sup>. The hybrid is

$$
h = 0.7 \, e_{\text{code}} + 0.3 \, e_{\text{text}},
$$

with the 384-dimensional text vector zero-padded to CodeBERT's 768 dimensions.

### Why the Encoder's Geometry Sets the Scale

Cosine distance only spreads out if the encoder spreads unrelated inputs across the sphere. Sentence-transformer models are trained to do that. Mean-pooled hidden states of a masked language model are not. Contextual embeddings from such models are known to occupy a narrow cone, where unrelated inputs still have high cosine<sup>[[5]](#ref-5)</sup>. An encoder like that pushes every drift toward zero, so the same formula reads on a completely different scale depending on which model sits under it.

### Breaking Changes Without a Model

The other detector doesn't use embeddings at all. It compares the API surface of a file before and after a commit. For Python it parses both versions with the standard `ast` module and collects a signature for every function, async function and class. For JavaScript, Java and C++ it does the comparison with regular expressions. A removed name, a renamed name or a changed parameter list is a candidate breaking change, and a separate behavioral-change category covers edits that keep the signature. It's a heuristic with no trained classifier behind it.

### Architecture

One FastAPI service sits between the clients and everything else. `main.py` is 3,913 lines with 36 route decorators (`@app.get`, `@app.post`, `@app.put`, `@app.delete`). The routes include clone-repository, repository stats and commits, an enhanced RAG query, per-commit analysis, analyze-range and a dashboard. The 11 modules under `src/` total about 8,700 lines, and the whole backend with `main.py` is about 12,600.

<figure class="excal" data-diagram="diffsense-architecture"><a href="/img/diagrams/diffsense-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/diffsense-architecture.webp" alt="DiffSense architecture: a React, Vite and Tailwind dashboard and a VS Code chat extension both call one FastAPI service (main.py, 3,913 lines, 36 routes) that uses seven backend modules by Jalen Francis: github_service, git_analyzer, embedding_engine, breaking_change_detector, rag_system, claude_analyzer and a SQLite database." width="2400" height="1193" loading="lazy" decoding="async"></a></figure>

Git access goes through GitPython and a GitHub service with OAuth. Analysis is split into the embedding engine and the breaking-change detector. Chat is a RAG system plus a Claude analyzer, and SQLite (the standard `sqlite3` module) stores users, chats, commits, analyses and cached file contents. The React dashboard and the VS Code chat webview both call the same API.

The drift path in detail:

<figure class="excal" data-diagram="diffsense-hybrid-drift"><a href="/img/diagrams/diffsense-hybrid-drift.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/diffsense-hybrid-drift.webp" alt="The hybrid drift score blends CodeBERT on the diff lines (0.7) with MiniLM on the commit message (0.3, zero-padded to 768 dimensions) as 1 minus cosine similarity against a 0.3 threshold, and measured on 17 commits MiniLM alone gives median drift 0.679 with 16 of 16 over threshold while CodeBERT gives median 0.024 with 0 of 16 over." width="2400" height="1352" loading="lazy" decoding="async"></a></figure>

## Implementation

The backend excerpts below are short and attributed. The original repository is [jalenfran/DiffSense](https://github.com/jalenfran/DiffSense), and my fork is [SadeekFarhan21/DiffSense](https://github.com/SadeekFarhan21/DiffSense).

### Embedding a Diff (Jalen Francis)

A diff becomes one string, with removed lines first and added lines after, each behind a marker token. That string is embedded like any other code, so a 2,000-line commit is truncated at 512 tokens.

```python
def embed_diff(self, added_lines, removed_lines):
    diff_text = ""
    if removed_lines:
        diff_text += "[REMOVED] " + "\n".join(removed_lines) + " "
    if added_lines:
        diff_text += "[ADDED] " + "\n".join(added_lines)
    if not diff_text.strip():
        diff_text = "[EMPTY_DIFF]"
    return self.embed_code(diff_text)
```

`embed_code` runs the tokenizer with `max_length=512, truncation=True`, then takes `outputs.last_hidden_state.mean(dim=1)`. The engine can also run `all-MiniLM-L6-v2` as the code encoder in CodeBERT's place, and that path is what I measure as "MiniLM" below.

### The Blend and the Score (Jalen Francis)

```python
hybrid_embedding = code_weight * code_embedding + text_weight * text_embedding
```

```python
drift_score = 1 - similarity  # Higher drift = lower similarity
```

### Signature Extraction With `ast` (Jalen Francis)

The Python half of the breaking-change detector visits every function, async function and class and stores a signature keyed by qualified name.

```python
def visit_FunctionDef(self, node):
    sig = self.detector._create_function_signature(node, self.current_class)
    signatures[sig.name] = sig
    self.generic_visit(node)

def visit_ClassDef(self, node):
    old_class = self.current_class
    self.current_class = node.name
    sig = self.detector._create_class_signature(node)
    signatures[sig.name] = sig
    self.generic_visit(node)
    self.current_class = old_class
```

Comparing the before and after dictionaries gives removals, renames and parameter changes. The detector's typed output uses `ChangeType`, `ImpactSeverity` and `ChangeIntent` enums, and it can hand its findings to Claude when `enable_ai_analysis` is set. That module is 2,824 lines.

### Claude and Retrieval (Jalen Francis)

`claude_analyzer.py` wraps the Anthropic SDK and defaults to `claude-3-sonnet-20240229`, overridable with `CLAUDE_MODEL`. Retrieval for the chat lives in `rag_system.py`, which is 1,511 lines.

### The Frontend (Jayson Clark, and My Parts)

Jayson wrote most of the dashboard, the API integration and the extension. The extension is a webview chat, with `extension.ts` and a React `ChatApp.tsx` under `media/`.

**Mobile layout.** My first change touches `Dashboard.jsx`, `MainContent.jsx` and `Sidebar.jsx` (+166/-50). The core of it is a sidebar that is off canvas below the `lg` breakpoint and slides in over a dimmed overlay.

```jsx
{isMobileSidebarOpen && (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
         onClick={() => setIsMobileSidebarOpen(false)} />
)}
<div className={`
    fixed lg:relative lg:translate-x-0 z-50 lg:z-auto
    transform transition-transform duration-300 ease-in-out
    ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
`}>
```

On desktop the sidebar is a normal flex child. On small screens it's `fixed` and translated out of view until the top bar toggles it, and a tap on the overlay closes it.

**Markdown in chat.** My second change added `MarkdownRenderer.jsx` (178 lines), wired it into `ChatInterface.jsx` (+21 lines), and added `react-markdown`, `remark-gfm`, `rehype-highlight` and `highlight.js`, among other packages (`axios` and `lucide-react` came in too). It swaps the highlight theme with the page's dark mode by injecting a stylesheet and removing it on unmount.

```jsx
const isDark = document.documentElement.classList.contains('dark')
existingStyles.forEach(style => style.remove())
link.href = isDark
  ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css'
  : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css'
```

Jayson later built on the same file with file and commit-hash linkifying: a mention like `src/App.jsx` or a 7-character hash in a reply becomes a click target that opens the file viewer. That code is his.

## Problems

### 1. Knowing What a Big Drift Looks Like

Drift is only useful if you can tell a big step from a small one, and the formula alone doesn't say. So I ran the identical 17 commits through the identical 0.7/0.3 blend with two code encoders and looked at where the numbers land. **CodeBERT's drift stays between 0.0135 and 0.0762, while MiniLM's runs from 0.501 to 0.985.**

A small diagnostic on five unrelated code snippets (Python, SQL, Java, C and NumPy) shows why. **Raw CodeBERT vectors for unrelated code have pairwise cosine between 0.935 and 0.963, while MiniLM's range from -0.078 to 0.284.** CodeBERT sees all code as similar, so every drift it reports is small. That's the narrow-cone picture from [Technical Details](#why-the-encoder-s-geometry-sets-the-scale). The answer is that **a drift value has to be read against its own encoder's range**, never as an absolute number.

### 2. Finding Breaks Without a Model

Asking Claude whether every commit breaks callers costs a call per commit. The detector instead reduces each file to a dictionary of signatures and diffs the dictionaries. Python gets an exact parse through `ast`, and JavaScript, Java and C++ get regular expressions, so one pass covers the languages the dashboard shows. To check it, I built a small repo with 14 one-commit edits in Python and JS and labeled each one. **It caught every one of the 8 real breaks**, from a removed class to a changed JS parameter count, with no model call at all.

## Experiments

I ran the parts that need no paid key, on CPU, with newer libraries than the repo pins (torch 2.14.0, transformers 4.57.6, sentence-transformers 5.7.0 and gitpython 3.1.62, against pinned 2.1.1, 4.36.0, 2.2.2 and 3.1.40). The measurement scripts, `scripts/measure.py` and `scripts/diagnose.py`, are mine.

1. **Drift.** Take each of the 17 commits that change Python, JS or TS files, oldest first. Build the patch, embed the diff and the message with the hybrid, and compute 1 minus cosine to the previous commit and to the first (what DiffSense itself does). Run it once with CodeBERT as the code encoder and once with MiniLM. That's 16 comparisons each.
2. **Diagnostic.** Embed five unrelated snippets with both encoders and record the min and max pairwise cosine over the 10 pairs.
3. **Planted breaking changes.** Build a small synthetic repo, apply 14 one-commit edits (Python and JS), and label each breaking or not. The detector's heuristic path runs with no Claude. 8 cases are breaking and 6 are not.
4. **History.** Run the detector over every commit of DiffSense itself and count findings. There are no labels.

## Results

### Drift Under Two Encoders

<figure data-figure="chart:projects/diffsense/diffsense-drift"></figure>

| Code encoder | Dimensions | Min | Median | Max |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.501 | 0.679 | 0.985 |
| microsoft/codebert-base (default) | 768 | 0.0135 | 0.0242 | 0.0762 |

**The two encoders put drift on scales that don't overlap, and CodeBERT's largest step (0.0762) is far below MiniLM's smallest (0.501).** CodeBERT's biggest steps are the last two commits, Jalen's that integrated change detection into more models (0.062) and Jayson's "More complete frontend" (0.076). MiniLM's biggest is Jalen's "major updates to architecture" (0.985). **The two encoders don't pick the same commit as the biggest move.**

The raw pairwise cosine on unrelated snippets:

| Encoder | Min cosine | Max cosine |
|---|---|---|
| CodeBERT | 0.935 | 0.963 |
| MiniLM | -0.078 | 0.284 |

### Breaking-Change Detector on Planted Cases

<figure data-figure="chart:projects/diffsense/diffsense-planted"></figure>

**The detector caught all 8 real breaks and missed none.** That covers every removed function, renamed function, removed method, removed class, added or removed required parameter, removed JS export and changed JS parameter count, all from the heuristic path with no Claude call.

### Findings Over Real History

<figure data-figure="chart:projects/diffsense/diffsense-history"></figure>

**Over DiffSense's own history, 14 of 17 commits had at least one finding, for 717 in total, and one commit accounts for 440 of them**: Jalen's "integrated change deteciton in further models". That's also where CodeBERT registers one of its two biggest drift steps. My mobile-layout commit gets 3 findings, on `MainContent` and `Sidebar`, and my markdown commit gets 0.

## What I Would Change

### Read Drift Against the Repository's Own History

Since the scale belongs to the encoder, I'd report drift relative to the data. Compute drift for many commits of many repositories, look at its distribution for the chosen encoder, and rank a commit by percentile, or mark it as an outlier against the repository's own drift history.

### Use a Code Encoder Trained for Similarity

Mean-pooled CodeBERT vectors all sit close together. A model trained with a contrastive objective for code retrieval, or CodeBERT with whitening or mean-centering of the vectors, would spread them out. I'd try mean-centering first because it's one line and testable with the same 10-pair diagnostic.

### Label Commits Before Adding Features

I'd write down 50 commits from a public repository with a human label for "behavior changed" and "API broke", then score the drift and the detector against it. Without labels every improvement is a guess. The 14 planted cases are a starting test file.

## References

1. <span id="ref-1"></span>Anthropic. *Claude API documentation*. Anthropic, 2025. [link](https://docs.anthropic.com/)
2. <span id="ref-2"></span>Zhangyin Feng, Daya Guo, Duyu Tang, et al. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*. Findings of EMNLP, 2020. [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
3. <span id="ref-3"></span>Wenhui Wang, Furu Wei, Li Dong, et al. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS, 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)
4. <span id="ref-4"></span>Nils Reimers and Iryna Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP, 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
5. <span id="ref-5"></span>Kawin Ethayarajh. *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings*. EMNLP-IJCNLP, 2019. [arXiv:1909.00512](https://arxiv.org/abs/1909.00512)
