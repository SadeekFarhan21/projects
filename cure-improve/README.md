> **Attribution (read first).** This is a copy of a team project, not solo work by the blog author.
> - Code (`cure/`, `cure_seq/`, `cure_dit/`, evaluation scripts, docs, quick-proof runs): **Arses Prasai** (16 commits, [arses-ui/CURE-Improve](https://github.com/arses-ui/CURE-Improve)).
> - Figure-6 sweeps (`evaluation/figure6_results*`, `FIGURE6_RESULTS.md`): committed by **Jeffrey Xie** (3 commits). Git shows who committed, not who ran them. Quick-proof runs were committed by Arses.
> - Farhan Sadeek ([SadeekFarhan21/CURE-Improve](https://github.com/SadeekFarhan21/CURE-Improve), a fork): 1 commit (authored under a different name and email, message 'Refactor code structure...') that only adds four PNG copies of Jeffrey's plots (not copied here). Any further role is not visible in git history.
> - The method is from the CURE paper (arXiv:2505.12677; Biswas, Roy, Roy, Purdue). `cure/` is a reimplementation, not the authors' release.
> - The upstream repo has no LICENSE file; code is copied here with the consent question to Arses still open. Some upstream docs were AI-assisted (the results doc lists "Reviewer: Codex").
> - Snapshot: upstream HEAD `5648737` (2026-05-23). Removed: `CLAUDE.md` files, `cure/docs/` debug notes, slide notes, `docs/assets/` PNG copies. Hardcoded personal output paths were replaced by `<OUTPUT_DIR>`, and a server cache path in the two results.json files by `<CACHE_DIR>`.
> - `results/` was written by the blog port with AI help (both scripts rerun 2026-09-29, output identical to the committed logs): `synthetic_orthogonality.py` (weight-free linear-algebra check on synthetic embeddings) and `figure6_deltas.py` (re-derives cure_seq minus cure deltas from Jeffrey's results.json). Neither needs SD weights; nothing needing weights was re-run.
> - TODO before publishing: get Arses's consent, or strip this folder to `results/`, `evaluation/figure6_results*` (results.json, summary, plots) and short excerpts.
> - Known caveats: at k=100 CURE-Seq is worse than CURE on LPIPS_u and CLIP_u (see `results/figure6_deltas.log`), which the upstream README does not mention. `unerased_artists_10.txt` overlaps `erased_artists_100.txt` (Artemisia Gentileschi, Paul Signac). CURE-DiT has never been run against SD3.

# CURE Improvements

Extensions of [CURE: Concept Unlearning via Orthogonal Representation Editing](https://arxiv.org/abs/2505.12677) (Biswas, Roy, Roy — Purdue University).

This repo contains the original CURE implementation and two extensions that address its key limitations.

---

## Repository Structure

```
├── cure/           # Original CURE implementation (SD v1.4)
├── cure_seq/       # CURE-Sequential: interference-free sequential erasure
├── cure_dit/       # CURE-DiT: concept erasure for SD3 (MM-DiT)
```

### `cure/` — Original Implementation

The base CURE method: training-free concept erasure for Stable Diffusion v1.4 via SVD of CLIP text embeddings and closed-form weight edits to cross-attention Wk/Wv. Erases a concept in ~2 seconds.

### `cure_seq/` — CURE-Sequential

**Problem:** Sequential CURE edits accumulate cross-term interference, which primarily appears as collateral degradation on untargeted generations. In the paper, perceptual divergence starts around 50 erasures, while stronger untargeted interference appears beyond ~100 (Figure 6).

**Solution:** Orthogonalize each new concept's projector against all previously erased subspaces, guaranteeing `Pi @ Pj = 0` for all `i ≠ j`. This removes the interference source for the forget subspaces, enabling cleaner sequential composition.

See [`cure_seq/README.md`](cure_seq/README.md) for details.

### `cure_dit/` — CURE-DiT

**Problem:** CURE only works on SD v1.4's UNet architecture. SD3 and Flux use MM-DiT (Multi-Modal Diffusion Transformer) with joint attention — no cross-attention layers to target.

**Solution:** Identify the analogous text-stream projections (`add_k_proj`/`add_v_proj`) in SD3's `JointTransformerBlock` and apply CURE's spectral projection in the 1152-dim context space.

See [`cure_dit/README.md`](cure_dit/README.md) for details.

---

## Paper Summary

See [`CURE_PAPER_SUMMARY.md`](CURE_PAPER_SUMMARY.md) for a detailed breakdown of the original CURE paper including the algorithm, equations, experimental results, and limitations.

---

## Shared Evaluation Protocol

For cross-branch re-baselining with a unified config and JSON result schema, use:

```bash
python evaluation/run_shared_eval.py --branch cure --concept-set objects10
python evaluation/run_shared_eval.py --branch cure_seq --concept-set objects10 --erasure-mode sequential
python evaluation/run_shared_eval.py --branch cure_dit --concept-set objects10
```

Details: [`evaluation/README.md`](evaluation/README.md)
