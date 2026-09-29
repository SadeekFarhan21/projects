# Shared Evaluation Protocol

This folder standardizes experimental runs across:
- `cure` (SD1.4 base CURE)
- `cure_seq` (sequential orthogonalized CURE)
- `cure_dit` (SD3/MM-DiT port)

## Files

- `protocol.py`: Shared concept sets, alpha rules, config helpers, JSON writer.
- `run_shared_eval.py`: Unified runner that generates before/after samples, applies erasure, and writes one result schema.

## Result Schema

Each run writes:
- `run_config.json`: Full run configuration snapshot.
- `results.partial.json`: Incremental checkpoint while running.
- `results.json`: Final output.

Top-level fields in `results.json`:
- `schema_version`
- `created_at_utc`
- `run_id`
- `git_commit`
- `config`
- `environment`
- `concept_results`
- `summary`

Each `concept_results[*]` entry includes:
- concept metadata (`concept`, `alpha`, prompt counts)
- `before`: image paths + generation time
- `erase`: erase time + raw eraser stats
- `after`: image paths + generation time

## Example Commands

```bash
# Base CURE, isolated per-concept runs (default for cure)
python evaluation/run_shared_eval.py \
  --branch cure \
  --concept-set objects10 \
  --embedding-mode mean_masked \
  --samples-per-concept 2

# CURE-Sequential, accumulated sequential erasures
python evaluation/run_shared_eval.py \
  --branch cure_seq \
  --concept-set objects10 \
  --erasure-mode sequential \
  --embedding-mode mean_masked

# CURE-DiT on SD3 (requires model access)
python evaluation/run_shared_eval.py \
  --branch cure_dit \
  --concept-set objects10 \
  --model-id stabilityai/stable-diffusion-3.5-medium \
  --embedding-mode mean_masked
```

## Notes

- `mean_masked` is the default embedding aggregation mode.
- Use `--embedding-mode token_flat` only for ablation.
- `cure_seq` supports only `sequential` mode.

## Figure-6 Style Metrics (LPIPS + CLIP)

Detailed runbook:
- `evaluation/FIGURE6_RUNBOOK.md`

To reproduce the paper-style sequential-degradation analysis (Figure-6-style curve), use:

```bash
python evaluation/paper_figure6_metrics.py \
  --methods "cure,cure_seq" \
  --erased-concepts-file evaluation/artist_lists/erased_artists_50.txt \
  --unerased-artists-file evaluation/artist_lists/unerased_artists_10.txt \
  --checkpoints "1,5,10,25,50" \
  --seeds "11,22,33" \
  --alpha 2.0 \
  --steps 20 \
  --height 384 \
  --width 384 \
  --embedding-mode mean_masked \
  --spectral-mode tikhonov \
  --max-prompts-per-group 180 \
  --cache-dir ./models \
  --output-dir <OUTPUT_DIR>/figure6_eval_50
```

Provided lists:
- `evaluation/artist_lists/erased_artists_50.txt`: 50 erase targets for scalability stress.
- `evaluation/artist_lists/erased_artists_100.txt`: 100 erase targets for high-load extension.
- `evaluation/artist_lists/unerased_artists_10.txt`: hold-out styles for preservation tracking.

100-artist extension command:

```bash
python evaluation/paper_figure6_metrics.py \
  --methods "cure,cure_seq" \
  --erased-concepts-file evaluation/artist_lists/erased_artists_100.txt \
  --unerased-artists-file evaluation/artist_lists/unerased_artists_10.txt \
  --checkpoints "1,5,10,25,50,100" \
  --seeds "11,22,33" \
  --alpha 2.0 \
  --steps 20 \
  --height 384 \
  --width 384 \
  --embedding-mode mean_masked \
  --spectral-mode tikhonov \
  --max-prompts-per-group 180 \
  --cache-dir ./models \
  --output-dir <OUTPUT_DIR>/figure6_eval_100
```

Outputs:
- `results.json`: full numeric payload
- `summary.md`: method-by-checkpoint table
- `plots/figure6_like_<method>.png`: dual-axis curve (`CLIP_u` and `LPIPSu`)
- `plots/lpips_u_comparison.png`: cross-method LPIPSu comparison
- each run is written to a unique timestamped folder under `--output-dir`

Metric conventions:
- `LPIPSe` (erased prompts): higher = stronger target divergence
- `LPIPSu` (unerased prompts): lower = less collateral perceptual drift
- `CLIP_u` (unerased prompts): higher = better prompt-image alignment

Install extras if needed:
- `lpips`
- `matplotlib`
