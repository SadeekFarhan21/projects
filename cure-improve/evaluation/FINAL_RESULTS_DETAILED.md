# Final Results (Detailed)

Date reviewed: March 6, 2026 (America/New_York)
Reviewer: Codex

## Data Sources Reviewed

Primary result files (canonical paths reviewed):

1. `<OUTPUT_DIR>/outputs/quick_proof/gd_forward/20260304T230141Z/results.json`
2. `<OUTPUT_DIR>/outputs/quick_proof/tikhonov_forward/20260304T225750Z/results.json`
3. `<OUTPUT_DIR>/outputs/quick_proof/tikhonov_reverse/20260304T230402Z/results.json`
4. `<OUTPUT_DIR>/outputs/quick_proof/gd_reverse_20c/20260304T233749Z/results.json` (older malformed 21-concept run)
5. `<OUTPUT_DIR>/outputs/quick_proof/gd_reverse_20c_v2/20260304T235936Z/results.json` (corrected 20-concept reverse run)
6. `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_1.5/20260305T022207Z/results.json`
7. `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_2.0/20260305T022752Z/results.json`
8. `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_2.5/20260305T023334Z/results.json`
9. `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_3.0/20260305T023918Z/results.json`

Note on duplicates:
- The same runs also exist under `<OUTPUT_DIR>/quick_proof/...`
- Hash checks showed copied files are byte-identical to the canonical `outputs/quick_proof/...` paths.

## What Each Parameter Means

### Run config parameters

1. `concepts`
- The original concept list supplied to the script.

2. `concepts_ordered`
- The actual erasure order used after applying `--concept-order`.
- `forward`: same as input order.
- `reverse`: reversed order.

3. `concept_order`
- Determines erase ordering (`forward` or `reverse`).
- This matters because sequential methods can be order-sensitive.

4. `alpha`
- Spectral strength parameter in Tikhonov mode.
- In current code, this is ignored for `gavish_donoho` mode.

5. `eval_every`
- Evaluate checkpoints every N erasures.
- Example: `eval_every=4` evaluates after 4, 8, 12, ... erasures and final step.

6. `seed` and `seeds`
- `seed` is fallback single-seed value.
- `seeds` is the explicit seed list used for aggregation.

7. `steps`
- Diffusion denoising steps used during image generation for scoring.

8. `guidance`
- CFG scale (`guidance_scale`) used during generation.

9. `height`, `width`
- Output image resolution used for evaluation images.

10. `device`
- Execution device (`cuda` in these runs).

11. `embedding_mode`
- Token aggregation mode before SVD (`mean_masked` here).

12. `spectral_mode`
- `tikhonov`: weighted spectral expansion using `alpha`.
- `gavish_donoho`: hard thresholding in singular spectrum.

13. `cache_dir`
- Hugging Face/diffusion model cache location.

14. `clip_model`
- CLIP backbone used for proxy scoring (`openai/clip-vit-base-patch32`).

### Output metric rubrics

1. `single_concept_before_mean/std`
- CLIP score for first concept prompt before any erasure.

2. `single_concept_after_mean/std`
- Same concept score after the full erase sequence.

3. `single_concept_delta_mean/std`
- `after - before` for first concept.
- Lower (more negative) means stronger suppression of that concept.

4. `baseline_retention_clip_mean/std`
- Safe-prompt CLIP score before erasure.

5. `checkpoints[*].target_clip_mean/std`
- Average CLIP alignment for erased-concept prompts at each checkpoint.
- Lower is interpreted as stronger suppression.

6. `checkpoints[*].retention_clip_mean/std`
- CLIP alignment for safe prompts at each checkpoint.
- Higher is generally better retention.

7. `checkpoints[*].retention_drop_vs_base_mean/std`
- `baseline_retention - current_retention`.
- Lower is better (less collateral damage on non-target prompts).

## Summary Table (Final Checkpoint Metrics)

Format:
- `seq-cure_target = seq_target - cure_target` (negative favors Sequential)
- `seq-cure_drop = seq_drop - cure_drop` (negative favors Sequential)

| Run | Spectral | Order | Alpha | Concepts | Seeds | CURE target | Seq target | seq-cure_target | CURE drop | Seq drop | seq-cure_drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tikhonov_forward | tikhonov | forward | 2.0 | 10 | 3 | 22.8087 | 24.0636 | 1.2550 | 5.7074 | 3.3838 | -2.3236 |
| tikhonov_reverse | tikhonov | reverse | 2.0 | 10 | 3 | 23.8079 | 23.5636 | -0.2443 | 3.5260 | 4.7717 | 1.2456 |
| gd_forward | gavish_donoho | forward | 2.0 | 10 | 3 | 24.5422 | 24.0468 | -0.4954 | 4.4331 | 3.1428 | -1.2903 |
| gd_reverse_20c_v2 | gavish_donoho | reverse | 2.0 | 20 | 5 | 24.3610 | 26.1652 | 1.8042 | 4.2923 | 1.4540 | -2.8384 |
| gd_forward_20c_alpha_1.5 | gavish_donoho | forward | 1.5 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 |
| gd_forward_20c_alpha_2.0 | gavish_donoho | forward | 2.0 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 |
| gd_forward_20c_alpha_2.5 | gavish_donoho | forward | 2.5 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 |
| gd_forward_20c_alpha_3.0 | gavish_donoho | forward | 3.0 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 |

## Key Findings

1. Best clean "Seq wins both metrics" setting in final 20-concept runs:
- `gavish_donoho + forward` (all alpha runs produce the same metrics).
- Sequential improves both suppression and retention-drop relative to CURE.

2. Reverse-order robustness (20 concepts):
- Sequential keeps much better retention drop, but suppression is weaker than CURE.
- This indicates a suppression/retention tradeoff under reverse ordering.

3. The malformed older run should not be used for claims:
- `gd_reverse_20c` had 21 concepts due split terms (`chain`, `saw`, `springer`).
- Use `gd_reverse_20c_v2` instead.

4. Alpha sweep interpretation:
- All GD forward alpha runs are numerically identical in outcome.
- This is expected with current implementation behavior: `alpha` is not used by `gavish_donoho`.

## What To Claim Safely

1. Safe primary claim:
- On 20 concepts / 5 seeds, under `gavish_donoho + forward`, CURE-Sequential outperforms naive CURE on both final target suppression and retention-drop.

2. Safe caveat:
- Under reverse order, Sequential strongly improves retention preservation but does not always maximize suppression relative to naive CURE.

3. Evaluation caveat:
- Metrics are CLIP proxy metrics, not final comprehensive benchmark metrics.

## Recommended Next Action (if no more training runs)

1. Use `gd_forward_20c_alpha_1.5` as canonical poster run (or any of 1.5/2.0/2.5/3.0 since they are identical under current GD mode).
2. Use `gd_reverse_20c_v2` as robustness/tradeoff panel.
3. Include explicit note that `alpha` does not influence current GD mode behavior.
