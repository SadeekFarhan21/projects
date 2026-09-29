# Consolidated Quick-Proof Summary

- Generated UTC: 2026-03-06T18:14:26.206033+00:00
- Unique runs summarized: 9
- Dedupe: 14 discovered paths -> 9 unique result files

## Final Checkpoint Comparison

| Run | Spectral | Order | Alpha | Concepts | Seeds | CURE target | Seq target | seq-cure target | CURE drop | Seq drop | seq-cure drop | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gd_forward/20260304T230141Z | gavish_donoho | forward | 2 | 10 | 3 | 24.5422 | 24.0468 | -0.4954 | 4.4331 | 3.1428 | -1.2903 | Seq wins both |
| gd_forward_20c_alpha_1.5/20260305T022207Z | gavish_donoho | forward | 1.5 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 | Seq wins both |
| gd_forward_20c_alpha_2.0/20260305T022752Z | gavish_donoho | forward | 2 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 | Seq wins both |
| gd_forward_20c_alpha_2.5/20260305T023334Z | gavish_donoho | forward | 2.5 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 | Seq wins both |
| gd_forward_20c_alpha_3.0/20260305T023918Z | gavish_donoho | forward | 3 | 20 | 5 | 26.7340 | 24.8841 | -1.8498 | 2.4178 | 1.9197 | -0.4981 | Seq wins both |
| gd_reverse_20c_v2/20260304T235936Z | gavish_donoho | reverse | 2 | 20 | 5 | 24.3610 | 26.1652 | 1.8042 | 4.2923 | 1.4540 | -2.8384 | Seq better retention |
| gd_reverse_20c/20260304T233749Z | gavish_donoho | reverse | 2 | 21 | 5 | 24.0288 | 25.8641 | 1.8354 | 5.7730 | 0.9727 | -4.8003 | Seq better retention (malformed concepts) |
| tikhonov_forward/20260304T225750Z | tikhonov | forward | 2 | 10 | 3 | 22.8087 | 24.0636 | 1.2550 | 5.7074 | 3.3838 | -2.3236 | Seq better retention |
| tikhonov_reverse/20260304T230402Z | tikhonov | reverse | 2 | 10 | 3 | 23.8079 | 23.5636 | -0.2443 | 3.5260 | 4.7717 | 1.2456 | Seq better suppression |

## Key Findings

1. Best Seq-on-both run: `gd_forward_20c_alpha_1.5/20260305T022207Z` (target delta -1.8498, drop delta -0.4981).
2. Strongest reverse-order retention improvement: `gd_reverse_20c_v2/20260304T235936Z` (seq-cure drop -2.8384).
3. GD forward (20 concepts, 5 seeds) is alpha-invariant across [1.5, 2.0, 2.5, 3.0].

## Notes

- `target` lower is better suppression.
- `drop` lower is better retention preservation.
- `seq-cure` values below 0 favor Sequential.
- CLIP scores here are fast proxy metrics, not final benchmark metrics.

## Canonical Result Paths

- `<OUTPUT_DIR>/outputs/quick_proof/gd_forward/20260304T230141Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_1.5/20260305T022207Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_2.0/20260305T022752Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_2.5/20260305T023334Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/gd_forward_20c_alpha_3.0/20260305T023918Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/gd_reverse_20c_v2/20260304T235936Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/gd_reverse_20c/20260304T233749Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/tikhonov_forward/20260304T225750Z/results.json`
- `<OUTPUT_DIR>/outputs/quick_proof/tikhonov_reverse/20260304T230402Z/results.json`
