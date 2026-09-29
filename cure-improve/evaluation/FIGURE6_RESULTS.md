# Figure-6 Style Metrics — Results

Run date: 2026-03-07
Script: `evaluation/paper_figure6_metrics.py`
Full runbook: `evaluation/FIGURE6_RUNBOOK.md`

## Run Configuration

| Parameter | Value |
|---|---|
| Methods | `cure`, `cure_seq` |
| Erased artists | 50 (see `artist_lists/erased_artists_50.txt`) |
| Unerased artists | 10 (see `artist_lists/unerased_artists_10.txt`) |
| Checkpoints | 1, 5, 10, 25, 50 |
| Seeds | 11, 22, 33 |
| Steps | 20 |
| Guidance | 7.5 |
| Resolution | 384×384 |
| Embedding mode | `mean_masked` |
| Spectral mode | `tikhonov` |
| Alpha | 2.0 |
| Max prompts / group | 180 |
| Model | CompVis/stable-diffusion-v1-4 |

## Results

### CURE (naive sequential)

| # Erased | LPIPSe (↑) | LPIPSu (↓) | CLIP_u (↑) |
|---:|---:|---:|---:|
| 1  | 0.7324 ± 0.1028 | 0.7094 ± 0.0876 | 25.21 ± 2.90 |
| 5  | 0.7746 ± 0.0823 | 0.7735 ± 0.0641 | 21.29 ± 3.36 |
| 10 | 0.7943 ± 0.0925 | 0.7701 ± 0.0696 | 20.25 ± 3.33 |
| 25 | 0.7660 ± 0.0803 | 0.7593 ± 0.0719 | 21.62 ± 2.83 |
| 50 | 0.7869 ± 0.0907 | 0.7612 ± 0.0783 | 21.52 ± 3.06 |

### CURE-Sequential (orthogonal projector composition)

| # Erased | LPIPSe (↑) | LPIPSu (↓) | CLIP_u (↑) |
|---:|---:|---:|---:|
| 1  | 0.7324 ± 0.1028 | 0.7094 ± 0.0876 | 25.21 ± 2.90 |
| 5  | 0.7587 ± 0.0746 | 0.7375 ± 0.0630 | 22.68 ± 3.09 |
| 10 | 0.7678 ± 0.0726 | 0.7397 ± 0.0635 | 22.48 ± 2.95 |
| 25 | 0.7570 ± 0.0897 | 0.7350 ± 0.0694 | 22.66 ± 2.89 |
| 50 | 0.7577 ± 0.0831 | 0.7470 ± 0.0734 | 22.16 ± 3.01 |

### Delta (cure_seq − cure)

Negative LPIPSu and positive CLIP_u both favor cure_seq.

| # Erased | ΔLPIPSu | ΔCLIP_u |
|---:|---:|---:|
| 1  | 0.000 | 0.00 |
| 5  | **−0.036** | **+1.39** |
| 10 | **−0.030** | **+2.23** |
| 25 | **−0.024** | **+1.04** |
| 50 | **−0.014** | **+0.64** |

## Interpretation

- At k=1 both methods are identical (first erasure is always the same).
- From k=5 onward, **CURE-seq consistently preserves unerased styles better** (lower LPIPSu)
  and maintains stronger text-image alignment (higher CLIP_u).
- The largest gap occurs at **k=10** (ΔCLIP_u = +2.23), consistent with interference
  accumulating under naive sequential application.
- CURE produces slightly higher LPIPSe (stronger perceptual divergence on erased artists)
  but at the cost of greater collateral drift — CURE-seq is more surgical.
- The gap narrows at k=25 and k=50, suggesting orthogonal composition partially saturates
  the 768-dim CLIP subspace budget at scale. Extending to k=100 would better reproduce
  the full Figure-6 degradation curve from the paper.

## Caveats

- Proxy metrics only (LPIPS + CLIP), not a full benchmark.
- n=3 seeds — confidence intervals are approximate.
- 20 diffusion steps and 384×384 resolution reduce absolute metric precision.
- Paper's Figure 6 inflection point is ~50–100 concepts; extending the run to k=100
  is recommended before final claims.
