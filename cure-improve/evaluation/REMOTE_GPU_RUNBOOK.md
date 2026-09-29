# Remote GPU Runbook (Quick Proof Benchmark)

This runbook is for generating fast slide-ready evidence for:
1. Base SD1.4 CURE implementation works
2. CURE-Sequential does better over multiple erasures
3. Numeric comparison table between methods

The benchmark script:
- `evaluation/quick_proof_benchmark.py`

Outputs:
- `outputs/quick_proof/<timestamp>/results.json`
- `outputs/quick_proof/<timestamp>/summary.md`

---

## 1) Environment Setup

```bash
git clone git@github.com:arses-ui/Diffusion-Concept-Removal.git
cd Diffusion-Concept-Removal

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch diffusers transformers accelerate Pillow
```

If your GPU has CUDA, ensure the installed PyTorch build matches CUDA.

---

## 2) Run the Benchmark (Fast Preset)

```bash
python evaluation/quick_proof_benchmark.py \
  --device cuda \
  --embedding-mode mean_masked \
  --concepts "car,dog,cat,french horn,golf ball,chain saw,cassette player,parachute,church,garbage truck" \
  --alpha 2.0 \
  --eval-every 2 \
  --steps 15 \
  --height 320 \
  --width 320 \
  --cache-dir ./models \
  --output-dir outputs/quick_proof
```

This is tuned for speed (<~1 hour on a typical remote GPU if model downloads are not too slow).

---

## 3) What to Put in Slides

Open:
- `outputs/quick_proof/<timestamp>/summary.md`

Use:
- **Single-Concept Delta** table for "base CURE works"
- **Final Multi-Concept Checkpoint** table for "CURE-Sequential better over time"

Interpretation:
- Lower target CLIP is better suppression on erased concepts.
- Lower retention drop vs baseline is better untargeted-quality preservation.

---

## 4) Optional: Higher-Quality (Slower) Run

Increase:
- `--steps 20`
- `--height 384 --width 384`
- `--eval-every 1` (more checkpoints)

---

## 5) Common Failure Modes

1. **Model download/auth issues**
- Ensure internet access and sufficient disk.

2. **CUDA OOM**
- Lower `--height/--width` to 256 or 320.
- Reduce concept count.

3. **Slow runtime**
- Keep `--steps 15`, `--eval-every 2`, `--height/--width 320`.
