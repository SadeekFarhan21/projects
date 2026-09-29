Commands (from project root, Python 3.x with torch; the venv also had pillow, diffusers, transformers because `cure_seq/__init__` imports them):
- `python results/figure6_deltas.py` -> figure6_deltas.json / .log (stdlib only)
- `python results/synthetic_orthogonality.py` -> synthetic_orthogonality.json / .log (torch 2.10.0, CPU, seconds)

Synthetic caveat: random 768-d concepts are already nearly orthogonal, so base projectors have small cross terms (Frobenius ~0.016-0.024) anyway; the orth version reduces them to ~1e-6. This confirms the construction (Pi Pj ~ 0 up to float error) and bank sizing (8 dims/concept here), not real-CLIP behavior.
