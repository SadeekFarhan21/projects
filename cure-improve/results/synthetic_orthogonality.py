"""Weight-free check of CURE-Sequential's orthogonality claim on SYNTHETIC embeddings.

Added by the blog port (not part of the original repo). No Stable Diffusion, no CLIP:
each "concept" is a random low-rank cluster in R^768 (rank 8 signal + small noise,
40 samples), standing in for CLIP token embeddings. This only tests the linear algebra
(Pi @ Pj = 0 by construction), not image quality.

Run from the project root:  python results/synthetic_orthogonality.py
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from cure_seq.spectral import compute_discriminative_projector, compute_discriminative_projector_orth
from cure_seq.subspace_bank import SubspaceBank

D, RANK, N, ALPHA = 768, 8, 40, 2.0
torch.manual_seed(0)

def concept(rng):
    basis = torch.randn(RANK, D, generator=rng)
    coef = torch.randn(N, RANK, generator=rng)
    return coef @ basis + 0.05 * torch.randn(N, D, generator=rng)

def offdiag_stats(Ps):
    vals = [torch.linalg.matrix_norm(Ps[i] @ Ps[j]).item()
            for i in range(len(Ps)) for j in range(len(Ps)) if i != j]
    return max(vals), sum(vals) / len(vals)

out = []
W0 = torch.randn(320, D, generator=torch.Generator().manual_seed(1))
for K in (2, 5, 10, 20, 40):
    rng = torch.Generator().manual_seed(123)
    embs = [concept(rng) for _ in range(K)]
    P_base = [compute_discriminative_projector(e, None, ALPHA) for e in embs]
    bank = SubspaceBank(hidden_dim=D)
    P_orth = []
    import io, contextlib
    for k, e in enumerate(embs):
        with contextlib.redirect_stdout(io.StringIO()):  # silence adaptive-alpha prints
            P, Vo, er, lam = compute_discriminative_projector_orth(e, None, ALPHA, bank)
        bank.add_concept(f"c{k}", Vo, er, lambda_diag=None if lam is None else lam)
        P_orth.append(P)
    mb, ma = offdiag_stats(P_base)
    ob, oa = offdiag_stats(P_orth)
    # naive sequential cross-term (i=0,j=1): ||W0 P0 P1||_F for base projectors vs orth projectors
    cross_base = torch.linalg.matrix_norm(W0 @ P_base[0] @ P_base[1]).item()
    cross_orth = torch.linalg.matrix_norm(W0 @ P_orth[0] @ P_orth[1]).item()
    out.append(dict(n_concepts=K, base_max_offdiag_fro=mb, base_mean_offdiag_fro=ma,
                    orth_max_offdiag_fro=ob, orth_mean_offdiag_fro=oa,
                    cross_term_W0P0P1_base=cross_base, cross_term_W0P0P1_orth=cross_orth,
                    bank_dims_used=bank.dims_used, bank_budget_fraction=bank.budget_fraction_used))
    print(out[-1])
json.dump(dict(setup=dict(hidden_dim=D, concept_rank=RANK, samples_per_concept=N, alpha=ALPHA,
                          data="synthetic random low-rank Gaussian, seed 0/123"), runs=out),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_orthogonality.json"), "w"), indent=2)
