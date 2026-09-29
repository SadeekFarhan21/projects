"""
Spectral operations for CURE-Sequential.

Extends CURE's spectral.py with orthogonalization-aware projector computation.
The key new function is compute_discriminative_projector_orth(), which:
  1. SVDs the forget embeddings
  2. Orthogonalizes the right singular vectors against the SubspaceBank
  3. Recomputes effective singular values in the orthogonalized subspace
  4. Optionally boosts alpha to compensate for energy loss
  5. Builds and returns the discriminative projector + orthogonalized basis for registration

Original CURE functions (compute_svd, spectral_expansion, build_projector,
compute_discriminative_projector) are preserved unchanged for use as baselines.
"""

import torch
from typing import Optional, Tuple
from .subspace_bank import SubspaceBank


SPECTRAL_MODES = ("tikhonov", "gavish_donoho")


# ── Original CURE functions (unchanged) ──────────────────────────────────────

def compute_svd(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD on embedding matrix. Returns U [n,k], S [k], Vh [k,d]."""
    return torch.linalg.svd(embeddings, full_matrices=False)


def _matrix_aspect_ratio(matrix: torch.Tensor) -> float:
    """Return beta=min(m,n)/max(m,n) in (0,1]."""
    m, n = matrix.shape
    bigger = max(m, n)
    if bigger == 0:
        return 1.0
    return min(m, n) / bigger


def _gavish_donoho_lambda_star(beta: float) -> float:
    """Optimal hard-threshold coefficient lambda*(beta)."""
    beta = float(max(min(beta, 1.0), 1e-8))
    numerator = 8.0 * beta
    denom = beta + 1.0 + (beta * beta + 14.0 * beta + 1.0) ** 0.5
    return (2.0 * (beta + 1.0) + (numerator / denom)) ** 0.5


def spectral_expansion(
    singular_values: torch.Tensor,
    alpha: float,
    mode: str = "tikhonov",
    aspect_ratio: Optional[float] = None,
) -> torch.Tensor:
    """
    Tikhonov-inspired spectral expansion (CURE Equation 4).
    f(ri; α) = α*ri / ((α-1)*ri + 1)   where ri = σi² / Σ σj²
    """
    if mode not in SPECTRAL_MODES:
        raise ValueError(f"Unknown spectral mode '{mode}'. Choose from {SPECTRAL_MODES}")

    if singular_values.numel() == 0:
        return singular_values

    if mode == "gavish_donoho":
        beta = 1.0 if aspect_ratio is None else aspect_ratio
        lambda_star = _gavish_donoho_lambda_star(beta)
        tau = lambda_star * torch.median(singular_values)
        return (singular_values >= tau).to(dtype=singular_values.dtype)

    sigma_sq = singular_values ** 2
    total_energy = sigma_sq.sum()
    r = sigma_sq / (total_energy + 1e-10)
    return (alpha * r) / ((alpha - 1) * r + 1)


def build_projector(
    U: torch.Tensor,
    singular_values: torch.Tensor,
    alpha: float,
    mode: str = "tikhonov",
    aspect_ratio: Optional[float] = None,
) -> torch.Tensor:
    """
    Build energy-scaled projection matrix (CURE Equation 3).
    P = U @ diag(f(ri;α)) @ U.T

    Args:
        U: Column basis [hidden_dim, k]  (i.e. Vhf.T)
        singular_values: [k]
        alpha: Spectral expansion parameter
    Returns:
        Projector [hidden_dim, hidden_dim]
    """
    lambda_diag = spectral_expansion(
        singular_values,
        alpha,
        mode=mode,
        aspect_ratio=aspect_ratio,
    )
    scaled_U = U * lambda_diag.unsqueeze(0)   # [hidden_dim, k]
    return scaled_U @ U.T                      # [hidden_dim, hidden_dim]


def compute_discriminative_projector(
    forget_embeddings: torch.Tensor,
    retain_embeddings: Optional[torch.Tensor],
    alpha: float,
    spectral_mode: str = "tikhonov",
) -> torch.Tensor:
    """Original CURE discriminative projector. Pdis = Pf - Pf @ Pr."""
    _, Sf, Vhf = compute_svd(forget_embeddings)
    Pf = build_projector(
        Vhf.T,
        Sf,
        alpha,
        mode=spectral_mode,
        aspect_ratio=_matrix_aspect_ratio(forget_embeddings),
    )

    if retain_embeddings is not None and retain_embeddings.shape[0] > 0:
        _, Sr, Vhr = compute_svd(retain_embeddings)
        Pr = build_projector(
            Vhr.T,
            Sr,
            alpha,
            mode=spectral_mode,
            aspect_ratio=_matrix_aspect_ratio(retain_embeddings),
        )
        return Pf - Pf @ Pr

    return Pf


# ── New CURE-Sequential functions ─────────────────────────────────────────────

def compute_discriminative_projector_orth(
    forget_embeddings: torch.Tensor,
    retain_embeddings: Optional[torch.Tensor],
    alpha: float,
    bank: SubspaceBank,
    adaptive_alpha: bool = True,
    alpha_max: float = 10.0,
    spectral_mode: str = "tikhonov",
) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
    """
    Compute discriminative projector with subspace orthogonalization.

    Same output as compute_discriminative_projector(), but the forget subspace
    is first projected onto the orthogonal complement of all previously erased
    subspaces (via the SubspaceBank). This guarantees that the resulting
    projector Pdis is orthogonal to all prior projectors, eliminating or strongly
    suppressing cross-term interference in sequential unlearning.

    Args:
        forget_embeddings: Forget concept embeddings [n, hidden_dim]
        retain_embeddings: Retain concept embeddings [m, hidden_dim] or None
        alpha: Spectral expansion parameter (base value)
        bank: SubspaceBank tracking all previously erased subspaces
        adaptive_alpha: If True, boost alpha when significant energy is lost to
                        orthogonalization (compensates for reduced subspace coverage)
        alpha_max: Cap for adaptive alpha scaling

    Returns:
        Pdis: Discriminative projector [hidden_dim, hidden_dim]
        Vhf_orth: Orthogonalized forget basis [k', hidden_dim] — register into bank after use
        energy_retained: Fraction of concept energy that survived orthogonalization
        lambda_diag: Spectral weights for the orthogonalized basis (or None if empty)
    """
    # 1. SVD on forget embeddings
    _, Sf, Vhf = compute_svd(forget_embeddings)

    # 2. Orthogonalize against bank (core new step)
    Vhf_orth, S_eff = bank.orthogonalize(Vhf, forget_embeddings)

    # 3. Compute fraction of energy retained
    energy_retained = bank.compute_energy_retained(Vhf, Vhf_orth, forget_embeddings)

    # 4. Adaptive alpha: boost if significant energy was consumed by orthogonalization
    alpha_effective = alpha
    if adaptive_alpha and energy_retained < 1.0 and energy_retained > 0.0:
        # Scale alpha inversely to energy retained, capped at alpha_max
        boost = 1.0 / max(energy_retained, 0.1)
        alpha_effective = min(alpha * boost, alpha_max)
        if abs(alpha_effective - alpha) > 0.1:
            print(f"  Adaptive alpha: {alpha:.1f} → {alpha_effective:.2f} "
                  f"(energy retained: {energy_retained:.1%})")

    # 5. Handle fully-consumed concept (no dimensions left)
    if Vhf_orth.shape[0] == 0:
        hidden_dim = forget_embeddings.shape[1]
        return (
            torch.zeros(hidden_dim, hidden_dim, device=forget_embeddings.device),
            Vhf_orth,
            0.0,
            None,
        )

    # 6. Move Vhf_orth to same device/dtype as embeddings for projector computation
    dev = forget_embeddings.device
    Vhf_orth_dev = Vhf_orth.to(device=dev, dtype=forget_embeddings.dtype)
    S_eff_dev = S_eff.to(device=dev, dtype=forget_embeddings.dtype)

    # 7. Build forget projector from orthogonalized subspace
    Pf = build_projector(
        Vhf_orth_dev.T,
        S_eff_dev,
        alpha_effective,
        mode=spectral_mode,
        aspect_ratio=_matrix_aspect_ratio(forget_embeddings),
    )

    # 8. Handle retain (unchanged — retain is NOT orthogonalized against bank)
    if retain_embeddings is not None and retain_embeddings.shape[0] > 0:
        _, Sr, Vhr = compute_svd(retain_embeddings)
        Pr = build_projector(
            Vhr.T,
            Sr,
            alpha,
            mode=spectral_mode,
            aspect_ratio=_matrix_aspect_ratio(retain_embeddings),
        )
        Pdis = Pf - Pf @ Pr
    else:
        Pdis = Pf

    # Also return lambda_diag so caller can filter significant dirs for bank registration
    lambda_diag = spectral_expansion(
        S_eff_dev,
        alpha_effective,
        mode=spectral_mode,
        aspect_ratio=_matrix_aspect_ratio(forget_embeddings),
    )

    return Pdis, Vhf_orth, energy_retained, lambda_diag
