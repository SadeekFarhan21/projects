"""
SequentialCURE: Sequential concept unlearning with orthogonal projector composition.

Extends the original CURE class with SubspaceBank-based orthogonalization.
Each new concept's discriminative projector is guaranteed to be orthogonal
to all previously applied projectors, removing the cross-term interference
source that causes collateral degradation on untargeted content in naive
sequential CURE (Figure 6 of the paper: divergence starts ~50, stronger beyond ~100).

Usage:
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    eraser = SequentialCURE(pipe)

    # Erase concepts one by one — no interference
    eraser.erase_concept(["car", "automobile", "vehicle"], concept_name="car")
    eraser.erase_concept(["truck", "lorry"], concept_name="truck")
    eraser.erase_concept(["Van Gogh", "van gogh style"], concept_name="van_gogh")
    ...

    # Check how much budget remains
    print(eraser.bank.summary())
"""

import torch
import time
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline

from .subspace_bank import SubspaceBank
from .spectral import (
    compute_discriminative_projector_orth,
    SPECTRAL_MODES,
)

# Re-use original CURE's attention utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cure.attention import (
    get_cross_attention_layers,
    apply_weight_update,
    count_cross_attention_layers,
)
from cure.utils import aggregate_embeddings, EMBEDDING_MODES


class SequentialCURE:
    """
    Sequential concept unlearning with orthogonal projector composition.

    Wraps the original CURE algorithm and adds a SubspaceBank that tracks
    the cumulative erased subspace. Each call to erase_concept() produces a
    projector orthogonal to all prior erasures, guaranteeing:

        Pi_orth @ Pj_orth = 0   for all i != j

    So the total edit after n concepts is:
        W_n = W0 @ (I - P1_orth - P2_orth - ... - Pn_orth)

    — a single clean projection with cross-term interference eliminated or strongly suppressed.
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        device: Optional[str] = None,
        hidden_dim: int = 768,
        orth_threshold: float = 1e-3,
        embedding_mode: str = "mean_masked",
        spectral_mode: str = "tikhonov",
    ):
        self.pipe = pipe

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.pipe = self.pipe.to(self.device)

        if embedding_mode not in EMBEDDING_MODES:
            raise ValueError(f"Unknown embedding_mode '{embedding_mode}'. Choose from {EMBEDDING_MODES}")
        self.embedding_mode = embedding_mode
        if spectral_mode not in SPECTRAL_MODES:
            raise ValueError(f"Unknown spectral_mode '{spectral_mode}'. Choose from {SPECTRAL_MODES}")
        self.spectral_mode = spectral_mode

        # Core new component
        self.bank = SubspaceBank(hidden_dim=hidden_dim, orth_threshold=orth_threshold)

        # History
        self.erased_concepts: List[str] = []
        self._original_weights: Optional[list] = None

    # ── Text embeddings ────────────────────────────────────────────────────────

    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """CLIP embeddings for subspace computation, aggregated by embedding_mode."""
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        with torch.no_grad():
            embeddings = text_encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state  # [b, 77, 768]

        return aggregate_embeddings(
            embeddings.float(), attention_mask, mode=self.embedding_mode
        )

    # ── Weight save/restore ───────────────────────────────────────────────────

    def save_original_weights(self) -> None:
        self._original_weights = [
            {"Wk": l.to_k.weight.data.clone(), "Wv": l.to_v.weight.data.clone()}
            for l in get_cross_attention_layers(self.pipe.unet)
        ]

    def restore_original_weights(self) -> None:
        if self._original_weights is None:
            raise ValueError("No weights saved. Call save_original_weights() first.")
        for layer, w in zip(get_cross_attention_layers(self.pipe.unet), self._original_weights):
            layer.to_k.weight.data = w["Wk"].clone()
            layer.to_v.weight.data = w["Wv"].clone()
        self.bank = SubspaceBank(hidden_dim=self.bank.hidden_dim)
        self.erased_concepts = []
        print("Weights restored. Bank cleared.")

    # ── Main erasure (orthogonalized) ─────────────────────────────────────────

    def erase_concept(
        self,
        forget_prompts: List[str],
        retain_prompts: Optional[List[str]] = None,
        alpha: float = 2.0,
        concept_name: Optional[str] = None,
        save_original: bool = True,
        adaptive_alpha: bool = True,
    ) -> dict:
        """
        Erase a concept using an orthogonalized spectral projector.

        The resulting projector is guaranteed orthogonal to all prior projectors,
        so sequential application eliminates or strongly suppresses cross-term interference.

        Args:
            forget_prompts: Prompts describing the concept to erase
            retain_prompts: Optional prompts describing concepts to preserve
            alpha: Base spectral expansion parameter (adaptive alpha may boost this)
            concept_name: Label for tracking (defaults to first forget_prompt)
            save_original: Save weights before first erasure for restoration
            adaptive_alpha: Boost alpha when orthogonalization reduces concept energy

        Returns:
            Dict with erasure stats: n_dims_consumed, energy_retained, alpha_effective,
                                     elapsed_s, budget_remaining
        """
        if concept_name is None:
            concept_name = forget_prompts[0]

        if save_original and self._original_weights is None:
            self.save_original_weights()

        # Budget check
        if self.bank.remaining_budget == 0:
            raise RuntimeError(
                f"Subspace budget exhausted ({self.bank.hidden_dim} dims used). "
                "Cannot erase more concepts without restoring weights."
            )

        start = time.time()

        # Step 1: Get embeddings
        forget_emb = self.get_text_embeddings(forget_prompts)
        retain_emb = self.get_text_embeddings(retain_prompts) if retain_prompts else None

        # Step 2: Compute orthogonalized discriminative projector
        Pdis, Vhf_orth, energy_retained, lambda_diag = compute_discriminative_projector_orth(
            forget_embeddings=forget_emb,
            retain_embeddings=retain_emb,
            alpha=alpha,
            bank=self.bank,
            adaptive_alpha=adaptive_alpha,
            spectral_mode=self.spectral_mode,
        )

        # Step 3: Apply to all cross-attention Wk/Wv (same as original CURE)
        n_layers = 0
        for layer in get_cross_attention_layers(self.pipe.unet):
            apply_weight_update(layer, Pdis, device=self.device)
            n_layers += 1

        elapsed = time.time() - start

        # Step 4: Register only significant directions into bank
        # (filters out near-zero spectral weight directions to preserve budget)
        self.bank.add_concept(
            concept_name, Vhf_orth, energy_retained,
            lambda_diag=lambda_diag.cpu() if lambda_diag is not None else None,
        )
        dims_consumed = self.bank.concepts[-1].n_dims  # actual dims registered (after filtering)
        self.erased_concepts.append(concept_name)

        stats = {
            "concept": concept_name,
            "n_dims_consumed": dims_consumed,
            "energy_retained": energy_retained,
            "elapsed_s": elapsed,
            "budget_remaining": self.bank.remaining_budget,
            "total_erased": len(self.erased_concepts),
        }

        print(
            f"[{len(self.erased_concepts):3d}] Erased '{concept_name}' | "
            f"dims={dims_consumed} | energy={energy_retained:.1%} | "
            f"budget left={self.bank.remaining_budget}/768 | "
            f"t={elapsed:.2f}s"
        )

        return stats

    # ── Generation (identical to CURE) ────────────────────────────────────────

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )
        return output.images

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def bank_summary(self) -> str:
        return self.bank.summary()

    def __repr__(self) -> str:
        n_layers = count_cross_attention_layers(self.pipe.unet)
        return (
            f"SequentialCURE("
            f"device={self.device}, "
            f"layers={n_layers}, "
            f"erased={len(self.erased_concepts)}, "
            f"budget={self.bank.remaining_budget}/768, "
            f"spectral_mode={self.spectral_mode})"
        )
