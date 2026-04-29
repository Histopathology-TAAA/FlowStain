"""
Tissue-aware loss weighting from CONCH zero-shot tissue probabilities.

During training, CONCH classifies each H&E crop into a tissue category.
A per-sample weight is computed as a weighted average of per-class weights,
then blended back toward 1.0 proportionally to the prediction entropy.

This effectively upweights clinically meaningful tissue (invasive carcinoma,
benign epithelium) and downweights background/adipose/necrosis where IHC
signal is not diagnostically relevant and consecutive-section misalignment
is most harmful.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig

from ..foundation.conch_tissue import TISSUE_CLASSES


def build_tissue_weight_vector(cfg_weights: DictConfig) -> torch.Tensor:
    """Build a [num_classes] tensor of per-tissue weights from config."""
    weights = []
    for cls in TISSUE_CLASSES:
        w = cfg_weights.get(cls, 1.0)
        weights.append(float(w))
    return torch.tensor(weights, dtype=torch.float32)


class TissueWeightedLoss(torch.nn.Module):
    """Applies tissue-aware weighting to a per-sample loss tensor.

    Given CONCH tissue probabilities, computes:
        w_raw = sum_c p[c] * tissue_weights[c]
        uncertainty = normalized_entropy(p)
        w = (1 - entropy_blend * uncertainty) * clamp(w_raw, min, max)
                + entropy_blend * uncertainty * 1.0
    """

    def __init__(
        self,
        weight_vector: torch.Tensor,
        min_weight: float = 0.5,
        max_weight: float = 1.5,
        entropy_blend: float = 0.5,
    ):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.entropy_blend = entropy_blend
        self.register_buffer("weight_vector", weight_vector)

    def compute_weights(self, tissue_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tissue_probs: [B, num_classes]

        Returns:
            weights: [B]
        """
        w_vector = self.weight_vector.to(tissue_probs.device)
        w_raw = (tissue_probs * w_vector.unsqueeze(0)).sum(dim=-1)
        w_clamped = w_raw.clamp(self.min_weight, self.max_weight)

        # Normalized entropy [0, 1]
        eps = 1e-8
        H = -(tissue_probs * (tissue_probs + eps).log()).sum(dim=-1)
        H_max = math.log(tissue_probs.shape[-1])
        uncertainty = (H / H_max).clamp(0.0, 1.0)

        # Blend toward 1.0 under uncertainty
        alpha = self.entropy_blend * uncertainty
        weights = (1.0 - alpha) * w_clamped + alpha * 1.0
        return weights

    def forward(
        self,
        per_sample_loss: torch.Tensor,
        tissue_probs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            per_sample_loss: [B]  unreduced per-sample loss values
            tissue_probs:    [B, num_classes] or None (returns plain mean if None)

        Returns:
            scalar weighted mean loss
        """
        if tissue_probs is None:
            return per_sample_loss.mean()
        weights = self.compute_weights(tissue_probs).to(per_sample_loss.device)
        return (weights * per_sample_loss).mean()
