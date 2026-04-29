"""
Stain-specific losses:
  - DAB top-percentile intensity matching (mean of top-k% pixels)
  - DAB histogram KL supervision

DAB intensity is extracted via Beer–Lambert color deconvolution.
Operates in image space on decoded IHC outputs.
Both losses are misalignment-tolerant: they use aggregate statistics
(distribution-level) rather than per-pixel correspondence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Beer–Lambert color deconvolution stain matrix for H-DAB ──────────────────
# Standard OD stain vectors: Hematoxylin and DAB (from Ruifrok & Johnston 2001)
_HE_STAIN_MATRIX = torch.tensor([
    [0.6500286, 0.7044536, 0.2860126],   # Hematoxylin
    [0.2684200, 0.5706605, 0.7765708],   # DAB
], dtype=torch.float32)


def rgb_to_dab(images: torch.Tensor) -> torch.Tensor:
    """Extract DAB optical density channel from RGB images.

    Args:
        images: [B, 3, H, W] in [-1, 1] or [0, 1]

    Returns:
        dab_od: [B, H, W]  DAB optical density (higher = more DAB staining)
    """
    # Ensure [0, 1]
    if images.min() < -0.1:
        images = (images + 1.0) / 2.0

    eps = 1e-6
    # Optical density
    od = -torch.log(images.clamp(min=eps))  # [B, 3, H, W]

    # Project onto stain matrix using pseudo-inverse
    stain = _HE_STAIN_MATRIX.to(images.device)  # [2, 3]
    # Solve: od ≈ conc @ stain  (least squares per pixel)
    # stain_pinv: [3, 2]
    stain_pinv = torch.linalg.pinv(stain)  # [3, 2]

    B, C, H, W = od.shape
    od_flat = od.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
    conc = od_flat @ stain_pinv  # [B*H*W, 2]
    dab = conc[:, 1].reshape(B, H, W).clamp(min=0.0)
    return dab


def dab_top_percentile(dab: torch.Tensor, percentile: float = 0.10) -> torch.Tensor:
    """Mean of the top-k% DAB intensities per image.

    Args:
        dab:        [B, H, W]
        percentile: fraction of pixels to average (0.10 = top 10%)

    Returns:
        scores: [B]
    """
    B = dab.shape[0]
    flat = dab.reshape(B, -1)             # [B, H*W]
    k = max(1, int(flat.shape[1] * percentile))
    topk, _ = flat.topk(k, dim=1)
    return topk.mean(dim=1)


def dab_histogram(dab: torch.Tensor, bins: int = 256, dab_max: float = 3.0) -> torch.Tensor:
    """Soft per-image DAB histogram using Gaussian kernel density.

    Args:
        dab:    [B, H, W]
        bins:   number of histogram bins
        dab_max: upper bound for OD range

    Returns:
        hist: [B, bins] normalized histogram (sums to 1 per image)
    """
    B = dab.shape[0]
    flat = dab.reshape(B, -1).unsqueeze(2)  # [B, N, 1]

    edges = torch.linspace(0, dab_max, bins, device=dab.device).unsqueeze(0).unsqueeze(0)  # [1, 1, bins]
    bw = dab_max / bins
    weights = torch.exp(-0.5 * ((flat - edges) / bw) ** 2)  # [B, N, bins]
    hist = weights.sum(dim=1)  # [B, bins]
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
    return hist


class DABLoss(nn.Module):
    """DAB intensity and histogram matching losses.

    Both operate on aggregate statistics → misalignment-tolerant.
    """

    def __init__(
        self,
        top_percentile: float = 0.10,
        histogram_bins: int = 256,
        histogram_weight: float = 0.5,
    ):
        super().__init__()
        self.top_percentile = top_percentile
        self.histogram_bins = histogram_bins
        self.histogram_weight = histogram_weight

    def forward(
        self,
        pred_images: torch.Tensor,
        real_images: torch.Tensor,
        reduce: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pred_images: [B, 3, H, W] in [-1, 1] generated IHC
            real_images: [B, 3, H, W] in [-1, 1] real IHC
            reduce:      if True return scalar, else [B]

        Returns:
            per-sample or mean DAB loss
        """
        dab_pred = rgb_to_dab(pred_images)
        dab_real = rgb_to_dab(real_images)

        # Top-percentile L1
        tp_pred = dab_top_percentile(dab_pred, self.top_percentile)
        tp_real = dab_top_percentile(dab_real, self.top_percentile)
        tp_loss = (tp_pred - tp_real).abs()

        # Histogram KL
        hist_pred = dab_histogram(dab_pred, self.histogram_bins)
        hist_real = dab_histogram(dab_real, self.histogram_bins)
        eps = 1e-8
        kl_loss = F.kl_div(
            (hist_pred + eps).log(), hist_real + eps, reduction="none"
        ).sum(dim=-1)  # [B]

        per_sample = tp_loss + self.histogram_weight * kl_loss
        return per_sample.mean() if reduce else per_sample
