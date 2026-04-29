"""
Validation metrics: SSIM, LPIPS, DAB Pearson/KL.
All are computed as per-sample scalars and later aggregated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..losses.stain_losses import rgb_to_dab, dab_top_percentile, dab_histogram


def compute_ssim_batch(pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Mean SSIM over a batch, computed via skimage per sample.

    Args:
        pred, real: [B, 3, H, W] in [-1, 1]

    Returns:
        [B] SSIM scores
    """
    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError:
        # Fallback: return zeros to avoid crash
        return torch.zeros(pred.shape[0])

    pred_np = ((pred.detach().cpu().clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    real_np = ((real.detach().cpu().clamp(-1, 1) + 1) / 2 * 255).byte().numpy()

    scores = []
    for p, r in zip(pred_np, real_np):
        p = p.transpose(1, 2, 0)  # [H, W, 3]
        r = r.transpose(1, 2, 0)
        s = _ssim(p, r, data_range=255, channel_axis=-1)
        scores.append(float(s))
    return torch.tensor(scores)


def compute_lpips_batch(pred: torch.Tensor, real: torch.Tensor, scale: int = 256) -> torch.Tensor:
    """Per-sample LPIPS at a given scale.

    Args:
        pred, real: [B, 3, H, W] in [-1, 1]
        scale: resize target for comparison

    Returns:
        [B]
    """
    try:
        import lpips
        _lpips_fn = lpips.LPIPS(net="vgg").to(pred.device).eval()
        p = F.interpolate(pred, (scale, scale), mode="bilinear", align_corners=False)
        r = F.interpolate(real, (scale, scale), mode="bilinear", align_corners=False)
        with torch.no_grad():
            dist = _lpips_fn(p, r).squeeze()
        if dist.dim() == 0:
            dist = dist.unsqueeze(0).expand(pred.shape[0])
        return dist.cpu()
    except ImportError:
        return torch.zeros(pred.shape[0])


def compute_dab_metrics_batch(
    pred: torch.Tensor,
    real: torch.Tensor,
    top_percentile: float = 0.10,
    histogram_bins: int = 256,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample DAB metrics.

    Returns dict with keys:
        pred_top_dab, real_top_dab, dab_kl  (all [B])
    """
    dab_pred = rgb_to_dab(pred)
    dab_real = rgb_to_dab(real)

    tp_pred = dab_top_percentile(dab_pred, top_percentile)
    tp_real = dab_top_percentile(dab_real, top_percentile)

    hist_pred = dab_histogram(dab_pred, histogram_bins)
    hist_real = dab_histogram(dab_real, histogram_bins)
    eps = 1e-8
    kl = F.kl_div(
        (hist_pred + eps).log(), hist_real + eps, reduction="none"
    ).sum(dim=-1)

    return {
        "pred_top_dab": tp_pred.cpu(),
        "real_top_dab": tp_real.cpu(),
        "dab_kl": kl.cpu(),
    }
