"""
FID and KID computation on a list of image tensors.
Wraps pytorch_fid for FID and uses torchmetrics for KID.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def compute_fid(
    pred_images: List[torch.Tensor],
    real_images: List[torch.Tensor],
    device: torch.device = None,
) -> float:
    """Compute FID between two lists of [3, H, W] tensors in [-1, 1].

    Concatenates all images, computes Inception activations,
    and returns FID.
    """
    try:
        from pytorch_fid.inception import InceptionV3
        from pytorch_fid.fid_score import calculate_frechet_distance

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

        def get_activations(images: List[torch.Tensor]) -> np.ndarray:
            acts = []
            batch = torch.stack(images).to(device)
            # Convert [-1, 1] → [0, 1]
            batch = (batch.clamp(-1, 1) + 1) / 2
            with torch.no_grad():
                pred = inception(batch)[0]
            pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
            return pred

        act_pred = get_activations(pred_images)
        act_real = get_activations(real_images)

        mu1, sig1 = act_pred.mean(0), np.cov(act_pred, rowvar=False)
        mu2, sig2 = act_real.mean(0), np.cov(act_real, rowvar=False)
        return float(calculate_frechet_distance(mu1, sig1, mu2, sig2))
    except Exception:
        return float("nan")


def compute_kid(
    pred_images: List[torch.Tensor],
    real_images: List[torch.Tensor],
    subset_size: int = 1000,
    device: torch.device = None,
) -> Tuple[float, float]:
    """Compute KID (mean, std) between two lists of image tensors.

    Returns (kid_mean * 1000, kid_std * 1000).
    """
    try:
        from torchmetrics.image.kid import KernelInceptionDistance

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kid = KernelInceptionDistance(subset_size=min(subset_size, len(pred_images))).to(device)

        def to_uint8(images):
            imgs = torch.stack(images).clamp(-1, 1)
            imgs = ((imgs + 1) / 2 * 255).byte()
            return imgs.to(device)

        kid.update(to_uint8(real_images), real=True)
        kid.update(to_uint8(pred_images), real=False)
        mean, std = kid.compute()
        return float(mean) * 1000, float(std) * 1000
    except Exception:
        return float("nan"), float("nan")
