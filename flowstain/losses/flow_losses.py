"""
Core rectified-flow training loss.

L_flow = weighted_mean( ||v_theta - v_target||  )
       = weighted_mean( ||v_theta - (z_ihc - z_he)||  )

v_target is constant given z_he and z_ihc.
Mid-trajectory bridge noise is added as:
    z_t = (1 - t) * z_he + t * z_ihc + sigma(t) * eps
    sigma(t) = sigma_max * sin(pi * t)

This matches the plan's source-anchored stochastic interpolant.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample continuous timesteps uniformly in (0, 1)."""
    return torch.rand(batch_size, device=device)


def bridge_noise_sigma(t: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """sigma(t) = sigma_max * sin(pi * t).  Zero at both endpoints."""
    return sigma_max * torch.sin(math.pi * t)


def build_noisy_latent(
    z_he: torch.Tensor,
    z_ihc: torch.Tensor,
    t: torch.Tensor,
    sigma_max: float = 0.0,
    noise_source: str = "he",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute z_t and v_target for a training step.

    Args:
        z_he:         [B, C, H, W]
        z_ihc:        [B, C, H, W]
        t:            [B]
        sigma_max:    bridge noise amplitude (0 = pure linear interpolation)
        noise_source: "he" (H&E-start) or "noise" (sample z_he from N(0,1))

    Returns:
        z_t:      [B, C, H, W]  interpolated latent
        v_target: [B, C, H, W]  z_ihc - z_he  (constant velocity target)
    """
    t4 = t[:, None, None, None]

    if noise_source == "noise":
        # Ablation baseline: treat z_he as pure Gaussian noise
        z_src = torch.randn_like(z_he)
    else:
        z_src = z_he

    v_target = z_ihc - z_src  # [B, C, H, W]
    z_t = (1 - t4) * z_src + t4 * z_ihc

    if sigma_max > 0:
        eps = torch.randn_like(z_t)
        sigma_t = bridge_noise_sigma(t, sigma_max)[:, None, None, None]
        z_t = z_t + sigma_t * eps

    return z_t, v_target


def flow_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 0.5,
    reduce: bool = False,
) -> torch.Tensor:
    """Per-sample or reduced velocity prediction loss.

    Args:
        v_pred:     [B, C, H, W]
        v_target:   [B, C, H, W]
        loss_type:  l1 | l2 | huber
        reduce:     if True return scalar; if False return [B]

    Returns:
        [B] per-sample mean losses, or scalar if reduce=True
    """
    diff = v_pred - v_target
    if loss_type == "l1":
        loss_map = diff.abs()
    elif loss_type == "l2":
        loss_map = diff.pow(2)
    else:  # huber
        loss_map = F.huber_loss(v_pred, v_target, reduction="none", delta=huber_delta)

    # Mean over C, H, W → [B]
    per_sample = loss_map.flatten(1).mean(1)
    return per_sample.mean() if reduce else per_sample
