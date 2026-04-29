"""
Full FlowStain model: autoencoder + UNI + flow U-Net combined.

This module ties together the frozen components (VAE, UNI extractor, CONCH)
and the trainable flow network.  It is the object that gets passed to
accelerator.prepare() during training.

Only self.flow_net and self.uni_processor are trainable.
The uni_extractor and vae are always frozen.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..foundation.uni import UNIExtractor, UNIFeatureProcessorHighRes
from .autoencoder import LatentAutoencoder
from .flow_unet import FlowUNet


class FlowStainModel(nn.Module):
    """End-to-end FlowStain model.

    Trainable params: FlowUNet + UNIFeatureProcessorHighRes (adapter weights)
    Frozen:           LatentAutoencoder + UNIExtractor
    """

    def __init__(
        self,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        vae_scale_factor: float = 0.18215,
        uni_model_name: str = "hf-hub:MahmoodLab/uni",
        uni_grid_size: int = 4,
        uni_base_channels: int = 512,
        flow_base_channels: int = 320,
        flow_channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        flow_num_res_blocks: int = 2,
        stain_num_embeddings: int = 5,
        stain_emb_dim: int = 64,
        cond_mode: str = "hybrid",
    ):
        super().__init__()

        self.vae = LatentAutoencoder(vae_model_name, vae_scale_factor)
        self.uni_extractor = UNIExtractor(uni_model_name, uni_grid_size)
        uni_dim = self.uni_extractor.feature_dim
        self.uni_processor = UNIFeatureProcessorHighRes(
            uni_dim=uni_dim,
            base_channels=uni_base_channels,
            spatial_size=uni_grid_size,
        )

        self.flow_net = FlowUNet(
            latent_channels=4,
            base_channels=flow_base_channels,
            channel_mults=flow_channel_mults,
            num_res_blocks=flow_num_res_blocks,
            stain_num_embeddings=stain_num_embeddings,
            stain_emb_dim=stain_emb_dim,
            uni_map_channels=self.uni_processor.channel_dims,
            uni_token_dim=uni_dim,
            cond_mode=cond_mode,
        )

    # ── frozen encoders ───────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_pair(
        self, he: torch.Tensor, ihc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_he = self.vae.encode(he)
        z_ihc = self.vae.encode(ihc)
        return z_he, z_ihc

    @torch.no_grad()
    def encode_uni(self, uni_subcrops: torch.Tensor) -> torch.Tensor:
        """Extract UNI tokens from pre-split sub-crops [B, N, 3, 224, 224]."""
        return self.uni_extractor(uni_subcrops)  # [B, N, D]

    # ── flow forward ─────────────────────────────────────────────────────────

    def get_uni_maps(
        self, uni_tokens: torch.Tensor, dropout_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Process UNI tokens through the trainable adapter.

        Args:
            uni_tokens:    [B, N, D]
            dropout_mask:  [B] bool – True means drop UNI features for this sample

        Returns:
            uni_maps:   {res: tensor}
            uni_tokens: [B, N, D] (possibly zeroed for dropout)
        """
        if dropout_mask is not None and dropout_mask.any():
            uni_tokens = uni_tokens.clone()
            uni_tokens[dropout_mask] = 0.0

        uni_maps = self.uni_processor(uni_tokens)
        return uni_maps, uni_tokens

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_he: torch.Tensor,
        uni_maps: Dict[int, torch.Tensor],
        uni_tokens: torch.Tensor,
        stain_id: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field v_theta."""
        return self.flow_net(z_t, t, z_he, uni_maps, uni_tokens, stain_id)

    # ── ODE sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_euler(
        self,
        z_he: torch.Tensor,
        uni_maps: Dict[int, torch.Tensor],
        uni_tokens: torch.Tensor,
        stain_id: torch.Tensor,
        num_steps: int = 16,
        sigma_bridge: float = 0.0,
    ) -> torch.Tensor:
        """Deterministic Euler ODE from z_he to z_ihc_hat."""
        z = z_he.clone()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
            v = self.flow_net(z, t, z_he, uni_maps, uni_tokens, stain_id)
            if sigma_bridge > 0:
                noise_scale = sigma_bridge * math.sin(math.pi * t_val) * dt
                z = z + v * dt + noise_scale * torch.randn_like(z)
            else:
                z = z + v * dt
        return z

    @torch.no_grad()
    def sample_heun(
        self,
        z_he: torch.Tensor,
        uni_maps: Dict[int, torch.Tensor],
        uni_tokens: torch.Tensor,
        stain_id: torch.Tensor,
        num_steps: int = 16,
    ) -> torch.Tensor:
        """Heun's method (2nd-order) ODE sampler."""
        z = z_he.clone()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_val = i * dt
            t_cur = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
            t_nxt = torch.full((z.shape[0],), t_val + dt, device=z.device, dtype=z.dtype)
            v1 = self.flow_net(z, t_cur, z_he, uni_maps, uni_tokens, stain_id)
            z_pred = z + v1 * dt
            v2 = self.flow_net(z_pred, t_nxt, z_he, uni_maps, uni_tokens, stain_id)
            z = z + 0.5 * (v1 + v2) * dt
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)
