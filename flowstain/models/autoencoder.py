"""
VAE autoencoder wrapper for FlowStain.

Uses diffusers AutoencoderKL. The encoder and decoder are frozen.
Provides encode/decode helpers with the standard SD latent scale factor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class LatentAutoencoder(nn.Module):
    """Thin wrapper around a frozen diffusers AutoencoderKL.

    All parameters are frozen at init; only the encode/decode interface
    is exposed. Fine-tuning the VAE decoder is possible by calling
    unfreeze_decoder() if color accuracy becomes a bottleneck.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/sd-vae-ft-mse",
        scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.freeze()

    def freeze(self):
        for p in self.vae.parameters():
            p.requires_grad = False

    def unfreeze_decoder(self):
        for p in self.vae.decoder.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images [-1, 1] → scaled latents.

        Args:
            x: [B, 3, H, W] in [-1, 1]

        Returns:
            z: [B, 4, H/8, W/8] scaled latents
        """
        dist = self.vae.encode(x).latent_dist
        z = dist.sample() * self.scale_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents → images.

        Args:
            z: [B, 4, H/8, W/8]

        Returns:
            x: [B, 3, H, W] in [-1, 1]
        """
        return self.vae.decode(z / self.scale_factor).sample
