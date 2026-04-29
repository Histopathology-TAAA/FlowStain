"""
Frozen UNI feature extractor.

Replicates UNIStainNet's 4×4 sub-crop strategy:
  - Each 512×512 patch is split into a 4×4 grid of 128×128 sub-crops.
  - Each sub-crop is resized to 224×224 and passed through UNI.
  - All patch tokens are reassembled into a dense [B, S*S, D] tensor
    where S = grid_size (default 32 patch tokens per sub-crop, but
    sub-crops are independent, so we average-pool to one token each,
    giving a 4×4 = 16-token grid).  For higher resolution we instead
    keep all 196 patch tokens per sub-crop and pool to 8×8 or 4×4.

For simplicity (and matching UNIStainNet), each sub-crop produces one
CLS/global token, reassembled to a [B, 16, D] dense grid (4×4).  The
embedding dimension D is inferred from the loaded timm model because
different UNI-family checkpoints can return 1024-d or 1536-d features.

Caching: features for validation (deterministic crops) are optionally
written to disk as .pt files keyed by a stable hash.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


class UNIExtractor(nn.Module):
    """Wraps a frozen UNI ViT and returns per-sub-crop patch-token grids.

    For every image in the batch, it expects a pre-split sub-crop tensor
    of shape [B, N, 3, 224, 224] produced by the dataset (uni_subcrops),
    where N = grid_size * grid_size.

    Returns spatial tokens [B, N, D], where D is inferred from the model.
    These are then fed to UNIFeatureProcessorHighRes.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:MahmoodLab/uni",
        grid_size: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.grid_size = grid_size
        self.model, self.transform = self._load_model(model_name)
        self.feature_dim = self._infer_feature_dim(self.model)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @staticmethod
    def _load_model(model_name: str) -> nn.Module:
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        model.eval()
        return model, transform

    @staticmethod
    def _infer_feature_dim(model: nn.Module) -> int:
        """Infer output embedding dimension from a timm model."""
        for attr in ("num_features", "embed_dim"):
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        raise AttributeError(
            "Could not infer UNI feature dimension from the loaded model. "
            "Expected timm model to expose `num_features` or `embed_dim`."
        )

    @torch.no_grad()
    def forward(self, uni_subcrops: torch.Tensor) -> torch.Tensor:
        """Extract features from pre-split sub-crops.

        Args:
            uni_subcrops: [B, N, 3, 224, 224]  N = grid_size^2

        Returns:
            tokens: [B, N, D]  one CLS/global token per sub-crop
        """
        B, N, C, H, W = uni_subcrops.shape
        flat = uni_subcrops.view(B * N, C, H, W)
        feats = self.model(flat)       # [B*N, D]  — CLS/global token
        tokens = feats.view(B, N, -1)  # [B, N, D]
        return tokens


class UNIFeatureProcessorHighRes(nn.Module):
    """Project [B, S*S, D] dense tokens into multi-scale spatial maps.

    Mirrors UNIStainNet's UNIFeatureProcessorHighRes with a few tweaks:
    – Input spatial_size = grid_size (4) for 4×4 tokens.
    – Output maps: {16, 32, 64, 128, 256} for use in flow U-Net decoder.
    """

    def __init__(
        self,
        uni_dim: int = 1024,
        base_channels: int = 512,
        spatial_size: int = 4,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.spatial_size = spatial_size
        ch = base_channels

        self.proj = nn.Sequential(
            nn.Linear(uni_dim, ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # process at native resolution (4×4)
        self.proc_4 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4→8→16
        self.up_16 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
        )
        # 16→32
        self.up_32 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
        )
        # 32→64
        ch64 = ch // 2
        self.up_64 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch64), nn.LeakyReLU(0.2, inplace=True),
        )
        # 64→128
        ch128 = ch // 4
        self.up_128 = nn.Sequential(
            nn.ConvTranspose2d(ch64, ch128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch128), nn.LeakyReLU(0.2, inplace=True),
        )
        # 128→256
        ch256 = ch // 8
        self.up_256 = nn.Sequential(
            nn.ConvTranspose2d(ch128, ch256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch256), nn.LeakyReLU(0.2, inplace=True),
        )

        self._channel_dims = {16: ch, 32: ch, 64: ch64, 128: ch128, 256: ch256}

    @property
    def channel_dims(self) -> Dict[int, int]:
        return self._channel_dims

    def forward(self, tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Args:
            tokens: [B, S*S, D]  e.g. [B, 16, 1536] for 4×4 grid

        Returns:
            dict {resolution: feature_map}
        """
        B, N, D = tokens.shape
        S = self.spatial_size
        assert N == S * S, f"Expected {S*S} tokens, got {N}"
        expected_dim = self.proj[0].in_features
        if D != expected_dim:
            raise ValueError(
                f"UNI token dimension mismatch: got {D}, but processor expects "
                f"{expected_dim}. Build UNIFeatureProcessorHighRes with "
                f"uni_dim=uni_extractor.feature_dim."
            )

        x = self.proj(tokens)                              # [B, S*S, C]
        x = x.permute(0, 2, 1).reshape(B, -1, S, S)       # [B, C, S, S]
        x = self.proc_4(x) + x                             # residual at native res

        f16 = self.up_16(x)
        f32 = self.up_32(f16)
        f64 = self.up_64(f32)
        f128 = self.up_128(f64)
        f256 = self.up_256(f128)

        return {16: f16, 32: f32, 64: f64, 128: f128, 256: f256}
