"""
Source-anchored Rectified-Flow U-Net (FlowUNet).

Architecture:
  Encoder: [z_t || z_he] → 5 strided conv blocks → bottleneck (with self-attn)
  Decoder: 5 upsample blocks with skip connections, UNI conditioning, stain FiLM
  Output:  velocity prediction v_theta ~ (z_ihc - z_he)

Inputs:
  z_t        – noisy interpolated latent [B, latent_C, H, W]
  t          – continuous timestep in [0, 1] [B]
  z_he       – H&E source latent (anchors trajectory) [B, latent_C, H, W]
  uni_maps   – {res: [B, C, H', W']} from UNIFeatureProcessorHighRes
  uni_tokens – [B, N, D] for cross-attention (= raw UNI output, optional)
  stain_id   – [B] long tensor

Both z_t and z_he are concatenated at the input channel dim, so the first
conv sees 2 * latent_C channels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import FlowCondBlock


# ─── helpers ─────────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding.  t in [0, 1]. Returns [B, dim]."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0  # scale to ~[0, 1000]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttnBlock(nn.Module):
    """Lightweight self-attention block for bottleneck."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 n_res: int = 2, downsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(n_res)
        ])
        self.down = (
            nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1) if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for res in self.resnets:
            x = res(x, t)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        time_emb_dim: int,
        uni_map_ch: int,
        uni_token_dim: int,
        stain_emb_dim: int,
        cond_mode: str,
        n_res: int = 2,
        upsample: bool = True,
    ):
        super().__init__()
        # First resblock takes in_ch + skip_ch
        self.resnets = nn.ModuleList()
        self.resnets.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim))
        for _ in range(n_res - 1):
            self.resnets.append(ResBlock(out_ch, out_ch, time_emb_dim))

        self.cond = FlowCondBlock(
            out_ch, uni_map_ch, uni_token_dim, stain_emb_dim, mode=cond_mode
        )
        self.up = (
            nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1) if upsample
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t: torch.Tensor,
        uni_map: Optional[torch.Tensor],
        uni_tokens: Optional[torch.Tensor],
        stain_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        for res in self.resnets:
            x = res(x, t)
        x = self.cond(x, uni_map, uni_tokens, stain_emb)
        x = self.up(x)
        return x


# ─── main model ──────────────────────────────────────────────────────────────

class FlowUNet(nn.Module):
    """Source-anchored rectified-flow velocity network.

    Key design choices:
    - z_he is concatenated to z_t at input (source anchoring).
    - Multi-scale UNI maps condition each decoder block (SPADE / xattn / hybrid).
    - Stain FiLM at every decoder block for multi-stain unified generation.
    - Sinusoidal time embedding (t in [0, 1]).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 320,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        time_emb_dim: int = 256,
        stain_num_embeddings: int = 5,
        stain_emb_dim: int = 64,
        uni_map_channels: Dict[int, int] = None,
        uni_token_dim: int = 1024,
        cond_mode: str = "hybrid",
        dropout: float = 0.0,
    ):
        super().__init__()
        # uni_map_channels: {resolution: channels} from UNIFeatureProcessorHighRes
        self.uni_map_channels = uni_map_channels or {
            16: 512, 32: 512, 64: 256, 128: 128, 256: 64
        }

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        self._time_emb_dim = time_emb_dim
        self._time_proj_dim = time_emb_dim * 4

        # Stain embedding (includes null slot for CFG dropout)
        self.stain_emb = nn.Embedding(stain_num_embeddings, stain_emb_dim)
        nn.init.normal_(self.stain_emb.weight, std=0.02)

        # Input channels = z_t + z_he concatenated
        in_ch = latent_channels * 2
        channels = [base_channels * m for m in channel_mults]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        ch = channels[0]
        for i, out_ch in enumerate(channels[:-1]):
            self.down_blocks.append(
                DownBlock(ch, out_ch, self._time_proj_dim, num_res_blocks, downsample=True)
            )
            ch = out_ch
        # Last encoder block without downsampling
        self.down_blocks.append(
            DownBlock(ch, channels[-1], self._time_proj_dim, num_res_blocks, downsample=False)
        )
        ch = channels[-1]

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, self._time_proj_dim),
            SelfAttnBlock(ch),
            ResBlock(ch, ch, self._time_proj_dim),
        ])

        # ── Decoder ───────────────────────────────────────────────────────────
        # Spatial resolutions for each decoder level (assuming 64x64 latent from 512 image)
        # level0=64, level1=32, level2=16, level3=8  (downsampled by 2 each)
        decoder_resolutions = self._decoder_resolutions(64, len(channels))
        rev_channels = list(reversed(channels))

        self.up_blocks: nn.ModuleList = nn.ModuleList()
        for i, (in_c, skip_c, out_c) in enumerate(
            zip(rev_channels, rev_channels[1:] + [channels[0]], rev_channels)
        ):
            res = decoder_resolutions[i]
            uni_map_ch = self.uni_map_channels.get(res, 64)
            is_last = i == len(rev_channels) - 1
            self.up_blocks.append(
                UpBlock(
                    in_c, skip_c, out_c,
                    self._time_proj_dim,
                    uni_map_ch, uni_token_dim, stain_emb_dim,
                    cond_mode, num_res_blocks,
                    upsample=not is_last,
                )
            )

        # Output conv
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _decoder_resolutions(latent_size: int, n_levels: int) -> List[int]:
        """Return spatial resolutions at each decoder level (coarse → fine)."""
        # encoder goes latent_size → latent_size/2 → ... → latent_size/2^(n-2)
        resolutions = []
        r = latent_size // (2 ** (n_levels - 2))
        for _ in range(n_levels):
            resolutions.append(r)
            r *= 2
        return resolutions

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_he: torch.Tensor,
        uni_maps: Optional[Dict[int, torch.Tensor]],
        uni_tokens: Optional[torch.Tensor],
        stain_id: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity v_theta.

        Args:
            z_t:        [B, C, H, W] interpolated latent
            t:          [B] timestep in [0, 1]
            z_he:       [B, C, H, W] source H&E latent
            uni_maps:   dict {res: tensor} or None
            uni_tokens: [B, N, D] or None
            stain_id:   [B] long

        Returns:
            v_pred: [B, C, H, W]
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self._time_emb_dim)
        t_emb = self.time_emb(t_emb)  # [B, time_proj_dim]

        # Stain embedding
        s_emb = self.stain_emb(stain_id)  # [B, stain_emb_dim]

        # Concatenate source latent at input (source anchoring)
        x = torch.cat([z_t, z_he], dim=1)  # [B, 2C, H, W]
        x = self.input_conv(x)

        # Encoder
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t_emb)
            skips.append(skip)

        # Bottleneck
        for blk in self.bottleneck:
            if isinstance(blk, SelfAttnBlock):
                x = blk(x)
            else:
                x = blk(x, t_emb)

        # Decoder
        for block, skip in zip(self.up_blocks, reversed(skips)):
            res = x.shape[-1]
            u_map = uni_maps.get(res) if uni_maps else None
            x = block(x, skip, t_emb, u_map, uni_tokens, s_emb)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x
