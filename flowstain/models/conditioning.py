"""
Conditioning blocks for FlowStain's rectified-flow U-Net.

Three modes (config-switchable):
  spade           – UNI spatial maps modulate each decoder block.
  cross_attention  – UNI CLS-pooled vector attends to flow features.
  hybrid           – both SPADE modulation and cross-attention.

Stain identity is injected through FiLM at every block.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── FiLM ────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """Feature-wise Linear Modulation from a 1-D conditioning vector."""

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_channels)
        self.beta = nn.Linear(cond_dim, num_channels)
        # Identity initialization
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, C, H, W] or [B, C]
            cond: [B, cond_dim]
        """
        g = self.gamma(cond)  # [B, C]
        b = self.beta(cond)   # [B, C]
        if x.dim() == 4:
            g = g.unsqueeze(-1).unsqueeze(-1)
            b = b.unsqueeze(-1).unsqueeze(-1)
        return g * x + b


# ─── SPADE block ─────────────────────────────────────────────────────────────

class SPADEBlock(nn.Module):
    """Spatially-adaptive denormalization conditioned on a UNI feature map.

    Parameters are zero-initialized (ControlNet style) so the block starts
    as identity and conditioning is learned gradually.
    """

    def __init__(self, num_channels: int, uni_channels: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(uni_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gamma_conv = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.beta_conv = nn.Conv2d(hidden, num_channels, 3, padding=1)
        # zero-init so block starts as identity
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, x: torch.Tensor, uni_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, C, H, W] feature map
            uni_map: [B, uni_C, H', W'] – will be bilinearly upsampled to x size
        """
        x_norm = self.norm(x)
        uni_up = F.interpolate(uni_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        shared = self.shared(uni_up)
        gamma = self.gamma_conv(shared)
        beta = self.beta_conv(shared)
        return (1 + gamma) * x_norm + beta


# ─── Cross-attention block ────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention from flow features to UNI token sequence."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, C, H, W] – flattened to [B, H*W, C] internally
            context: [B, N, context_dim] – UNI tokens

        Returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        x_norm = self.norm(x_flat)

        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out(out)

        # residual
        out = out + x_flat
        return out.permute(0, 2, 1).reshape(B, C, H, W)


# ─── Combined conditioning block ─────────────────────────────────────────────

class FlowCondBlock(nn.Module):
    """Applies UNI conditioning (SPADE / cross-attention / hybrid) + stain FiLM.

    mode options:
        spade           – spatial only
        cross_attention  – token attention only
        hybrid           – spatial SPADE then cross-attention
    """

    def __init__(
        self,
        num_channels: int,
        uni_map_channels: int,
        uni_token_dim: int,
        stain_emb_dim: int,
        mode: str = "hybrid",
        num_attn_heads: int = 8,
    ):
        super().__init__()
        self.mode = mode

        if mode in ("spade", "hybrid"):
            self.spade = SPADEBlock(num_channels, uni_map_channels)

        if mode in ("cross_attention", "hybrid"):
            self.cross_attn = CrossAttentionBlock(
                num_channels, uni_token_dim, num_heads=num_attn_heads
            )

        self.film = FiLM(stain_emb_dim, num_channels)

    def forward(
        self,
        x: torch.Tensor,
        uni_map: Optional[torch.Tensor],
        uni_tokens: Optional[torch.Tensor],
        stain_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          [B, C, H, W]
            uni_map:    [B, uni_C, H', W'] or None
            uni_tokens: [B, N, uni_token_dim] or None
            stain_emb:  [B, stain_emb_dim]
        """
        if self.mode in ("spade", "hybrid") and uni_map is not None:
            x = self.spade(x, uni_map)

        if self.mode in ("cross_attention", "hybrid") and uni_tokens is not None:
            x = self.cross_attn(x, uni_tokens)

        x = self.film(x, stain_emb)
        return x
