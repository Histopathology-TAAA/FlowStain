"""
Structure losses operating on the aligned H&E → generated IHC axis.

Because the generated image and H&E input are pixel-aligned (no section
gap between them), structure supervision against H&E is safe at full
resolution.  Paired losses against real IHC should remain at low resolution.

L_edge: Sobel gradient consistency between H&E and generated IHC.
L_perceptual: LPIPS at 128 and 256 (misalignment-tolerant scale).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def sobel_gradients(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude from [B, 3, H, W] image."""
    # Convert to grayscale
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [B, 1, H, W]

    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                       dtype=x.dtype, device=x.device).unsqueeze(0)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                       dtype=x.dtype, device=x.device).unsqueeze(0)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()  # [B, 1, H, W]


class EdgeLoss(nn.Module):
    """Pixel-aligned edge consistency between H&E and generated IHC.

    Computes at two scales (512 and 256) to capture both fine and
    medium-scale structural alignment.
    """

    def __init__(self, scales: tuple = (512, 256)):
        super().__init__()
        self.scales = scales

    def forward(
        self,
        pred_ihc: torch.Tensor,
        he: torch.Tensor,
        reduce: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pred_ihc: [B, 3, H, W] generated IHC
            he:       [B, 3, H, W] H&E input (pixel-aligned to pred_ihc)
            reduce:   scalar if True, else [B]
        """
        total = torch.zeros(pred_ihc.shape[0], device=pred_ihc.device)
        for s in self.scales:
            if pred_ihc.shape[-1] != s:
                p = F.interpolate(pred_ihc, size=(s, s), mode="bilinear", align_corners=False)
                h = F.interpolate(he, size=(s, s), mode="bilinear", align_corners=False)
            else:
                p, h = pred_ihc, he
            g_pred = sobel_gradients(p)
            g_he = sobel_gradients(h)
            diff = (g_pred - g_he).abs().flatten(1).mean(1)  # [B]
            total = total + diff
        total = total / len(self.scales)
        return total.mean() if reduce else total


class PerceptualLoss(nn.Module):
    """Multi-scale LPIPS between generated and real IHC.

    Low-resolution patches make this misalignment-tolerant.
    Falls back to VGG feature L1 if lpips is unavailable.
    """

    def __init__(self, scales: tuple = (128, 256), weight_128: float = 1.0, weight_256: float = 0.5):
        super().__init__()
        self.scales = scales
        self.weights = {128: weight_128, 256: weight_256}

        if _LPIPS_AVAILABLE:
            self.lpips = _lpips_lib.LPIPS(net="vgg").eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
            self._use_lpips = True
        else:
            self._build_vgg_fallback()
            self._use_lpips = False

    def _build_vgg_fallback(self):
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),
            nn.Sequential(*list(vgg.children())[4:9]),
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _vgg_features(self, x: torch.Tensor) -> list:
        x = (x + 1) / 2
        x = (x - self.vgg_mean) / self.vgg_std
        feats = []
        for sl in self.vgg_slices:
            x = sl(x)
            feats.append(x)
        return feats

    def forward(
        self,
        pred: torch.Tensor,
        real: torch.Tensor,
        reduce: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            pred, real: [B, 3, H, W] in [-1, 1]
            reduce: scalar if True, else [B]
        """
        total = torch.zeros(pred.shape[0], device=pred.device)
        for s in self.scales:
            p = F.interpolate(pred, size=(s, s), mode="bilinear", align_corners=False)
            r = F.interpolate(real, size=(s, s), mode="bilinear", align_corners=False)
            w = self.weights.get(s, 1.0)
            if self._use_lpips:
                loss_map = self.lpips(p, r).squeeze()  # [B] or scalar
                if loss_map.dim() == 0:
                    loss_map = loss_map.unsqueeze(0).expand(pred.shape[0])
            else:
                fp = self._vgg_features(p)
                fr = self._vgg_features(r)
                loss_map = sum(
                    (a - b).abs().flatten(1).mean(1) for a, b in zip(fp, fr)
                )
            total = total + w * loss_map
        return total.mean() if reduce else total
