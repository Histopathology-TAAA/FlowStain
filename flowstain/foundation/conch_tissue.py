"""
Frozen CONCH tissue classifier for zero-shot H&E tissue classification.

Uses the official CONCH package API:
  from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

Model can be loaded from HF hub (preferred) or a local checkpoint:
  model, preprocess = create_model_from_pretrained(
      'conch_ViT-B-16', 'hf_hub:MahmoodLab/conch', hf_auth_token=token
  )
  model, preprocess = create_model_from_pretrained(
      'conch_ViT-B-16', './checkpoints/CONCH/pytorch_model.bin'
  )

Tissue probabilities are computed as:
  sim = model.encode_image(img) @ model.encode_text(prompts).T * model.logit_scale.exp()
  probs = sim.softmax(dim=-1)

Results are optionally cached to disk (keyed by filename + stain + model config)
for validation crops where crops are deterministic.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


TISSUE_CLASSES = [
    "invasive_carcinoma",
    "benign_epithelium",
    "stroma",
    "inflammation",
    "adipose",
    "necrosis",
    "background",
]


def _load_tissue_prompts(yaml_path: str | Path) -> Dict[str, List[str]]:
    cfg = OmegaConf.load(yaml_path)
    return {cls: list(cfg.prompts[cls]) for cls in cfg.tissue_classes}


class CONCHTissueClassifier(nn.Module):
    """Zero-shot CONCH tissue classifier.

    Produces per-crop tissue probability vectors used for:
    1. Loss weighting during training (tissue-aware weighted mean).
    2. Tissue-stratified validation metrics and W&B reporting.

    Args:
        model_cfg:             CONCH model config name, e.g. 'conch_ViT-B-16'.
        checkpoint_path:       Path to the CONCH checkpoint .bin file.
        tissue_prompts_yaml:   Path to tissue_prompts.yaml.
        cache_dir:             Optional directory to cache per-sample tissue probs.
        device:                Optional device. Defaults to current CUDA device.
    """

    def __init__(
        self,
        model_cfg: str = "conch_ViT-B-16",
        checkpoint_path: str | Path = "hf_hub:MahmoodLab/conch",
        hf_auth_token: Optional[str] = None,
        tissue_prompts_yaml: str | Path = "configs/tissue_prompts.yaml",
        cache_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._model_cfg = model_cfg
        self._checkpoint_path = str(checkpoint_path)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._device = device

        self.tissue_classes = TISSUE_CLASSES
        self.num_classes = len(TISSUE_CLASSES)

        # Load frozen model + preprocessing transform
        self.model, self.preprocess = self._load_conch(
            model_cfg, checkpoint_path, device, hf_auth_token
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Pre-compute averaged text prototype embeddings once at init
        prompts = _load_tissue_prompts(tissue_prompts_yaml)
        text_protos = self._encode_text_prototypes(prompts)  # [num_classes, D]
        self.register_buffer("text_protos", text_protos)

    # ── model loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_conch(
        model_cfg: str,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        hf_auth_token: Optional[str] = None,
    ):
        """Load CONCH model and preprocessing transform.

        ``checkpoint_path`` accepts:
          - HuggingFace hub string:  ``'hf_hub:MahmoodLab/conch'``
          - Local file path:         ``'./checkpoints/CONCH/pytorch_model.bin'``

        For gated HF repos supply the user access token via ``hf_auth_token``.
        """
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError:
            raise ImportError(
                "The 'conch' package is required for tissue classification. "
                "Install it from https://github.com/mahmoodlab/CONCH or via pip."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {}
        if hf_auth_token:
            kwargs["hf_auth_token"] = hf_auth_token

        model, preprocess = create_model_from_pretrained(
            model_cfg, str(checkpoint_path), device=device, **kwargs
        )
        model.eval()
        return model, preprocess

    # ── text encoding ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text_prototypes(self, prompts: Dict[str, List[str]]) -> torch.Tensor:
        """Encode all prompts and average per class.

        Returns: [num_classes, D]
        """
        try:
            from conch.open_clip_custom import tokenize, get_tokenizer
        except ImportError:
            raise ImportError("conch package required for text encoding.")

        device = next(self.model.parameters()).device
        tokenizer = get_tokenizer()

        protos = []
        for cls in TISSUE_CLASSES:
            cls_prompts = prompts.get(
                cls, [f"an H&E image of {cls.replace('_', ' ')}"]
            )
            tokens = tokenize(texts=cls_prompts, tokenizer=tokenizer).to(device)
            embeds = self.model.encode_text(tokens)        # [P, D]
            embeds = F.normalize(embeds.float(), dim=-1)
            proto = embeds.mean(0)
            proto = F.normalize(proto, dim=-1)
            protos.append(proto)

        return torch.stack(protos)  # [num_classes, D]

    # ── image encoding ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_images(self, he_01: torch.Tensor) -> torch.Tensor:
        """Encode a batch of H&E images [B, 3, H, W] (values 0–1).

        Converts each image to PIL, applies CONCH's preprocess, and runs
        through the image encoder.

        Returns: [B, D]  L2-normalised embeddings
        """
        from PIL import Image as PILImage
        import torchvision.transforms.functional as TF

        device = next(self.model.parameters()).device
        preprocessed = []
        for i in range(he_01.shape[0]):
            # Convert tensor [0,1] → PIL
            pil = TF.to_pil_image(he_01[i].cpu().clamp(0, 1))
            preprocessed.append(self.preprocess(pil))

        batch = torch.stack(preprocessed).to(device)  # [B, 3, H', W']
        embeds = self.model.encode_image(batch)
        return F.normalize(embeds.float(), dim=-1)     # [B, D]

    # ── classification ───────────────────────────────────────────────────────

    @torch.no_grad()
    def classify(
        self,
        he_tensor: torch.Tensor,
        filenames: Optional[List[str]] = None,
        stain_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Classify H&E crops into tissue categories.

        Args:
            he_tensor:   [B, 3, H, W] in [-1, 1]
            filenames:   list[str] used for disk caching
            stain_names: list[str] used as part of cache key

        Returns dict:
            probs:      [B, num_classes]  softmax tissue probabilities
            top_labels: [B]               int64 class indices
            top_names:  list[str]         class name per sample
            entropy:    [B]               normalised entropy in [0, 1]
        """
        he_01 = (he_tensor.detach() + 1.0) / 2.0  # [-1,1] → [0,1]
        B = he_01.shape[0]

        all_probs: List[Optional[torch.Tensor]] = [None] * B
        uncached_indices: List[int] = []

        # ── cache lookup ─────────────────────────────────────────────────
        if self._cache_dir is not None and filenames is not None:
            for i, fname in enumerate(filenames):
                sname = stain_names[i] if stain_names else ""
                cpath = self._cache_dir / f"{self._cache_key(fname, sname)}.pt"
                if cpath.exists():
                    all_probs[i] = torch.load(cpath, map_location="cpu")["probs"].to(he_01.device)
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(B))

        # ── forward pass for uncached samples ────────────────────────────
        if uncached_indices:
            subset = he_01[uncached_indices]
            embeds = self._encode_images(subset)                      # [N, D]
            logit_scale = self.model.logit_scale.exp()
            logits = embeds @ self.text_protos.T * logit_scale        # [N, num_classes]
            probs = logits.softmax(dim=-1)

            for out_i, src_i in enumerate(uncached_indices):
                all_probs[src_i] = probs[out_i]
                if self._cache_dir is not None and filenames is not None:
                    sname = stain_names[src_i] if stain_names else ""
                    self._write_cache(filenames[src_i], sname, probs[out_i].cpu())

        probs_t = torch.stack(all_probs, dim=0)       # [B, num_classes]
        top_labels = probs_t.argmax(dim=-1)            # [B]
        entropy = self._normalized_entropy(probs_t)

        return {
            "probs": probs_t,
            "top_labels": top_labels,
            "top_names": [TISSUE_CLASSES[int(l)] for l in top_labels.cpu().tolist()],
            "entropy": entropy,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalized_entropy(probs: torch.Tensor) -> torch.Tensor:
        """Normalised Shannon entropy in [0, 1]."""
        eps = 1e-8
        H = -(probs * (probs + eps).log()).sum(dim=-1)
        return (H / math.log(probs.shape[-1])).clamp(0.0, 1.0)

    def _cache_key(self, filename: str, stain: str) -> str:
        raw = f"{filename}|{stain}|{self._model_cfg}|{self._checkpoint_path}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _write_cache(self, filename: str, stain: str, probs: torch.Tensor):
        if self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(filename, stain)
        torch.save({"probs": probs}, self._cache_dir / f"{key}.pt")
