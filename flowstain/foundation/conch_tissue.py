"""
Frozen CONCH image+text encoder for zero-shot tissue classification.

CONCH is loaded via open_clip. The image encoder is used to classify
H&E crops into tissue categories using per-class text prompt ensembles.

Tissue probabilities are optionally cached to disk for validation crops.
During training, CONCH runs on the fly on the H&E batch but results are
not backpropagated (frozen, no_grad).

Cache format: {cache_dir}/{key}.pt  where each file is a dict:
    {"probs": tensor[num_classes], "top_class": str, "entropy": float}
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """

    def __init__(
        self,
        model_name: str = "hf-hub:MahmoodLab/conch",
        tissue_prompts_yaml: str | Path = "configs/tissue_prompts.yaml",
        cache_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._device = device

        self.tissue_classes = TISSUE_CLASSES
        self.num_classes = len(TISSUE_CLASSES)

        # Load model
        self.model, self.preprocess, self.tokenizer = self._load_conch(model_name)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # Pre-compute text prototype embeddings
        prompts = _load_tissue_prompts(tissue_prompts_yaml)
        text_protos = self._encode_text_prototypes(prompts)
        self.register_buffer("text_protos", text_protos)  # [num_classes, D]

    # ── model loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_conch(model_name: str):
        try:
            import open_clip
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            return model, preprocess_val, tokenizer
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CONCH model '{model_name}'. "
                f"Ensure open-clip-torch>=2.24 and network access.\nError: {e}"
            )

    @torch.no_grad()
    def _encode_text_prototypes(self, prompts: Dict[str, List[str]]) -> torch.Tensor:
        """Encode all prompts and average per class.

        Returns: [num_classes, D]
        """
        device = next(self.model.parameters()).device
        protos = []
        for cls in TISSUE_CLASSES:
            cls_prompts = prompts.get(cls, [f"H&E histopathology tissue: {cls.replace('_', ' ')}"])
            tokens = self.tokenizer(cls_prompts)
            if hasattr(tokens, "to"):
                tokens = tokens.to(device)
            with torch.no_grad():
                embeds = self.model.encode_text(tokens)  # [P, D]
                embeds = F.normalize(embeds, dim=-1)
                proto = embeds.mean(0)
                proto = F.normalize(proto, dim=-1)
            protos.append(proto)
        return torch.stack(protos)  # [num_classes, D]

    # ── image encoding ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_images(self, images_0_1: torch.Tensor) -> torch.Tensor:
        """Encode images [B, 3, H, W] (values 0–1) into CONCH embeddings.

        CONCH expects input normalized with its own stats.  We apply the
        torchvision functional version of CONCH's validation preprocess
        (resize + center-crop + normalize).

        Returns: [B, D]
        """
        import torchvision.transforms.functional as TF

        # CONCH preprocess: resize to 448, center-crop to 448, normalize
        imgs = TF.resize(images_0_1, [448], antialias=True)
        if imgs.shape[-1] > 448 or imgs.shape[-2] > 448:
            imgs = TF.center_crop(imgs, [448, 448])

        # CONCH normalization params (OpenAI CLIP stats)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=images_0_1.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=images_0_1.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std

        embeds = self.model.encode_image(imgs)
        return F.normalize(embeds, dim=-1)  # [B, D]

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
            he_tensor: [B, 3, H, W] in [-1, 1]
            filenames: used for disk caching
            stain_names: used as part of cache key

        Returns dict with:
            probs:      [B, num_classes]  softmax tissue probabilities
            top_labels: [B]               int64 class indices
            top_names:  list[str]         class name per sample
            entropy:    [B]               normalized entropy in [0, 1]
        """
        he_01 = (he_tensor + 1.0) / 2.0  # [-1,1] → [0,1]

        results = {}
        cached_probs = []
        uncached_indices = []

        # Try loading from cache
        if self._cache_dir is not None and filenames is not None:
            for i, fname in enumerate(filenames):
                key = self._cache_key(fname, stain_names[i] if stain_names else "")
                cpath = self._cache_dir / f"{key}.pt"
                if cpath.exists():
                    cached_probs.append((i, torch.load(cpath, map_location="cpu")["probs"]))
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(he_01.shape[0]))

        # Compute for uncached samples
        all_probs = [None] * he_01.shape[0]
        for (i, p) in cached_probs:
            all_probs[i] = p.to(he_01.device)

        if uncached_indices:
            subset = he_01[uncached_indices]
            embeds = self._encode_images(subset)
            logits = embeds @ self.text_protos.T  # [B_sub, num_classes]
            probs = F.softmax(logits / 0.07, dim=-1)
            for out_i, src_i in enumerate(uncached_indices):
                all_probs[src_i] = probs[out_i]
                # Write cache
                if self._cache_dir is not None and filenames is not None:
                    self._write_cache(
                        filenames[src_i],
                        stain_names[src_i] if stain_names else "",
                        probs[out_i].cpu(),
                    )

        probs_t = torch.stack(all_probs, dim=0)  # [B, num_classes]
        top_labels = probs_t.argmax(dim=-1)       # [B]
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
        """Normalized Shannon entropy in [0, 1]."""
        eps = 1e-8
        H = -(probs * (probs + eps).log()).sum(dim=-1)
        H_max = math.log(probs.shape[-1])
        return H / H_max

    def _cache_key(self, filename: str, stain: str) -> str:
        raw = f"{filename}|{stain}|{self._model_name}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _write_cache(self, filename: str, stain: str, probs: torch.Tensor):
        if self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(filename, stain)
        torch.save({"probs": probs}, self._cache_dir / f"{key}.pt")
