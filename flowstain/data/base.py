"""
Shared crop/augment/tensorize logic for FlowStain paired datasets.

Returns a dict (not a tuple) so training code is explicit about field names.
No Lightning DataModule dependency – pure PyTorch Dataset.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Subset


# Stain label mapping shared by MIST datasets
STAIN_TO_LABEL: Dict[str, int] = {"HER2": 0, "Ki67": 1, "ER": 2, "PR": 3}
LABEL_TO_STAIN: Dict[int, str] = {v: k for k, v in STAIN_TO_LABEL.items()}
NULL_STAIN_LABEL = 4  # used for CFG dropout


def _uni_crop_transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CropPairedDataset(Dataset):
    """Base class – common crop, augment, tensorize, UNI subcrop logic."""

    def __init__(
        self,
        crop_size: int = 512,
        augment: bool = False,
        uni_grid_size: int = 4,
        uni_subcrop_size: int = 224,
    ):
        self.crop_size = crop_size
        self.augment = augment
        self.uni_grid_size = uni_grid_size
        self._uni_transform = _uni_crop_transform(uni_subcrop_size)

    # ── crop helpers ─────────────────────────────────────────────────────────

    def _center_crop_pair(
        self, he: Image.Image, ihc: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Deterministic center crop used for validation."""
        w, h = he.size
        if w == self.crop_size and h == self.crop_size:
            return he, ihc
        left = (w - self.crop_size) // 2
        top = (h - self.crop_size) // 2
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return he.crop(box), ihc.crop(box)

    def _random_crop_pair(
        self, he: Image.Image, ihc: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
        """Random crop shared between H&E and IHC. Returns coords for caching."""
        w, h = he.size
        if w == self.crop_size and h == self.crop_size:
            return he, ihc, (0, 0)
        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return he.crop(box), ihc.crop(box), (left, top)

    # ── UNI sub-crops ────────────────────────────────────────────────────────

    def _prepare_uni_subcrops(self, he_pil: Image.Image) -> torch.Tensor:
        """Split into NxN sub-crops, each resized to 224x224 (ImageNet-norm).

        Returns: [N*N, 3, 224, 224]
        """
        w, h = he_pil.size
        n = self.uni_grid_size
        cw, ch = w // n, h // n
        tiles = []
        for i in range(n):
            for j in range(n):
                tile = he_pil.crop((j * cw, i * ch, (j + 1) * cw, (i + 1) * ch))
                tiles.append(self._uni_transform(tile))
        return torch.stack(tiles)  # [n*n, 3, 224, 224]

    # ── augmentation ─────────────────────────────────────────────────────────

    def _paired_augment(
        self, he: Image.Image, ihc: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() > 0.5:
            he, ihc = TF.hflip(he), TF.hflip(ihc)
        if random.random() > 0.5:
            he, ihc = TF.vflip(he), TF.vflip(ihc)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            he = TF.rotate(he, k * 90)
            ihc = TF.rotate(ihc, k * 90)
        if random.random() > 0.7:
            angle = random.uniform(-15, 15)
            trans = [random.uniform(-0.05, 0.05) * self.crop_size] * 2
            scale = random.uniform(0.9, 1.1)
            he = TF.affine(he, angle, trans, scale, 0, T.InterpolationMode.BILINEAR)
            ihc = TF.affine(ihc, angle, trans, scale, 0, T.InterpolationMode.BILINEAR)
        return he, ihc

    def _he_color_augment(self, he: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            he = TF.adjust_brightness(he, random.uniform(0.85, 1.15))
        if random.random() > 0.5:
            he = TF.adjust_contrast(he, random.uniform(0.85, 1.15))
        if random.random() > 0.5:
            he = TF.adjust_saturation(he, random.uniform(0.85, 1.15))
        if random.random() > 0.7:
            he = TF.adjust_hue(he, random.uniform(-0.05, 0.05))
        return he

    # ── tensorize ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        """PIL → [3, H, W] float tensor in [-1, 1]."""
        return TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # ── process pair ─────────────────────────────────────────────────────────

    def _process_pair(
        self,
        he_img: Image.Image,
        ihc_img: Image.Image,
        stain_id: int,
        stain_name: str,
        filename: str,
        dataset_name: str,
        is_train: bool,
    ) -> dict:
        """Full pipeline: crop → augment → tensorize → UNI subcrops.

        Returns a dict with keys:
            he, ihc, uni_subcrops, stain_id, stain_name, filename,
            dataset, crop_coords
        """
        crop_coords = (0, 0)
        if is_train:
            he_crop, ihc_crop, crop_coords = self._random_crop_pair(he_img, ihc_img)
            he_crop, ihc_crop = self._paired_augment(he_crop, ihc_crop)
            he_input = self._he_color_augment(he_crop)
        else:
            he_crop, ihc_crop = self._center_crop_pair(he_img, ihc_img)
            he_input = he_crop

        uni_subcrops = self._prepare_uni_subcrops(he_input)
        he_t = self._to_tensor(he_input)
        ihc_t = self._to_tensor(ihc_crop)

        return {
            "he": he_t,
            "ihc": ihc_t,
            "uni_subcrops": uni_subcrops,
            "stain_id": torch.tensor(stain_id, dtype=torch.long),
            "stain_name": stain_name,
            "filename": filename,
            "dataset": dataset_name,
            "crop_coords": torch.tensor(list(crop_coords), dtype=torch.long),
        }


def build_val_subset(
    dataset: Dataset,
    max_samples: int,
    subset_path: Optional[Path],
    reuse: bool,
    seed: int = 42,
) -> Dataset:
    """Return a (possibly cached) deterministic validation subset.

    If subset_path exists and reuse=True, reload saved indices.
    Otherwise sample max_samples and save indices to subset_path.
    If max_samples < 0, return the full dataset.
    """
    if max_samples < 0 or max_samples >= len(dataset):
        return dataset

    if subset_path is not None and subset_path.exists() and reuse:
        info = json.loads(subset_path.read_text())
        indices = info["indices"]
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), min(max_samples, len(dataset)))
        indices.sort()
        if subset_path is not None:
            subset_path.parent.mkdir(parents=True, exist_ok=True)
            subset_path.write_text(json.dumps({"indices": indices, "seed": seed}))

    return Subset(dataset, indices)
