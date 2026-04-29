"""
MIST multi-stain paired dataset.

Layout on disk:
  {mist_dir}/{stain}/TrainValAB/{trainA,trainB,valA,valB}/
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .base import CropPairedDataset, STAIN_TO_LABEL, LABEL_TO_STAIN, NULL_STAIN_LABEL


class MISTMultiStainDataset(CropPairedDataset):
    """Unified dataset that loads all requested MIST stains together.

    Stain identity is encoded as an integer label and a string name
    returned in each sample dict so training code can log per-stain metrics.
    """

    def __init__(
        self,
        base_dir: str | Path,
        stains: List[str] | None = None,
        split: str = "train",
        crop_size: int = 512,
        augment: bool = False,
        uni_grid_size: int = 4,
        uni_subcrop_size: int = 224,
    ):
        is_train = split == "train"
        super().__init__(crop_size, augment and is_train, uni_grid_size, uni_subcrop_size)
        self.base_dir = Path(base_dir)
        self.stains = stains or list(STAIN_TO_LABEL.keys())
        self._is_train = is_train
        self._split = split

        split_he = "trainA" if is_train else "valA"
        split_ihc = "trainB" if is_train else "valB"
        valid_exts = (".jpg", ".jpeg", ".png")

        self.samples: List[Dict] = []
        for stain in self.stains:
            if stain not in STAIN_TO_LABEL:
                raise ValueError(f"Unknown stain '{stain}'. Available: {list(STAIN_TO_LABEL)}")
            label = STAIN_TO_LABEL[stain]
            he_dir = self.base_dir / stain / "TrainValAB" / split_he
            ihc_dir = self.base_dir / stain / "TrainValAB" / split_ihc

            for d in (he_dir, ihc_dir):
                if not d.exists():
                    raise FileNotFoundError(f"Directory not found: {d}")

            he_files = {
                Path(f).stem: f
                for f in os.listdir(he_dir)
                if f.lower().endswith(valid_exts)
            }
            ihc_files = {
                Path(f).stem: f
                for f in os.listdir(ihc_dir)
                if f.lower().endswith(valid_exts)
            }
            common = sorted(set(he_files) & set(ihc_files))
            for stem in common:
                self.samples.append(
                    dict(
                        he_path=he_dir / he_files[stem],
                        ihc_path=ihc_dir / ihc_files[stem],
                        stain_id=label,
                        stain_name=stain,
                    )
                )
            print(f"  MIST {stain} ({split}): {len(common)} pairs")

        dist = Counter(s["stain_id"] for s in self.samples)
        counts = {LABEL_TO_STAIN[k]: v for k, v in sorted(dist.items())}
        print(f"MISTMultiStainDataset ({split}): {len(self.samples)} total | {counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        he_img = Image.open(s["he_path"]).convert("RGB")
        ihc_img = Image.open(s["ihc_path"]).convert("RGB")
        return self._process_pair(
            he_img, ihc_img,
            stain_id=s["stain_id"],
            stain_name=s["stain_name"],
            filename=s["he_path"].name,
            dataset_name="mist",
            is_train=self._is_train,
        )
