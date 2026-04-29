"""
BCI HER2 paired dataset.

Layout on disk:
  {bci_dir}/HE/{train,test}/  *.png
  {bci_dir}/IHC/{train,test}/ *.png

Filenames encode HER2 grade: <id>_<id>_<grade>_<...>.png
Grades: 0, 1+, 2+, 3+  → labels 0, 1, 2, 3
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from PIL import Image

from .base import CropPairedDataset


HER2_LABEL_MAP = {"0": 0, "1+": 1, "2+": 2, "3+": 3}
LABEL_TO_HER2 = {v: k for k, v in HER2_LABEL_MAP.items()}


class BCIDataset(CropPairedDataset):
    """BCI HER2 dataset with optional HER2-grade label."""

    def __init__(
        self,
        bci_dir: str | Path,
        split: str = "train",
        crop_size: int = 512,
        augment: bool = False,
        uni_grid_size: int = 4,
        uni_subcrop_size: int = 224,
        use_grade_label: bool = True,
    ):
        is_train = split == "train"
        super().__init__(crop_size, augment and is_train, uni_grid_size, uni_subcrop_size)
        self.bci_dir = Path(bci_dir)
        self._is_train = is_train
        self._use_grade_label = use_grade_label

        folder = "train" if is_train else "test"
        he_dir = self.bci_dir / "HE" / folder
        ihc_dir = self.bci_dir / "IHC" / folder

        for d in (he_dir, ihc_dir):
            if not d.exists():
                raise FileNotFoundError(f"BCI directory not found: {d}")

        he_files = sorted(f for f in os.listdir(he_dir) if f.endswith(".png"))
        ihc_files = sorted(f for f in os.listdir(ihc_dir) if f.endswith(".png"))
        assert len(he_files) == len(ihc_files), (
            f"BCI H&E/IHC count mismatch: {len(he_files)} vs {len(ihc_files)}"
        )

        self.he_paths = [he_dir / f for f in he_files]
        self.ihc_paths = [ihc_dir / f for f in ihc_files]
        self.labels = [self._parse_label(f) for f in he_files]

        dist = Counter(self.labels)
        counts = {LABEL_TO_HER2[k]: v for k, v in sorted(dist.items())}
        print(f"BCIDataset ({split}): {len(self)} images | {counts}")

    def _parse_label(self, filename: str) -> int:
        parts = filename.replace(".png", "").split("_")
        if len(parts) >= 3 and parts[2] in HER2_LABEL_MAP:
            return HER2_LABEL_MAP[parts[2]]
        raise ValueError(f"Cannot parse HER2 grade from filename: {filename}")

    def __len__(self) -> int:
        return len(self.he_paths)

    def __getitem__(self, idx: int) -> dict:
        he_img = Image.open(self.he_paths[idx]).convert("RGB")
        ihc_img = Image.open(self.ihc_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return self._process_pair(
            he_img, ihc_img,
            stain_id=label,
            stain_name=f"HER2_{LABEL_TO_HER2[label]}",
            filename=self.he_paths[idx].name,
            dataset_name="bci",
            is_train=self._is_train,
        )
