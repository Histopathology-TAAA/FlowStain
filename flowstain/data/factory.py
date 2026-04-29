"""
Dataset factory and DataLoader builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .base import build_val_subset
from .bci import BCIDataset
from .mist import MISTMultiStainDataset


def build_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) from config."""
    name = cfg.dataset

    if name == "mist":
        stains = list(cfg.get("stains", ["HER2", "Ki67", "ER", "PR"]))
        train_ds = MISTMultiStainDataset(
            cfg.mist_dir, stains=stains, split="train",
            crop_size=cfg.crop_size, augment=cfg.get("augment", True),
            uni_grid_size=cfg.foundation.uni.grid_size,
            uni_subcrop_size=cfg.foundation.uni.subcrop_size,
        )
        val_ds = MISTMultiStainDataset(
            cfg.mist_dir, stains=stains, split="val",
            crop_size=cfg.crop_size, augment=False,
            uni_grid_size=cfg.foundation.uni.grid_size,
            uni_subcrop_size=cfg.foundation.uni.subcrop_size,
        )

    elif name == "bci":
        train_ds = BCIDataset(
            cfg.bci_dir, split="train",
            crop_size=cfg.crop_size, augment=cfg.get("augment", True),
            uni_grid_size=cfg.foundation.uni.grid_size,
            uni_subcrop_size=cfg.foundation.uni.subcrop_size,
        )
        val_ds = BCIDataset(
            cfg.bci_dir, split="test",
            crop_size=cfg.crop_size, augment=False,
            uni_grid_size=cfg.foundation.uni.grid_size,
            uni_subcrop_size=cfg.foundation.uni.subcrop_size,
        )

    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'mist' or 'bci'.")

    return train_ds, val_ds


def build_val_subset_from_cfg(
    val_ds: Dataset, cfg: DictConfig, output_dir: Path
) -> Dataset:
    max_samples = cfg.validation.val_max_samples
    reuse = cfg.validation.get("reuse_val_subset", True)
    subset_path = output_dir / "val_subset.json"
    return build_val_subset(val_ds, max_samples, subset_path, reuse)


def build_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader]:
    bs = cfg.training.batch_size
    nw = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader
