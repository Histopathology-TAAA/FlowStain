"""
Checkpoint save/load utilities for FlowStain Accelerate training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
    ema_model: Optional[EMAModel] = None,
    cfg=None,
    keep_last_n: int = 3,
):
    """Save a training checkpoint to output_dir/checkpoints/step_{step:07d}/."""
    ckpt_dir = output_dir / "checkpoints" / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accelerator.save_state(str(ckpt_dir))

    if ema_model is not None:
        torch.save(
            ema_model.state_dict(),
            ckpt_dir / "ema_model.pt",
        )

    meta = {"step": step, "epoch": epoch}
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))

    if cfg is not None:
        (ckpt_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Pointer to latest checkpoint
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt_dir))

    # Prune old checkpoints
    ckpt_root = output_dir / "checkpoints"
    all_ckpts = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in all_ckpts[:-keep_last_n]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(
    output_dir: Path,
    accelerator: Accelerator,
    ema_model: Optional[EMAModel] = None,
    specific_step: Optional[int] = None,
):
    """Load the latest (or specific) checkpoint.

    Returns (step, epoch) from checkpoint metadata.
    """
    if specific_step is not None:
        ckpt_dir = output_dir / "checkpoints" / f"step_{specific_step:07d}"
    else:
        latest = output_dir / "latest_checkpoint.txt"
        if not latest.exists():
            return 0, 0
        ckpt_dir = Path(latest.read_text().strip())

    if not ckpt_dir.exists():
        return 0, 0

    accelerator.load_state(str(ckpt_dir))

    if ema_model is not None and (ckpt_dir / "ema_model.pt").exists():
        ema_model.load_state_dict(
            torch.load(ckpt_dir / "ema_model.pt", map_location="cpu")
        )

    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("step", 0), meta.get("epoch", 0)
    return 0, 0
