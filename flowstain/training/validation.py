"""
Validation runner for FlowStain.

Runs on the main Accelerate process.  Generates IHC for the validation
subset, computes all enabled metrics, saves per-sample CSV, and logs
image grids and tissue distributions to W&B.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..foundation.conch_tissue import CONCHTissueClassifier
from ..metrics.fid_kid import compute_fid, compute_kid
from ..metrics.image_quality import compute_ssim_batch, compute_lpips_batch, compute_dab_metrics_batch
from ..metrics.retrieval import RetrievalEvaluator
from ..metrics.tissue_stratified import aggregate_per_tissue, compute_dab_pearson, save_per_sample_csv
from ..models.flowstain_model import FlowStainModel


def make_dab_heatmap(images: torch.Tensor) -> torch.Tensor:
    """Generate a DAB heatmap [B, 3, H, W] from IHC images (jet colormap)."""
    from ..losses.stain_losses import rgb_to_dab
    dab = rgb_to_dab(images)  # [B, H, W]
    dab_norm = (dab / (dab.max() + 1e-8)).clamp(0, 1)

    # Approximate jet colormap
    r = (dab_norm * 4 - 1.5).clamp(0, 1)
    g = (1 - (dab_norm * 4 - 2.0).abs().clamp(0, 1))
    b = (1 - (dab_norm * 4 - 0.5).clamp(0, 1))

    heatmap = torch.stack([r, g, b], dim=1) * 2 - 1  # to [-1, 1]
    return heatmap


@torch.no_grad()
def run_validation(
    model: FlowStainModel,
    val_loader: DataLoader,
    cfg: DictConfig,
    accelerator: Accelerator,
    step: int,
    output_dir: Path,
    conch_classifier: Optional[CONCHTissueClassifier] = None,
    retrieval_evaluator: Optional[RetrievalEvaluator] = None,
) -> Dict:
    """Full validation pass.  Returns a flat dict of scalar metrics."""
    model.eval()

    sampling_cfg = cfg.flow.sampling
    num_steps = cfg.validation.get("num_inference_steps", cfg.flow.num_inference_steps)
    sample_method = sampling_cfg.method
    sigma_bridge = (
        cfg.flow.bridge_noise.sigma_max
        if cfg.flow.bridge_noise.enabled
        else 0.0
    )

    pred_images: List[torch.Tensor] = []
    real_images: List[torch.Tensor] = []
    he_images: List[torch.Tensor] = []
    all_records: List[dict] = []

    all_per_sample: Dict[str, list] = defaultdict(list)
    tissue_labels_all: List[str] = []

    num_wandb_images = cfg.validation.wandb_num_val_images
    wandb_samples: List[dict] = []  # {he, real, pred, dab_real, dab_pred, meta}

    for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):
        he = batch["he"].to(accelerator.device)
        ihc = batch["ihc"].to(accelerator.device)
        uni_sc = batch["uni_subcrops"].to(accelerator.device)
        stain_id = batch["stain_id"].to(accelerator.device)
        filenames = batch["filename"]
        stain_names = batch["stain_name"]
        B = he.shape[0]

        # Encode to latent space
        z_he = model.vae.encode(he)

        # Extract UNI features (no CFG dropout at validation)
        uni_tokens = model.encode_uni(uni_sc)
        uni_maps, uni_tokens_cond = model.get_uni_maps(uni_tokens)

        # Generate IHC via ODE
        if sample_method == "heun":
            z_pred = model.sample_heun(z_he, uni_maps, uni_tokens_cond, stain_id, num_steps)
        else:
            z_pred = model.sample_euler(
                z_he, uni_maps, uni_tokens_cond, stain_id, num_steps, sigma_bridge
            )

        pred_ihc = model.decode(z_pred).clamp(-1, 1)

        # CONCH tissue classification
        tissue_result = None
        if conch_classifier is not None and cfg.foundation.conch.enabled:
            tissue_result = conch_classifier.classify(
                he, filenames=list(filenames), stain_names=list(stain_names)
            )

        # Per-sample metrics
        ssim_scores = compute_ssim_batch(pred_ihc.cpu(), ihc.cpu())
        lpips_scores = compute_lpips_batch(pred_ihc, ihc.cpu(), scale=256)
        dab = compute_dab_metrics_batch(pred_ihc.cpu(), ihc.cpu())

        for i in range(B):
            tissue_name = (
                tissue_result["top_names"][i] if tissue_result else "unknown"
            )
            record = {
                "filename": filenames[i],
                "stain_name": stain_names[i] if isinstance(stain_names[i], str) else stain_names[i],
                "step": step,
                "tissue_label": tissue_name,
                "ssim": float(ssim_scores[i]),
                "lpips_256": float(lpips_scores[i]),
                "dab_kl": float(dab["dab_kl"][i]),
                "pred_top_dab": float(dab["pred_top_dab"][i]),
                "real_top_dab": float(dab["real_top_dab"][i]),
                "failure": float(dab["dab_kl"][i]) > cfg.validation.dab_kl_failure_threshold,
            }
            if tissue_result is not None:
                record["tissue_entropy"] = float(tissue_result["entropy"][i])
            all_records.append(record)
            tissue_labels_all.append(tissue_name)

        for k, v in dab.items():
            all_per_sample[k].extend(v.tolist())
        all_per_sample["ssim"].extend(ssim_scores.tolist())
        all_per_sample["lpips_256"].extend(lpips_scores.tolist())

        pred_images.extend([pred_ihc[i].cpu() for i in range(B)])
        real_images.extend([ihc[i].cpu() for i in range(B)])
        he_images.extend([he[i].cpu() for i in range(B)])

        # Collect W&B image samples
        if len(wandb_samples) < num_wandb_images:
            batch_record_start = len(all_records) - B
            for i in range(min(B, num_wandb_images - len(wandb_samples))):
                wandb_samples.append({
                    "he": he[i].cpu(),
                    "real": ihc[i].cpu(),
                    "pred": pred_ihc[i].cpu(),
                    "dab_real": make_dab_heatmap(ihc[i:i+1].cpu())[0],
                    "dab_pred": make_dab_heatmap(pred_ihc[i:i+1].cpu())[0],
                    "meta": all_records[batch_record_start + i],
                })

    # ── aggregate scalars ────────────────────────────────────────────────────
    per_sample_tensors = {k: torch.tensor(v) for k, v in all_per_sample.items()}

    metrics: Dict = {}
    metrics["val/ssim"] = float(per_sample_tensors["ssim"].mean())
    metrics["val/lpips_256"] = float(per_sample_tensors["lpips_256"].mean())
    metrics["val/dab_kl_mean"] = float(per_sample_tensors["dab_kl"].mean())
    metrics["val/dab_pearson"] = compute_dab_pearson(
        per_sample_tensors["pred_top_dab"],
        per_sample_tensors["real_top_dab"],
    )
    failure_thresh = cfg.validation.dab_kl_failure_threshold
    metrics["val/failure_rate"] = float(
        (per_sample_tensors["dab_kl"] > failure_thresh).float().mean()
    )

    # ── FID / KID ────────────────────────────────────────────────────────────
    if cfg.metrics.get("fid", True):
        metrics["val/fid"] = compute_fid(pred_images, real_images, accelerator.device)
    if cfg.metrics.get("kid", True):
        kid_mean, kid_std = compute_kid(pred_images, real_images, device=accelerator.device)
        metrics["val/kid_mean"] = kid_mean
        metrics["val/kid_std"] = kid_std

    # ── retrieval / MRA ───────────────────────────────────────────────────────
    if retrieval_evaluator is not None and cfg.metrics.retrieval.get("enabled", True):
        metrics["val/mra"] = retrieval_evaluator.compute_mra(pred_images, real_images)

    # ── per-stain ────────────────────────────────────────────────────────────
    stain_names_all = [r["stain_name"] for r in all_records]
    stain_groups = defaultdict(list)
    for i, sn in enumerate(stain_names_all):
        stain_groups[sn].append(i)
    for stain, idxs in stain_groups.items():
        safe = stain.replace("+", "plus").replace(" ", "_")
        metrics[f"val/stain/{safe}/dab_kl"] = float(
            per_sample_tensors["dab_kl"][idxs].mean()
        )
        metrics[f"val/stain/{safe}/dab_pearson"] = compute_dab_pearson(
            per_sample_tensors["pred_top_dab"][idxs],
            per_sample_tensors["real_top_dab"][idxs],
        )
        metrics[f"val/stain/{safe}/ssim"] = float(
            per_sample_tensors["ssim"][idxs].mean()
        )

    # ── per-tissue ────────────────────────────────────────────────────────────
    if tissue_labels_all:
        tissue_summary = aggregate_per_tissue(
            per_sample_tensors,
            tissue_labels_all,
            failure_threshold_dab_kl=failure_thresh,
        )
        for tissue, tm in tissue_summary.items():
            for metric_name, val in tm.items():
                metrics[f"val/tissue/{tissue}/{metric_name}"] = val

    # ── save per-sample CSV ───────────────────────────────────────────────────
    if accelerator.is_main_process:
        csv_path = output_dir / f"val_metrics_step_{step:07d}.csv"
        save_per_sample_csv(all_records, csv_path)

        json_path = output_dir / f"val_metrics_step_{step:07d}.json"
        json_path.write_text(json.dumps(metrics, indent=2))

    return metrics, wandb_samples


def log_val_images_to_wandb(
    wandb_samples: list,
    step: int,
    failure_threshold: float = 0.5,
):
    """Create W&B image grids from validation samples.

    Each sample row: H&E | Real IHC | Generated IHC | DAB heatmap real | DAB heatmap pred | DAB diff
    """
    try:
        import wandb

        def to_pil(t: torch.Tensor):
            """[3, H, W] in [-1, 1] → PIL"""
            from PIL import Image
            import numpy as np
            arr = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
            return Image.fromarray(arr)

        images = []
        for s in wandb_samples:
            meta = s["meta"]
            fail = meta.get("failure", False)
            caption = (
                f"{meta.get('stain_name', '')} | {meta.get('tissue_label', '')} | "
                f"DAB KL={meta.get('dab_kl', 0):.3f} | "
                f"{'FAIL' if fail else 'OK'}"
            )
            # Stitch: H&E | real | pred | DAB real | DAB pred
            row_imgs = [
                wandb.Image(to_pil(s["he"]), caption=f"H&E {caption}"),
                wandb.Image(to_pil(s["real"]), caption=f"Real IHC"),
                wandb.Image(to_pil(s["pred"]), caption=f"Generated IHC"),
                wandb.Image(to_pil(s["dab_real"]), caption="DAB real"),
                wandb.Image(to_pil(s["dab_pred"]), caption="DAB pred"),
            ]
            images.extend(row_imgs)

        log_dict = {"val/image_grid": images, "step": step}

        # Tissue-grouped subsets (first image per tissue)
        tissue_first: Dict[str, dict] = {}
        for s in wandb_samples:
            t = s["meta"].get("tissue_label", "unknown")
            if t not in tissue_first:
                tissue_first[t] = s

        for tissue, s in tissue_first.items():
            log_dict[f"val/tissue_examples/{tissue}"] = [
                wandb.Image(to_pil(s["he"]), caption="H&E"),
                wandb.Image(to_pil(s["pred"]), caption="Generated"),
            ]

        wandb.log(log_dict, step=step)

    except ImportError:
        pass
