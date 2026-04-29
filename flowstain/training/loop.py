"""
FlowStain Accelerate training loop.

Usage:
    accelerate launch scripts/train_flowstain.py --config configs/mist_flowstain.yaml

Supports:
    - Mixed-precision (bf16/fp16)
    - Gradient accumulation
    - EMA model
    - W&B logging (scalars + image grids)
    - Subset validation with per-sample CSV and JSON
    - Three training phases:
        Phase 1: L_flow only
        Phase 2: + decoded rollout losses (DAB, edge, perceptual)
        Phase 3: + adversarial (future, not implemented)
    - Checkpoint save/resume
    - CFG dropout for conditioning signals
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.factory import build_datasets, build_dataloaders, build_val_subset_from_cfg
from ..foundation.conch_tissue import CONCHTissueClassifier
from ..losses.flow_losses import build_noisy_latent, flow_loss, sample_timesteps
from ..losses.stain_losses import DABLoss
from ..losses.structure_losses import EdgeLoss, PerceptualLoss
from ..losses.tissue_weighting import TissueWeightedLoss, build_tissue_weight_vector
from ..metrics.retrieval import RetrievalEvaluator
from ..models.flowstain_model import FlowStainModel
from .checkpointing import load_checkpoint, save_checkpoint
from .validation import run_validation, log_val_images_to_wandb

logger = get_logger(__name__, log_level="INFO")


def build_model(cfg: DictConfig) -> FlowStainModel:
    return FlowStainModel(
        vae_model_name=cfg.autoencoder.model_name,
        vae_scale_factor=cfg.autoencoder.scale_factor,
        uni_model_name=cfg.foundation.uni.model_name,
        uni_grid_size=cfg.foundation.uni.grid_size,
        uni_base_channels=cfg.conditioning.uni.base_channels,
        flow_base_channels=cfg.flow.base_channels,
        flow_channel_mults=tuple(cfg.flow.channel_mults),
        flow_num_res_blocks=cfg.flow.num_res_blocks,
        stain_num_embeddings=cfg.conditioning.stain_embedding.num_stains,
        stain_emb_dim=cfg.conditioning.stain_embedding.dim,
        cond_mode=cfg.conditioning.uni.mode,
    )


def build_losses(cfg: DictConfig):
    losses = {}
    if cfg.losses.dab.enabled:
        losses["dab"] = DABLoss(
            top_percentile=cfg.losses.dab.top_percentile,
            histogram_bins=cfg.losses.dab.histogram_bins,
        )
    if cfg.losses.edge.enabled:
        losses["edge"] = EdgeLoss()
    if cfg.losses.perceptual.enabled:
        losses["perceptual"] = PerceptualLoss(
            scales=tuple(cfg.losses.perceptual.scales)
        )
    return losses


def _cfg_dropout_masks(batch_size: int, dropout_cfg: DictConfig, device) -> tuple:
    """Return (drop_stain_mask [B], drop_uni_mask [B]) boolean tensors."""
    sp = dropout_cfg.stain_prob
    up = dropout_cfg.uni_prob
    bp = dropout_cfg.both_prob

    r = torch.rand(batch_size, device=device)
    drop_both = r < bp
    drop_stain = (r >= bp) & (r < bp + sp)
    drop_uni = (r >= bp + sp) & (r < bp + sp + up)

    return drop_both | drop_stain, drop_both | drop_uni


NULL_STAIN_LABEL = 4


def train(cfg: DictConfig):
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_cfg = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to if cfg.logging.report_to != "none" else None,
        project_config=proj_cfg,
    )

    set_seed(42)

    if accelerator.is_main_process:
        run_name = cfg.logging.get("wandb_run_name") or None
        accelerator.init_trackers(
            cfg.logging.wandb_project,
            config=dict(cfg),
            init_kwargs={"wandb": {"name": run_name}},
        )

    # ── datasets & loaders ───────────────────────────────────────────────────
    train_ds, val_ds_full = build_datasets(cfg)
    val_ds = build_val_subset_from_cfg(val_ds_full, cfg, output_dir)

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg)

    # ── model ────────────────────────────────────────────────────────────────
    model = build_model(cfg)

    # Only flow_net + uni_processor are trainable
    trainable_params = list(model.flow_net.parameters()) + list(model.uni_processor.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.learning_rate)

    total_steps = (
        len(train_loader) * cfg.training.num_epochs
        // cfg.training.gradient_accumulation_steps
    )
    scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema_model = None
    if cfg.training.ema.enabled:
        ema_model = EMAModel(
            model.flow_net.parameters(),
            decay=cfg.training.ema.decay,
        )

    # ── losses ────────────────────────────────────────────────────────────────
    extra_losses = build_losses(cfg)
    for loss_fn in extra_losses.values():
        loss_fn.to(accelerator.device)

    # Tissue weighting
    tissue_weighter = None
    if cfg.tissue_weighting.enabled:
        wv = build_tissue_weight_vector(cfg.tissue_weighting.weights)
        tissue_weighter = TissueWeightedLoss(
            wv,
            min_weight=cfg.tissue_weighting.min_weight,
            max_weight=cfg.tissue_weighting.max_weight,
            entropy_blend=cfg.tissue_weighting.entropy_blend,
        ).to(accelerator.device)

    # CONCH tissue classifier
    conch = None
    if cfg.foundation.conch.enabled and cfg.tissue_weighting.enabled:
        tissue_cache_dir = (
            Path(cfg.foundation.conch.tissue_cache_dir)
            if cfg.foundation.conch.tissue_cache_dir
            else output_dir / "tissue_cache"
        )
        conch = CONCHTissueClassifier(
            model_name=cfg.foundation.conch.model_name,
            tissue_prompts_yaml=cfg.foundation.conch.tissue_prompts,
            cache_dir=tissue_cache_dir if cfg.foundation.conch.cache_tissue_probs else None,
        ).to(accelerator.device)

    # Retrieval evaluator (main process only, not prepared)
    retrieval_eval = None
    if cfg.metrics.retrieval.get("enabled", True) and accelerator.is_main_process:
        retrieval_eval = RetrievalEvaluator(
            model_name=cfg.metrics.retrieval.model,
            device=accelerator.device,
        )

    # ── accelerate prepare ───────────────────────────────────────────────────
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if ema_model is not None:
        ema_model.to(accelerator.device)

    # ── resume ───────────────────────────────────────────────────────────────
    global_step, start_epoch = load_checkpoint(output_dir, accelerator, ema_model)

    # ── training ─────────────────────────────────────────────────────────────
    flow_cfg = cfg.losses.flow
    bridge_sigma = cfg.flow.bridge_noise.sigma_max if cfg.flow.bridge_noise.enabled else 0.0
    noise_source = cfg.flow.source

    log_every = cfg.logging.log_every_steps
    save_every_epochs = cfg.logging.save_every_epochs
    val_every_epochs = cfg.validation.val_every_epochs
    phase2_start = cfg.training.phase2_start_step

    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
        ):
            with accelerator.accumulate(model):
                he = batch["he"]
                ihc = batch["ihc"]
                uni_sc = batch["uni_subcrops"]
                stain_id = batch["stain_id"]
                filenames = list(batch["filename"])
                stain_names_batch = batch.get("stain_name", [""] * he.shape[0])
                B = he.shape[0]
                device = he.device

                # ── encode latents (frozen VAE) ───────────────────────────
                unwrapped: FlowStainModel = accelerator.unwrap_model(model)
                with torch.no_grad():
                    z_he, z_ihc = unwrapped.encode_pair(he, ihc)
                    uni_tokens = unwrapped.encode_uni(uni_sc)

                # ── CFG dropout ───────────────────────────────────────────
                drop_stain_mask, drop_uni_mask = _cfg_dropout_masks(
                    B, cfg.training.cfg_dropout, device
                )
                stain_id_in = stain_id.clone()
                stain_id_in[drop_stain_mask] = NULL_STAIN_LABEL
                uni_maps, uni_tokens_cond = unwrapped.get_uni_maps(
                    uni_tokens, dropout_mask=drop_uni_mask
                )

                # ── tissue probabilities ──────────────────────────────────
                tissue_probs = None
                tissue_top_names = None
                if conch is not None:
                    with torch.no_grad():
                        tissue_result = conch.classify(
                            he,
                            filenames=filenames,
                            stain_names=list(stain_names_batch),
                        )
                    tissue_probs = tissue_result["probs"]
                    tissue_top_names = tissue_result["top_names"]

                # ── flow forward ──────────────────────────────────────────
                t = sample_timesteps(B, device)
                z_t, v_target = build_noisy_latent(
                    z_he, z_ihc, t,
                    sigma_max=bridge_sigma,
                    noise_source=noise_source,
                )
                v_pred = model(z_t, t, z_he, uni_maps, uni_tokens_cond, stain_id_in)

                # ── losses ────────────────────────────────────────────────
                flow_per_sample = flow_loss(
                    v_pred, v_target,
                    loss_type=flow_cfg.loss_type,
                    huber_delta=flow_cfg.huber_delta,
                    reduce=False,
                )

                if tissue_weighter is not None:
                    L_flow = tissue_weighter(flow_per_sample, tissue_probs)
                else:
                    L_flow = flow_per_sample.mean()

                total_loss = flow_cfg.weight * L_flow
                loss_log = {"train/loss_flow": float(L_flow.item())}

                # Phase 2: decoded image-space losses
                if global_step >= phase2_start and (
                    cfg.losses.dab.enabled
                    or cfg.losses.edge.enabled
                    or cfg.losses.perceptual.enabled
                ):
                    with torch.no_grad():
                        # Quick 4-step rollout for gradient target
                        z_pred_img = unwrapped.sample_euler(
                            z_he, uni_maps, uni_tokens_cond, stain_id, num_steps=4
                        )
                        pred_ihc_img = unwrapped.decode(z_pred_img).clamp(-1, 1)

                    # DAB loss
                    if "dab" in extra_losses:
                        dab_per_sample = extra_losses["dab"](
                            pred_ihc_img.detach(), ihc, reduce=False
                        )
                        if tissue_weighter is not None:
                            L_dab = tissue_weighter(dab_per_sample, tissue_probs)
                        else:
                            L_dab = dab_per_sample.mean()
                        total_loss = total_loss + cfg.losses.dab.weight * L_dab
                        loss_log["train/loss_dab"] = float(L_dab.item())

                    # Edge loss (H&E-aligned, no tissue weighting needed)
                    if "edge" in extra_losses:
                        L_edge = extra_losses["edge"](pred_ihc_img.detach(), he, reduce=True)
                        total_loss = total_loss + cfg.losses.edge.weight * L_edge
                        loss_log["train/loss_edge"] = float(L_edge.item())

                    # Perceptual loss
                    if "perceptual" in extra_losses:
                        perc_per_sample = extra_losses["perceptual"](
                            pred_ihc_img.detach(), ihc, reduce=False
                        )
                        if tissue_weighter is not None:
                            L_perc = tissue_weighter(perc_per_sample, tissue_probs)
                        else:
                            L_perc = perc_per_sample.mean()
                        total_loss = total_loss + cfg.losses.perceptual.weight * L_perc
                        loss_log["train/loss_perceptual"] = float(L_perc.item())

                loss_log["train/loss_total"] = float(total_loss.item())

                # ── backward ──────────────────────────────────────────────
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_losses.append(float(total_loss.item()))

            if accelerator.sync_gradients:
                if ema_model is not None:
                    ema_model.step(unwrapped.flow_net.parameters())
                global_step += 1

                # ── W&B scalar logging ───────────────────────────────────
                if global_step % log_every == 0 and accelerator.is_main_process:
                    loss_log["train/lr"] = float(scheduler.get_last_lr()[0])
                    if tissue_probs is not None and tissue_weighter is not None:
                        w = tissue_weighter.compute_weights(tissue_probs)
                        loss_log["train/tissue_weight_mean"] = float(w.mean().item())
                        from ..foundation.conch_tissue import TISSUE_CLASSES
                        import math
                        eps = 1e-8
                        H = -(tissue_probs * (tissue_probs + eps).log()).sum(dim=-1)
                        H_norm = (H / math.log(len(TISSUE_CLASSES))).mean()
                        loss_log["train/tissue_entropy_mean"] = float(H_norm.item())
                    accelerator.log(loss_log, step=global_step)

        # ── end of epoch ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            accelerator.log(
                {"train/epoch_loss": sum(epoch_losses) / max(len(epoch_losses), 1)},
                step=global_step,
            )

        # ── validation ───────────────────────────────────────────────────
        if (epoch + 1) % val_every_epochs == 0:
            if accelerator.is_main_process:
                unwrapped_eval = accelerator.unwrap_model(model)
                if ema_model is not None:
                    ema_model.store(unwrapped_eval.flow_net.parameters())
                    ema_model.copy_to(unwrapped_eval.flow_net.parameters())

                val_metrics, wandb_samples = run_validation(
                    unwrapped_eval,
                    val_loader,
                    cfg,
                    accelerator,
                    step=global_step,
                    output_dir=output_dir,
                    conch_classifier=conch,
                    retrieval_evaluator=retrieval_eval,
                )

                accelerator.log(val_metrics, step=global_step)
                log_val_images_to_wandb(
                    wandb_samples,
                    step=global_step,
                    failure_threshold=cfg.validation.dab_kl_failure_threshold,
                )

                if ema_model is not None:
                    ema_model.restore(unwrapped_eval.flow_net.parameters())

        # ── checkpoint ───────────────────────────────────────────────────
        if (epoch + 1) % save_every_epochs == 0 and accelerator.is_main_process:
            save_checkpoint(
                output_dir, global_step, epoch,
                model, optimizer, scheduler,
                accelerator, ema_model, cfg,
            )

    accelerator.end_training()
