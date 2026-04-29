"""
Ablation helper: quickly produce a modified config for a named ablation.

Each ablation function takes the base DictConfig and returns a copy with
the relevant switches flipped.  Pass the returned config to train().

Usage:
    from flowstain.training.ablations import ablation_noise_source
    cfg = load_config("configs/mist_flowstain.yaml")
    ablation_cfg = ablation_noise_source(cfg)
    train(ablation_cfg)

All ablations are also achievable via CLI overrides:
    accelerate launch scripts/train_flowstain.py \\
        --config configs/mist_flowstain.yaml \\
        flow.source=noise \\
        logging.wandb_run_name=ablation_noise_source
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def _copy(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


# ── flow source ──────────────────────────────────────────────────────────────

def ablation_noise_source(cfg: DictConfig) -> DictConfig:
    """Baseline: flow from Gaussian noise instead of H&E latent."""
    c = _copy(cfg)
    c.flow.source = "noise"
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_noise_source"
    return c


# ── UNI conditioning ─────────────────────────────────────────────────────────

def ablation_no_uni(cfg: DictConfig) -> DictConfig:
    """Remove all UNI conditioning."""
    c = _copy(cfg)
    c.conditioning.uni.enabled = False
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_no_uni"
    return c


def ablation_uni_spade_only(cfg: DictConfig) -> DictConfig:
    """Use SPADE conditioning only (no cross-attention)."""
    c = _copy(cfg)
    c.conditioning.uni.mode = "spade"
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_spade_only"
    return c


def ablation_uni_xattn_only(cfg: DictConfig) -> DictConfig:
    """Use cross-attention conditioning only (no SPADE)."""
    c = _copy(cfg)
    c.conditioning.uni.mode = "cross_attention"
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_xattn_only"
    return c


# ── bridge noise ─────────────────────────────────────────────────────────────

def ablation_no_bridge_noise(cfg: DictConfig) -> DictConfig:
    """Disable mid-trajectory bridge noise (pure linear interpolation)."""
    c = _copy(cfg)
    c.flow.bridge_noise.enabled = False
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_no_bridge_noise"
    return c


# ── tissue weighting ─────────────────────────────────────────────────────────

def ablation_no_tissue_weighting(cfg: DictConfig) -> DictConfig:
    """Disable tissue-aware loss weighting."""
    c = _copy(cfg)
    c.tissue_weighting.enabled = False
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_no_tissue_w"
    return c


# ── DAB loss ─────────────────────────────────────────────────────────────────

def ablation_no_dab(cfg: DictConfig) -> DictConfig:
    """Disable DAB stain supervision."""
    c = _copy(cfg)
    c.losses.dab.enabled = False
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_no_dab"
    return c


# ── sampling method ───────────────────────────────────────────────────────────

def ablation_heun_sampler(cfg: DictConfig) -> DictConfig:
    """Use Heun's method for ODE integration."""
    c = _copy(cfg)
    c.flow.sampling.method = "heun"
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_heun"
    return c


def ablation_stochastic_euler(cfg: DictConfig) -> DictConfig:
    """Euler sampler with stochastic bridge noise at inference."""
    c = _copy(cfg)
    c.flow.sampling.method = "stochastic_euler"
    c.logging.wandb_run_name = (c.logging.get("wandb_run_name") or "run") + "_stochastic"
    return c


# ── Registry of all ablations ────────────────────────────────────────────────

ABLATION_REGISTRY = {
    "noise_source": ablation_noise_source,
    "no_uni": ablation_no_uni,
    "uni_spade_only": ablation_uni_spade_only,
    "uni_xattn_only": ablation_uni_xattn_only,
    "no_bridge_noise": ablation_no_bridge_noise,
    "no_tissue_weighting": ablation_no_tissue_weighting,
    "no_dab": ablation_no_dab,
    "heun_sampler": ablation_heun_sampler,
    "stochastic_euler": ablation_stochastic_euler,
}
