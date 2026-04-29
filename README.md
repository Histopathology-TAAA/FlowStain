# FlowStain

**Source-Anchored Rectified Flow for H&E→IHC Virtual Staining**

FlowStain learns to transport H&E latents directly into IHC latents via a
rectified-flow ODE, guided by dense UNI morphology tokens and tissue-aware
loss weighting from CONCH.

## Project structure

```
FlowStain/
├── configs/
│   ├── mist_flowstain.yaml        # Main MIST config (all switches documented inline)
│   ├── bci_flowstain.yaml         # BCI config (overrides MIST defaults)
│   └── tissue_prompts.yaml        # CONCH zero-shot tissue prompts
├── flowstain/
│   ├── config.py                  # OmegaConf config loader + CLI arg parser
│   ├── data/
│   │   ├── base.py                # Shared crop/augment/tensorize logic + val subset
│   │   ├── mist.py                # MISTMultiStainDataset
│   │   ├── bci.py                 # BCIDataset
│   │   └── factory.py             # build_datasets / build_dataloaders
│   ├── foundation/
│   │   ├── uni.py                 # Frozen UNI extractor + UNIFeatureProcessorHighRes
│   │   └── conch_tissue.py        # Frozen CONCH zero-shot tissue classifier
│   ├── models/
│   │   ├── autoencoder.py         # Frozen diffusers AutoencoderKL wrapper
│   │   ├── conditioning.py        # FiLM, SPADE, CrossAttention, FlowCondBlock
│   │   ├── flow_unet.py           # Source-anchored rectified-flow U-Net
│   │   └── flowstain_model.py     # Full model: VAE + UNI + FlowUNet
│   ├── losses/
│   │   ├── flow_losses.py         # Velocity loss, noisy latent builder
│   │   ├── stain_losses.py        # DAB deconvolution, top-% loss, histogram KL
│   │   ├── structure_losses.py    # Edge loss, perceptual (LPIPS/VGG)
│   │   └── tissue_weighting.py    # TissueWeightedLoss from CONCH probs
│   ├── metrics/
│   │   ├── image_quality.py       # SSIM, LPIPS per batch
│   │   ├── fid_kid.py             # FID and KID
│   │   ├── retrieval.py           # MRA-style retrieval (CONCH/UNI)
│   │   └── tissue_stratified.py   # Per-tissue metric aggregation
│   └── training/
│       ├── loop.py                # Main Accelerate training loop
│       ├── validation.py          # Validation runner + W&B image logging
│       ├── checkpointing.py       # Save/resume checkpoints
│       └── ablations.py           # Named ablation config modifiers
└── scripts/
    ├── train_flowstain.py         # Entry point
    └── run_ablation.py            # Ablation launcher
```

## Installation (uv + CUDA 13)

```bash
# Install uv once (if needed)
pip install uv

# Create venv + install dependencies (includes torch 2.10 from CUDA 13 index)
uv sync
```

## Data layout

**MIST:**
```
/data/MIST/
  HER2/TrainValAB/{trainA,trainB,valA,valB}/
  Ki67/TrainValAB/{trainA,trainB,valA,valB}/
  ER/...
  PR/...
```

**BCI:**
```
/data/BCI/
  HE/{train,test}/*.png
  IHC/{train,test}/*.png
```

## Training

```bash
# MIST unified 4-stain model (single GPU)
uv run python scripts/train_flowstain.py --config configs/mist_flowstain.yaml \
    mist_dir=/data/MIST \
    logging.output_dir=runs/flowstain_mist \
    logging.wandb_run_name=mist_baseline

# MIST with accelerate (multi-GPU)
uv run accelerate launch scripts/train_flowstain.py \
    --config configs/mist_flowstain.yaml \
    mist_dir=/data/MIST \
    training.batch_size=8 \
    validation.val_max_samples=256 \
    wandb_num_val_images=16

# BCI HER2 model
uv run accelerate launch scripts/train_flowstain.py \
    --config configs/bci_flowstain.yaml \
    bci_dir=/data/BCI
```

## Key config switches (all via CLI override)

| Switch | Default | Effect |
|---|---|---|
| `flow.source` | `he` | `noise` = ablation baseline from Gaussian |
| `conditioning.uni.mode` | `hybrid` | `spade` / `cross_attention` / `hybrid` |
| `conditioning.uni.enabled` | `true` | Remove all UNI conditioning |
| `flow.bridge_noise.enabled` | `true` | Disable mid-trajectory bridge noise |
| `tissue_weighting.enabled` | `true` | Disable tissue-aware loss weighting |
| `losses.dab.enabled` | `true` | Phase 2 onwards |
| `losses.perceptual.enabled` | `false` | Enable perceptual loss |
| `losses.edge.enabled` | `false` | Enable edge consistency loss |
| `flow.sampling.method` | `euler` | `heun` / `stochastic_euler` |
| `validation.val_max_samples` | `256` | `-1` = full validation set |
| `validation.wandb_num_val_images` | `16` | Number of image grid samples on W&B |

## Running ablations

```bash
# List all ablations
uv run python scripts/run_ablation.py --list

# Run a specific ablation
uv run accelerate launch scripts/run_ablation.py \
    --config configs/mist_flowstain.yaml \
    --ablation no_uni \
    mist_dir=/data/MIST
```

Available ablations: `noise_source`, `no_uni`, `uni_spade_only`, `uni_xattn_only`,
`no_bridge_noise`, `no_tissue_weighting`, `no_dab`, `heun_sampler`, `stochastic_euler`.

## W&B logging

Each validation run logs:
- Scalar metrics: `val/fid`, `val/kid_mean`, `val/ssim`, `val/lpips_256`,
  `val/dab_kl_mean`, `val/dab_pearson`, `val/mra`, `val/failure_rate`
- Per-stain metrics: `val/stain/HER2/...`, `val/stain/Ki67/...`, etc.
- Per-tissue metrics: `val/tissue/invasive_carcinoma/...`, `val/tissue/adipose/...`, etc.
- Image grids: H&E | Real IHC | Generated IHC | DAB heatmap (real) | DAB heatmap (pred)
- Tissue-grouped examples: `val/tissue_examples/{tissue_class}`

Per-sample results are saved to `{output_dir}/val_metrics_step_*.csv` and `.json`.
