"""
Config loading utilities.

Loads YAML configs via OmegaConf and merges with CLI overrides.
A base MIST or BCI config can be extended via --config, with any
key overridden on the command line as --key=value (dot-notation).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load config from YAML and apply dot-notation CLI overrides.

    Args:
        config_path: Path to the base YAML config file.
        overrides: List of 'key=value' strings from CLI (e.g. ['training.batch_size=8']).

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    OmegaConf.set_readonly(cfg, False)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the training script."""
    parser = argparse.ArgumentParser(
        description="FlowStain training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/mist_flowstain.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dot-notation format, e.g. training.batch_size=8",
    )
    return parser


def config_from_args(args: argparse.Namespace | None = None) -> DictConfig:
    """Parse CLI args and return merged config."""
    parser = build_arg_parser()
    parsed = parser.parse_args(args)
    return load_config(parsed.config, parsed.overrides)


def cfg_get(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Safe dot-path access with a default."""
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default
