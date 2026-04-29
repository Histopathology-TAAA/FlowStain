#!/usr/bin/env python
"""
Launch a named ablation from ABLATION_REGISTRY.

Usage:
    accelerate launch scripts/run_ablation.py \\
        --config configs/mist_flowstain.yaml \\
        --ablation no_uni

    # List available ablations:
    python scripts/run_ablation.py --list

    # Override additional config values:
    accelerate launch scripts/run_ablation.py \\
        --config configs/mist_flowstain.yaml \\
        --ablation noise_source \\
        training.num_epochs=20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowstain.config import load_config
from flowstain.training.ablations import ABLATION_REGISTRY
from flowstain.training.loop import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/mist_flowstain.yaml")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List available ablations")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if args.list:
        print("Available ablations:")
        for name in ABLATION_REGISTRY:
            print(f"  {name}")
        return

    if args.ablation is None:
        print("Specify --ablation <name> or --list")
        sys.exit(1)

    if args.ablation not in ABLATION_REGISTRY:
        print(f"Unknown ablation '{args.ablation}'. Use --list to see options.")
        sys.exit(1)

    cfg = load_config(args.config, args.overrides)
    ablation_fn = ABLATION_REGISTRY[args.ablation]
    ablation_cfg = ablation_fn(cfg)
    train(ablation_cfg)


if __name__ == "__main__":
    main()
