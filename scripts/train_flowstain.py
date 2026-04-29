#!/usr/bin/env python
"""
FlowStain training entry point.

Usage:
    # Single GPU:
    python scripts/train_flowstain.py --config configs/mist_flowstain.yaml

    # Multi-GPU / multi-node via accelerate:
    accelerate launch scripts/train_flowstain.py --config configs/mist_flowstain.yaml

    # Override config values on the CLI:
    accelerate launch scripts/train_flowstain.py \\
        --config configs/mist_flowstain.yaml \\
        training.batch_size=8 \\
        training.num_epochs=50 \\
        logging.wandb_run_name=my_experiment \\
        validation.val_max_samples=512
"""

import sys
from pathlib import Path

# Allow importing flowstain from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowstain.config import config_from_args
from flowstain.training.loop import train


def main():
    cfg = config_from_args()
    train(cfg)


if __name__ == "__main__":
    main()
