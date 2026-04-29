"""
Tissue-stratified metrics aggregation.

After computing per-sample metrics, aggregate by CONCH tissue label to
produce per-tissue summaries logged to W&B.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..foundation.conch_tissue import TISSUE_CLASSES


def aggregate_per_tissue(
    per_sample: Dict[str, torch.Tensor],
    tissue_labels: List[str],
    failure_threshold_dab_kl: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Group per-sample metrics by tissue label.

    Args:
        per_sample: dict of {metric_name: [N] tensor}
        tissue_labels: list[str] of length N

    Returns:
        tissue_metrics: {tissue_class: {metric: mean_value}}
    """
    by_tissue: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    failure_by_tissue: Dict[str, list] = defaultdict(list)

    for i, tissue in enumerate(tissue_labels):
        for key, vals in per_sample.items():
            by_tissue[tissue][key].append(float(vals[i]))
        if "dab_kl" in per_sample:
            fail = float(per_sample["dab_kl"][i]) > failure_threshold_dab_kl
            failure_by_tissue[tissue].append(fail)

    result = {}
    for tissue in TISSUE_CLASSES:
        metrics = by_tissue[tissue]
        if not metrics:
            continue
        summary = {k: sum(v) / len(v) for k, v in metrics.items()}
        if tissue in failure_by_tissue:
            fails = failure_by_tissue[tissue]
            summary["failure_rate"] = sum(fails) / len(fails)
            summary["n_samples"] = len(fails)
        result[tissue] = summary

    return result


def compute_dab_pearson(
    pred_top_dab: torch.Tensor,
    real_top_dab: torch.Tensor,
) -> float:
    """Pearson correlation between per-image top-DAB intensities."""
    p = pred_top_dab.float()
    r = real_top_dab.float()
    p_m = p - p.mean()
    r_m = r - r.mean()
    corr = (p_m * r_m).sum() / (
        (p_m.pow(2).sum().sqrt() * r_m.pow(2).sum().sqrt()) + 1e-8
    )
    return float(corr.item())


def save_per_sample_csv(
    records: List[dict],
    output_path: Path,
):
    """Save per-sample metric records to a CSV file via pandas."""
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    except ImportError:
        # Write as JSONL if pandas unavailable
        output_path.write_text("\n".join(json.dumps(r) for r in records))
