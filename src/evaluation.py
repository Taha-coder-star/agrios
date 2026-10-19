"""Per-DPI metrics and pot-level uncertainty (CONSTRAINTS R5, R6).

Out-of-fold frames have columns: experiment, pot_id, dpi, label, fold, prob.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.splits import pot_key


def per_dpi_f1(y_true: np.ndarray, y_pred: np.ndarray, dpi: np.ndarray) -> dict[int, float]:
    """Image-level F1 for each DPI."""
    return {
        int(d): float(f1_score(y_true[dpi == d], y_pred[dpi == d], zero_division=0))
        for d in np.unique(dpi)
    }


def f1(label: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    """Binary F1 in plain NumPy: same result as sklearn, without its per-call overhead, which
    dominates a 2,000-draw bootstrap."""
    pred = prob >= threshold
    tp = np.sum(pred & (label == 1))
    denom = pred.sum() + (label == 1).sum()
    return float(2 * tp / denom) if denom else 0.0


def per_dpi_pot_accuracy(oof: pd.DataFrame, threshold: float = 0.5) -> dict[int, float]:
    """A pot's verdict at a DPI is its mean image probability against the threshold."""
    pots = oof.assign(key=pot_key(oof)).groupby(["dpi", "key"]).agg(
        prob=("prob", "mean"), label=("label", "first"))
    correct = (pots.prob >= threshold).astype(int) == pots.label
    return {int(d): float(v) for d, v in correct.groupby(level="dpi").mean().items()}


def pot_bootstrap_ci(
    oof: pd.DataFrame,
    metric: Callable[[np.ndarray, np.ndarray], float] = f1,
    n: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point, lo, hi) of metric(label, prob): resamples whole pots, stratified by label, never
    single images.

    Images of one pot are near-duplicates, so an image bootstrap would give falsely narrow
    intervals.
    """
    keys = pot_key(oof)
    label, prob = oof.label.to_numpy(), oof.prob.to_numpy()
    rows_of = {k: np.flatnonzero(keys.to_numpy() == k) for k in sorted(keys.unique())}
    label_of = oof.groupby(keys).label.first()
    strata = [sorted(label_of.index[label_of == c]) for c in sorted(label_of.unique())]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        rows = np.concatenate([rows_of[k] for pots in strata
                               for k in rng.choice(pots, size=len(pots))])
        stats.append(metric(label[rows], prob[rows]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(label, prob)), float(lo), float(hi)


def write_results(out_dir: Path, config: dict, metrics: dict, per_dpi: pd.DataFrame) -> None:
    """Every experiment's numbers go here, so the report can trace each one (R8)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    per_dpi.to_csv(out_dir / "per_dpi.csv", index=False)
