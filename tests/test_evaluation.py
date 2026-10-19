import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.evaluation import f1, per_dpi_f1, per_dpi_pot_accuracy, pot_bootstrap_ci, write_results


def _oof(seed: int = 0, images_per_pot: int = 4) -> pd.DataFrame:
    """6 inoculated + 6 control pots per DPI; pots differ in how separable they are."""
    rng = np.random.default_rng(seed)
    rows = []
    for pot in range(12):
        label = int(pot < 6)
        skill = rng.uniform(0.2, 0.9)  # per-pot probability of a correct prediction
        for dpi in range(4, 10):
            for _ in range(images_per_pot):
                correct = rng.random() < skill
                prob = 0.8 if correct == bool(label) else 0.2
                rows.append({"experiment": "A", "pot_id": pot, "dpi": dpi, "label": label,
                             "fold": pot % 6, "prob": prob})
    return pd.DataFrame(rows)


def test_per_dpi_f1_reports_every_dpi():  # R5
    oof = _oof()
    scores = per_dpi_f1(oof.label.to_numpy(), (oof.prob >= 0.5).to_numpy(), oof.dpi.to_numpy())
    assert sorted(scores) == list(range(4, 10))
    d4 = oof[oof.dpi == 4]
    assert scores[4] == pytest.approx(f1_score(d4.label, d4.prob >= 0.5))


def test_pot_accuracy_uses_mean_prob_per_pot():
    oof = pd.DataFrame({
        "experiment": "A", "dpi": 4, "label": [1, 1, 0, 0],
        "pot_id": [0, 0, 1, 1], "prob": [0.9, 0.3, 0.6, 0.6],
    })
    # pot 0: mean 0.6 -> infected (correct); pot 1: mean 0.6 -> infected (wrong)
    assert per_dpi_pot_accuracy(oof) == {4: 0.5}


def test_bootstrap_resamples_pots_not_images():  # R6
    base = _oof(images_per_pot=2)
    dup = pd.concat([base] * 10, ignore_index=True)  # 10x more images, same pots
    assert pot_bootstrap_ci(base, seed=1) == pytest.approx(pot_bootstrap_ci(dup, seed=1))


def test_bootstrap_known_answer_and_order():
    oof = _oof()
    point, lo, hi = pot_bootstrap_ci(oof, seed=0)
    assert lo <= point <= hi and lo < hi
    perfect = oof.assign(prob=oof.label.astype(float))
    assert pot_bootstrap_ci(perfect, seed=0) == (1.0, 1.0, 1.0)


def test_numpy_f1_matches_sklearn():
    oof = _oof(seed=3)
    assert f1(oof.label.to_numpy(), oof.prob.to_numpy()) == pytest.approx(
        f1_score(oof.label, oof.prob >= 0.5))
    assert f1(oof.label.to_numpy() * 0, oof.prob.to_numpy() * 0) == 0.0


def test_write_results(tmp_path):
    per_dpi = pd.DataFrame({"dpi": [4, 5], "f1": [0.7, 0.8]})
    write_results(tmp_path / "exp", {"model": "lr"}, {"f1": 0.75}, per_dpi)
    assert json.loads((tmp_path / "exp" / "config.json").read_text()) == {"model": "lr"}
    assert json.loads((tmp_path / "exp" / "metrics.json").read_text()) == {"f1": 0.75}
    assert pd.read_csv(tmp_path / "exp" / "per_dpi.csv").equals(per_dpi)
