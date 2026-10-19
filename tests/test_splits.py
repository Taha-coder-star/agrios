import json

import pandas as pd
import pytest

from src.data.splits import fold_roles, make_splits, pot_key

ROLES = ("train", "val", "test")


def _pots(index: pd.DataFrame, experiment: str) -> set[str]:
    return set(pot_key(index[index.experiment == experiment]))


def _labels(index: pd.DataFrame) -> dict[str, int]:
    return dict(zip(pot_key(index), index.label, strict=True))


def test_no_pot_in_two_roles_in_any_fold(synthetic_index):  # R1
    splits = make_splits(synthetic_index, seed=42)
    for exp, s in splits["experiments"].items():
        for fold in [s["single"], *s["folds"]]:
            sets = [set(fold[r]) for r in ROLES]
            assert sum(len(x) for x in sets) == len(set().union(*sets)), "pot in two roles"
            assert set().union(*sets) == _pots(synthetic_index, exp), "pot missing from fold"


def test_role_sizes_and_class_balance(synthetic_index):
    labels = _labels(synthetic_index)
    s = make_splits(synthetic_index, seed=42)["experiments"]["A"]
    for fold in [s["single"], *s["folds"]]:
        for role, size in {"train": 8, "val": 2, "test": 2}.items():
            assert len(fold[role]) == size
            assert sum(labels[p] for p in fold[role]) == size // 2, f"{role} unbalanced"


def test_every_pot_is_test_exactly_once(synthetic_index):
    s = make_splits(synthetic_index, seed=42)["experiments"]["A"]
    tested = [p for fold in s["folds"] for p in fold["test"]]
    assert len(s["folds"]) == 6
    assert sorted(tested) == sorted(_pots(synthetic_index, "A"))


def test_class_balance_per_dpi_per_split(synthetic_index):
    s = make_splits(synthetic_index, seed=42)["experiments"]["A"]["single"]
    index = synthetic_index[synthetic_index.experiment == "A"]
    roles = fold_roles(index, s)
    for role in ROLES:
        counts = index[roles == role].groupby("dpi").label.agg(["sum", "count"])
        assert (counts["sum"] * 2 == counts["count"]).all(), f"{role} unbalanced within a DPI"


def test_deterministic_under_seed_and_json_roundtrip(synthetic_index):
    a = make_splits(synthetic_index, seed=7)
    assert a == make_splits(synthetic_index, seed=7)
    assert a != make_splits(synthetic_index, seed=8)
    assert json.loads(json.dumps(a)) == a


def test_unbalanced_pots_rejected(synthetic_index):
    with pytest.raises(ValueError, match="balanced"):
        make_splits(synthetic_index[synthetic_index.pot_id != 0], seed=42)


@pytest.mark.requires_data
def test_real_splits_file_has_no_leak():  # R1 on the committed file
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    index = pd.read_csv(root / "data" / "index.csv")
    splits = json.loads((root / "data" / "splits.json").read_text())
    for exp, s in splits["experiments"].items():
        for fold in [s["single"], *s["folds"]]:
            assert sum(len(fold[r]) for r in ROLES) == len(set().union(*map(set, fold.values())))
            assert set().union(*map(set, fold.values())) == _pots(index, exp)


def test_cli_writes_splits(synthetic_index, tmp_path, monkeypatch):
    import sys

    from src.data import splits as mod

    synthetic_index.to_csv(tmp_path / "index.csv", index=False)
    out = tmp_path / "splits.json"
    argv = ["splits", "--index", str(tmp_path / "index.csv"), "--out", str(out), "--seed", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    assert json.loads(out.read_text()) == make_splits(synthetic_index, seed=3)
