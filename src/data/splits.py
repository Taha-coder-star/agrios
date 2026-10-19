"""Pot-level splits: one single 8/2/2 split and 6 outer folds with inner validation pairs.

Every fold, split and role in the project comes from the JSON this module writes (hard rule 1).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pot_key(index: pd.DataFrame) -> pd.Series:
    """Pots are only unique within an experiment, so the key carries both."""
    return index.experiment.astype(str) + ":" + index.pot_id.astype(str)


def _experiment_folds(pots: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    inoc = rng.permutation(sorted(pots.loc[pots.label == 1, "key"]))
    ctrl = rng.permutation(sorted(pots.loc[pots.label == 0, "key"]))
    if len(inoc) != len(ctrl) or len(inoc) < 3:
        raise ValueError(f"need balanced inoculated/control pots (>= 3 each), got "
                         f"{len(inoc)}/{len(ctrl)}")
    n = len(inoc)
    folds = []
    for i in range(n):
        j = (i + 1) % n  # the next pair validates, so every pot is test once and val once
        test = [str(inoc[i]), str(ctrl[i])]
        val = [str(inoc[j]), str(ctrl[j])]
        train = sorted(set(pots.key) - set(test) - set(val))
        folds.append({"train": train, "val": val, "test": test})
    return folds


def make_splits(index: pd.DataFrame, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    pots = index.assign(key=pot_key(index)).drop_duplicates("key")
    experiments = {}
    for exp in sorted(pots.experiment.astype(str).unique()):
        folds = _experiment_folds(pots[pots.experiment.astype(str) == exp], rng)
        experiments[exp] = {"single": folds[0], "folds": folds}
    return {"seed": seed, "experiments": experiments}


def fold_roles(index: pd.DataFrame, fold: dict) -> pd.Series:
    """Role ('train' / 'val' / 'test') of every image row for one fold; NaN if absent."""
    role_of = {pot: role for role, pots in fold.items() for pot in pots}
    return pot_key(index).map(role_of)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=Path("data/index.csv"))
    p.add_argument("--out", type=Path, default=Path("data/splits.json"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    splits = make_splits(pd.read_csv(args.index), args.seed)
    args.out.write_text(json.dumps(splits, indent=2))
    print(f"wrote {args.out}: {', '.join(splits['experiments'])}")


if __name__ == "__main__":
    main()
