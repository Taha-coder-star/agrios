from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_INDEX = Path(__file__).resolve().parents[1] / "data" / "index.csv"


def pytest_collection_modifyitems(config, items):
    if DATA_INDEX.exists():
        return
    skip = pytest.mark.skip(reason="real dataset not downloaded yet (data/index.csv missing)")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def synthetic_index() -> pd.DataFrame:
    """Mirrors the real layout: 2 experiments x 12 pots (6 inoculated) x DPI 4-9 x 2 images."""
    rows = [
        {"experiment": exp, "pot_id": pot, "label": int(pot < 6), "dpi": dpi, "image": img}
        for exp in ("A", "B")
        for pot in range(12)
        for dpi in range(4, 10)
        for img in range(2)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_cube() -> np.ndarray:
    """8 x 8 x 106 float32 cube: plant pixels in the centre, background at the edges."""
    cube = np.full((8, 8, 106), 0.05, dtype=np.float32)
    cube[2:6, 2:6, 80:] = 0.5  # near-infrared plateau of vegetation
    return cube
