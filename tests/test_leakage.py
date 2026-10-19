import numpy as np
import pytest

from src.features.preprocess import BandScaler, OutlierFilter


def _spectra(n: int = 20, bands: int = 106, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    curve = np.linspace(0.05, 0.5, bands)
    return (curve + rng.normal(0, 0.01, (n, bands))).astype(np.float32)


def test_scaler_ignores_test_rows():  # R2
    X = _spectra()
    train, test = np.arange(15), np.arange(15, 20)
    a = BandScaler().fit(X, train)
    X_poisoned = X.copy()
    X_poisoned[test] = 1000.0
    b = BandScaler().fit(X_poisoned, train)
    np.testing.assert_array_equal(a.min_, b.min_)
    np.testing.assert_array_equal(a.max_, b.max_)


def test_scaler_requires_train_indices():  # R2: no fit on the whole dataset by accident
    with pytest.raises(TypeError):
        BandScaler().fit(_spectra())


def test_scaler_maps_train_to_unit_range_and_handles_cubes():
    X = _spectra()
    s = BandScaler().fit(X, np.arange(20))
    Z = s.transform(X)
    assert Z.min() == pytest.approx(0.0) and Z.max() == pytest.approx(1.0)
    cubes = np.stack([np.broadcast_to(x, (4, 4, 106)) for x in X])
    np.testing.assert_allclose(BandScaler().fit(cubes, np.arange(20)).transform(cubes)[:, 0, 0], Z)


def test_scaler_constant_band_does_not_divide_by_zero():
    X = _spectra()
    X[:, 3] = 0.2
    Z = BandScaler().fit(X, np.arange(20)).transform(X)
    assert np.isfinite(Z).all()


def test_outlier_filter_touches_train_only_and_reports_count():  # R3
    X = _spectra()
    X[2] = X[2][::-1]   # train outlier: reversed curve, correlation ~ -1
    X[17] = X[17][::-1]  # test outlier: must survive, the filter never sees it
    train = np.arange(15)
    f = OutlierFilter(threshold=0.85)
    kept = f.fit(X, train)
    assert f.n_removed == 1
    assert list(f.removed_idx) == [2]
    assert set(kept) == set(train) - {2}
    assert set(kept) <= set(train)


def test_outlier_filter_keeps_everything_when_clean():
    f = OutlierFilter()
    kept = f.fit(_spectra(), np.arange(10))
    assert f.n_removed == 0 and len(kept) == 10
