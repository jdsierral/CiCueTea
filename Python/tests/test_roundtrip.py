import numpy as np
import pytest

from cicuetea import NsgfCQT, NsgfVQT


def _rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2.0))


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("frac", [1.0, 1 / 12])
def test_cqt_roundtrip(mode, frac):
    fs = 48000
    n_samples = 2**14
    x = np.random.default_rng(0).standard_normal(n_samples)

    cqt = NsgfCQT(mode, fs, n_samples, frac=frac, f_min=100, f_max=10000)
    y = cqt.inverse(cqt.forward(x))

    assert _rms(x - y) < 1e-10


def test_vqt_roundtrip():
    fs = 48000
    n_samples = 2**14
    x = np.random.default_rng(1).standard_normal(n_samples)

    vqt = NsgfVQT("dense", fs, n_samples, f_map=np.log2, frac=1 / 12)
    y = vqt.inverse(vqt.forward(x))

    assert _rms(x - y) < 1e-10


def test_rasterize_matches_dense():
    fs = 48000
    n_samples = 2**14
    x = np.random.default_rng(2).standard_normal(n_samples)

    dense = NsgfCQT("dense", fs, n_samples, frac=1 / 12)
    sparse = NsgfCQT("sparse", fs, n_samples, frac=1 / 12)

    X_dense = dense.forward(x)
    X_sparse = sparse.rasterize(sparse.forward(x))

    assert _rms(X_dense - X_sparse) < 1e-5


def test_rejects_invalid_range():
    with pytest.raises(AssertionError):
        NsgfCQT("dense", 48000, 2**14, frac=1 / 12, f_min=10000, f_max=100)
