"""Dataset contract: save/load roundtrip, split sizes/ordering, shape enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from vguitar.data import Dataset


def _toy(n: int = 1000, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    y = np.tanh(x).astype(np.float32)
    return Dataset(x, y, sr=44_100, name="toy", meta={"k": 1})


def test_save_load_roundtrip(tmp_path) -> None:
    ds = _toy()
    path = ds.save(tmp_path / "ds.npz")
    loaded = Dataset.load(path)
    np.testing.assert_array_equal(loaded.x, ds.x)
    np.testing.assert_array_equal(loaded.y, ds.y)
    assert loaded.sr == ds.sr
    assert loaded.name == ds.name
    assert loaded.meta == ds.meta
    assert loaded.x.dtype == np.float32 and loaded.y.dtype == np.float32


def test_split_sizes_sum_and_ordering() -> None:
    n = 1000
    ds = _toy(n)
    train, val, test = ds.split(val_fraction=0.1, test_fraction=0.1)
    # Partitions tile the signal exactly, with no overlap or gaps.
    assert len(train) + len(val) + len(test) == n
    assert len(val) == int(n * 0.1)
    assert len(test) == int(n * 0.1)
    # Contiguous, time-ordered concatenation must reproduce the original.
    np.testing.assert_array_equal(np.concatenate([train.x, val.x, test.x]), ds.x)
    np.testing.assert_array_equal(np.concatenate([train.y, val.y, test.y]), ds.y)
    assert (train.name, val.name, test.name) == ("toy:train", "toy:val", "toy:test")


def test_split_empty_partition_raises() -> None:
    ds = _toy(10)
    with pytest.raises(ValueError):
        ds.split(val_fraction=0.0, test_fraction=0.0)


def test_unequal_length_raises() -> None:
    x = np.zeros(10, dtype=np.float32)
    y = np.zeros(9, dtype=np.float32)
    with pytest.raises(ValueError):
        Dataset(x, y, sr=44_100)


def test_dtype_enforced() -> None:
    # Integer inputs are coerced to float32 mono regardless of original dtype/shape.
    ds = Dataset(np.arange(8, dtype=np.int64), np.arange(8, dtype=np.int64), sr=8000)
    assert ds.x.dtype == np.float32 and ds.y.dtype == np.float32
    assert ds.x.ndim == 1 and len(ds) == 8
