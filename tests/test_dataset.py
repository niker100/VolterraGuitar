"""Dataset contract: save/load roundtrip, split sizes/ordering, shape enforcement.

Also covers the optional exogenous ``controls`` matrix: validation, slice/split
alignment, save/load round-trip, and backward compatibility with control-free
``.npz`` files written before the field existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vguitar.data import Dataset


def _toy(n: int = 1000, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    y = np.tanh(x).astype(np.float32)
    return Dataset(x, y, sr=44_100, name="toy", meta={"k": 1})


def _toy_conditioned(n: int = 1000, c: int = 3, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    y = np.tanh(x).astype(np.float32)
    controls = rng.standard_normal((n, c)).astype(np.float32)
    return Dataset(
        x,
        y,
        sr=44_100,
        name="toy_c",
        meta={"k": 1},
        controls=controls,
        control_names=["gain", "tone", "temp"][:c],
        control_kinds=["continuous", "continuous", "drift"][:c],
    )


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


# --- controls: defaults & validation -------------------------------------------------


def test_controls_default_none() -> None:
    ds = _toy()
    assert ds.controls is None
    assert ds.control_names is None
    assert ds.control_kinds is None
    assert ds.n_controls == 0


def test_controls_basic() -> None:
    ds = _toy_conditioned(n=100, c=3)
    assert ds.controls is not None
    assert ds.controls.shape == (100, 3)
    assert ds.controls.dtype == np.float32
    assert ds.n_controls == 3
    assert ds.control_names == ["gain", "tone", "temp"]
    assert ds.control_kinds == ["continuous", "continuous", "drift"]


def test_controls_1d_reshaped_to_column() -> None:
    # A 1-D (N,) control is accepted and treated as a single column.
    ds = Dataset(np.zeros(8), np.zeros(8), sr=8000, controls=np.arange(8, dtype=np.float32))
    assert ds.controls is not None and ds.controls.shape == (8, 1)
    assert ds.control_names == ["c0"]  # auto-generated


def test_controls_dtype_coerced() -> None:
    ds = Dataset(np.zeros(4), np.zeros(4), sr=8000, controls=np.arange(4, dtype=np.int64))
    assert ds.controls is not None and ds.controls.dtype == np.float32


def test_controls_default_names_generated() -> None:
    ds = Dataset(np.zeros(5), np.zeros(5), sr=8000, controls=np.zeros((5, 2)))
    assert ds.control_names == ["c0", "c1"]
    assert ds.control_kinds is None  # genuinely optional


def test_controls_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="controls length mismatch"):
        Dataset(np.zeros(10), np.zeros(10), sr=8000, controls=np.zeros((9, 2)))


def test_controls_zero_columns_raises() -> None:
    with pytest.raises(ValueError, match="zero columns"):
        Dataset(np.zeros(5), np.zeros(5), sr=8000, controls=np.zeros((5, 0)))


def test_controls_3d_raises() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        Dataset(np.zeros(5), np.zeros(5), sr=8000, controls=np.zeros((5, 2, 1)))


def test_control_names_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="control_names"):
        Dataset(
            np.zeros(5), np.zeros(5), sr=8000, controls=np.zeros((5, 2)), control_names=["only"]
        )


def test_control_kinds_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="control_kinds"):
        Dataset(
            np.zeros(5),
            np.zeros(5),
            sr=8000,
            controls=np.zeros((5, 2)),
            control_kinds=["continuous"],
        )


def test_control_kinds_unknown_value_raises() -> None:
    with pytest.raises(ValueError, match="unknown control_kinds"):
        Dataset(
            np.zeros(5),
            np.zeros(5),
            sr=8000,
            controls=np.zeros((5, 1)),
            control_kinds=["nonsense"],
        )


def test_control_metadata_without_controls_raises() -> None:
    with pytest.raises(ValueError, match="controls is None"):
        Dataset(np.zeros(5), np.zeros(5), sr=8000, control_names=["gain"])


# --- controls: slice & split ---------------------------------------------------------


def test_controls_slice_alignment() -> None:
    ds = _toy_conditioned(n=100, c=3)
    sl = ds.slice(10, 40)
    assert sl.controls is not None
    assert ds.controls is not None
    np.testing.assert_array_equal(sl.controls, ds.controls[10:40])
    assert sl.control_names == ds.control_names
    assert sl.control_kinds == ds.control_kinds
    # Slicing copies the metadata lists rather than aliasing them.
    assert sl.control_names is not ds.control_names


def test_controls_split_partitions_and_reassembles() -> None:
    ds = _toy_conditioned(n=1000, c=2)
    train, val, test = ds.split(val_fraction=0.1, test_fraction=0.1)
    for part in (train, val, test):
        assert part.n_controls == 2
        assert part.controls is not None
        assert part.controls.shape[0] == len(part)
        assert part.control_names == ds.control_names
    rebuilt = np.concatenate([train.controls, val.controls, test.controls], axis=0)
    np.testing.assert_array_equal(rebuilt, ds.controls)


# --- controls: from_segments (piecewise-constant SPICE sweeps) ------------------------


def test_from_segments_expands_to_dense() -> None:
    x = np.zeros(10, dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    ds = Dataset.from_segments(
        x,
        y,
        sr=8000,
        boundaries=[0, 4, 10],
        values=np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32),
        control_names=["pot", "vcc"],
        control_kinds=["continuous", "drift"],
    )
    assert ds.controls is not None and ds.controls.shape == (10, 2)
    # First 4 samples hold segment 0, the rest hold segment 1.
    np.testing.assert_array_equal(ds.controls[:4], np.tile([0.0, 1.0], (4, 1)))
    np.testing.assert_array_equal(ds.controls[4:], np.tile([0.5, 2.0], (6, 1)))
    assert ds.control_names == ["pot", "vcc"]


def test_from_segments_boundary_validation() -> None:
    x = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="boundaries must run"):
        Dataset.from_segments(
            x, x, sr=8000, boundaries=[0, 4, 8], values=np.zeros((2, 1), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="S\\+1"):
        Dataset.from_segments(
            x, x, sr=8000, boundaries=[0, 10], values=np.zeros((2, 1), dtype=np.float32)
        )


# --- controls: save / load round-trip & backward compatibility -----------------------


def test_controls_save_load_roundtrip(tmp_path) -> None:
    ds = _toy_conditioned(n=200, c=3)
    path = ds.save(tmp_path / "ds_c.npz")
    loaded = Dataset.load(path)
    np.testing.assert_array_equal(loaded.x, ds.x)
    np.testing.assert_array_equal(loaded.y, ds.y)
    np.testing.assert_array_equal(loaded.controls, ds.controls)
    assert loaded.controls is not None
    assert loaded.controls.dtype == np.float32
    assert loaded.control_names == ds.control_names
    assert loaded.control_kinds == ds.control_kinds
    assert loaded.meta == ds.meta


def test_controls_roundtrip_without_kinds(tmp_path) -> None:
    # control_kinds is optional; its absence must round-trip as None (not "").
    ds = Dataset(
        np.zeros(20), np.zeros(20), sr=8000, controls=np.ones((20, 2)), control_names=["a", "b"]
    )
    loaded = Dataset.load(ds.save(tmp_path / "no_kinds.npz"))
    np.testing.assert_array_equal(loaded.controls, ds.controls)
    assert loaded.control_names == ["a", "b"]
    assert loaded.control_kinds is None


def test_unconditioned_save_has_no_control_keys(tmp_path) -> None:
    # A control-free dataset must serialize to exactly the legacy key set.
    path = _toy().save(tmp_path / "plain.npz")
    with np.load(path, allow_pickle=False) as f:
        assert set(f.files) == {"x", "y", "sr", "name", "meta"}


def test_load_legacy_npz_without_controls(tmp_path) -> None:
    # Simulate a file written by the pre-conditioning code (no control keys at all).
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        x=np.arange(16, dtype=np.float32),
        y=np.arange(16, dtype=np.float32),
        sr=np.int64(44_100),
        name="legacy",
        meta=repr({"old": True}),
    )
    loaded = Dataset.load(path)
    assert loaded.controls is None
    assert loaded.control_names is None
    assert loaded.control_kinds is None
    assert loaded.n_controls == 0
    assert loaded.meta == {"old": True}
    assert len(loaded) == 16
