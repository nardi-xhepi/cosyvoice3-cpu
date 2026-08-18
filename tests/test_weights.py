"""Weight I/O: torch pickles, the safetensors container, quantisation."""

import json
import os
import pickle
import struct
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import FIXTURES  # noqa: E402

from cv3cpu.io.weights import WeightFile, WeightWriter  # noqa: E402
from cv3cpu.io.safetensors_io import SafeTensors  # noqa: E402
from cv3cpu.io.torch_pickle import load_pt  # noqa: E402
from cv3cpu.quant import (Q4_BLOCK, Q8_BLOCK, QTensor, qmatmul, quantize_q4,  # noqa: E402
                          quantize_q8)


# --------------------------------------------------------------------------
# reading torch checkpoints without torch
# --------------------------------------------------------------------------

def test_pt_reader_handles_every_storage_type():
    """float/half/bfloat16/int/bool, plus views with an offset and a stride."""
    got = load_pt(os.path.join(FIXTURES, "dtypes.pt"))
    with np.load(os.path.join(FIXTURES, "dtypes.npz")) as expected:
        assert set(got) == set(expected.files)
        for name in expected.files:
            ref = expected[name]
            assert got[name].shape == ref.shape, name
            assert np.array_equal(got[name].astype(np.float64),
                                  ref.astype(np.float64)), name
    assert got["strided_view"].shape == (4, 2)
    assert got["bf16"].dtype == np.float32      # numpy has no bfloat16


def test_pt_reader_arrays_are_owned_and_writable():
    """Views into the zip buffer must not leak out — the converter mutates these."""
    got = load_pt(os.path.join(FIXTURES, "dtypes.pt"))
    arr = got["f32"]
    assert arr.flags.owndata or arr.flags.writeable
    arr[0, 0] = 1234.0                            # must not raise


def _write_pickle_zip(path, payload_bytes):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", payload_bytes)
        zf.writestr("archive/version", "3\n")


def test_pt_reader_refuses_arbitrary_globals(tmp_path):
    """A checkpoint is data; nothing in it may name a callable to import."""
    p = str(tmp_path / "evil.pt")
    _write_pickle_zip(p, pickle.dumps({"x": os.system}, protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        load_pt(p)


def test_pt_reader_rejects_non_zip_checkpoints(tmp_path):
    p = str(tmp_path / "legacy.pt")
    with open(p, "wb") as fh:
        fh.write(b"not a zip archive")
    with pytest.raises(ValueError, match="zip-format"):
        load_pt(p)


def test_pt_reader_rejects_a_pickle_without_a_mapping(tmp_path):
    p = str(tmp_path / "list.pt")
    _write_pickle_zip(p, pickle.dumps([1, 2, 3], protocol=2))
    with pytest.raises(ValueError, match="state_dict"):
        load_pt(p)


# --------------------------------------------------------------------------
# safetensors
# --------------------------------------------------------------------------

def test_safetensors_reader(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1, 2], dtype=np.int64)
    header = {
        "a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, a.nbytes]},
        "b": {"dtype": "I64", "shape": [2], "data_offsets": [a.nbytes, a.nbytes + b.nbytes]},
    }
    raw = json.dumps(header).encode()
    p = str(tmp_path / "m.safetensors")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + a.tobytes() + b.tobytes())
    with SafeTensors(p) as st:
        assert set(st.keys()) == {"a", "b"}
        assert st.shape_dtype("a") == ((2, 3), "F32")
        assert np.array_equal(st["a"], a)
        assert np.array_equal(st["b"], b)


# --------------------------------------------------------------------------
# quantisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,tol", [("q8", 0.02), ("q4", 0.15)])
@pytest.mark.parametrize("shape", [(128, 257), (4864, 896), (896, 4864), (64, 64)])
def test_quantised_matmul_error(kind, tol, shape):
    rng = np.random.default_rng(0)
    n, k = shape
    w = rng.normal(size=(n, k)).astype(np.float32)
    q, sc, _ = (quantize_q8(w) if kind == "q8" else quantize_q4(w))
    t = QTensor(kind, q, sc, (n, k), Q8_BLOCK if kind == "q8" else Q4_BLOCK)
    for m in (1, 7, 64):
        x = rng.normal(size=(m, k)).astype(np.float32)
        ref = x @ w.T
        err = np.abs(qmatmul(x, t) - ref).mean() / np.abs(ref).mean()
        assert err < tol, (kind, shape, m, err)


@pytest.mark.parametrize("kind", ["q8", "q4"])
def test_quantisation_survives_a_ragged_reduction_axis(kind):
    """K is padded up to the block size; the padding must not leak into the result."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=(9, 100)).astype(np.float32)     # 100 % 64 != 0, 100 % 32 != 0
    q, sc, _ = (quantize_q8(w) if kind == "q8" else quantize_q4(w))
    t = QTensor(kind, q, sc, (9, 100), Q8_BLOCK if kind == "q8" else Q4_BLOCK)
    assert t.dequant().shape == w.shape
    x = rng.normal(size=(3, 100)).astype(np.float32)
    err = np.abs(qmatmul(x, t) - x @ w.T).mean() / np.abs(x @ w.T).mean()
    assert err < (0.03 if kind == "q8" else 0.2)


def test_zero_rows_do_not_divide_by_zero():
    w = np.zeros((4, 64), dtype=np.float32)
    for quant, kind, block in ((quantize_q8, "q8", Q8_BLOCK), (quantize_q4, "q4", Q4_BLOCK)):
        q, sc, _ = quant(w)
        t = QTensor(kind, q, sc, (4, 64), block)
        assert np.isfinite(t.dequant()).all()
        assert np.abs(t.dequant()).max() == 0.0


def test_qmatmul_is_tile_size_and_thread_invariant():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(1000, 640)).astype(np.float32)
    q, sc, _ = quantize_q8(w)
    t = QTensor("q8", q, sc, (1000, 640), Q8_BLOCK)
    x = rng.normal(size=(3, 640)).astype(np.float32)
    base = qmatmul(x, t, tile_bytes=1 << 23, threads=1)
    for tb in (1 << 14, 1 << 16, 1 << 20):
        for th in (1, 2, 4):
            assert np.abs(qmatmul(x, t, tile_bytes=tb, threads=th) - base).max() < 1e-4


def test_qmatmul_accepts_plain_arrays_and_bias():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(5, 7)).astype(np.float32)
    b = rng.normal(size=5).astype(np.float32)
    x = rng.normal(size=(2, 7)).astype(np.float32)
    assert np.abs(qmatmul(x, w, b) - (x @ w.T + b)).max() < 1e-5


# --------------------------------------------------------------------------
# the .safetensors container
# --------------------------------------------------------------------------

def test_weightfile_roundtrip_and_alignment(tmp_path):
    rng = np.random.default_rng(2)
    w = rng.normal(size=(200, 301)).astype(np.float32)
    b = rng.normal(size=200).astype(np.float32)
    conv = rng.normal(size=(64, 32, 5)).astype(np.float32)
    p = str(tmp_path / "t.safetensors")
    with WeightWriter(p, meta={"hello": "world"}) as writer:
        writer.add_quant("lin.weight", w, "q8")
        writer.add_raw("lin.bias", b, "f32")
        writer.add_quant("conv.weight", conv.reshape(64, -1), "q8", orig_shape=conv.shape)
        writer.add_raw("half", w, "f16")
    f = WeightFile(p)
    assert f.meta["hello"] == "world"
    assert np.array_equal(f.get("lin.bias"), b)
    assert f.get("conv.weight").kernel_size == 5
    assert f.entry("conv.weight")["shape"] == [64, 32, 5]
    assert np.abs(f.get_f32("half") - w).max() < 1e-2
    x = rng.normal(size=(3, 301)).astype(np.float32)
    ref = x @ w.T + b
    got = qmatmul(x, f.get("lin.weight"), f.get("lin.bias"))
    assert np.abs(got - ref).mean() / np.abs(ref).mean() < 0.02
    assert set(f.subset("lin.")) == {"weight", "bias"}
    f.close()


def test_weightfile_rejects_a_foreign_file(tmp_path):
    p = str(tmp_path / "nope.safetensors")
    with open(p, "wb") as fh:
        fh.write(b"XXXX" + b"\0" * 128)
    with pytest.raises(Exception):
        WeightFile(p)


def test_weightfile_header_overflow_is_reported(tmp_path):
    p = str(tmp_path / "small.safetensors")
    writer = WeightWriter(p, header_reserve=32)
    writer.add_raw("a_very_long_tensor_name_" * 4, np.zeros(4, np.float32))
    with pytest.raises(RuntimeError, match="header"):
        writer.close()


def test_weightfile_evict_releases_pages(tmp_path):
    rng = np.random.default_rng(3)
    p = str(tmp_path / "big.safetensors")
    with WeightWriter(p) as writer:
        writer.add_quant("stage1.w", rng.normal(size=(4000, 4000)).astype(np.float32), "q8")
        writer.add_quant("stage2.w", rng.normal(size=(1000, 1000)).astype(np.float32), "q8")
    f = WeightFile(p)
    x = np.ones((1, 4000), np.float32)
    before = qmatmul(x, f.get("stage1.w"))       # fault the pages in
    assert f.evict("stage1.") > 8 << 20          # ~16 MB of int8 handed back
    after = qmatmul(x, f.get("stage1.w"))        # re-read on demand
    assert np.array_equal(before, after)
    f.close()


# --------------------------------------------------------------------------
# interoperability with the reference safetensors implementation
# --------------------------------------------------------------------------

def test_files_are_readable_by_the_official_library(tmp_path):
    """The whole reason for using safetensors: other tools can open these files.

    Skipped unless ``safetensors`` happens to be installed — the suite itself
    needs nothing but NumPy.
    """
    stnp = pytest.importorskip("safetensors.numpy")
    from safetensors import safe_open

    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 96)).astype(np.float32)
    conv = rng.normal(size=(8, 4, 5)).astype(np.float32)
    p = str(tmp_path / "m.safetensors")
    with WeightWriter(p, meta={"sample_rate": 24000}) as writer:
        writer.add_quant("llm.w", w, "q8")
        writer.add_quant("hift.conv.weight", conv.reshape(8, -1), "q4",
                         orig_shape=conv.shape)
        writer.add_raw("hift.bias", np.arange(8, dtype=np.float32))

    loaded = stnp.load_file(p)
    assert loaded["llm.w"].dtype == np.int8
    assert loaded["llm.w.__scales__"].dtype == np.float16
    assert loaded["hift.conv.weight"].dtype == np.uint8      # packed q4 nibbles

    with safe_open(p, framework="numpy") as fh:
        blob = json.loads(fh.metadata()["cv3cpu"])
    assert blob["meta"]["sample_rate"] == 24000
    assert blob["quant"]["hift.conv.weight"]["shape"] == [8, 4, 5]

    ours = WeightFile(p)
    assert np.array_equal(ours.get("llm.w").q, loaded["llm.w"])
    assert np.array_equal(ours.get("hift.bias"), loaded["hift.bias"])
    ours.close()


def test_we_can_read_what_the_official_library_writes(tmp_path):
    stnp = pytest.importorskip("safetensors.numpy")

    p = str(tmp_path / "official.safetensors")
    stnp.save_file({"a": np.arange(6, dtype=np.float32).reshape(2, 3)}, p,
                   metadata={"note": "hi"})
    f = WeightFile(p)
    assert np.array_equal(f.get("a"), np.arange(6, dtype=np.float32).reshape(2, 3))
    assert f.entry("a")["shape"] == [2, 3]
    assert f.meta == {}                     # no cv3cpu blob in a foreign file
    f.close()


def test_misaligned_tensors_are_still_read_correctly(tmp_path):
    """safetensors packs with no gaps, so an odd-length int8 tensor shifts the rest."""
    rng = np.random.default_rng(9)
    w = rng.normal(size=(16, 64)).astype(np.float32)
    p = str(tmp_path / "odd.safetensors")
    with WeightWriter(p) as writer:
        writer.add_raw("odd_bytes", np.arange(7, dtype=np.uint8), "u8")   # odd length
        writer.add_quant("w", w, "q8")
        writer.add_raw("f", w, "f32")
    f = WeightFile(p)
    assert f.get("w").scales.ctypes.data % 2 == 0, "scales must end up aligned"
    assert f.get("f").ctypes.data % 4 == 0, "float32 must end up aligned"
    assert np.abs(f.get_f32("f") - w).max() == 0.0
    x = rng.normal(size=(2, 64)).astype(np.float32)
    assert np.abs(qmatmul(x, f.get("w")) - x @ w.T).mean() / np.abs(x @ w.T).mean() < 0.02
    f.close()
