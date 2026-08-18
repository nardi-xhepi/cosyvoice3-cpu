"""Kernel tests against naive reference implementations written here in NumPy.

The point of the kernels in ``cv3cpu.ops`` is that they tile, stream and reuse
buffers; the point of the references below is that they do none of that.  Each
one is the textbook definition, written for clarity, so agreement means the fast
path's blocking and masking are right.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv3cpu.ops as ops  # noqa: E402


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(1e-9, np.abs(b).max()))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------
# activations, elementwise, against scalar math
# --------------------------------------------------------------------------

def _scalar(fn, x):
    return np.array([fn(float(v)) for v in np.asarray(x).reshape(-1)]).reshape(np.shape(x))


def test_activations_match_scalar_definitions(rng):
    import math

    x = rng.normal(size=(5, 17)).astype(np.float32) * 3.0
    assert rel(ops.silu(x), _scalar(lambda v: v / (1 + math.exp(-v)), x)) < 1e-6
    assert rel(ops.sigmoid(x), _scalar(lambda v: 1 / (1 + math.exp(-v)), x)) < 1e-6
    assert rel(ops.gelu(x), _scalar(lambda v: 0.5 * v * (1 + math.erf(v / math.sqrt(2))), x)) < 1e-6
    assert rel(ops.gelu_tanh(x), _scalar(
        lambda v: 0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3))), x)) < 1e-6
    assert rel(ops.mish(x), _scalar(lambda v: v * math.tanh(math.log1p(math.exp(v))), x)) < 1e-5
    assert rel(ops.elu(x), _scalar(lambda v: v if v > 0 else math.expm1(v), x)) < 1e-6
    assert rel(ops.leaky_relu(x, 0.1), _scalar(lambda v: v if v >= 0 else 0.1 * v, x)) < 1e-6


def test_mish_is_stable_for_large_inputs():
    """The softplus guard must not overflow where exp(x) does."""
    x = np.array([-50.0, 0.0, 20.0, 100.0, 1e4], dtype=np.float32)
    got = ops.mish(x)
    assert np.isfinite(got).all()
    assert rel(got[-2:], x[-2:]) < 1e-6      # mish(x) -> x for large x


def test_snake_matches_definition(rng):
    a = np.abs(rng.normal(size=5)).astype(np.float32) + 0.5
    x = rng.normal(size=(5, 13)).astype(np.float32)
    ref = x + (1.0 / (a[:, None] + 1e-9)) * np.sin(a[:, None] * x) ** 2
    assert rel(ops.snake(x, a), ref) < 1e-6


def test_norms_match_definitions(rng):
    x = rng.normal(size=(5, 17)).astype(np.float32)
    w = rng.normal(size=17).astype(np.float32)
    b = rng.normal(size=17).astype(np.float32)
    ln = np.stack([(r - r.mean()) / np.sqrt(r.var() + 1e-6) for r in x.astype(np.float64)])
    assert rel(ops.layer_norm(x, None, None, 1e-6), ln) < 1e-5
    assert rel(ops.layer_norm(x, w, b, 1e-6), ln * w + b) < 1e-5
    rms = np.stack([r / np.sqrt((r ** 2).mean() + 1e-6) for r in x.astype(np.float64)])
    assert rel(ops.rms_norm(x, w, 1e-6), rms * w) < 1e-5


def test_softmax_is_stable_and_normalised(rng):
    """Large logits must not overflow, and log_softmax must stay exact where
    softmax itself underflows to zero."""
    x = rng.normal(size=(4, 9)).astype(np.float32) * 50
    p = ops.softmax(x)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-6)
    x64 = x.astype(np.float64)
    m = x64.max(axis=-1, keepdims=True)
    ref = x64 - (m + np.log(np.exp(x64 - m).sum(axis=-1, keepdims=True)))
    assert np.abs(ops.log_softmax(x) - ref).max() < 1e-3


# --------------------------------------------------------------------------
# convolution, against a direct triple loop
# --------------------------------------------------------------------------

def _naive_conv1d(x, w, b, stride, dilation, groups, pad):
    xp = np.pad(np.asarray(x, np.float64), ((0, 0), pad))
    cout, cin_g, k = w.shape
    cout_g = cout // groups
    span = dilation * (k - 1) + 1
    t_out = (xp.shape[1] - span) // stride + 1
    out = np.zeros((cout, t_out))
    for o in range(cout):
        g = o // cout_g
        for t in range(t_out):
            acc = 0.0
            for c in range(cin_g):
                for j in range(k):
                    acc += w[o, c, j] * xp[g * cin_g + c, t * stride + j * dilation]
            out[o, t] = acc + (b[o] if b is not None else 0.0)
    return out


@pytest.mark.parametrize("cin,cout,k,stride,dil,groups,T,pad", [
    (4, 6, 3, 1, 1, 1, 12, (1, 1)),
    (4, 6, 5, 1, 2, 1, 16, (4, 0)),          # left-causal, dilated
    (8, 8, 3, 1, 1, 4, 13, (2, 2)),          # grouped
    (3, 6, 4, 2, 1, 1, 17, (0, 0)),          # strided
    (18, 8, 30, 15, 1, 1, 61, (14, 0)),      # HiFT source downsample
    (16, 16, 7, 1, 1, 4, 20, (6, 0)),        # DiT-style grouped causal
])
def test_conv1d_matches_naive(rng, cin, cout, k, stride, dil, groups, T, pad):
    w = rng.normal(size=(cout, cin // groups, k)).astype(np.float32) * 0.1
    b = rng.normal(size=cout).astype(np.float32)
    x = rng.normal(size=(cin, T)).astype(np.float32)
    ref = _naive_conv1d(x, w, b, stride, dil, groups, pad)
    for tile in (1 << 8, 1 << 20):           # force many tiles, then one
        got = ops.conv1d(x, w, b, stride=stride, dilation=dil, groups=groups,
                         pad=pad, tile_bytes=tile)
        assert got.shape == ref.shape
        assert rel(got, ref) < 1e-5


def test_depthwise_matches_naive(rng):
    c, k, t = 12, 7, 20
    w = rng.normal(size=(c, 1, k)).astype(np.float32)
    b = rng.normal(size=c).astype(np.float32)
    x = rng.normal(size=(c, t)).astype(np.float32)
    ref = _naive_conv1d(x, w, b, 1, 1, c, (3, 3))
    assert rel(ops.depthwise_conv1d(x, w, b, pad=(3, 3)), ref) < 1e-5


def test_quantised_conv_weight_carries_its_kernel_size(rng):
    """A folded (out, in*k) conv weight must still convolve, not just matmul."""
    from cv3cpu.quant import Q8_BLOCK, QTensor, quantize_q8

    w = (rng.normal(size=(6, 4, 3)) * 0.1).astype(np.float32)
    x = rng.normal(size=(4, 15)).astype(np.float32)
    q, sc, _ = quantize_q8(w.reshape(6, -1))
    qt = QTensor("q8", q, sc, (6, 12), Q8_BLOCK, orig_shape=(6, 4, 3))
    assert qt.kernel_size == 3
    ref = _naive_conv1d(x, w, None, 1, 1, 1, (2, 0))
    assert rel(ops.conv1d(x, qt, None, pad=(2, 0)), ref) < 0.02


# --------------------------------------------------------------------------
# attention, against a materialised softmax
# --------------------------------------------------------------------------

def _naive_attention(q, k, v, mask=None):
    h, n, d = q.shape
    kk = np.repeat(k, h // k.shape[0], axis=0) if k.shape[0] != h else k
    vv = np.repeat(v, h // v.shape[0], axis=0) if v.shape[0] != h else v
    out = np.zeros((h, n, d))
    for i in range(h):
        logits = q[i].astype(np.float64) @ kk[i].astype(np.float64).T / np.sqrt(d)
        if mask is not None:
            logits = np.where(mask, logits, -np.inf)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        out[i] = p @ vv[i].astype(np.float64)
    return out


def _chunk_mask(n, chunk, num_left=-1):
    m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        start = 0 if num_left < 0 else max((i // chunk - num_left) * chunk, 0)
        m[i, start:min((i // chunk + 1) * chunk, n)] = True
    return m


def test_attention_full_and_causal(rng):
    h, n, d = 4, 23, 16
    q, k, v = (rng.normal(size=(h, n, d)).astype(np.float32) for _ in range(3))
    for tile in (1 << 10, 1 << 22):
        assert rel(ops.sdpa(q, k, v, tile_bytes=tile), _naive_attention(q, k, v)) < 1e-5
        causal = np.tril(np.ones((n, n), dtype=bool))
        assert rel(ops.sdpa(q, k, v, ("causal",), tile_bytes=tile),
                   _naive_attention(q, k, v, causal)) < 1e-5


@pytest.mark.parametrize("chunk,left", [(8, -1), (8, 1), (5, 0)])
def test_attention_chunk_mask(rng, chunk, left):
    h, n, d = 4, 23, 16
    q, k, v = (rng.normal(size=(h, n, d)).astype(np.float32) for _ in range(3))
    ref = _naive_attention(q, k, v, _chunk_mask(n, chunk, left))
    assert rel(ops.sdpa(q, k, v, ("chunk", chunk, left), tile_bytes=1 << 10), ref) < 1e-5


def test_attention_grouped_query(rng):
    h, n, d = 4, 19, 16
    q = rng.normal(size=(h, n, d)).astype(np.float32)
    k = rng.normal(size=(2, n, d)).astype(np.float32)
    v = rng.normal(size=(2, n, d)).astype(np.float32)
    assert rel(ops.sdpa(q, k, v), _naive_attention(q, k, v)) < 1e-5


def test_attention_with_a_cache_offset(rng):
    """One decode step attends to the whole cache plus itself."""
    h, past, d = 4, 9, 16
    q = rng.normal(size=(h, 1, d)).astype(np.float32)
    k = rng.normal(size=(h, past + 1, d)).astype(np.float32)
    v = rng.normal(size=(h, past + 1, d)).astype(np.float32)
    assert rel(ops.sdpa(q, k, v, ("causal", past)), _naive_attention(q, k, v)) < 1e-5


# --------------------------------------------------------------------------
# STFT / ISTFT
# --------------------------------------------------------------------------

def _naive_stft(x, n_fft, hop, win):
    xp = np.pad(np.asarray(x, np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
    n = 1 + (len(xp) - n_fft) // hop
    out = np.zeros((n_fft // 2 + 1, n), dtype=np.complex128)
    idx = np.arange(n_fft)
    for t in range(n):
        frame = xp[t * hop: t * hop + n_fft] * win
        for f in range(n_fft // 2 + 1):
            out[f, t] = (frame * np.exp(-2j * np.pi * f * idx / n_fft)).sum()
    return out


def test_stft_matches_direct_dft(rng):
    n_fft, hop = 16, 4
    win = ops.hann_window(n_fft)
    sig = rng.normal(size=200).astype(np.float32)
    got = ops.stft(sig, n_fft, hop, win)
    ref = _naive_stft(sig, n_fft, hop, win)
    assert got.shape == ref.shape
    assert rel(np.stack([got.real, got.imag]), np.stack([ref.real, ref.imag])) < 1e-5


def test_istft_inverts_stft(rng):
    """Hann at 4x overlap satisfies COLA, so the round trip is the identity."""
    n_fft, hop = 16, 4
    win = ops.hann_window(n_fft)
    sig = rng.normal(size=2000).astype(np.float32)
    back = ops.istft(ops.stft(sig, n_fft, hop, win), n_fft, hop, win, length=len(sig))
    assert rel(back[n_fft:-n_fft], sig[n_fft:-n_fft]) < 1e-5


def test_istft_length_matches_torch_convention(rng):
    """torch.istft returns (frames - 1) * hop samples when no length is given."""
    n_fft, hop, frames = 16, 4, 51
    spec = (rng.normal(size=(9, frames)) + 1j * rng.normal(size=(9, frames))).astype(np.complex64)
    assert len(ops.istft(spec, n_fft, hop, ops.hann_window(n_fft))) == (frames - 1) * hop


def test_hann_window_is_periodic():
    w = ops.hann_window(8)
    assert w[0] == 0.0
    assert rel(w, np.array([0, 0.1464466, 0.5, 0.8535534, 1, 0.8535534, 0.5, 0.1464466])) < 1e-6


# --------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------

def test_linear_interpolation_half_pixel_convention(rng):
    """align_corners=False: output i samples input at (i + 0.5) * scale - 0.5."""
    x = np.arange(10, dtype=np.float32)[None, :]
    got = ops.interpolate_linear(x, 5)[0]
    assert rel(got, np.array([0.5, 2.5, 4.5, 6.5, 8.5])) < 1e-6
    assert rel(ops.interpolate_linear(x, 10), x) < 1e-6


def test_nearest_upsample_repeats(rng):
    x = rng.normal(size=(3, 5)).astype(np.float32)
    got = ops.interpolate_nearest(x, 4)
    assert got.shape == (3, 20)
    assert np.array_equal(got[:, ::4], x)
    assert np.array_equal(got[:, 1::4], x)


def test_embedding_rejects_out_of_range_ids():
    table = np.zeros((10, 4), dtype=np.float32)
    assert ops.embedding(table, np.array([0, 9])).shape == (2, 4)
    with pytest.raises(IndexError, match="outside"):
        ops.embedding(table, np.array([10]))
