"""NumPy kernels for the CosyVoice 3 graph.

Two rules shape everything here:

1. No temporary may scale with ``len(weights)`` or ``T**2``.  Convolutions are
   im2col'd in time tiles and attention uses a streaming (flash-style) softmax,
   so every intermediate is bounded by a byte budget rather than by the input.
2. Weights are consumed straight from the ``.safetensors`` mmap, expanded one tile at a
   time by :func:`cv3cpu.quant.qmatmul`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from .quant import QTensor, qmatmul

Weight = Union[np.ndarray, QTensor]

TILE_BYTES = 8 << 20

# --------------------------------------------------------------------------
# elementwise
# --------------------------------------------------------------------------


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """``nn.GELU(approximate='tanh')``."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x * x * x)))


def gelu(x: np.ndarray) -> np.ndarray:
    """``nn.GELU()`` — the exact erf form."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26: max absolute error 1.5e-7, i.e. below float32
    # resolution, and fully vectorised (math.erf through np.vectorize is ~100x
    # slower).
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def mish(x: np.ndarray) -> np.ndarray:
    """``nn.Mish``: ``x * tanh(softplus(x))`` with the same threshold=20 guard."""
    sp = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))
    return x * np.tanh(sp)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0.0)))


def leaky_relu(x: np.ndarray, slope: float = 0.01) -> np.ndarray:
    return np.where(x >= 0, x, x * slope)


def snake(x: np.ndarray, alpha: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """BigVGAN Snake over ``(C, T)`` with per-channel alpha."""
    a = alpha.reshape(-1, 1)
    s = np.sin(a * x)
    return x + (1.0 / (a + eps)) * (s * s)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


# --------------------------------------------------------------------------
# norms and linears
# --------------------------------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * (1.0 / np.sqrt(var + eps))) * weight


def layer_norm(x: np.ndarray, weight: Optional[np.ndarray] = None,
               bias: Optional[np.ndarray] = None, eps: float = 1e-5) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    mu = np.mean(x32, axis=-1, keepdims=True)
    xc = x32 - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    y = xc / np.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def linear(x: np.ndarray, weight: Weight, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """``x @ weight.T + bias`` with ``weight`` shaped ``(out, in)``."""
    return qmatmul(x, weight, bias)


def embedding(table: Weight, ids: np.ndarray) -> np.ndarray:
    """Row gather that never materialises the full table."""
    ids = np.asarray(ids)
    rows = table.shape[0] if isinstance(table, QTensor) else np.shape(table)[0]
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= rows):
        raise IndexError(
            f"token id {int(ids.max())} is outside the {rows}-row embedding table")
    if isinstance(table, QTensor):
        flat = ids.reshape(-1)
        out = np.empty((flat.size, table.shape[1]), dtype=np.float32)
        # Gather unique rows so a repeated token costs one dequantisation.
        uniq, inv = np.unique(flat, return_inverse=True)
        rows = np.empty((uniq.size, table.shape[1]), dtype=np.float32)
        for i, r in enumerate(uniq):
            rows[i] = table.dequant_rows(int(r), int(r) + 1)[0]
        out[:] = rows[inv]
        return out.reshape(*ids.shape, table.shape[1])
    # Gather first, widen second: a float16 table stays float16 on disk.
    return np.asarray(table[ids], dtype=np.float32)


# --------------------------------------------------------------------------
# convolution
# --------------------------------------------------------------------------


def _im2col(xp: np.ndarray, k: int, stride: int, dilation: int,
            t0: int, t1: int) -> np.ndarray:
    """``(C, k, t1 - t0)`` patch view of padded ``xp`` for output frames [t0, t1)."""
    c, _ = xp.shape
    sc, st = xp.strides
    return np.lib.stride_tricks.as_strided(
        xp[:, t0 * stride:],
        shape=(c, k, t1 - t0),
        strides=(sc, st * dilation, st * stride),
        writeable=False,
    )


def conv1d(x: np.ndarray, weight: Weight, bias: Optional[np.ndarray] = None,
           stride: int = 1, dilation: int = 1, groups: int = 1,
           pad: Tuple[int, int] = (0, 0), kernel_size: Optional[int] = None,
           tile_bytes: int = TILE_BYTES) -> np.ndarray:
    """1-D convolution over ``(C_in, T)`` returning ``(C_out, T_out)``.

    ``weight`` is either a ``(C_out, C_in // groups, k)`` array or a QTensor
    holding the same weight flattened to ``(C_out, C_in // groups * k)``.
    """
    if isinstance(weight, QTensor):
        cout = weight.shape[0]
        k = kernel_size or weight.kernel_size
        if k is None:
            raise ValueError("kernel_size is required for a quantised conv weight "
                             "that does not record its original 3-D shape")
        cin_g = weight.shape[1] // k
    else:
        weight = np.asarray(weight, dtype=np.float32)
        cout, cin_g, k = weight.shape
    cin = x.shape[0]
    assert cin == cin_g * groups, (cin, cin_g, groups)

    xp = x
    if pad[0] or pad[1]:
        xp = np.pad(x, ((0, 0), pad), mode="constant")
    xp = np.ascontiguousarray(xp, dtype=np.float32)
    tp = xp.shape[1]
    span = dilation * (k - 1) + 1
    t_out = (tp - span) // stride + 1
    if t_out <= 0:
        return np.zeros((cout, 0), dtype=np.float32)

    out = np.empty((cout, t_out), dtype=np.float32)
    per_frame = cin_g * k * 4
    tile = max(1, min(t_out, tile_bytes // max(1, per_frame)))

    cout_g = cout // groups
    for g in range(groups):
        xg = xp[g * cin_g:(g + 1) * cin_g]
        if isinstance(weight, QTensor):
            wg = weight  # groups==1 only for quantised convs
            assert groups == 1, "quantised grouped convs are not supported"
        else:
            wg = weight[g * cout_g:(g + 1) * cout_g].reshape(cout_g, cin_g * k)
        for t0 in range(0, t_out, tile):
            t1 = min(t_out, t0 + tile)
            cols = _im2col(xg, k, stride, dilation, t0, t1)
            cols = np.ascontiguousarray(cols).reshape(cin_g * k, t1 - t0)
            if isinstance(wg, QTensor):
                out[:, t0:t1] = qmatmul(cols.T, wg).T
            else:
                out[g * cout_g:(g + 1) * cout_g, t0:t1] = np.dot(wg, cols)
    if bias is not None:
        out += np.asarray(bias, dtype=np.float32).reshape(-1, 1)
    return out


def depthwise_conv1d(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
                     dilation: int = 1, pad: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """``groups == C`` convolution done as ``k`` shifted multiply-accumulates."""
    weight = np.asarray(weight, dtype=np.float32)
    c, one, k = weight.shape
    assert one == 1
    xp = np.pad(x, ((0, 0), pad), mode="constant") if (pad[0] or pad[1]) else x
    span = dilation * (k - 1) + 1
    t_out = xp.shape[1] - span + 1
    out = np.zeros((c, t_out), dtype=np.float32)
    for j in range(k):
        out += weight[:, 0, j:j + 1] * xp[:, j * dilation: j * dilation + t_out]
    if bias is not None:
        out += np.asarray(bias, dtype=np.float32).reshape(-1, 1)
    return out


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------


def _block_mask(q0: int, q1: int, k0: int, k1: int, spec) -> Optional[np.ndarray]:
    """Boolean ``(q1-q0, k1-k0)`` block of the attention mask; None means 'all on'."""
    if spec is None:
        return None
    kind = spec[0]
    qi = np.arange(q0, q1)[:, None]
    ki = np.arange(k0, k1)[None, :]
    if kind == "causal":
        offset = spec[1] if len(spec) > 1 else 0
        return ki <= qi + offset
    if kind == "chunk":
        # wenet subsequent_chunk_mask: attend to the end of one's own chunk and
        # (optionally a bounded number of) chunks to the left.
        size, num_left = spec[1], spec[2]
        end = (qi // size + 1) * size
        if num_left < 0:
            start = np.zeros_like(qi)
        else:
            start = np.maximum((qi // size - num_left) * size, 0)
        return (ki < end) & (ki >= start)
    raise ValueError(f"unknown mask spec {spec}")


def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask_spec=None,
         scale: Optional[float] = None, tile_bytes: int = TILE_BYTES) -> np.ndarray:
    """Scaled dot-product attention over ``(H, N, D)``, tiled over queries.

    Each query tile sees every key in one pass, so the softmax is the ordinary
    one -- no online rescaling -- while peak scratch stays at
    ``tile_q * M * H * 4`` bytes.  A 3000-frame DiT layer costs a few MB instead
    of the 500+ MB a materialised ``N x N`` map would take.
    ``k``/``v`` may have fewer heads than ``q`` (grouped-query attention).
    """
    h, n, d = q.shape
    hk, m, _ = k.shape
    if hk != h:
        if h % hk:
            raise ValueError(f"{h} query heads is not a multiple of {hk} kv heads")
        k = np.repeat(k, h // hk, axis=0)
        v = np.repeat(v, h // hk, axis=0)
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    tile_q = max(1, min(n, tile_bytes // max(1, h * m * 4)))
    kt = np.ascontiguousarray(np.swapaxes(k, 1, 2))
    out = np.empty((h, n, d), dtype=np.float32)
    for q0 in range(0, n, tile_q):
        q1 = min(n, q0 + tile_q)
        logits = (q[:, q0:q1] * np.float32(scale)) @ kt
        add = _additive_mask(q0, q1, m, mask_spec)
        if add is not None:
            logits += add
        logits -= logits.max(axis=-1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=-1, keepdims=True)
        out[:, q0:q1] = logits @ v
    return out


def _additive_mask(q0: int, q1: int, m: int, spec) -> Optional[np.ndarray]:
    """``(1, q1-q0, m)`` float mask with -inf where attention is forbidden."""
    if spec is None:
        return None
    bm = _block_mask(q0, q1, 0, m, spec)
    add = np.zeros((1, q1 - q0, m), dtype=np.float32)
    add[0][~bm] = -np.inf
    return add


# --------------------------------------------------------------------------
# resampling / signal helpers
# --------------------------------------------------------------------------


def interpolate_linear(x: np.ndarray, size: int, align_corners: bool = False) -> np.ndarray:
    """``F.interpolate(mode='linear')`` over the last axis of ``(C, T)``."""
    c, t = x.shape
    if t == size:
        return x.astype(np.float32, copy=True)
    if align_corners:
        pos = np.linspace(0.0, t - 1.0, size, dtype=np.float64)
    else:
        scale = t / size
        pos = (np.arange(size, dtype=np.float64) + 0.5) * scale - 0.5
        pos = np.clip(pos, 0.0, t - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (pos - i0).astype(np.float32)
    return x[:, i0] * (1.0 - w) + x[:, i1] * w


def interpolate_nearest(x: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(x, scale, axis=-1)


def hann_window(n: int, periodic: bool = True) -> np.ndarray:
    """``scipy.signal.get_window('hann', n, fftbins=True)``."""
    if periodic:
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def stft(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray,
         center: bool = True, pad_mode: str = "reflect") -> np.ndarray:
    """Match ``torch.stft(..., return_complex=True, onesided=True)`` for 1-D input."""
    x = np.asarray(x, dtype=np.float32)
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode=pad_mode)
    n = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, n_fft), strides=(x.strides[0] * hop, x.strides[0]), writeable=False)
    return np.fft.rfft(frames * win, n=n_fft, axis=1).T  # (F, n)


def istft(spec: np.ndarray, n_fft: int, hop: int, win: np.ndarray,
          center: bool = True, length: Optional[int] = None) -> np.ndarray:
    """Match ``torch.istft`` (onesided, normalized=False) for a ``(F, N)`` input."""
    frames = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)  # (n_fft, N)
    n = frames.shape[1]
    frames = frames * win[:, None]
    total = (n - 1) * hop + n_fft
    y = np.zeros(total, dtype=np.float32)
    env = np.zeros(total, dtype=np.float32)
    w2 = (win * win).astype(np.float32)
    for j in range(n_fft):
        y[j: j + (n - 1) * hop + 1: hop] += frames[j]
        env[j: j + (n - 1) * hop + 1: hop] += w2[j]
    nz = np.abs(env) > 1e-11
    y[nz] /= env[nz]
    if center:
        y = y[n_fft // 2: total - n_fft // 2]
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y
