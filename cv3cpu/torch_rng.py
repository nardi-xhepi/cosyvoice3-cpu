"""Bit-exact replication of PyTorch's CPU RNG, in NumPy.

CosyVoice 3 bakes two *fixed* random tensors into inference:
``CausalConditionalCFM.rand_noise`` (the flow-matching ODE initial condition,
drawn right after ``torch.manual_seed(0)``) and the NSF excitation noise inside
the vocoder.  They are plain attributes, so they are not in the checkpoint — a
PyTorch-free runtime has to be able to reproduce them.

PyTorch's CPU generator is MT19937 seeded with ``init_genrand`` (NumPy's
``RandomState`` uses ``init_by_array`` instead, so it does *not* match), 32-bit
outputs turned into float32 uniforms by taking the low 24 bits, and normals
produced by ``normal_fill``: Box–Muller over 16-element chunks pairing element
``j`` with element ``j + 8``.
"""

from __future__ import annotations

import numpy as np

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF


class TorchMT19937:
    """MT19937 with PyTorch's seeding and output scaling."""

    def __init__(self, seed: int = 0):
        mt = np.zeros(_N, dtype=np.uint32)
        mt[0] = np.uint32(seed & 0xFFFFFFFF)
        for i in range(1, _N):
            prev = int(mt[i - 1])
            mt[i] = np.uint32((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self._mt = mt
        self._idx = _N

    def _twist(self) -> None:
        mt = self._mt
        nxt = np.empty(_N, dtype=np.uint32)
        upper = np.uint32(_UPPER)
        lower = np.uint32(_LOWER)
        mag = np.uint32(_MATRIX_A)

        def step(dst_lo, dst_hi, src):
            y = (mt[dst_lo:dst_hi] & upper) | (mt[dst_lo + 1:dst_hi + 1] & lower)
            nxt[dst_lo:dst_hi] = src ^ (y >> np.uint32(1)) ^ \
                np.where(y & np.uint32(1), mag, np.uint32(0))

        # i in [0, N-M): reads only untouched words.
        step(0, _N - _M, mt[_M:_N])
        # i in [N-M, N-1): reads nxt[i - (N-M)], which lags by N-M, so it can be
        # done in two vectorised slabs.
        lag = _N - _M
        lo = _N - _M
        while lo < _N - 1:
            hi = min(lo + lag, _N - 1)
            step(lo, hi, nxt[lo - lag:hi - lag])
            lo = hi
        # i = N-1 reads the freshly written nxt[0] and nxt[M-1].
        y = (int(mt[_N - 1]) & _UPPER) | (int(nxt[0]) & _LOWER)
        v = int(nxt[_M - 1]) ^ (y >> 1)
        if y & 1:
            v ^= _MATRIX_A
        nxt[_N - 1] = np.uint32(v & 0xFFFFFFFF)

        self._mt = nxt
        self._idx = 0

    def next_u32(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.uint32)
        filled = 0
        while filled < n:
            if self._idx >= _N:
                self._twist()
            take = min(n - filled, _N - self._idx)
            out[filled:filled + take] = self._mt[self._idx:self._idx + take]
            self._idx += take
            filled += take
        y = out.copy()
        y ^= y >> np.uint32(11)
        y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
        y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
        y ^= y >> np.uint32(18)
        return y

    def random_float32(self, n: int) -> np.ndarray:
        """``at::uniform_real_distribution<float>`` over [0, 1)."""
        u = self.next_u32(n) & np.uint32((1 << 24) - 1)
        return (u.astype(np.float32) * np.float32(1.0 / (1 << 24)))

    def random_float64(self, n: int) -> np.ndarray:
        """``at::uniform_real_distribution<double>``: ``random64() & (2**53-1)``.

        ``CPUGeneratorImpl::random64`` puts the *first* 32-bit draw in the high
        half.
        """
        raw = self.next_u32(2 * n).astype(np.uint64)
        v = (raw[0::2] << np.uint64(32)) | raw[1::2]
        mask = (np.uint64(1) << np.uint64(53)) - np.uint64(1)
        return (v & mask).astype(np.float64) * (1.0 / (1 << 53))


def _normal_fill_16(block: np.ndarray) -> np.ndarray:
    d = block.astype(np.float64)
    u1 = 1.0 - d[..., :8]
    u2 = d[..., 8:]
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    out = np.empty_like(d)
    out[..., :8] = radius * np.cos(theta)
    out[..., 8:] = radius * np.sin(theta)
    return out.astype(np.float32)


def randn(shape, seed: int = 0, gen: "TorchMT19937 | None" = None) -> np.ndarray:
    """Reproduce ``torch.manual_seed(seed); torch.randn(shape)`` on CPU."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    n = int(np.prod(shape)) if shape else 1
    g = gen if gen is not None else TorchMT19937(seed)
    if n < 16:
        # Below 16 elements ATen skips normal_fill and runs the scalar
        # ``normal_distribution<double>``: two *double* uniforms per pair, theta
        # from the first, radius from the second, second value cached.
        pairs = (n + 1) // 2
        u = g.random_float64(2 * pairs).reshape(pairs, 2)
        th = 2.0 * np.pi * u[:, 0]
        r = np.sqrt(-2.0 * np.log1p(-u[:, 1]))
        both = np.empty(2 * pairs, dtype=np.float32)
        both[0::2] = (r * np.cos(th)).astype(np.float32)
        both[1::2] = (r * np.sin(th)).astype(np.float32)
        return both[:n].reshape(shape)
    flat = g.random_float32(n)
    whole = n - n % 16
    out = np.empty(n, dtype=np.float32)
    out[:whole] = _normal_fill_16(flat[:whole].reshape(-1, 16)).reshape(-1)
    if n % 16:
        # torch re-fills the *last* 16 elements, overlapping what it just wrote.
        tail = g.random_float32(16)
        out[n - 16:] = _normal_fill_16(tail.reshape(1, 16)).reshape(-1)
    return out.reshape(shape)


def rand(shape, seed: int = 0, gen: "TorchMT19937 | None" = None) -> np.ndarray:
    """Reproduce ``torch.manual_seed(seed); torch.rand(shape)`` on CPU."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    n = int(np.prod(shape)) if shape else 1
    g = gen if gen is not None else TorchMT19937(seed)
    return g.random_float32(n).reshape(shape)
