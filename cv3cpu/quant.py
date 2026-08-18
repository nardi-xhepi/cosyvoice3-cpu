"""Block-wise weight quantisation (q8 / q4) and the dequantising GEMM.

The whole point of this module is that a weight never exists in float form for
longer than one row-block.  ``qmatmul`` walks the output rows of ``W`` in tiles,
expands one tile into a small reusable float32 scratch buffer and hands that to
BLAS.  Peak extra memory is ``tile_rows * K * 4`` bytes (a few MB), not the size
of the weight.

Layout, for a weight of logical shape ``(N, K)``:

* ``K`` is padded up to a multiple of ``block``; the padding is quantised as
  zeros and sliced off after dequantisation.
* ``q8``: ``int8[N, Kp]`` payload, ``float16[N, Kp // block]`` scales.
* ``q4``: ``uint8[N, Kp // 2]`` payload packed llama.cpp-Q4_0 style (byte ``j``
  of a block holds value ``j`` in the low nibble and value ``j + block//2`` in
  the high nibble, both biased by +8), same scale layout.

Quantisation is symmetric and per block along the reduction axis, which is the
axis a matmul sums over, so the scale factors out of the dot product cleanly.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

Q8_BLOCK = 64
Q4_BLOCK = 32

# How many float32 bytes one dequantisation tile may occupy.  1 MB keeps the
# expanded tile inside L2, which measured ~1.8x faster than an L3-sized tile:
# the cast is bandwidth-bound on its *output*, so the smaller the round trip the
# better.  Matmuls with many activation rows are compute-bound instead and get a
# larger tile (see ``qmatmul``).
TILE_BYTES = 1 << 20
BIG_TILE_BYTES = 8 << 20
# Above this many activation rows the GEMM dominates, BLAS parallelises it
# itself, and adding our own threads only oversubscribes the machine.
THREAD_ROW_LIMIT = 4


def _pad_last(x: np.ndarray, block: int) -> np.ndarray:
    k = x.shape[-1]
    rem = (-k) % block
    if rem == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[-1] = (0, rem)
    return np.pad(x, pad, mode="constant")


def quantize_q8(w: np.ndarray, block: int = Q8_BLOCK):
    """Return ``(int8 payload, float16 scales, padded_k)``."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    n = w.shape[0]
    wp = _pad_last(w, block)
    kp = wp.shape[-1]
    blocks = wp.reshape(n, kp // block, block)
    amax = np.abs(blocks).max(axis=2)
    scale = (amax / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.rint(blocks / scale[:, :, None]).clip(-127, 127).astype(np.int8)
    return q.reshape(n, kp), scale.astype(np.float16), kp


def quantize_q4(w: np.ndarray, block: int = Q4_BLOCK):
    """Return ``(uint8 payload, float16 scales, padded_k)``."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    n = w.shape[0]
    wp = _pad_last(w, block)
    kp = wp.shape[-1]
    nb = kp // block
    blocks = wp.reshape(n, nb, block)
    # llama.cpp Q4_0: anchor on the signed extreme so it maps exactly onto -8.
    idx = np.abs(blocks).argmax(axis=2)
    ext = np.take_along_axis(blocks, idx[:, :, None], axis=2)[:, :, 0]
    scale = (ext / -8.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.rint(blocks / scale[:, :, None] + 8.0).clip(0, 15).astype(np.uint8)
    half = block // 2
    packed = (q[:, :, :half] | (q[:, :, half:] << 4)).astype(np.uint8)
    return packed.reshape(n, kp // 2), scale.astype(np.float16), kp


@dataclass
class QTensor:
    """A quantised 2-D weight plus everything needed to expand it."""

    kind: str            # "q8" | "q4"
    q: np.ndarray        # payload view (usually into an mmap)
    scales: np.ndarray   # float16 (N, nblocks)
    shape: tuple         # logical (N, K) of the matrix
    block: int
    orig_shape: Optional[tuple] = None   # e.g. (out, in, k) for a folded conv

    @property
    def kernel_size(self) -> Optional[int]:
        if self.orig_shape is not None and len(self.orig_shape) == 3:
            return int(self.orig_shape[2])
        return None

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scales.nbytes

    def dequant_rows(self, r0: int, r1: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Expand rows ``[r0, r1)`` into float32, writing straight into ``out``."""
        k = self.shape[1]
        nb = self.scales.shape[1]
        rows = r1 - r0
        padded = nb * self.block
        if out is None:
            out = np.empty((rows, padded), dtype=np.float32)
        elif out.shape[1] < padded:
            raise ValueError("scratch buffer is narrower than the padded row")
        view = out[:rows, :padded].reshape(rows, nb, self.block)
        if self.kind == "q8":
            np.copyto(view, self.q[r0:r1].reshape(rows, nb, self.block))
        else:
            half = self.block // 2
            b = self.q[r0:r1].reshape(rows, nb, half)
            # Fold the +8 bias removal into the widening cast so the expanded
            # tile is written once rather than written and then walked again.
            np.subtract(b & np.uint8(0x0F), np.float32(8.0), out=view[:, :, :half],
                        casting="unsafe")
            np.subtract(b >> np.uint8(4), np.float32(8.0), out=view[:, :, half:],
                        casting="unsafe")
        view *= self.scales[r0:r1][:, :, None].astype(np.float32)
        return out[:rows, :k]

    def dequant(self) -> np.ndarray:
        return self.dequant_rows(0, self.shape[0])


def _default_threads() -> int:
    env = os.environ.get("CV3_MATMUL_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(4, os.cpu_count() or 1))


_POOL: Optional[ThreadPoolExecutor] = None


def _pool(n: int) -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None or getattr(_POOL, "_max_workers", 0) < n:
        _POOL = ThreadPoolExecutor(max_workers=n)
    return _POOL


def qmatmul(x: np.ndarray, w, bias: Optional[np.ndarray] = None,
            tile_bytes: Optional[int] = None, threads: Optional[int] = None
            ) -> np.ndarray:
    """``x @ w.T`` where ``w`` is a :class:`QTensor` or a plain ``(N, K)`` array.

    ``x`` has shape ``(..., K)``; the result is ``(..., N)``.  Only one tile of
    ``w`` is ever float, so peak scratch is a megabyte regardless of how big the
    weight is.
    """
    if not isinstance(w, QTensor):
        wf = np.asarray(w, dtype=np.float32)
        y = x.astype(np.float32, copy=False) @ wf.T
        if bias is not None:
            y += bias
        return y

    n, k = w.shape
    lead = x.shape[:-1]
    xf = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    m = xf.shape[0]
    padded = w.scales.shape[1] * w.block
    if tile_bytes is None:
        tile_bytes = TILE_BYTES if m <= THREAD_ROW_LIMIT else BIG_TILE_BYTES
    rows = max(1, min(n, tile_bytes // max(1, padded * 4)))
    # Accumulate transposed so each tile's destination stays C-contiguous and
    # BLAS can write straight into it.
    yt = np.empty((n, m), dtype=np.float32)
    starts = range(0, n, rows)

    def work(chunk) -> None:
        scratch = np.empty((rows, padded), dtype=np.float32)
        for r0 in chunk:
            r1 = min(n, r0 + rows)
            tile = w.dequant_rows(r0, r1, scratch)
            np.dot(tile, xf.T, out=yt[r0:r1])

    nthreads = _default_threads() if threads is None else threads
    if nthreads > 1 and m <= THREAD_ROW_LIMIT and len(starts) >= 2 * nthreads:
        # Decoding one token at a time is bound by streaming the weights in and
        # casting them; that parallelises where a single BLAS gemv does not.
        chunks = [list(starts)[i::nthreads] for i in range(nthreads)]
        list(_pool(nthreads).map(work, chunks))
    else:
        work(starts)

    y = np.ascontiguousarray(yt.T)
    if bias is not None:
        y += bias
    return y.reshape(*lead, n)
