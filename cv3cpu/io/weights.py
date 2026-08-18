"""The model-facing weight file: a plain ``.safetensors`` plus a quantisation index.

Quantised weights need two tensors and a little bookkeeping, which safetensors
carries without any format extension:

* the payload goes under the tensor's own name, as ``I8`` (q8) or ``U8`` (packed
  q4 nibbles);
* its per-block scales go under ``<name>.__scales__`` as ``F16``;
* kind, block size, and the weight's logical shape live in a JSON blob under the
  ``__metadata__`` key ``cv3cpu``.

So the file loads in any safetensors reader — you just see the raw payload and
scale tensors — while :class:`WeightFile` reassembles them into a
:class:`~cv3cpu.quant.QTensor` the kernels can consume.

The mapping is ``MAP_PRIVATE`` read-only, so pages are file-backed: they count as
page cache, are shared between processes, and can be dropped by the kernel under
pressure instead of pushing the process into swap.  :meth:`WeightFile.evict` asks
for that explicitly once a stage is done with its weights, which is what keeps
the pipeline's memory profile flat.
"""

from __future__ import annotations

import json
import mmap
from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..quant import Q4_BLOCK, Q8_BLOCK, QTensor, quantize_q4, quantize_q8
from .safetensors_io import DTYPES, SafeTensors, SafeTensorsWriter

SCALES_SUFFIX = ".__scales__"
METADATA_KEY = "cv3cpu"
FORMAT_VERSION = 1

# raw storage kinds -> numpy dtype
RAW_KINDS = {"f32": np.dtype("<f4"), "f16": np.dtype("<f2"), "i32": np.dtype("<i4"),
             "i64": np.dtype("<i8"), "i8": np.dtype("i1"), "u8": np.dtype("u1")}
QUANT_KINDS = {"q8": (quantize_q8, Q8_BLOCK, np.dtype("i1")),
               "q4": (quantize_q4, Q4_BLOCK, np.dtype("u1"))}


class WeightWriter:
    """Streaming writer.  Tensors are appended; the header lands at :meth:`close`."""

    def __init__(self, path: str, meta: Optional[Dict[str, Any]] = None,
                 header_reserve: int = 1 << 20):
        self.path = path
        self.meta = dict(meta or {})
        self._quant: Dict[str, Dict[str, Any]] = {}
        self._st = SafeTensorsWriter(path, header_reserve=header_reserve)

    def add_raw(self, name: str, arr: np.ndarray, kind: str = "f32") -> None:
        if kind not in RAW_KINDS:
            raise ValueError(f"unknown raw kind {kind!r}")
        self._st.add(name, np.asarray(arr), dtype=RAW_KINDS[kind])

    def add_quant(self, name: str, w: np.ndarray, kind: str = "q8",
                  orig_shape=None) -> None:
        """Store a 2-D matrix block-quantised.

        ``orig_shape`` keeps a convolution's ``(out, in, k)`` so readers can still
        recover the kernel size after it has been folded to ``(out, in * k)``.
        """
        if kind not in QUANT_KINDS:
            raise ValueError(f"unknown quant kind {kind!r}")
        w = np.asarray(w, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"{name}: quantised tensors must be 2-D, got {w.shape}")
        quantize, block, _ = QUANT_KINDS[kind]
        q, scales, _ = quantize(w)
        self._st.add(name, q)
        self._st.add(name + SCALES_SUFFIX, scales)
        self._quant[name] = {
            "kind": kind, "block": block,
            "shape": list(orig_shape if orig_shape is not None else w.shape),
            "mshape": list(w.shape),
        }

    def close(self) -> None:
        self._st.metadata[METADATA_KEY] = json.dumps(
            {"version": FORMAT_VERSION, "meta": self.meta, "quant": self._quant},
            separators=(",", ":"))
        self._st.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class WeightFile:
    """Read side.  ``get(name)`` returns an ndarray view or a :class:`QTensor`."""

    def __init__(self, path: str):
        self.path = path
        self._st = SafeTensors(path)
        blob = json.loads(self._st.metadata.get(METADATA_KEY, "{}"))
        self.meta: Dict[str, Any] = blob.get("meta", {})
        self._quant: Dict[str, Dict[str, Any]] = blob.get("quant", {})
        self._names = [k for k in self._st.keys() if not k.endswith(SCALES_SUFFIX)]
        self._cache: Dict[str, Any] = {}

    # -- introspection -----------------------------------------------------
    def keys(self) -> Iterable[str]:
        return list(self._names)

    def __contains__(self, name: str) -> bool:
        return name in self._st and not name.endswith(SCALES_SUFFIX)

    def entry(self, name: str) -> Dict[str, Any]:
        """Logical description: storage kind, original shape, bytes on disk."""
        if name not in self:
            raise KeyError(name)
        q = self._quant.get(name)
        nbytes = self._st.nbytes(name)
        if q is not None:
            return {"kind": q["kind"], "shape": list(q["shape"]), "block": q["block"],
                    "mshape": list(q["mshape"]),
                    "nbytes": nbytes + self._st.nbytes(name + SCALES_SUFFIX)}
        shape, tag = self._st.shape_dtype(name)
        kind = next((k for k, d in RAW_KINDS.items() if DTYPES.get(tag) == d), tag.lower())
        return {"kind": kind, "shape": list(shape), "nbytes": nbytes}

    def total_bytes(self) -> int:
        return sum(self._st.nbytes(k) for k in self._st.keys())

    def stage_bytes(self, prefix: str) -> int:
        return sum(self._st.nbytes(k) for k in self._st.keys() if k.startswith(prefix))

    # -- access ------------------------------------------------------------
    @staticmethod
    def _aligned(arr: np.ndarray) -> np.ndarray:
        """Copy out of the mapping only if the view is misaligned for its dtype.

        safetensors packs tensors with no gaps, so a 1-byte tensor of odd length
        would leave everything after it on an odd address, and NumPy falls back
        to a buffered path for unaligned arrays -- measured at ~18% on the
        float16 scale multiply.  No tensor in this model triggers it (quantised
        payloads always have an even byte count, because the reduction axis is
        padded to a multiple of the block size), but a copy is the cheap
        insurance.
        """
        if arr.itemsize > 1 and arr.ctypes.data % arr.itemsize:
            # np.ascontiguousarray would hand the same misaligned view straight
            # back -- it only copies when the layout is non-contiguous.
            return np.array(arr, copy=True)
        return arr

    def get(self, name: str):
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        q = self._quant.get(name)
        if q is None:
            out = self._aligned(self._st[name])
        else:
            out = QTensor(q["kind"], self._st[name],
                          self._aligned(self._st[name + SCALES_SUFFIX]),
                          tuple(q["mshape"]), q["block"], orig_shape=tuple(q["shape"]))
        self._cache[name] = out
        return out

    def get_f32(self, name: str) -> np.ndarray:
        """Materialise a tensor as float32, dequantising if needed."""
        t = self.get(name)
        return t.dequant() if isinstance(t, QTensor) else np.asarray(t, dtype=np.float32)

    def subset(self, prefix: str) -> Dict[str, Any]:
        n = len(prefix)
        return {k[n:]: self.get(k) for k in self._names if k.startswith(prefix)}

    # -- lifetime ----------------------------------------------------------
    def release(self, prefix: str = "") -> None:
        """Drop cached views so nothing keeps the mapped pages referenced."""
        for k in [k for k in self._cache if k.startswith(prefix)]:
            del self._cache[k]

    def evict(self, prefix: str = "") -> int:
        """Hand the pages of every tensor under ``prefix`` back to the kernel.

        This is what makes the staged pipeline hold a flat memory profile: once
        the LLM has produced its speech tokens, its ~500 MB of weights are pure
        page cache, and ``MADV_DONTNEED`` drops them instead of leaving them to
        compete with the flow stage.  Returns the number of bytes released.
        """
        lo = hi = None
        for name in self._st.keys():
            if not name.startswith(prefix):
                continue
            start, end = self._st.byte_range(name)
            lo = start if lo is None else min(lo, start)
            hi = end if hi is None else max(hi, end)
        if lo is None:
            return 0
        self.release(prefix)
        page = mmap.PAGESIZE
        start = (lo + page - 1) // page * page
        end = hi // page * page
        if end <= start:
            return 0
        return end - start if self._st.madvise(mmap.MADV_DONTNEED, start, end - start) else 0

    def close(self) -> None:
        self._cache.clear()
        self._st.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
