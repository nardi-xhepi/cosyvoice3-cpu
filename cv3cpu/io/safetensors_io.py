"""Reading and writing ``.safetensors`` files with NumPy only.

The format is a u64 header length, that many bytes of UTF-8 JSON naming every
tensor's dtype, shape and byte range, then the packed tensor data.  That is
already everything a memory-mapped weight file needs, which is why this project
uses it rather than a container of its own.

:class:`SafeTensorsWriter` streams: it reserves header space up front, appends
tensor data as it goes, and patches the JSON in at close, padding it with spaces
(valid JSON whitespace) so the reserved length is exact and the data section
stays gapless.  That lets a multi-gigabyte file be written one tensor at a time
without holding them all in memory or making a second pass.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

# safetensors dtype tag <-> numpy dtype
DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}
_TAGS = {v: k for k, v in DTYPES.items()}
_BF16 = "BF16"  # read-only: widened to float32, numpy has no bfloat16


def dtype_tag(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype).newbyteorder("<")
    if dtype not in _TAGS:
        raise ValueError(f"no safetensors dtype for {dtype}")
    return _TAGS[dtype]


class SafeTensorsWriter:
    """Append tensors in order; the header is patched in at :meth:`close`."""

    def __init__(self, path: str, metadata: Optional[Dict[str, str]] = None,
                 header_reserve: int = 1 << 20):
        self.path = path
        self.metadata = dict(metadata or {})
        # Keep (8 + reserve) a multiple of 8 so the data section stays aligned.
        self._reserve = (header_reserve + 7) // 8 * 8
        self._index: Dict[str, Dict[str, Any]] = {}
        self._pos = 0
        self._fh = open(path, "wb")
        self._fh.write(b"\0" * (8 + self._reserve))

    def add(self, name: str, arr: np.ndarray, dtype=None) -> None:
        if name in self._index:
            raise ValueError(f"duplicate tensor name {name!r}")
        arr = np.ascontiguousarray(arr if dtype is None else np.asarray(arr).astype(dtype, copy=False))
        start = self._pos
        self._fh.write(arr.tobytes())
        self._pos += arr.nbytes
        self._index[name] = {"dtype": dtype_tag(arr.dtype), "shape": list(arr.shape),
                             "data_offsets": [start, self._pos]}

    def close(self) -> None:
        header: Dict[str, Any] = dict(self._index)
        if self.metadata:
            header["__metadata__"] = self.metadata
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(raw) > self._reserve:
            raise RuntimeError(
                f"header of {len(raw)} bytes exceeds the reserved {self._reserve}; "
                f"raise header_reserve")
        self._fh.seek(0)
        self._fh.write(struct.pack("<Q", self._reserve))
        # Trailing spaces are JSON whitespace, so the padding parses away.
        self._fh.write(raw + b" " * (self._reserve - len(raw)))
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SafeTensors:
    """Lazily map a safetensors file; ``__getitem__`` returns a view into it."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        header = json.loads(self._fh.read(n).decode("utf-8"))
        self.data_start = 8 + n
        self.metadata: Dict[str, str] = header.pop("__metadata__", {})
        self._index: Dict[str, Dict[str, Any]] = header
        self._mm = mmap.mmap(self._fh.fileno(), os.path.getsize(path),
                             access=mmap.ACCESS_READ)
        self._view = memoryview(self._mm)

    def keys(self) -> Iterator[str]:
        return iter(self._index)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def shape_dtype(self, key: str) -> Tuple[Tuple[int, ...], str]:
        e = self._index[key]
        return tuple(e["shape"]), e["dtype"]

    def byte_range(self, key: str) -> Tuple[int, int]:
        """Absolute file offsets of a tensor's data, for madvise and friends."""
        start, end = self._index[key]["data_offsets"]
        return self.data_start + start, self.data_start + end

    def nbytes(self, key: str) -> int:
        start, end = self._index[key]["data_offsets"]
        return end - start

    def __getitem__(self, key: str) -> np.ndarray:
        e = self._index[key]
        start, end = e["data_offsets"]
        raw = self._view[self.data_start + start: self.data_start + end]
        tag = e["dtype"]
        shape = tuple(e["shape"])
        if tag == _BF16:
            u16 = np.frombuffer(raw, dtype="<u2")
            return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
        if tag not in DTYPES:
            raise ValueError(f"unsupported safetensors dtype {tag} for {key}")
        return np.frombuffer(raw, dtype=DTYPES[tag]).reshape(shape)

    def madvise(self, what: int, start: int, length: int) -> bool:
        try:
            self._mm.madvise(what, start, length)
        except (AttributeError, OSError, ValueError):
            return False
        return True

    def close(self) -> None:
        try:
            self._view.release()
            self._mm.close()
        except BufferError:
            # Views handed out are still alive; drop our reference and let the
            # mapping go when the last one does.
            pass
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SafeTensorsDir(dict):
    """``{name: ndarray}`` over one or more shards, keeping the mappings alive."""

    def __init__(self, handles):
        super().__init__()
        self._handles = handles
        for st in handles:
            for k in st.keys():
                self[k] = st[k]

    def close(self) -> None:
        self.clear()
        for st in self._handles:
            st.close()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_dir(path: str) -> SafeTensorsDir:
    """Map every ``*.safetensors`` shard under ``path`` as one name->array dict."""
    if os.path.isfile(path):
        files = [path]
    else:
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as fh:
                shards = sorted(set(json.load(fh)["weight_map"].values()))
            files = [os.path.join(path, s) for s in shards]
        else:
            files = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found under {path}")
    return SafeTensorsDir([SafeTensors(f) for f in files])
