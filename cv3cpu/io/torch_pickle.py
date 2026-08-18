"""Read PyTorch ``.pt`` checkpoints without PyTorch.

``torch.save`` writes a zip archive that contains a pickle (``data.pkl``)
describing the object graph plus one flat file per storage under ``data/``.
Tensors are reconstructed by the pickle through ``torch._utils._rebuild_tensor_v2``
with a *persistent id* referring to the storage file.

We re-implement just enough of that protocol to hand back NumPy arrays.  Nothing
is executed from the pickle stream: the unpickler refuses every global except a
small allow-list of the ``torch`` rebuild helpers.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from typing import Any, Dict

import numpy as np

# torch storage class name -> numpy dtype
_STORAGE_DTYPE = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "BFloat16Storage": None,  # handled specially, numpy has no bfloat16
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}

_DTYPE_NAMES = {
    "torch.float32": "FloatStorage",
    "torch.float": "FloatStorage",
    "torch.float64": "DoubleStorage",
    "torch.double": "DoubleStorage",
    "torch.float16": "HalfStorage",
    "torch.half": "HalfStorage",
    "torch.bfloat16": "BFloat16Storage",
    "torch.int64": "LongStorage",
    "torch.long": "LongStorage",
    "torch.int32": "IntStorage",
    "torch.int": "IntStorage",
    "torch.int16": "ShortStorage",
    "torch.int8": "CharStorage",
    "torch.uint8": "ByteStorage",
    "torch.bool": "BoolStorage",
}


class _Storage:
    """Lazy handle on one ``data/<key>`` member of the archive."""

    __slots__ = ("zf", "name", "dtype", "key")

    def __init__(self, zf: zipfile.ZipFile, name: str, dtype, key: str):
        self.zf = zf
        self.name = name
        self.dtype = dtype
        self.key = key

    def read(self) -> np.ndarray:
        raw = self.zf.read(self.name)
        if self.dtype is None:  # bfloat16 -> float32 by zero-extending the mantissa
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype(np.uint32) << np.uint32(16)
            return u32.view(np.float32)
        return np.frombuffer(raw, dtype=self.dtype)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False,
                       backward_hooks=None, metadata=None):
    flat = storage.read()
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    n = 1
    for s in size:
        n *= s
    if n == 0:
        return np.zeros(size, dtype=flat.dtype)
    itemsize = flat.dtype.itemsize
    strides = tuple(s * itemsize for s in stride)
    base = flat[storage_offset:]
    arr = np.lib.stride_tricks.as_strided(base, shape=size, strides=strides)
    # as_strided views alias a read-only buffer; hand back an owned, writable copy.
    return np.array(arr, dtype=flat.dtype, copy=True)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None):
    return data


class _OrderedDictShim(dict):
    pass


_ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("collections", "OrderedDict"): _OrderedDictShim,
}


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, zf: zipfile.ZipFile, prefix: str):
        super().__init__(file, encoding="latin1")
        self._zf = zf
        self._prefix = prefix

    def find_class(self, module, name):
        key = (module, name)
        if key in _ALLOWED:
            return _ALLOWED[key]
        if module == "torch" and name in _DTYPE_NAMES.values():
            return name  # storage *class*, only used as a tag
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name} from checkpoint; only plain "
            f"tensor state_dicts are supported"
        )

    def persistent_load(self, pid):
        assert pid[0] == "storage", pid
        storage_type, key = pid[1], pid[2]
        name = str(storage_type)
        dtype = _STORAGE_DTYPE.get(name)
        if name not in _STORAGE_DTYPE:
            raise pickle.UnpicklingError(f"unsupported storage type {name}")
        member = f"{self._prefix}data/{key}"
        return _Storage(self._zf, member, dtype, key)


def load_pt(path: str) -> Dict[str, Any]:
    """Load a ``torch.save``-d state dict as ``{name: np.ndarray}``."""
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a zip-format torch checkpoint (legacy .pt files from "
            f"torch<1.6 are not supported)"
        )
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = next((n for n in names if n.endswith("data.pkl")), None)
    if pkl is None:
        raise ValueError(f"{path} has no data.pkl member")
    prefix = pkl[: -len("data.pkl")]
    with zf.open(pkl) as fh:
        buf = io.BytesIO(fh.read())
    try:
        obj = _Unpickler(buf, zf, prefix).load()
    finally:
        zf.close()
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a state_dict-like mapping")
    return {str(k): v for k, v in obj.items()}
