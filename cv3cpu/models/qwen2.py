"""Qwen2 decoder stack (the CosyVoice 3 text/speech LM body) in NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .. import ops
from ..io.weights import WeightFile


@dataclass
class Qwen2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 151936

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_meta(cls, meta: dict) -> "Qwen2Config":
        m = meta["qwen2"]
        return cls(**{k: m[k] for k in cls.__dataclass_fields__ if k in m})


class RopeCache:
    """cos/sin tables for HF-style (half-split) rotary embeddings."""

    def __init__(self, dim: int, theta: float, max_pos: int = 4096):
        self.dim = dim
        self.theta = theta
        self._n = 0
        self._cos = np.zeros((0, dim), np.float32)
        self._sin = np.zeros((0, dim), np.float32)
        self.extend(max_pos)

    def extend(self, n: int) -> None:
        if n <= self._n:
            return
        n = max(n, self._n * 2, 256)
        inv = 1.0 / (self.theta ** (np.arange(0, self.dim, 2, dtype=np.float64) / self.dim))
        t = np.arange(n, dtype=np.float64)
        freqs = np.outer(t, inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self._cos = np.cos(emb).astype(np.float32)
        self._sin = np.sin(emb).astype(np.float32)
        self._n = n

    def get(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if positions.max() >= self._n:
            self.extend(int(positions.max()) + 1)
        return self._cos[positions], self._sin[positions]


def _rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``x`` is ``(H, T, D)``, ``cos``/``sin`` are ``(T, D)``."""
    return x * cos[None] + _rotate_half(x) * sin[None]


class KVCache:
    """Per-layer key/value cache, grown geometrically.

    float32 by default: for a 24-layer/2-KV-head/64-dim model this is ~25 KB per
    token, so even a 2000-token utterance costs ~50 MB — not the thing to
    squeeze.  Pass ``dtype=np.float16`` to halve it anyway.
    """

    def __init__(self, layers: int, heads: int, head_dim: int, dtype=np.float32):
        self.k: List[Optional[np.ndarray]] = [None] * layers
        self.v: List[Optional[np.ndarray]] = [None] * layers
        self._len = 0
        self._cap = 0
        self.heads = heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.layers = layers

    def __len__(self) -> int:
        return self._len

    def _reserve(self, need: int) -> None:
        if need <= self._cap:
            return
        cap = max(need, max(256, self._cap * 2))
        for i in range(self.layers):
            for buf in ("k", "v"):
                old = getattr(self, buf)[i]
                new = np.zeros((self.heads, cap, self.head_dim), dtype=self.dtype)
                if old is not None:
                    new[:, : self._len] = old[:, : self._len]
                getattr(self, buf)[i] = new
        self._cap = cap

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = k.shape[1]
        if layer == 0:
            self._reserve(self._len + t)
        start = self._len
        self.k[layer][:, start:start + t] = k.astype(self.dtype)
        self.v[layer][:, start:start + t] = v.astype(self.dtype)
        end = start + t
        if layer == self.layers - 1:
            self._len = end
        return self.k[layer][:, :end], self.v[layer][:, :end]

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k + self.v if a is not None)


class Qwen2Model:
    """Decoder-only Qwen2 body.  ``forward`` consumes *embeddings*, not ids."""

    def __init__(self, weights: WeightFile, cfg: Qwen2Config, prefix: str = "llm."):
        self.w = weights
        self.cfg = cfg
        self.prefix = prefix
        self.rope = RopeCache(cfg.head_dim, cfg.rope_theta)

    def _t(self, name: str):
        return self.w.get(self.prefix + name)

    def embed(self, ids: np.ndarray) -> np.ndarray:
        return ops.embedding(self._t("embed_tokens"), ids)

    def forward(self, x: np.ndarray, cache: Optional[KVCache] = None,
                tile_bytes: int = ops.TILE_BYTES) -> np.ndarray:
        """``x`` is ``(T, hidden)``; returns the final hidden states ``(T, hidden)``."""
        cfg = self.cfg
        t = x.shape[0]
        past = len(cache) if cache is not None else 0
        pos = np.arange(past, past + t)
        cos, sin = self.rope.get(pos)
        # Prefill is causal; a single decoding step sees the whole cache.
        mask = ("causal", past) if t > 1 else None
        h = x.astype(np.float32, copy=False)
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}."
            res = h
            n = ops.rms_norm(h, self._t(p + "input_layernorm"), cfg.rms_norm_eps)
            q = ops.linear(n, self._t(p + "self_attn.q_proj.weight"), self._t(p + "self_attn.q_proj.bias"))
            k = ops.linear(n, self._t(p + "self_attn.k_proj.weight"), self._t(p + "self_attn.k_proj.bias"))
            v = ops.linear(n, self._t(p + "self_attn.v_proj.weight"), self._t(p + "self_attn.v_proj.bias"))
            q = np.ascontiguousarray(q.reshape(t, nh, hd).transpose(1, 0, 2))
            k = np.ascontiguousarray(k.reshape(t, nkv, hd).transpose(1, 0, 2))
            v = np.ascontiguousarray(v.reshape(t, nkv, hd).transpose(1, 0, 2))
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if cache is not None:
                k, v = cache.append(i, k, v)
                if k.dtype != np.float32:
                    k = k.astype(np.float32)
                    v = v.astype(np.float32)
            a = ops.sdpa(q, k, v, mask, tile_bytes=tile_bytes)
            a = a.transpose(1, 0, 2).reshape(t, nh * hd)
            h = res + ops.linear(a, self._t(p + "self_attn.o_proj.weight"))

            res = h
            n = ops.rms_norm(h, self._t(p + "post_attention_layernorm"), cfg.rms_norm_eps)
            g = ops.linear(n, self._t(p + "mlp.gate_proj.weight"))
            u = ops.linear(n, self._t(p + "mlp.up_proj.weight"))
            h = res + ops.linear(ops.silu(g) * u, self._t(p + "mlp.down_proj.weight"))

        return ops.rms_norm(h, self._t("norm"), cfg.rms_norm_eps)

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.num_hidden_layers, self.cfg.num_key_value_heads,
                       self.cfg.head_dim)
