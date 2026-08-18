"""The CosyVoice 3 flow-matching estimator: an F5-TTS style DiT, in NumPy."""

from __future__ import annotations

import math

import numpy as np

from .. import ops
from ..io.weights import WeightFile


def sinus_position_embedding(t: np.ndarray, dim: int = 256, scale: float = 1000.0) -> np.ndarray:
    half = dim // 2
    step = math.log(10000.0) / (half - 1)
    freqs = np.exp(np.arange(half, dtype=np.float32) * -step)
    emb = scale * t.reshape(-1, 1).astype(np.float32) * freqs[None, :]
    return np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)


def rope_freqs(seq_len: int, dim_head: int, theta: float = 10000.0) -> np.ndarray:
    """``x_transformers.RotaryEmbedding.forward_from_seq_len`` -> ``(T, dim_head)``.

    The inner frequencies are duplicated *interleaved* (``stack`` then flatten),
    which is what pairs with the GPT-J style ``rotate_half`` below.
    """
    inv = 1.0 / (theta ** (np.arange(0, dim_head, 2, dtype=np.float64) / dim_head))
    freqs = np.outer(np.arange(seq_len, dtype=np.float64), inv)      # (T, dim_head/2)
    return np.repeat(freqs, 2, axis=-1).astype(np.float32)           # (T, dim_head)


def _rotate_half_interleaved(x: np.ndarray) -> np.ndarray:
    y = x.reshape(*x.shape[:-1], -1, 2)
    out = np.empty_like(y)
    out[..., 0] = -y[..., 1]
    out[..., 1] = y[..., 0]
    return out.reshape(x.shape)


def apply_partial_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Rotate the leading ``freqs.shape[-1]`` channels of ``x`` and leave the rest.

    ``x`` is ``(..., T, C)`` with ``C`` the *whole* projection width (1024), while
    ``freqs`` covers only ``dim_head`` (64).  Upstream applies the rotation before
    splitting into heads, so only head 0 actually receives positional
    information.  That is a quirk of the reference implementation, and the
    weights were trained with it, so it is reproduced here deliberately.
    """
    rot = freqs.shape[-1]
    head, tail = x[..., :rot], x[..., rot:]
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    head = head * cos + _rotate_half_interleaved(head) * sin
    return np.concatenate([head, tail], axis=-1)


class DiT:
    """``cosyvoice.flow.DiT.dit.DiT`` with dim=1024, depth=22, heads=16."""

    def __init__(self, weights: WeightFile, prefix: str = "flow.decoder.estimator."):
        self.w = weights
        self.p = prefix
        m = weights.meta.get("dit", {})
        self.dim = int(m.get("dim", 1024))
        self.depth = int(m.get("depth", 22))
        self.heads = int(m.get("heads", 16))
        self.dim_head = int(m.get("dim_head", 64))
        self.static_chunk_size = int(m.get("static_chunk_size", 50))
        self.num_decoding_left_chunks = int(m.get("num_decoding_left_chunks", -1))
        self.conv_pos_kernel = int(m.get("conv_pos_kernel", 31))
        self.conv_pos_groups = int(m.get("conv_pos_groups", 16))
        self._rope_cache = (0, None)

    def _t(self, name: str):
        return self.w.get(self.p + name)

    def _has(self, name: str) -> bool:
        return (self.p + name) in self.w

    def _rope(self, n: int) -> np.ndarray:
        if self._rope_cache[0] < n:
            self._rope_cache = (n, rope_freqs(n, self.dim_head))
        return self._rope_cache[1][:n]

    # ------------------------------------------------------------------
    def time_embed(self, t: np.ndarray) -> np.ndarray:
        h = sinus_position_embedding(t, 256, 1000.0)
        h = ops.linear(h, self._t("time_embed.time_mlp.0.weight"), self._t("time_embed.time_mlp.0.bias"))
        h = ops.silu(h)
        return ops.linear(h, self._t("time_embed.time_mlp.2.weight"), self._t("time_embed.time_mlp.2.bias"))

    def input_embed(self, x: np.ndarray, cond: np.ndarray, mu: np.ndarray,
                    spks: np.ndarray) -> np.ndarray:
        """``x``/``cond``/``mu`` are ``(B, T, 80)``, ``spks`` is ``(B, 80)``."""
        b, t, _ = x.shape
        spk = np.broadcast_to(spks[:, None, :], (b, t, spks.shape[1]))
        cat = np.concatenate([x, cond, mu, spk], axis=-1)
        h = ops.linear(cat, self._t("input_embed.proj.weight"), self._t("input_embed.proj.bias"))
        pos = np.empty_like(h)
        k = self.conv_pos_kernel
        for i in range(b):
            z = np.ascontiguousarray(h[i].T)                        # (dim, T)
            z = ops.conv1d(z, self._t("input_embed.conv_pos_embed.conv1.0.weight"),
                           self._t("input_embed.conv_pos_embed.conv1.0.bias"),
                           groups=self.conv_pos_groups, pad=(k - 1, 0))
            z = ops.mish(z)
            z = ops.conv1d(z, self._t("input_embed.conv_pos_embed.conv2.0.weight"),
                           self._t("input_embed.conv_pos_embed.conv2.0.bias"),
                           groups=self.conv_pos_groups, pad=(k - 1, 0))
            z = ops.mish(z)
            pos[i] = z.T
        return pos + h

    def _attn(self, blk: str, x: np.ndarray, freqs: np.ndarray, mask_spec) -> np.ndarray:
        b, t, _ = x.shape
        q = ops.linear(x, self._t(blk + "attn.to_q.weight"), self._t(blk + "attn.to_q.bias"))
        k = ops.linear(x, self._t(blk + "attn.to_k.weight"), self._t(blk + "attn.to_k.bias"))
        v = ops.linear(x, self._t(blk + "attn.to_v.weight"), self._t(blk + "attn.to_v.bias"))
        q = apply_partial_rope(q, freqs)
        k = apply_partial_rope(k, freqs)
        h, d = self.heads, self.dim_head
        out = np.empty((b, t, h * d), dtype=np.float32)
        for i in range(b):
            qi = np.ascontiguousarray(q[i].reshape(t, h, d).transpose(1, 0, 2))
            ki = np.ascontiguousarray(k[i].reshape(t, h, d).transpose(1, 0, 2))
            vi = np.ascontiguousarray(v[i].reshape(t, h, d).transpose(1, 0, 2))
            a = ops.sdpa(qi, ki, vi, mask_spec)
            out[i] = a.transpose(1, 0, 2).reshape(t, h * d)
        return ops.linear(out, self._t(blk + "attn.to_out.0.weight"), self._t(blk + "attn.to_out.0.bias"))

    def _ff(self, blk: str, x: np.ndarray, seq_chunk: int) -> np.ndarray:
        b, t, c = x.shape
        out = np.empty_like(x)
        w1 = self._t(blk + "ff.ff.0.0.weight")
        b1 = self._t(blk + "ff.ff.0.0.bias")
        w2 = self._t(blk + "ff.ff.2.weight")
        b2 = self._t(blk + "ff.ff.2.bias")
        for t0 in range(0, t, seq_chunk):
            t1 = min(t, t0 + seq_chunk)
            h = ops.gelu_tanh(ops.linear(x[:, t0:t1], w1, b1))
            out[:, t0:t1] = ops.linear(h, w2, b2)
        return out

    def forward(self, x: np.ndarray, mu: np.ndarray, t: np.ndarray, spks: np.ndarray,
                cond: np.ndarray, streaming: bool = False,
                seq_chunk: int = 512) -> np.ndarray:
        """``x``/``mu``/``cond``: ``(B, 80, T)``; ``t``: ``(B,)``; ``spks``: ``(B, 80)``.

        Returns ``(B, 80, T)``.
        """
        x = np.ascontiguousarray(np.swapaxes(x, 1, 2))
        mu = np.ascontiguousarray(np.swapaxes(mu, 1, 2))
        cond = np.ascontiguousarray(np.swapaxes(cond, 1, 2))
        b, n, _ = x.shape

        temb = self.time_embed(np.asarray(t, dtype=np.float32))      # (B, dim)
        h = self.input_embed(x, cond, mu, spks)
        freqs = self._rope(n)
        mask_spec = ("chunk", self.static_chunk_size, self.num_decoding_left_chunks) \
            if streaming else None

        for i in range(self.depth):
            blk = f"transformer_blocks.{i}."
            mod = ops.linear(ops.silu(temb), self._t(blk + "attn_norm.linear.weight"),
                             self._t(blk + "attn_norm.linear.bias"))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                np.split(mod, 6, axis=1)
            norm = ops.layer_norm(h, None, None, 1e-6)
            norm = norm * (1.0 + scale_msa[:, None]) + shift_msa[:, None]
            h = h + gate_msa[:, None] * self._attn(blk, norm, freqs, mask_spec)
            norm = ops.layer_norm(h, None, None, 1e-6)
            norm = norm * (1.0 + scale_mlp[:, None]) + shift_mlp[:, None]
            h = h + gate_mlp[:, None] * self._ff(blk, norm, seq_chunk)

        mod = ops.linear(ops.silu(temb), self._t("norm_out.linear.weight"),
                         self._t("norm_out.linear.bias"))
        scale, shift = np.split(mod, 2, axis=1)
        h = ops.layer_norm(h, None, None, 1e-6) * (1.0 + scale)[:, None] + shift[:, None]
        out = ops.linear(h, self._t("proj_out.weight"), self._t("proj_out.bias"))
        return np.ascontiguousarray(np.swapaxes(out, 1, 2))
