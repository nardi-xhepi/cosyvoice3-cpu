"""CausalMaskedDiffWithDiT: speech tokens (+ prompt mel) -> mel spectrogram."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import ops
from ..io.weights import WeightFile
from ..torch_rng import randn
from .dit import DiT

# CausalConditionalCFM draws its ODE initial condition once, right after
# ``set_all_random_seed(0)``: torch.randn([1, 80, 50 * 300]).
NOISE_SHAPE = (1, 80, 50 * 300)
NOISE_SEED = 0


class Flow:
    def __init__(self, weights: WeightFile, prefix: str = "flow."):
        self.w = weights
        self.p = prefix
        m = weights.meta.get("flow", {})
        self.token_mel_ratio = int(m.get("token_mel_ratio", 2))
        self.pre_lookahead_len = int(m.get("pre_lookahead_len", 3))
        self.output_size = int(m.get("output_size", 80))
        self.inference_cfg_rate = float(m.get("inference_cfg_rate", 0.7))
        self.n_timesteps = int(m.get("n_timesteps", 10))
        self.estimator = DiT(weights, prefix + "decoder.estimator.")
        self._noise: Optional[np.ndarray] = None

    def _t(self, name: str):
        return self.w.get(self.p + name)

    def rand_noise(self, n: int) -> np.ndarray:
        if self._noise is None:
            self._noise = randn(NOISE_SHAPE, NOISE_SEED)
        if n > self._noise.shape[2]:
            raise ValueError(
                f"utterance needs {n} mel frames but the fixed CFM noise only has "
                f"{self._noise.shape[2]}; split the text into shorter segments")
        return self._noise[:, :, :n]

    # ------------------------------------------------------------------
    def pre_lookahead(self, tokens_emb: np.ndarray) -> np.ndarray:
        """``(T, 80)`` -> ``(T, 80)``; offline (``finalize=True``) path."""
        x = np.ascontiguousarray(tokens_emb.T)                       # (80, T)
        h = ops.conv1d(x, self._t("pre_lookahead_layer.conv1.weight"),
                       self._t("pre_lookahead_layer.conv1.bias"),
                       pad=(0, self.pre_lookahead_len))
        h = ops.leaky_relu(h, 0.01)
        k2 = self.w.entry(self.p + "pre_lookahead_layer.conv2.weight")["shape"][-1]
        h = ops.conv1d(h, self._t("pre_lookahead_layer.conv2.weight"),
                       self._t("pre_lookahead_layer.conv2.bias"), pad=(k2 - 1, 0))
        return h.T + tokens_emb

    def solve_euler(self, mu: np.ndarray, spks: np.ndarray, cond: np.ndarray,
                    streaming: bool = False, temperature: float = 1.0,
                    seq_chunk: int = 512, progress=None) -> np.ndarray:
        """Fixed-step Euler CFM solve with classifier-free guidance.

        ``mu``/``cond``: ``(1, 80, T)``, ``spks``: ``(1, 80)``.
        """
        n = mu.shape[2]
        x = self.rand_noise(n) * temperature
        steps = self.n_timesteps
        t_span = 1.0 - np.cos(np.linspace(0.0, 1.0, steps + 1, dtype=np.float64) * 0.5 * np.pi)
        t_span = t_span.astype(np.float32)

        # The CFG batch: slot 0 keeps the conditioning, slot 1 zeroes it out.
        mu_in = np.zeros((2, 80, n), dtype=np.float32)
        mu_in[0] = mu[0]
        spks_in = np.zeros((2, spks.shape[1]), dtype=np.float32)
        spks_in[0] = spks[0]
        cond_in = np.zeros((2, 80, n), dtype=np.float32)
        cond_in[0] = cond[0]

        t = t_span[0]
        dt = t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            x_in = np.broadcast_to(x, (2, 80, n))
            t_in = np.array([t, t], dtype=np.float32)
            d = self.estimator.forward(x_in, mu_in, t_in, spks_in, cond_in,
                                       streaming=streaming, seq_chunk=seq_chunk)
            dphi = (1.0 + self.inference_cfg_rate) * d[0:1] - self.inference_cfg_rate * d[1:2]
            x = x + dt * dphi
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            if progress is not None:
                progress(step, len(t_span) - 1)
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    def inference(self, token: Sequence[int], prompt_token: Sequence[int],
                  prompt_feat: np.ndarray, embedding: np.ndarray,
                  streaming: bool = False, seq_chunk: int = 512,
                  progress=None) -> np.ndarray:
        """Return the generated mel ``(80, T_mel)`` (prompt frames stripped).

        ``prompt_feat`` is ``(T_prompt_mel, 80)`` and ``embedding`` is the
        192-dim CAM++ speaker vector.
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        emb = ops.linear(emb, self._t("spk_embed_affine_layer.weight"),
                         self._t("spk_embed_affine_layer.bias"))          # (1, 80)

        ids = np.concatenate([np.asarray(prompt_token, dtype=np.int64),
                              np.asarray(token, dtype=np.int64)])
        ids = np.maximum(ids, 0)
        emb_tok = ops.embedding(self._t("input_embedding"), ids)          # (T, 80)
        h = self.pre_lookahead(emb_tok)
        h = np.repeat(h, self.token_mel_ratio, axis=0)                    # (2T, 80)

        prompt_feat = np.asarray(prompt_feat, dtype=np.float32)
        mel_len1 = prompt_feat.shape[0]
        total = h.shape[0]
        if mel_len1 > total:
            raise ValueError("prompt mel is longer than the token grid")
        cond = np.zeros((1, self.output_size, total), dtype=np.float32)
        cond[0, :, :mel_len1] = prompt_feat.T

        mu = np.ascontiguousarray(h.T)[None]                              # (1, 80, 2T)
        feat = self.solve_euler(mu, emb, cond, streaming=streaming,
                                seq_chunk=seq_chunk, progress=progress)
        return feat[0, :, mel_len1:]
