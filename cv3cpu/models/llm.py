"""CosyVoice3LM: text tokens -> FSQ speech tokens, autoregressively."""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np

from .. import ops
from ..io.weights import WeightFile
from .qwen2 import Qwen2Config, Qwen2Model

# CosyVoice 3 uses a 6561-entry FSQ speech codebook; the control tokens live in
# the 200 slots above it (see cosyvoice/llm/llm.py::CosyVoice3LM.__init__).
SPEECH_TOKEN_SIZE = 6561
ENDOFPROMPT_ID = 151646  # "<|endofprompt|>" in the Qwen tokenizer


class SamplingConfig:
    def __init__(self, top_p: float = 0.8, top_k: int = 25, win_size: int = 10,
                 tau_r: float = 0.1, seed: Optional[int] = None,
                 max_token_text_ratio: float = 20.0,
                 min_token_text_ratio: float = 2.0,
                 strict_min_len: bool = False):
        self.top_p = top_p
        self.top_k = top_k
        self.win_size = win_size
        self.tau_r = tau_r
        self.max_token_text_ratio = max_token_text_ratio
        self.min_token_text_ratio = min_token_text_ratio
        # Upstream masks only one id while below min_len, so generation can still
        # stop early; set this to mask every stop id, as the code there intends.
        self.strict_min_len = strict_min_len
        self.rng = np.random.default_rng(seed)


def nucleus_sampling(probs_sorted: np.ndarray, idx_sorted: np.ndarray,
                     rng: np.random.Generator, top_p: float, top_k: int) -> int:
    """Upstream ``nucleus_sampling``: take entries until cum-prob >= top_p, capped at top_k."""
    cum = 0.0
    n = 0
    for i in range(min(top_k, idx_sorted.shape[0])):
        if cum >= top_p:
            break
        cum += float(probs_sorted[i])
        n += 1
    p = probs_sorted[:n].astype(np.float64)
    p = p / p.sum()
    return int(idx_sorted[rng.choice(n, p=p)])


class CosyVoice3LM:
    def __init__(self, weights: WeightFile, prefix: str = "llm."):
        self.w = weights
        self.prefix = prefix
        cfg = Qwen2Config.from_meta(weights.meta)
        self.cfg = cfg
        self.body = Qwen2Model(weights, cfg, prefix=prefix + "model.")
        self.speech_token_size = int(weights.meta.get("speech_token_size", SPEECH_TOKEN_SIZE))
        self.sos = self.speech_token_size + 0
        self.eos_token = self.speech_token_size + 1
        self.task_id = self.speech_token_size + 2
        self.fill_token = self.speech_token_size + 3
        self.stop_token_ids = set(range(self.speech_token_size, self.speech_token_size + 200))
        self.endofprompt_id = int(weights.meta.get("endofprompt_id", ENDOFPROMPT_ID))

    # -- pieces ------------------------------------------------------------
    def _speech_emb(self, ids) -> np.ndarray:
        return ops.embedding(self.w.get(self.prefix + "speech_embedding"), np.asarray(ids))

    def _text_emb(self, ids) -> np.ndarray:
        return self.body.embed(np.asarray(ids))

    def _head(self, h: np.ndarray) -> np.ndarray:
        bias = self.w.get(self.prefix + "llm_decoder.bias") if \
            (self.prefix + "llm_decoder.bias") in self.w else None
        return ops.linear(h, self.w.get(self.prefix + "llm_decoder.weight"), bias)

    # -- generation --------------------------------------------------------
    def generate(self,
                 text_tokens: Sequence[int],
                 prompt_text_tokens: Sequence[int] = (),
                 prompt_speech_tokens: Sequence[int] = (),
                 sampling: Optional[SamplingConfig] = None,
                 strict_min_len: Optional[bool] = None,
                 progress=None) -> Iterator[int]:
        """Yield speech tokens one at a time (the upstream streaming contract)."""
        sampling = sampling or SamplingConfig()
        if strict_min_len is None:
            strict_min_len = sampling.strict_min_len
        text = list(prompt_text_tokens) + list(text_tokens)
        if self.endofprompt_id not in text:
            raise ValueError(
                f"CosyVoice 3 expects '<|endofprompt|>' (id {self.endofprompt_id}) "
                f"somewhere in the prompt text; prepend e.g. "
                f"'You are a helpful assistant.<|endofprompt|>'")

        sos = self._speech_emb([self.sos])          # (1, H)
        task = self._speech_emb([self.task_id])
        text_emb = self._text_emb(text)             # (T, H)
        parts = [sos, text_emb, task]
        if len(prompt_speech_tokens):
            parts.append(self._speech_emb(list(prompt_speech_tokens)))
        lm_input = np.concatenate(parts, axis=0).astype(np.float32)

        n_new_text = len(text) - len(prompt_text_tokens)
        min_len = int(n_new_text * sampling.min_token_text_ratio)
        max_len = int(n_new_text * sampling.max_token_text_ratio)

        cache = self.body.new_cache()
        out_tokens: List[int] = []
        for i in range(max_len):
            h = self.body.forward(lm_input, cache)
            logp = ops.log_softmax(self._head(h[-1:]).astype(np.float32), axis=-1)[0]
            ignore_eos = i < min_len
            top = self._sample(logp, out_tokens, sampling, ignore_eos, strict_min_len)
            if top in self.stop_token_ids:
                break
            yield top
            out_tokens.append(top)
            lm_input = self._speech_emb([top])
            if progress is not None:
                progress(i, top)

    def _sample(self, logp: np.ndarray, decoded: List[int], cfg: SamplingConfig,
                ignore_eos: bool, strict_min_len: bool) -> int:
        scores = logp.copy()
        if ignore_eos:
            # Upstream masks exactly one id here (``speech_token_size``); with
            # strict_min_len we mask every stop id, which is what the comment in
            # the reference intends but not what it does.
            if strict_min_len:
                for t in self.stop_token_ids:
                    scores[t] = -np.inf
            else:
                scores[self.speech_token_size] = -np.inf
        probs = ops.softmax(scores)
        order = np.argsort(-probs, kind="stable")
        top = nucleus_sampling(probs[order], order, cfg.rng, cfg.top_p, cfg.top_k)
        # Repetition-aware sampling: if the pick already dominates the recent
        # window, drop it and resample from the full distribution.
        window = decoded[-cfg.win_size:]
        rep = sum(1 for t in window if t == top)
        if rep >= cfg.win_size * cfg.tau_r:
            scores[top] = -np.inf
            p = ops.softmax(scores).astype(np.float64)
            p /= p.sum()
            top = int(cfg.rng.choice(p.shape[0], p=p))
        return top
