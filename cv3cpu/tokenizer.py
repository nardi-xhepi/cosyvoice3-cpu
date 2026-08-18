"""Qwen2 byte-level BPE tokenizer in pure Python.

Reproduces ``AutoTokenizer.from_pretrained(<qwen dir>)`` followed by
``add_special_tokens(...)``, which is what ``CosyVoice3Tokenizer`` does — no
``tokenizers``/``transformers``/``regex`` dependency.

The Qwen2 pre-tokenizer pattern is

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}
    | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

which needs Unicode property classes that :mod:`re` does not have.  Rather than
depend on ``regex``, :func:`pretokenize` walks the string and applies the same
alternatives in the same order, with the same greedy/backtracking behaviour.
"""

from __future__ import annotations

import json
import os
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .cv3_special_tokens import ADDITIONAL_SPECIAL_TOKENS, EOS_TOKEN


# --------------------------------------------------------------------------
# byte <-> unicode table (GPT-2)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def bytes_to_unicode() -> Dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> Dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


# --------------------------------------------------------------------------
# pre-tokenizer
# --------------------------------------------------------------------------

_CONTRACTIONS = ("re", "ve", "ll", "s", "t", "m", "d")  # longest-first per prefix


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "L"


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "N"


def _is_space(ch: str) -> bool:
    # \s in Oniguruma/Rust regex: ASCII + Unicode whitespace
    return ch.isspace()


def pretokenize(text: str) -> List[str]:
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]

        # (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == "'":
            hit = None
            for suf in _CONTRACTIONS:
                if text[i + 1: i + 1 + len(suf)].lower() == suf:
                    hit = suf
                    break
            if hit is not None:
                out.append(text[i: i + 1 + len(hit)])
                i += 1 + len(hit)
                continue

        # [^\r\n\p{L}\p{N}]?\p{L}+
        j = i
        if c not in "\r\n" and not _is_letter(c) and not _is_number(c):
            j = i + 1
        if j < n and _is_letter(text[j]):
            k = j
            while k < n and _is_letter(text[k]):
                k += 1
            out.append(text[i:k])
            i = k
            continue

        # \p{N}   (Qwen2 splits digits one at a time)
        if _is_number(c):
            out.append(c)
            i += 1
            continue

        # ?[^\s\p{L}\p{N}]+[\r\n]*
        j = i + 1 if c == " " else i
        if j < n and not _is_space(text[j]) and not _is_letter(text[j]) and not _is_number(text[j]):
            k = j
            while k < n and not _is_space(text[k]) and not _is_letter(text[k]) \
                    and not _is_number(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue

        # \s*[\r\n]+
        j = i
        while j < n and _is_space(text[j]) and text[j] not in "\r\n":
            j += 1
        if j < n and text[j] in "\r\n":
            k = j
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue

        # \s+(?!\S) then \s+
        if _is_space(c):
            k = i
            while k < n and _is_space(text[k]):
                k += 1
            if k < n and k - 1 > i:
                k -= 1  # the lookahead forces us to leave the last space behind
            out.append(text[i:k])
            i = k
            continue

        out.append(c)   # unreachable for well-formed input, but never loop forever
        i += 1
    return out


# --------------------------------------------------------------------------
# BPE
# --------------------------------------------------------------------------

class QwenTokenizer:
    def __init__(self, vocab: Dict[str, int], merges: Sequence[Tuple[str, str]],
                 added_tokens: Optional[Dict[str, int]] = None):
        self.vocab = dict(vocab)
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.added: Dict[str, int] = dict(added_tokens or {})
        self._cache: Dict[str, List[int]] = {}
        self._b2u = bytes_to_unicode()
        self._decoder = {i: t for t, i in self.vocab.items()}
        self._decoder.update({i: t for t, i in self.added.items()})
        self._u2b = unicode_to_bytes()

    # -- construction --------------------------------------------------
    @classmethod
    def from_dir(cls, path: str, extra_special_tokens: Optional[Iterable[str]] = None
                 ) -> "QwenTokenizer":
        tok_json = os.path.join(path, "tokenizer.json")
        if os.path.exists(tok_json):
            with open(tok_json, encoding="utf-8") as fh:
                blob = json.load(fh)
            vocab = blob["model"]["vocab"]
            raw_merges = blob["model"]["merges"]
            merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
                      for m in raw_merges]
            added = {t["content"]: t["id"] for t in blob.get("added_tokens", [])}
        else:
            with open(os.path.join(path, "vocab.json"), encoding="utf-8") as fh:
                vocab = json.load(fh)
            merges = []
            with open(os.path.join(path, "merges.txt"), encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#version"):
                        continue
                    a, b = line.split(" ")
                    merges.append((a, b))
            added = {}
            at = os.path.join(path, "added_tokens.json")
            if os.path.exists(at):
                with open(at, encoding="utf-8") as fh:
                    added = json.load(fh)
        tok = cls(vocab, merges, added)
        tok.add_special_tokens(list(extra_special_tokens)
                               if extra_special_tokens is not None
                               else [EOS_TOKEN] + ADDITIONAL_SPECIAL_TOKENS)
        return tok

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        """Mirror HF: unknown tokens get consecutive ids starting past the vocab."""
        next_id = max(list(self.vocab.values()) + list(self.added.values()) or [-1]) + 1
        for t in tokens:
            if t in self.added or t in self.vocab:
                continue
            self.added[t] = next_id
            self._decoder[next_id] = t
            next_id += 1

    def __len__(self) -> int:
        return max(list(self.vocab.values()) + list(self.added.values())) + 1

    # -- encode/decode --------------------------------------------------
    def _bpe(self, token: str) -> List[str]:
        hit = self._cache.get(token)
        if hit is not None:
            return hit  # type: ignore[return-value]
        word = list(token)
        while len(word) > 1:
            best, best_rank = None, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            word[best: best + 2] = [word[best] + word[best + 1]]
        self._cache[token] = word  # type: ignore[assignment]
        return word

    def _encode_plain(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in pretokenize(text):
            mapped = "".join(self._b2u[b] for b in piece.encode("utf-8"))
            for sub in self._bpe(mapped):
                if sub not in self.vocab:
                    raise KeyError(f"BPE produced an out-of-vocabulary piece {sub!r}")
                ids.append(self.vocab[sub])
        return ids

    def encode(self, text: str) -> List[int]:
        """Encode, splitting on added/special tokens first (``allowed_special='all'``)."""
        if not self.added:
            return self._encode_plain(text)
        specials = sorted(self.added, key=len, reverse=True)
        out: List[int] = []
        i = 0
        n = len(text)
        buf_start = 0
        while i < n:
            for s in specials:
                if text.startswith(s, i):
                    if i > buf_start:
                        out.extend(self._encode_plain(text[buf_start:i]))
                    out.append(self.added[s])
                    i += len(s)
                    buf_start = i
                    break
            else:
                i += 1
        if n > buf_start:
            out.extend(self._encode_plain(text[buf_start:]))
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special_ids = set(self.added.values())
        buf = bytearray()
        out: List[str] = []
        for i in ids:
            tok = self._decoder.get(int(i))
            if tok is None:
                continue
            if int(i) in special_ids:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special_tokens:
                    out.append(tok)
                continue
            buf.extend(self._u2b[c] for c in tok)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)
