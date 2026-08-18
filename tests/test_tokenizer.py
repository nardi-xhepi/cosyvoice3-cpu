"""Tokenizer tests: frozen pre-tokenizer splits plus an independent BPE reference.

The Qwen2 pre-tokenizer is a Unicode-property regex that :mod:`re` cannot express,
so :func:`cv3cpu.tokenizer.pretokenize` reimplements its alternatives by hand.
The expected splits in ``tests/reference_values.json`` were produced by the real
pattern under the ``regex`` engine, which keeps that check honest without making
``regex`` a dependency.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cv3cpu.tokenizer import (QwenTokenizer, bytes_to_unicode,  # noqa: E402
                              pretokenize, unicode_to_bytes)


def test_pretokenize_matches_the_frozen_regex_splits(reference_values):
    for text, expected in reference_values["pretokenize"].items():
        assert pretokenize(text) == expected, repr(text)


def test_pretokenize_is_lossless(reference_values):
    for text in reference_values["pretokenize"]:
        assert "".join(pretokenize(text)) == text


def test_byte_table_is_a_bijection():
    b2u, u2b = bytes_to_unicode(), unicode_to_bytes()
    assert len(b2u) == 256 and len(u2b) == 256
    assert all(u2b[b2u[i]] == i for i in range(256))


# --------------------------------------------------------------------------
# BPE
# --------------------------------------------------------------------------

def _reference_bpe(token, ranks):
    """The textbook merge loop: repeatedly apply the lowest-ranked adjacent pair."""
    word = list(token)
    while len(word) > 1:
        best, rank = None, None
        for i in range(len(word) - 1):
            r = ranks.get((word[i], word[i + 1]))
            if r is not None and (rank is None or r < rank):
                best, rank = i, r
        if best is None:
            break
        word[best: best + 2] = [word[best] + word[best + 1]]
    return word


def _build(tmpdir, merges, extra_tokens=()):
    b2u = bytes_to_unicode()
    vocab = {b2u[i]: i for i in range(256)}
    for a, b in merges:                       # merged pieces get the next free ids
        vocab.setdefault(a + b, len(vocab))
    blob = {"model": {"vocab": vocab, "merges": [f"{a} {b}" for a, b in merges]},
            "added_tokens": [{"id": len(vocab), "content": "<|endoftext|>"}]}
    path = os.path.join(tmpdir, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    return QwenTokenizer.from_dir(tmpdir, extra_special_tokens=list(extra_tokens)), vocab


MERGES = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"), ("Ġ", "w"),
          ("Ġw", "o"), ("l", "d"), ("Ġwo", "r")]


def test_bpe_matches_the_reference_loop(tmp_path):
    tok, vocab = _build(str(tmp_path), MERGES)
    ranks = {pair: i for i, pair in enumerate(MERGES)}
    b2u = bytes_to_unicode()
    rng = np.random.default_rng(0)
    alphabet = "helowrd "
    for _ in range(200):
        text = "".join(rng.choice(list(alphabet), size=int(rng.integers(1, 12))))
        expected = []
        for piece in pretokenize(text):
            mapped = "".join(b2u[b] for b in piece.encode("utf-8"))
            expected += [vocab[s] for s in _reference_bpe(mapped, ranks)]
        assert tok.encode(text) == expected, repr(text)


def test_bpe_prefers_the_earliest_merge(tmp_path):
    tok, vocab = _build(str(tmp_path), MERGES)
    assert tok.encode("hello") == [vocab["hello"]]
    assert tok.encode("hell") == [vocab["hell"]]
    assert tok.encode("held") == [vocab["hel"], vocab[bytes_to_unicode()[ord("d")]]]


def test_every_byte_survives_a_roundtrip(tmp_path):
    tok, _ = _build(str(tmp_path), MERGES)
    for text in ["Hello world", "你好，世界", "emoji 🙂", "tab\tnewline\n", "  ",
                 "Übermäßig", "русский", "混合 mixed 123"]:
        assert tok.decode(tok.encode(text)) == text


def test_special_tokens_are_matched_before_bpe(tmp_path):
    tok, vocab = _build(str(tmp_path), MERGES, extra_tokens=["<|endofprompt|>", "[breath]"])
    ids = tok.encode("hello<|endofprompt|>world[breath]")
    assert tok.added["<|endofprompt|>"] in ids
    assert tok.added["[breath]"] in ids
    assert ids[0] == vocab["hello"]
    # the special token must not be split into its characters
    assert len(ids) == 1 + 1 + len(tok.encode("world")) + 1


def test_longest_special_token_wins(tmp_path):
    tok, _ = _build(str(tmp_path), MERGES, extra_tokens=["<|a|>", "<|a|><|b|>"])
    assert tok.encode("<|a|><|b|>") == [tok.added["<|a|><|b|>"]]


def test_added_token_ids_are_sequential_past_the_vocab(tmp_path):
    tok, vocab = _build(str(tmp_path), MERGES, extra_tokens=["<|x|>", "<|y|>"])
    base_max = max(vocab.values()) + 1        # <|endoftext|> already took the next id
    assert tok.added["<|x|>"] == base_max + 1
    assert tok.added["<|y|>"] == base_max + 2
    # a token that already exists keeps its id
    again, _ = _build(str(tmp_path), MERGES, extra_tokens=["<|endoftext|>", "<|x|>"])
    assert again.added["<|endoftext|>"] == tok.added["<|endoftext|>"]
    assert again.added["<|x|>"] == base_max + 1


def test_decode_can_keep_special_tokens(tmp_path):
    tok, _ = _build(str(tmp_path), MERGES, extra_tokens=["<|endofprompt|>"])
    ids = tok.encode("hello<|endofprompt|>world")
    assert tok.decode(ids) == "helloworld"
    assert tok.decode(ids, skip_special_tokens=False) == "hello<|endofprompt|>world"


def test_unknown_piece_is_reported(tmp_path):
    """A vocabulary that cannot cover a byte must fail loudly, not silently."""
    path = str(tmp_path / "broken")
    os.makedirs(path)
    with open(os.path.join(path, "tokenizer.json"), "w") as fh:
        json.dump({"model": {"vocab": {"a": 0}, "merges": []}, "added_tokens": []}, fh)
    tok = QwenTokenizer.from_dir(path, extra_special_tokens=[])
    with pytest.raises(KeyError, match="out-of-vocabulary"):
        tok.encode("zzz")
