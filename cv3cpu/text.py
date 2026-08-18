"""Text segmentation, ported from ``cosyvoice/utils/frontend_utils.py``.

CosyVoice 3 needs no separate text-normalisation frontend (it reads numbers and
symbols directly), so only the paragraph splitter and its helpers are needed —
and they are reimplemented without the ``regex`` dependency.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, List, Optional, Sequence

CHINESE_CHAR = re.compile(r"[一-鿿]+")

ZH_PUNC = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
EN_PUNC = [".", "?", "!", ";", ":"]


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR.search(text))


def is_only_punctuation(text: str) -> bool:
    """``^[\\p{P}\\p{S}]*$`` without the regex module."""
    return all(unicodedata.category(c)[0] in ("P", "S") for c in text)


def replace_blank(text: str) -> str:
    """Drop spaces between CJK characters, keep them between ASCII words."""
    out = []
    for i, c in enumerate(text):
        if c == " ":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev.isascii() and prev not in ("", " ") and nxt.isascii() and nxt not in ("", " "):
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def replace_corner_mark(text: str) -> str:
    return text.replace("²", "平方").replace("³", "立方")


def remove_bracket(text: str) -> str:
    for a, b in (("（", ""), ("）", ""), ("【", ""), ("】", ""), ("`", ""), ("——", " ")):
        text = text.replace(a, b)
    return text


def split_paragraph(text: str, tokenize: Optional[Callable[[str], Sequence[int]]] = None,
                    lang: str = "zh", token_max_n: int = 80, token_min_n: int = 60,
                    merge_len: int = 20, comma_split: bool = False) -> List[str]:
    if not text:
        return []

    def length(s: str) -> int:
        if lang == "zh" or tokenize is None:
            return len(s)
        return len(tokenize(s))

    punc = list(ZH_PUNC if lang == "zh" else EN_PUNC)
    if comma_split:
        punc.extend(["，", ","])
    if text[-1] not in punc:
        text += "。" if lang == "zh" else "."

    utts: List[str] = []
    st = 0
    i = 0
    while i < len(text):
        if text[i] in punc:
            if i > st:
                utts.append(text[st:i] + text[i])
            if i + 1 < len(text) and text[i + 1] in ('"', "”"):
                utts[-1] = utts[-1] + text[i + 1]
                st = i + 2
            else:
                st = i + 1
        i += 1

    out: List[str] = []
    cur = ""
    for utt in utts:
        if length(cur + utt) > token_max_n and length(cur) > token_min_n:
            out.append(cur)
            cur = ""
        cur += utt
    if cur:
        if length(cur) < merge_len and out:
            out[-1] += cur
        else:
            out.append(cur)
    return out


def normalize_and_split(text: str, tokenize: Optional[Callable[[str], Sequence[int]]] = None,
                        split: bool = True) -> List[str]:
    """Light-touch cleanup + sentence splitting.

    Anything containing SSML-ish ``<|...|>`` markers is passed through untouched,
    matching ``CosyVoiceFrontEnd.text_normalize``.
    """
    if "<|" in text and "|>" in text:
        return [text]
    text = text.strip()
    if not text:
        return []
    if not split:
        return [text]
    if contains_chinese(text):
        text = text.replace("\n", "")
        text = replace_blank(text)
        text = replace_corner_mark(text)
        text = text.replace(".", "。").replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r"[，,、]+$", "。", text)
        parts = split_paragraph(text, tokenize, "zh", token_max_n=80, token_min_n=60,
                                merge_len=20, comma_split=False)
    else:
        parts = split_paragraph(text, tokenize, "en", token_max_n=80, token_min_n=60,
                                merge_len=20, comma_split=False)
    return [p for p in parts if not is_only_punctuation(p)]
