"""Voice packs: everything the runtime needs about a reference speaker.

Enrolling a speaker runs two ONNX encoders (CAM++ and the S3 speech tokeniser)
that ship with the checkpoint.  Those are only needed *once* per voice, so the
result is cached in a small ``.npz`` and the synthesis path stays pure NumPy —
and never pays their memory cost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np


@dataclass
class VoicePack:
    speech_token: np.ndarray      # (T,) int32 — FSQ tokens of the prompt audio
    mel: np.ndarray               # (2T, 80) float32 — prompt mel at 24 kHz
    embedding: np.ndarray         # (192,) float32 — CAM++ speaker vector
    prompt_text: str = ""
    sample_rate: int = 24000
    name: str = ""

    def __post_init__(self):
        self.speech_token = np.asarray(self.speech_token, dtype=np.int32).reshape(-1)
        self.mel = np.asarray(self.mel, dtype=np.float32)
        self.embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        n = min(self.mel.shape[0] // 2, self.speech_token.shape[0])
        # The flow decoder consumes exactly two mel frames per speech token.
        self.speech_token = self.speech_token[:n]
        self.mel = self.mel[: 2 * n]

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, speech_token=self.speech_token, mel=self.mel.astype(np.float32),
            embedding=self.embedding, prompt_text=np.array(self.prompt_text),
            sample_rate=np.array(self.sample_rate), name=np.array(self.name))

    @classmethod
    def load(cls, path: str) -> "VoicePack":
        with np.load(path, allow_pickle=False) as z:
            return cls(speech_token=z["speech_token"], mel=z["mel"],
                       embedding=z["embedding"],
                       prompt_text=str(z["prompt_text"]),
                       sample_rate=int(z["sample_rate"]),
                       name=str(z["name"]) if "name" in z else
                       os.path.splitext(os.path.basename(path))[0])

    def __repr__(self) -> str:
        return (f"VoicePack(name={self.name!r}, tokens={len(self.speech_token)}, "
                f"mel={self.mel.shape}, text={self.prompt_text[:40]!r})")
