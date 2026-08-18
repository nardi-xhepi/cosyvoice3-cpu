"""End-to-end zero-shot TTS with a staged, flat memory profile.

The three networks are never hot at the same time.  Text -> speech tokens runs
first and then the LLM's ~0.5 GB of mapped weights are handed back to the kernel;
the flow decoder produces every mel and is released in turn; only then does the
vocoder touch its weights.  Peak RSS is therefore ``max(stage)`` rather than the
sum, which is what brings a ~1 B parameter model inside a 2 GB budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from . import dsp, text as textutil
from .io.weights import WeightFile
from .memory import Tracker
from .models.flow import Flow
from .models.hift import HiFTGenerator
from .models.llm import CosyVoice3LM, SamplingConfig
from .tokenizer import QwenTokenizer

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.<|endofprompt|>"


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    speech_tokens: List[List[int]] = field(default_factory=list)
    segments: List[str] = field(default_factory=list)
    memory_report: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


class CosyVoice3CPU:
    """PyTorch-free CosyVoice 3 inference over a ``.safetensors`` weight file."""

    def __init__(self, weights_path: str, tokenizer_dir: Optional[str] = None,
                 seed: Optional[int] = None, release_stages: bool = True,
                 track_memory: bool = False, seq_chunk: int = 512,
                 hift_chunk: Optional[int] = 250):
        self.w = WeightFile(weights_path)
        self.sample_rate = int(self.w.meta.get("sample_rate", 24000))
        self.release_stages = release_stages
        self.seq_chunk = seq_chunk
        self.hift_chunk = hift_chunk
        self.seed = seed
        self.tracker = Tracker(track_memory)
        self.tokenizer: Optional[QwenTokenizer] = None
        if tokenizer_dir:
            self.tokenizer = QwenTokenizer.from_dir(tokenizer_dir)
        self._llm: Optional[CosyVoice3LM] = None
        self._flow: Optional[Flow] = None
        self._hift: Optional[HiFTGenerator] = None

    # -- lazily built stages ---------------------------------------------
    @property
    def llm(self) -> CosyVoice3LM:
        if self._llm is None:
            self._llm = CosyVoice3LM(self.w, prefix="llm.")
        return self._llm

    @property
    def flow(self) -> Flow:
        if self._flow is None:
            self._flow = Flow(self.w, prefix="flow.")
        return self._flow

    @property
    def hift(self) -> HiFTGenerator:
        if self._hift is None:
            self._hift = HiFTGenerator(self.w, prefix="hift.")
        return self._hift

    def _drop(self, stage: str) -> None:
        if not self.release_stages:
            return
        setattr(self, f"_{stage}", None)
        self.w.evict(f"{stage}.")

    # -- text ------------------------------------------------------------
    def encode_text(self, s: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError(
                "no tokenizer loaded; construct with tokenizer_dir=<the checkpoint's "
                "CosyVoice-BlankEN directory> or pass pre-tokenised ids")
        return self.tokenizer.encode(s)

    def split_text(self, s: str, split: bool = True) -> List[str]:
        tok = self.tokenizer.encode if self.tokenizer is not None else None
        return textutil.normalize_and_split(s, tok, split=split)

    # -- synthesis --------------------------------------------------------
    def synthesize(self, text: str, voice, split: bool = True,
                   system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                   sampling: Optional[SamplingConfig] = None,
                   on_progress: Optional[Callable[[str, int, int], None]] = None
                   ) -> SynthesisResult:
        """Zero-shot synthesis of ``text`` in the voice of ``voice``."""
        segments = self.split_text(text, split=split)
        if not segments:
            return SynthesisResult(np.zeros(0, np.float32), self.sample_rate)
        prompt_ids = self.encode_text(system_prompt + (voice.prompt_text or ""))
        result = self.synthesize_ids([self.encode_text(s) for s in segments],
                                     prompt_ids, voice, sampling=sampling,
                                     on_progress=on_progress)
        result.segments = segments
        return result

    def synthesize_ids(self, segment_ids: Sequence[Sequence[int]],
                       prompt_ids: Sequence[int], voice,
                       sampling: Optional[SamplingConfig] = None,
                       on_progress: Optional[Callable[[str, int, int], None]] = None
                       ) -> SynthesisResult:
        """Same as :meth:`synthesize` but takes already-tokenised text.

        The three stages run strictly one after another so each one's weights can
        be handed back to the kernel before the next one faults its own in.
        """
        sampling = sampling or SamplingConfig(seed=self.seed)
        segments = list(segment_ids)
        if not segments:
            return SynthesisResult(np.zeros(0, np.float32), self.sample_rate)

        self.tracker.mark("start")
        token_lists: List[List[int]] = []
        for i, ids in enumerate(segments):
            toks = list(self.llm.generate(
                list(ids), prompt_text_tokens=list(prompt_ids),
                prompt_speech_tokens=voice.speech_token.tolist(),
                sampling=sampling))
            token_lists.append(toks)
            if on_progress:
                on_progress("llm", i + 1, len(segments))
        self.tracker.mark("llm")
        self._drop("llm")
        self.tracker.mark("llm released")

        mels: List[np.ndarray] = []
        for i, toks in enumerate(token_lists):
            if not toks:
                mels.append(np.zeros((80, 0), np.float32))
                continue
            mels.append(self.flow.inference(
                toks, voice.speech_token, voice.mel, voice.embedding,
                streaming=False, seq_chunk=self.seq_chunk))
            if on_progress:
                on_progress("flow", i + 1, len(token_lists))
        self.tracker.mark("flow")
        self._drop("flow")
        self.tracker.mark("flow released")

        waves: List[np.ndarray] = []
        for i, mel in enumerate(mels):
            if mel.shape[1] == 0:
                continue
            waves.append(self.hift.inference(mel, chunk=self.hift_chunk))
            if on_progress:
                on_progress("hift", i + 1, len(mels))
        self.tracker.mark("hift")
        self._drop("hift")
        self.tracker.mark("done")

        audio = np.concatenate(waves) if waves else np.zeros(0, np.float32)
        return SynthesisResult(audio.astype(np.float32), self.sample_rate,
                               speech_tokens=token_lists,
                               memory_report=self.tracker.report())

    def save(self, result: SynthesisResult, path: str) -> None:
        dsp.write_wav(path, result.audio, result.sample_rate)

    def close(self) -> None:
        self._llm = self._flow = self._hift = None
        self.w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
