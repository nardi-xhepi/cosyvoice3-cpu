"""Turn a reference wav into a :class:`~cv3cpu.voice.VoicePack`.

This is the one place that touches the two ONNX encoders shipped with the
checkpoint (``campplus.onnx`` and ``speech_tokenizer_v3.onnx``).  It needs
``onnxruntime`` — which is not PyTorch, and is not loaded during synthesis.  Run
it once per speaker; the resulting ``.npz`` is a few hundred kB.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import dsp
from .voice import VoicePack

MAX_PROMPT_SECONDS = 30.0


def _session(path: str):
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "enrolling a new voice needs onnxruntime (pip install onnxruntime) to run "
            "campplus.onnx / speech_tokenizer_v3.onnx.  Synthesis itself does not.") from exc
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(path, sess_options=opts,
                                        providers=["CPUExecutionProvider"])


def extract_speech_token(wav16: np.ndarray, tokenizer_onnx: str) -> np.ndarray:
    feat = dsp.whisper_log_mel(wav16, n_mels=128)[None]          # (1, 128, T)
    sess = _session(tokenizer_onnx)
    names = [i.name for i in sess.get_inputs()]
    out = sess.run(None, {names[0]: feat.astype(np.float32),
                          names[1]: np.array([feat.shape[2]], dtype=np.int32)})[0]
    return np.asarray(out, dtype=np.int32).reshape(-1)


def extract_speaker_embedding(wav16: np.ndarray, campplus_onnx: str) -> np.ndarray:
    feat = dsp.kaldi_fbank(wav16, num_mel_bins=80, sample_frequency=16000)
    feat = feat - feat.mean(axis=0, keepdims=True)
    sess = _session(campplus_onnx)
    out = sess.run(None, {sess.get_inputs()[0].name: feat[None].astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def enroll(wav_path: str, prompt_text: str, model_dir: str,
           name: Optional[str] = None,
           tokenizer_onnx: Optional[str] = None,
           campplus_onnx: Optional[str] = None) -> VoicePack:
    """Build a voice pack from a reference recording and its transcript."""
    audio, sr = dsp.read_wav(wav_path)
    if audio.size == 0:
        raise ValueError(f"{wav_path} is empty")
    if len(audio) / sr > MAX_PROMPT_SECONDS:
        raise ValueError(
            f"prompt audio is {len(audio) / sr:.1f}s; the speech tokeniser only "
            f"supports up to {MAX_PROMPT_SECONDS:.0f}s — trim the reference clip")

    tokenizer_onnx = tokenizer_onnx or _find(model_dir, ("speech_tokenizer_v3.onnx",
                                                         "speech_tokenizer_v2.onnx"))
    campplus_onnx = campplus_onnx or _find(model_dir, ("campplus.onnx",))

    wav16 = dsp.resample(audio, sr, 16000)
    wav24 = dsp.resample(audio, sr, 24000)

    token = extract_speech_token(wav16, tokenizer_onnx)
    embedding = extract_speaker_embedding(wav16, campplus_onnx)
    mel = dsp.mel_spectrogram(wav24).T                       # (frames, 80)

    return VoicePack(speech_token=token, mel=mel, embedding=embedding,
                     prompt_text=prompt_text, sample_rate=24000,
                     name=name or os.path.splitext(os.path.basename(wav_path))[0])


def _find(model_dir: str, candidates) -> str:
    for c in candidates:
        p = os.path.join(model_dir, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"none of {list(candidates)} found in {model_dir}; pass an explicit path")
