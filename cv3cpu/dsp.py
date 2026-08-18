"""Signal processing for the CosyVoice 3 front end, in NumPy.

Everything here is a re-implementation of a specific reference:

* :func:`mel_spectrogram` — ``matcha.utils.audio.mel_spectrogram`` (the flow's
  ``feat_extractor``), including its librosa/slaney mel basis.
* :func:`whisper_log_mel` — ``whisper.log_mel_spectrogram``, the input to the
  speech tokeniser.
* :func:`kaldi_fbank` — ``torchaudio.compliance.kaldi.fbank``, the input to the
  CAM++ speaker encoder.
"""

from __future__ import annotations

import math
import wave
from typing import Optional, Tuple

import numpy as np

from . import ops


# --------------------------------------------------------------------------
# mel filterbank (librosa-compatible: htk=False, norm="slaney")
# --------------------------------------------------------------------------

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    big = f >= min_log_hz
    mels = np.where(big, min_log_mel + np.log(np.maximum(f, 1e-12) / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    big = m >= min_log_mel
    return np.where(big, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filters(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0,
                fmax: Optional[float] = None) -> np.ndarray:
    """``librosa.filters.mel(..., htk=False, norm='slaney')``."""
    if fmax is None:
        fmax = sr / 2.0
    fft_freqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2, dtype=np.float64)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    mel_f = _mel_to_hz(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


# --------------------------------------------------------------------------
# spectrograms
# --------------------------------------------------------------------------

def mel_spectrogram(y: np.ndarray, n_fft: int = 1920, num_mels: int = 80,
                    sampling_rate: int = 24000, hop_size: int = 480,
                    win_size: int = 1920, fmin: float = 0.0,
                    fmax: Optional[float] = None, center: bool = False) -> np.ndarray:
    """``(T,)`` waveform -> ``(num_mels, frames)`` log-mel, as the flow expects."""
    y = np.asarray(y, dtype=np.float32)
    pad = int((n_fft - hop_size) / 2)
    y = np.pad(y, (pad, pad), mode="reflect")
    win = ops.hann_window(win_size)
    if win_size < n_fft:
        win = np.pad(win, (0, n_fft - win_size))
    spec = ops.stft(y, n_fft, hop_size, win, center=center)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(np.float32)
    basis = mel_filters(sampling_rate, n_fft, num_mels, fmin, fmax)
    return np.log(np.maximum(basis @ mag, 1e-5)).astype(np.float32)


def whisper_log_mel(y: np.ndarray, n_mels: int = 128, sampling_rate: int = 16000,
                    n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """``whisper.log_mel_spectrogram`` -> ``(n_mels, frames)``."""
    y = np.asarray(y, dtype=np.float32)
    win = ops.hann_window(n_fft)
    spec = ops.stft(y, n_fft, hop, win, center=True)[:, :-1]
    mags = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    basis = mel_filters(sampling_rate, n_fft, n_mels, 0.0, None)
    mel = basis @ mags
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def _povey_window(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))) ** 0.85


def kaldi_fbank(y: np.ndarray, num_mel_bins: int = 80, sample_frequency: int = 16000,
                frame_length_ms: float = 25.0, frame_shift_ms: float = 10.0,
                low_freq: float = 20.0, high_freq: float = 0.0,
                preemphasis: float = 0.97, remove_dc_offset: bool = True) -> np.ndarray:
    """``torchaudio.compliance.kaldi.fbank(..., dither=0)`` -> ``(frames, num_mel_bins)``."""
    y = np.asarray(y, dtype=np.float32)
    win_len = int(sample_frequency * frame_length_ms / 1000)
    shift = int(sample_frequency * frame_shift_ms / 1000)
    if len(y) < win_len:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    n_frames = 1 + (len(y) - win_len) // shift          # snip_edges=True
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, win_len),
        strides=(y.strides[0] * shift, y.strides[0])).astype(np.float32)
    frames = frames.copy()
    if remove_dc_offset:
        frames -= frames.mean(axis=1, keepdims=True)
    if preemphasis:
        prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - preemphasis * prev
    frames *= _povey_window(win_len)
    n_fft = 1
    while n_fft < win_len:
        n_fft *= 2                                      # round_to_power_of_two=True
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    hi = high_freq if high_freq > 0 else sample_frequency / 2.0 + high_freq
    basis = _kaldi_mel_banks(num_mel_bins, n_fft, sample_frequency, low_freq, hi)
    mel = power @ basis.T
    return np.log(np.maximum(mel, np.finfo(np.float32).eps)).astype(np.float32)


def _kaldi_mel_banks(num_bins: int, n_fft: int, sr: float, low_freq: float,
                     high_freq: float) -> np.ndarray:
    """Kaldi's triangular mel banks (HTK mel scale, no area normalisation)."""
    def hz2mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    num_bins_fft = n_fft // 2
    fft_bin_width = sr / n_fft
    mel_low, mel_high = hz2mel(low_freq), hz2mel(high_freq)
    delta = (mel_high - mel_low) / (num_bins + 1)
    bins = np.zeros((num_bins, num_bins_fft + 1), dtype=np.float32)
    freqs = fft_bin_width * np.arange(num_bins_fft)
    mel = hz2mel(freqs)
    for b in range(num_bins):
        left, center, right = mel_low + b * delta, mel_low + (b + 1) * delta, mel_low + (b + 2) * delta
        up = (mel - left) / (center - left)
        down = (right - mel) / (right - center)
        w = np.maximum(0.0, np.minimum(up, down))
        w[(freqs < low_freq) | (freqs > high_freq)] = 0.0
        bins[b, :num_bins_fft] = w
    return bins


# --------------------------------------------------------------------------
# wav io and resampling
# --------------------------------------------------------------------------

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a PCM wav as float32 in [-1, 1], averaging channels down to mono."""
    with wave.open(path, "rb") as wf:
        ch, width, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width {width}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path: str, y: np.ndarray, sr: int) -> None:
    pcm = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def resample(y: np.ndarray, sr_in: int, sr_out: int, zeros: int = 24) -> np.ndarray:
    """Windowed-sinc (Kaiser) resampler; good enough for prompt conditioning."""
    if sr_in == sr_out:
        return np.asarray(y, dtype=np.float32)
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    cutoff = 0.5 * min(1.0 / up, 1.0 / down)
    half = int(zeros * max(up, down))
    n = np.arange(-half, half + 1, dtype=np.float64)
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    h *= np.kaiser(len(n), 8.6)
    h = (h / h.sum() * up).astype(np.float32)
    ups = np.zeros(len(y) * up, dtype=np.float32)
    ups[::up] = np.asarray(y, dtype=np.float32)
    filtered = np.convolve(ups, h, mode="same")
    return filtered[::down].astype(np.float32)
