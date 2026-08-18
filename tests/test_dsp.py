"""DSP tests against frozen reference values plus structural invariants.

``tests/reference_values.json`` holds exact samples of what librosa, torch.stft
and torchaudio.compliance.kaldi produce for fixed inputs.  Freezing them keeps
the checks meaningful without making librosa/torch a dependency of the suite.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cv3cpu import dsp  # noqa: E402


def check_against(golden, arr, tol=1e-5):
    arr = np.asarray(arr, dtype=np.float64)
    assert list(arr.shape) == golden["shape"]
    flat = arr.reshape(-1)
    scale = max(golden["absmax"], 1e-9)
    got = np.array([flat[i] for i in golden["idx"]])
    assert np.abs(got - np.array(golden["vals"])).max() / scale < tol
    assert abs(flat.sum() - golden["sum"]) / max(abs(golden["sum"]), 1.0) < tol


@pytest.fixture(scope="module")
def signals(reference_values):
    rng = np.random.default_rng(20240816)
    y24 = (rng.normal(size=24000) * 0.1).astype(np.float32)
    y16 = (rng.normal(size=16000) * 0.1).astype(np.float32)
    # guard against the fixture and the test drifting apart
    assert abs(float(y24.astype(np.float64).sum())
               - reference_values["input_checksums"]["y24"]) < 1e-6
    assert abs(float(y16.astype(np.float64).sum())
               - reference_values["input_checksums"]["y16"]) < 1e-6
    return y24, y16


# --------------------------------------------------------------------------
# mel filterbank
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,sr,n_fft,n_mels", [
    ("24k_1920_80", 24000, 1920, 80),
    ("16k_400_128", 16000, 400, 128),
])
def test_mel_filters_match_librosa(reference_values, key, sr, n_fft, n_mels):
    check_against(reference_values["mel_filters"][key],
                  dsp.mel_filters(sr, n_fft, n_mels, 0.0, None), tol=1e-6)


def test_mel_filters_are_well_formed():
    fb = dsp.mel_filters(24000, 1920, 80, 0.0, None)
    assert (fb >= 0).all()
    peaks = fb.argmax(axis=1)
    assert (np.diff(peaks) > 0).all(), "filter centres must increase with index"
    for i, row in enumerate(fb):         # each filter is one contiguous triangle
        nz = np.flatnonzero(row)
        assert np.array_equal(nz, np.arange(nz[0], nz[-1] + 1))
        assert np.all(np.diff(row[nz[0]: peaks[i] + 1]) >= -1e-9)
        assert np.all(np.diff(row[peaks[i]: nz[-1] + 1]) <= 1e-9)


def test_mel_scale_roundtrips():
    hz = np.array([0.0, 100.0, 999.0, 1000.0, 1001.0, 8000.0, 12000.0])
    assert np.abs(dsp._mel_to_hz(dsp._hz_to_mel(hz)) - hz).max() < 1e-6


# --------------------------------------------------------------------------
# spectrograms
# --------------------------------------------------------------------------

def test_mel_spectrogram_matches_matcha(reference_values, signals):
    y24, _ = signals
    check_against(reference_values["mel_spectrogram"], dsp.mel_spectrogram(y24))


def test_whisper_log_mel_matches_whisper(reference_values, signals):
    _, y16 = signals
    check_against(reference_values["whisper_log_mel"], dsp.whisper_log_mel(y16), tol=1e-4)


def test_kaldi_fbank_matches_torchaudio(reference_values, signals):
    _, y16 = signals
    check_against(reference_values["kaldi_fbank"], dsp.kaldi_fbank(y16), tol=1e-4)


def test_mel_spectrogram_locates_a_pure_tone():
    sr, f0 = 24000, 1000.0
    y = np.sin(2 * np.pi * f0 * np.arange(sr) / sr).astype(np.float32)
    mel = dsp.mel_spectrogram(y)
    fb = dsp.mel_filters(sr, 1920, 80, 0.0, None)
    expected_bin = int(np.argmax(fb[:, int(round(f0 / (sr / 1920)))]))
    assert abs(int(np.argmax(mel[:, mel.shape[1] // 2])) - expected_bin) <= 1


def test_whisper_log_mel_dynamic_range_is_clamped():
    y = (np.random.default_rng(1).normal(size=16000) * 0.1).astype(np.float32)
    m = dsp.whisper_log_mel(y)
    assert m.max() <= 1.0 + 1e-6
    assert m.min() >= m.max() - 2.0 - 1e-6      # 8 dB / 4, by construction


def test_kaldi_frame_count_and_windowing():
    """snip_edges=True: 25 ms window, 10 ms hop, no partial frames."""
    y = np.zeros(16000, dtype=np.float32)
    assert dsp.kaldi_fbank(y).shape == (1 + (16000 - 400) // 160, 80)
    assert dsp.kaldi_fbank(np.zeros(300, np.float32)).shape == (0, 80)


# --------------------------------------------------------------------------
# wav io and resampling
# --------------------------------------------------------------------------

def test_wav_roundtrip(tmp_path):
    y = (np.random.default_rng(3).normal(size=8000) * 0.2).astype(np.float32)
    p = str(tmp_path / "t.wav")
    dsp.write_wav(p, y, 24000)
    back, sr = dsp.read_wav(p)
    assert sr == 24000
    assert np.abs(back - y).max() < 1e-4


def test_wav_is_clipped_not_wrapped(tmp_path):
    p = str(tmp_path / "loud.wav")
    dsp.write_wav(p, np.array([2.0, -2.0, 0.0], np.float32), 16000)
    back, _ = dsp.read_wav(p)
    assert back[0] > 0.99 and back[1] < -0.99


def test_resample_preserves_a_sine():
    t = np.arange(24000) / 24000.0
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    down = dsp.resample(y, 24000, 16000)
    assert abs(len(down) - 16000) <= 2
    spec = np.abs(np.fft.rfft(down * np.hanning(len(down))))
    assert abs(np.argmax(spec) * 16000 / len(down) - 440) < 5


def test_resample_is_a_noop_at_the_same_rate():
    y = np.random.default_rng(4).normal(size=1000).astype(np.float32)
    assert np.array_equal(dsp.resample(y, 24000, 24000), y)
