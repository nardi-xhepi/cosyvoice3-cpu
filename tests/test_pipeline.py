"""End-to-end behaviour of the staged pipeline, on a small random model."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from cv3cpu.models.llm import SamplingConfig  # noqa: E402
from cv3cpu.pipeline import CosyVoice3CPU  # noqa: E402
from cv3cpu.voice import VoicePack  # noqa: E402

import make_dummy_model  # noqa: E402

SR = 24000


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("m") / "small.safetensors")
    make_dummy_model.build(path, "balanced", small=True)
    return path


@pytest.fixture
def voice():
    rng = np.random.default_rng(0)
    n = 20
    return VoicePack(speech_token=rng.integers(0, 200, size=n),
                     mel=rng.normal(size=(2 * n, 80)).astype(np.float32),
                     embedding=rng.normal(size=192).astype(np.float32),
                     prompt_text="a reference sentence.", name="test")


EOP = make_dummy_model.SMALL_QWEN["vocab_size"] - 1   # the model's <|endofprompt|>


def _ids(rng, n):
    """Text ids that include the mandatory <|endofprompt|> marker."""
    return [EOP] + rng.integers(0, 2000, size=n).tolist()


def test_end_to_end_produces_audio(model, voice):
    rng = np.random.default_rng(1)
    tts = CosyVoice3CPU(model, seed=7, track_memory=True)
    sampling = SamplingConfig(seed=7, max_token_text_ratio=3.0, min_token_text_ratio=1.0,
                              strict_min_len=True)
    out = tts.synthesize_ids([_ids(rng, 6)], _ids(rng, 4), voice, sampling=sampling)
    tts.close()

    assert out.sample_rate == SR
    assert out.audio.dtype == np.float32
    assert out.audio.ndim == 1
    assert len(out.audio) > 0
    assert np.isfinite(out.audio).all()
    assert np.abs(out.audio).max() <= 0.99 + 1e-6
    # every speech token becomes two mel frames, every mel frame 480 samples
    assert len(out.audio) == 2 * 480 * len(out.speech_tokens[0])
    assert "llm released" in out.memory_report


def test_multiple_segments_concatenate(model, voice):
    rng = np.random.default_rng(2)
    tts = CosyVoice3CPU(model, seed=3)
    sampling = SamplingConfig(seed=3, max_token_text_ratio=3.0, min_token_text_ratio=1.0,
                              strict_min_len=True)
    segs = [_ids(rng, 5), _ids(rng, 5)]
    out = tts.synthesize_ids(segs, _ids(rng, 3), voice, sampling=sampling)
    tts.close()
    assert len(out.speech_tokens) == 2
    total = sum(len(t) for t in out.speech_tokens)
    assert len(out.audio) == 2 * 480 * total


def test_seed_is_reproducible(model, voice):
    rng = np.random.default_rng(4)
    ids, prompt = _ids(rng, 6), _ids(rng, 3)
    outs = []
    for _ in range(2):
        tts = CosyVoice3CPU(model, seed=11)
        outs.append(tts.synthesize_ids(
            [ids], prompt, voice,
            sampling=SamplingConfig(seed=11, max_token_text_ratio=3.0,
                                    min_token_text_ratio=1.0, strict_min_len=True)))
        tts.close()
    assert outs[0].speech_tokens == outs[1].speech_tokens
    assert np.array_equal(outs[0].audio, outs[1].audio)


def test_stage_release_does_not_change_output(model, voice):
    rng = np.random.default_rng(5)
    ids, prompt = _ids(rng, 6), _ids(rng, 3)
    res = []
    for release in (True, False):
        tts = CosyVoice3CPU(model, seed=13, release_stages=release)
        res.append(tts.synthesize_ids(
            [ids], prompt, voice,
            sampling=SamplingConfig(seed=13, max_token_text_ratio=3.0,
                                    min_token_text_ratio=1.0, strict_min_len=True)))
        tts.close()
    assert np.array_equal(res[0].audio, res[1].audio)


def test_endofprompt_is_required(model, voice):
    tts = CosyVoice3CPU(model, seed=1)
    with pytest.raises(ValueError, match="endofprompt"):
        list(tts.llm.generate([1, 2, 3], prompt_text_tokens=[4, 5]))
    tts.close()


def test_voice_pack_roundtrip(tmp_path, voice):
    p = str(tmp_path / "v.npz")
    voice.save(p)
    back = VoicePack.load(p)
    assert np.array_equal(back.speech_token, voice.speech_token)
    assert np.allclose(back.mel, voice.mel)
    assert back.prompt_text == voice.prompt_text
    assert back.sample_rate == voice.sample_rate


def test_voice_pack_trims_to_token_grid():
    rng = np.random.default_rng(6)
    v = VoicePack(speech_token=rng.integers(0, 100, size=30),
                  mel=rng.normal(size=(41, 80)).astype(np.float32),
                  embedding=np.zeros(192, np.float32))
    assert v.mel.shape[0] == 2 * len(v.speech_token)
    assert len(v.speech_token) == 20


def test_enroll_reports_missing_encoders_clearly(tmp_path):
    from cv3cpu import dsp, enroll

    wav = str(tmp_path / "ref.wav")
    dsp.write_wav(wav, np.zeros(16000, np.float32), 16000)
    with pytest.raises(FileNotFoundError, match="speech_tokenizer"):
        enroll.enroll(wav, "hello", str(tmp_path))


def test_enroll_rejects_over_long_prompts(tmp_path):
    from cv3cpu import dsp, enroll

    wav = str(tmp_path / "long.wav")
    dsp.write_wav(wav, np.zeros(16000 * 31, np.float32), 16000)
    with pytest.raises(ValueError, match="30s"):
        enroll.enroll(wav, "hello", str(tmp_path))
