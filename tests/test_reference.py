"""Convert the committed checkpoint and check the runtime against frozen output.

``tests/fixtures/checkpoint`` is a small model saved exactly the way the release
saves one — a CosyVoice3LM ``llm.pt``, a ``flow.pt``, and a ``hift.pt`` whose keys
carry the ``generator.`` prefix and weight-norm parametrisations.
``reference_io.npz`` holds what the upstream PyTorch implementation produced for a
fixed set of inputs.  Neither PyTorch nor the upstream package is needed to run
this: the ground truth is frozen, and ``tools/make_reference_fixture.py`` records
how it was produced.
"""

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import CHECKPOINT, rel_err  # noqa: E402

from cv3cpu.convert import convert  # noqa: E402
from cv3cpu.io.weights import WeightFile  # noqa: E402
from cv3cpu.io.torch_pickle import load_pt  # noqa: E402
from cv3cpu.torch_rng import TorchMT19937  # noqa: E402


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("weights") / "ref.safetensors")
    convert(CHECKPOINT, out, profile="f32", verbose=False)
    f = WeightFile(out)
    yield f
    f.close()


# --------------------------------------------------------------------------
# the converter
# --------------------------------------------------------------------------

def test_metadata_is_inferred_from_the_checkpoint(converted, fixture_config):
    m = converted.meta
    assert m["qwen2"]["num_hidden_layers"] == fixture_config["layers"]
    assert m["qwen2"]["num_attention_heads"] == fixture_config["heads"]
    assert m["qwen2"]["num_key_value_heads"] == fixture_config["kv"]
    assert m["qwen2"]["intermediate_size"] == fixture_config["inter"]
    assert m["qwen2"]["hidden_size"] == fixture_config["hid"]
    assert m["speech_token_size"] == fixture_config["speech_tokens"]
    assert m["dit"]["depth"] == fixture_config["depth"]
    assert m["dit"]["dim"] == fixture_config["dim"]
    assert m["dit"]["heads"] == fixture_config["nhead"]


def test_unused_tensors_are_dropped(converted):
    """The LM head is replaced by llm_decoder, so it must not be carried along."""
    keys = list(converted.keys())
    assert not any("lm_head" in k for k in keys)
    assert not any("criterion_ce" in k for k in keys)
    assert not any(k.endswith("weight_g") or k.endswith("weight_v") for k in keys)
    assert not any("parametrizations" in k for k in keys)
    assert "llm.speech_embedding" in keys
    assert "llm.llm_decoder.weight" in keys
    assert "flow.input_embedding" in keys


def _target_name(raw_key: str) -> str:
    """Where a raw ``hift.pt`` weight-norm key ends up in the ``.safetensors``."""
    base = raw_key.split(".parametrizations.weight.original0")[0].split(".weight_g")[0]
    base = base[len("generator."):] if base.startswith("generator.") else base
    # condnet is an nn.Sequential with ELUs interleaved; convert densifies it
    m = re.match(r"^(f0_predictor\.condnet\.)(\d+)$", base)
    if m:
        base = f"{m.group(1)}{int(m.group(2)) // 2}"
    return "hift." + base + ".weight"


def test_weight_norm_is_folded(converted):
    """w = g * v / ||v||, recomputed here straight from the raw checkpoint."""
    raw = load_pt(os.path.join(CHECKPOINT, "hift.pt"))
    pairs = [(k, k.replace("original0", "original1")) for k in raw
             if k.endswith(".parametrizations.weight.original0")]
    pairs += [(k, k[: -len("_g")] + "_v") for k in raw if k.endswith(".weight_g")]
    assert pairs, "the fixture is supposed to contain weight-norm parametrisations"
    for gk, vk in pairs:
        g = np.asarray(raw[gk], dtype=np.float64)
        v = np.asarray(raw[vk], dtype=np.float64)
        norm = np.sqrt((v * v).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
        expected = g / norm * v
        got = converted.get_f32(_target_name(gk))
        assert got.shape == expected.shape, gk
        assert rel_err(got, expected) < 1e-6, gk


# --------------------------------------------------------------------------
# the models, against frozen upstream output
# --------------------------------------------------------------------------

def test_flow_matches_frozen_reference(converted, reference_io):
    from cv3cpu.models.flow import Flow

    got = Flow(converted, "flow.").inference(
        reference_io["token"], reference_io["prompt_token"],
        reference_io["prompt_feat"], reference_io["embedding"])
    ref = reference_io["flow_mel"]
    assert got.shape == ref.shape
    assert rel_err(got, ref) < 5e-4


def test_vocoder_matches_frozen_reference(converted, reference_io, fixture_config):
    from cv3cpu.models.hift import HiFTGenerator

    hift = HiFTGenerator(converted, "hift.")
    n = 480 * fixture_config["n_mel"]
    hift.set_source_noise(TorchMT19937(0).random_float32(n * 9).reshape(n, 9))
    got = hift.inference(reference_io["mel"], chunk=None)
    ref = reference_io["hift_wav"]
    assert got.shape == ref.shape
    assert rel_err(got, ref) < 2e-3


def test_vocoder_chunking_matches_full_decode(converted, reference_io, fixture_config):
    """Time-chunked decoding reproduces whole-utterance decoding, not an approximation."""
    from cv3cpu.models.hift import HiFTGenerator

    hift = HiFTGenerator(converted, "hift.")
    n = 480 * fixture_config["n_mel"]
    hift.set_source_noise(TorchMT19937(0).random_float32(n * 9).reshape(n, 9))
    full = hift.inference(reference_io["mel"], chunk=None)
    chunked = hift.inference(reference_io["mel"], chunk=8, context=64)
    # Same arithmetic, only regrouped, so what is left is float32 rounding from
    # BLAS blocking the work differently.  A boundary bug -- too little left
    # context, or forgetting conv_pre's 4-frame lookahead -- shows up at ~1e-1.
    assert rel_err(chunked, full) < 1e-5


def test_llm_logits_match_frozen_reference(converted, reference_io):
    """Every decode step, not just the prefill — this exercises the KV cache."""
    from cv3cpu.models.llm import CosyVoice3LM

    lm = CosyVoice3LM(converted, "llm.")
    emb = np.concatenate([
        lm._speech_emb([lm.sos]),
        lm._text_emb(reference_io["text"]),
        lm._speech_emb([lm.task_id]),
        lm._speech_emb(reference_io["prompt_speech"]),
    ], axis=0)
    cache = lm.body.new_cache()
    forced = list(reference_io["forced"])
    step_in = emb
    for i, ref in enumerate(reference_io["llm_logits"]):
        got = lm._head(lm.body.forward(step_in, cache)[-1:])[0]
        assert rel_err(got, ref) < 1e-3, f"decode step {i}"
        if i < len(forced):
            step_in = lm._speech_emb([int(forced[i])])


def test_incremental_decoding_equals_one_full_pass(converted):
    """A cached step-by-step run must equal a single whole-sequence forward."""
    from cv3cpu.models.llm import CosyVoice3LM

    lm = CosyVoice3LM(converted, "llm.")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(17, lm.cfg.hidden_size)).astype(np.float32) * 0.2
    full = lm.body.forward(x)
    cache = lm.body.new_cache()
    parts = [lm.body.forward(x[:5], cache)]
    parts += [lm.body.forward(x[i:i + 1], cache) for i in range(5, 17)]
    assert rel_err(np.concatenate(parts), full) < 5e-5


# --------------------------------------------------------------------------
# quantisation, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile,tol", [("quality", 5e-3), ("balanced", 2e-2),
                                         ("tiny", 2e-2)])
def test_quantised_profiles_track_the_reference(tmp_path, reference_io, profile, tol,
                                                fixture_config):
    from cv3cpu.models.flow import Flow
    from cv3cpu.models.hift import HiFTGenerator

    out = str(tmp_path / f"{profile}.safetensors")
    convert(CHECKPOINT, out, profile=profile, verbose=False)
    f = WeightFile(out)
    mel = Flow(f, "flow.").inference(
        reference_io["token"], reference_io["prompt_token"],
        reference_io["prompt_feat"], reference_io["embedding"])
    assert rel_err(mel, reference_io["flow_mel"]) < tol

    hift = HiFTGenerator(f, "hift.")
    n = 480 * fixture_config["n_mel"]
    hift.set_source_noise(TorchMT19937(0).random_float32(n * 9).reshape(n, 9))
    wav = hift.inference(reference_io["mel"], chunk=None)
    assert rel_err(wav, reference_io["hift_wav"]) < tol
    f.close()
