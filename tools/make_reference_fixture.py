"""Regenerate ``tests/fixtures`` — MAINTAINER ONLY, never run by the test suite.

The test suite needs nothing but NumPy: it converts a small checkpoint that is
committed under ``tests/fixtures`` and compares the runtime's output against
frozen ground truth in the same directory.

This script is how that ground truth was produced.  It needs PyTorch and a
checkout of the upstream reference implementation, and exists so the frozen
numbers are reproducible rather than magic:

    pip install torch transformers==4.51.3 x-transformers==2.11.24 einops omegaconf
    git clone https://github.com/QwenAudio/CosyVoice /tmp/cosyvoice_ref
    COSYVOICE_REF=/tmp/cosyvoice_ref python tools/make_reference_fixture.py

Nothing else in the project imports it.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
sys.path.insert(0, ROOT)

from cv3cpu.torch_rng import TorchMT19937  # noqa: E402

# Deliberately small: the fixture is committed, so it has to stay a couple of MB
# while keeping every structural feature of the real model.
HID, LAYERS, HEADS, KV, INTER, VOCAB = 64, 2, 2, 1, 128, 256
DIM, DEPTH, NHEAD, DHEAD = 128, 2, 2, 64   # dim_head must match the release
SPEECH_TOKENS, BASE_CH, F0_CH = 48, 32, 32
SEED = 20240816


def _stub(name, **attrs):
    import importlib.machinery

    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_reference():
    """Make the upstream package importable, stubbing the deps it never uses here."""
    ref = os.environ.get("COSYVOICE_REF")
    if not ref or not os.path.isdir(ref):
        raise SystemExit("set COSYVOICE_REF to a clone of "
                         "https://github.com/QwenAudio/CosyVoice")
    sys.path.insert(0, ref)
    import torch  # noqa: F401

    try:
        importlib.import_module("torchaudio.compliance.kaldi")
    except Exception:
        ta = _stub("torchaudio")
        ta.transforms = types.SimpleNamespace(MelSpectrogram=object)
        _stub("torchaudio.transforms", MelSpectrogram=object)
        _stub("torchaudio.compliance")
        _stub("torchaudio.compliance.kaldi", fbank=None)
    for name in ("onnxruntime", "whisper", "inflect", "hyperpyyaml", "tiktoken",
                 "wetext", "deepspeed"):
        try:
            importlib.import_module(name)
        except Exception:
            _stub(name)
    _stub("cosyvoice.utils.onnx", SpeechTokenExtractor=object,
          online_feature=False, onnx_path="")

    class BASECFM(torch.nn.Module):
        def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
            super().__init__()
            self.n_feats, self.n_spks, self.spk_emb_dim = n_feats, n_spks, spk_emb_dim
            self.solver = cfm_params.solver
            self.sigma_min = getattr(cfm_params, "sigma_min", 1e-4)
            self.estimator = None

    _stub("matcha")
    _stub("matcha.models")
    _stub("matcha.models.components")
    _stub("matcha.models.components.flow_matching", BASECFM=BASECFM)
    return ref


def _qwen_config():
    from transformers import Qwen2Config
    return Qwen2Config(vocab_size=VOCAB, hidden_size=HID, num_hidden_layers=LAYERS,
                       num_attention_heads=HEADS, num_key_value_heads=KV,
                       intermediate_size=INTER, max_position_embeddings=2048,
                       rope_theta=1000000.0, rms_norm_eps=1e-6,
                       tie_word_embeddings=True, use_cache=True)


def build() -> None:
    import torch
    from omegaconf import DictConfig
    from transformers import Qwen2ForCausalLM

    from cosyvoice.flow.DiT.dit import DiT as RefDiT
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    os.makedirs(os.path.join(FIXTURES, "checkpoint", "CosyVoice-BlankEN"), exist_ok=True)
    ckpt = os.path.join(FIXTURES, "checkpoint")

    torch.manual_seed(SEED)
    enc = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(enc)
    enc.model = Qwen2ForCausalLM(_qwen_config())
    lm = CosyVoice3LM(llm_input_size=HID, llm_output_size=HID,
                      speech_token_size=SPEECH_TOKENS, llm=enc,
                      sampling=lambda *a, **k: 0).eval()

    cfm_params = DictConfig(content={
        "sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
        "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"})
    est = RefDiT(dim=DIM, depth=DEPTH, heads=NHEAD, dim_head=DHEAD, mel_dim=80,
                 mu_dim=80, spk_dim=80, out_channels=80, static_chunk_size=50,
                 num_decoding_left_chunks=-1, ff_mult=2, dropout=0.0)
    flow = CausalMaskedDiffWithDiT(
        input_size=80, output_size=80, spk_embed_dim=192, output_type="mel",
        vocab_size=SPEECH_TOKENS, input_frame_rate=25, only_mask_loss=True,
        token_mel_ratio=2, pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(80, 128, 3),
        decoder=CausalConditionalCFM(in_channels=240, n_spks=1, spk_emb_dim=80,
                                     cfm_params=cfm_params, estimator=est)).eval()

    hift = CausalHiFTGenerator(
        in_channels=80, base_channels=BASE_CH, nb_harmonics=8, sampling_rate=24000,
        nsf_alpha=0.1, nsf_sigma=0.003, nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(1, 80, F0_CH)).eval()

    torch.save(lm.state_dict(), os.path.join(ckpt, "llm.pt"))
    torch.save(flow.state_dict(), os.path.join(ckpt, "flow.pt"))
    # the release ships the vocoder inside a HiFiGan wrapper, hence "generator."
    torch.save({"generator." + k: v for k, v in hift.state_dict().items()},
               os.path.join(ckpt, "hift.pt"))
    with open(os.path.join(ckpt, "CosyVoice-BlankEN", "config.json"), "w") as fh:
        json.dump(_qwen_config().to_dict(), fh, indent=1)

    # ---- reference outputs -------------------------------------------------
    rng = np.random.default_rng(SEED)
    n_tok, n_prompt, n_mel = 20, 6, 24
    inputs = {
        "token": rng.integers(0, SPEECH_TOKENS, size=n_tok).astype(np.int64),
        "prompt_token": rng.integers(0, SPEECH_TOKENS, size=n_prompt).astype(np.int64),
        "prompt_feat": rng.normal(size=(2 * n_prompt, 80)).astype(np.float32),
        "embedding": rng.normal(size=(1, 192)).astype(np.float32),
        "mel": rng.normal(size=(80, n_mel)).astype(np.float32),
        "text": rng.integers(0, VOCAB, size=9).astype(np.int64),
        "prompt_speech": rng.integers(0, SPEECH_TOKENS, size=4).astype(np.int64),
    }

    # The vocoder's NSF dither is a random buffer the reference draws at
    # construction time.  Swap in one our own RNG can reproduce, so the fixture
    # does not have to carry half a megabyte of noise.
    up = 480
    noise = TorchMT19937(0).random_float32(up * n_mel * 9).reshape(up * n_mel, 9)
    hift.m_source.l_sin_gen.sine_waves = torch.tensor(noise)[None]

    with torch.no_grad():
        mel, _ = flow.inference(
            token=torch.tensor(inputs["token"], dtype=torch.int32)[None],
            token_len=torch.tensor([n_tok], dtype=torch.int32),
            prompt_token=torch.tensor(inputs["prompt_token"], dtype=torch.int32)[None],
            prompt_token_len=torch.tensor([n_prompt], dtype=torch.int32),
            prompt_feat=torch.tensor(inputs["prompt_feat"])[None],
            prompt_feat_len=torch.tensor([2 * n_prompt], dtype=torch.int32),
            embedding=torch.tensor(inputs["embedding"]), streaming=False, finalize=True)
        wav, _ = hift.inference(torch.tensor(inputs["mel"])[None], finalize=True)

        sos = lm.speech_embedding.weight[lm.sos].reshape(1, 1, -1)
        task = lm.speech_embedding.weight[lm.task_id].reshape(1, 1, -1)
        temb = lm.llm.model.model.embed_tokens(torch.tensor([inputs["text"]]))
        pemb = lm.speech_embedding(torch.tensor([inputs["prompt_speech"]]))
        lm_input = torch.concat([sos, temb, task, pemb], dim=1)
        cache, logits = None, []
        forced = rng.integers(0, SPEECH_TOKENS, size=3).astype(np.int64)
        step_in = lm_input
        for i in range(len(forced) + 1):
            y, cache = lm.llm.forward_one_step(
                step_in,
                masks=torch.tril(torch.ones((1, step_in.shape[1], step_in.shape[1]))).bool(),
                cache=cache)
            logits.append(lm.llm_decoder(y[:, -1])[0].numpy())
            if i < len(forced):
                step_in = lm.speech_embedding.weight[forced[i]].reshape(1, 1, -1)
    inputs["forced"] = forced

    np.savez_compressed(
        os.path.join(FIXTURES, "reference_io.npz"),
        flow_mel=mel[0].numpy(), hift_wav=wav[0].numpy(),
        llm_logits=np.stack(logits), noise_seed=np.array(0), **inputs)

    _dtype_fixture()
    _value_fixture()

    with open(os.path.join(FIXTURES, "config.json"), "w") as fh:
        json.dump({"hid": HID, "layers": LAYERS, "heads": HEADS, "kv": KV,
                   "inter": INTER, "vocab": VOCAB, "dim": DIM, "depth": DEPTH,
                   "nhead": NHEAD, "dhead": DHEAD, "speech_tokens": SPEECH_TOKENS,
                   "base_ch": BASE_CH, "f0_ch": F0_CH, "seed": SEED,
                   "n_mel": n_mel, "source": "tools/make_reference_fixture.py"},
                  fh, indent=1)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(FIXTURES) for f in fs)
    print(f"wrote {FIXTURES} ({total / 1e6:.2f} MB)")


QWEN2_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?"""
    r"""[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

PRETOKENIZE_SAMPLES = [
    "Hello world!", "  leading and   multiple    spaces  ",
    "It's a test, isn't it? I'LL say I'd've...", "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002",
    "\u6df7\u5408 mixed \u4e2d\u82f1\u6587 text 123 456.789",
    "line one\nline two\r\n\r\nline three", "tabs\tand\t\tspaces\n",
    "emoji \U0001f642\U0001f643 and symbols \xb1\xa7\xb6",
    "numbers1234mixedwithletters", "trailing whitespace   ", "", "\n\n\n", "   ",
    "a" * 50 + " " + "0" * 20,
    "\xdcberm\xe4\xdfig gro\xdfe Pr\xfcfung \u2014 \xabguillemets\xbb \u2039and\u203a \u201equotes\u201c",
    "\u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442, \u0435\u0449\u0451 \u043d\u0435\u043c\u043d\u043e\u0433\u043e",
    "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8\uc785\ub2c8\ub2e4",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002",
    "You are a helpful assistant.Hello there",
]


def _sample(arr, n=16):
    """A deterministic spread of exact values, plus shape and summary stats."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    idx = np.linspace(0, flat.size - 1, n).astype(int).tolist()
    return {"shape": list(np.shape(arr)), "idx": idx,
            "vals": [float(flat[i]) for i in idx],
            "sum": float(flat.sum()), "absmax": float(np.abs(flat).max())}


def _value_fixture() -> None:
    """Freeze librosa / torch.stft / torchaudio / regex ground truth as JSON."""
    import librosa
    import regex
    import torch
    import torchaudio

    pat = regex.compile(QWEN2_PATTERN)
    out = {"pretokenize": {s: pat.findall(s) for s in PRETOKENIZE_SAMPLES}}

    out["mel_filters"] = {
        "24k_1920_80": _sample(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80,
                                                   fmin=0, fmax=None)),
        "16k_400_128": _sample(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128,
                                                   fmin=0, fmax=None)),
    }

    rng = np.random.default_rng(20240816)
    y24 = (rng.normal(size=24000) * 0.1).astype(np.float32)
    y16 = (rng.normal(size=16000) * 0.1).astype(np.float32)
    out["input_checksums"] = {"y24": float(y24.astype(np.float64).sum()),
                              "y16": float(y16.astype(np.float64).sum())}

    basis = torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80,
                                                 fmin=0, fmax=None)).float()
    t = torch.nn.functional.pad(torch.tensor(y24)[None].unsqueeze(1), (720, 720),
                                mode="reflect").squeeze(1)
    spec = torch.stft(t, 1920, hop_length=480, win_length=1920,
                      window=torch.hann_window(1920), center=False, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    out["mel_spectrogram"] = _sample(torch.log(torch.clamp(basis @ spec, min=1e-5))[0].numpy())

    st = torch.stft(torch.tensor(y16), 400, 160, window=torch.hann_window(400),
                    return_complex=True)
    filt = torch.tensor(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
    log = torch.clamp(filt @ (st[..., :-1].abs() ** 2), min=1e-10).log10()
    out["whisper_log_mel"] = _sample(((torch.maximum(log, log.max() - 8.0) + 4.0) / 4.0).numpy())

    out["kaldi_fbank"] = _sample(torchaudio.compliance.kaldi.fbank(
        torch.tensor(y16)[None], num_mel_bins=80, dither=0, sample_frequency=16000).numpy())

    # PyTorch's CPU RNG, which cv3cpu.torch_rng reproduces without torch.  Both
    # the >=16 vectorised path and the <16 scalar one, across seeds.
    rngs = {}
    for seed in (0, 1, 42, 1986):
        for shape in ((5,), (17,), (1000,), (1, 80, 15000)):
            torch.manual_seed(seed)
            rngs[f"randn_{seed}_{'x'.join(map(str, shape))}"] = _sample(
                torch.randn(shape).numpy(), n=8)
        torch.manual_seed(seed)
        rngs[f"rand_{seed}_1000"] = _sample(torch.rand(1000).numpy(), n=8)
    out["torch_rng"] = rngs

    with open(os.path.join(ROOT, "tests", "reference_values.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)


def _dtype_fixture() -> None:
    """A tiny state dict covering every storage type the .pt reader supports."""
    import torch

    torch.manual_seed(1)
    sd = {
        "f32": torch.randn(3, 4),
        "f16": torch.randn(5).half(),
        "bf16": torch.randn(2, 3).bfloat16(),
        "i64": torch.arange(6, dtype=torch.int64).reshape(2, 3),
        "i32": torch.arange(4, dtype=torch.int32),
        "i8": torch.tensor([-1, 0, 127], dtype=torch.int8),
        "u8": torch.tensor([0, 128, 255], dtype=torch.uint8),
        "bool": torch.tensor([True, False, True]),
        "strided_view": torch.randn(4, 4)[:, :2],      # non-contiguous storage view
        "shared_storage": torch.randn(8)[2:6],         # non-zero storage offset
    }
    torch.save(sd, os.path.join(FIXTURES, "dtypes.pt"))
    np.savez_compressed(
        os.path.join(FIXTURES, "dtypes.npz"),
        **{k: (v.float() if v.dtype == torch.bfloat16 else v).numpy() for k, v in sd.items()})


if __name__ == "__main__":
    _install_reference()
    build()
