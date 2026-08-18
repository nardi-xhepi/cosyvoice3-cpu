"""Convert a released Fun-CosyVoice3-0.5B checkpoint into one ``.safetensors`` file.

Runs without PyTorch: ``llm.pt`` / ``flow.pt`` / ``hift.pt`` are read with
:mod:`cv3cpu.io.torch_pickle`, weight-norm parametrisations are folded, and each
tensor is written raw or block-quantised according to the chosen profile.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

import numpy as np

from .io.weights import WeightWriter
from .io.safetensors_io import load_dir
from .io.torch_pickle import load_pt

# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

PROFILES = {
    # name       default kind   LLM body kind   min elements to quantise
    "balanced": ("q8", "q8", 1 << 16),
    "tiny":     ("q8", "q4", 1 << 16),
    "quality":  ("f16", "f16", 1 << 18),
    "f32":      ("f32", "f32", 1 << 62),
}

# Tensors that stay in float32 whatever the profile: normalisation scales,
# biases, Snake alphas, and the F0 predictor (upstream runs it in float64 and
# notes that its precision drives the vocoder).
_KEEP_F32 = re.compile(
    r"(\.bias$)|(layernorm)|(\.norm$)|(alpha)|(f0_predictor)|(l_linear)|"
    r"(spk_embed_affine)|(conv_pos_embed)")

# Tensors that never get quantised, only narrowed to float16.  These are either
# quality-critical or too small for quantisation to buy anything:
#
#  * the whole vocoder -- 25 M parameters, so float16 costs ~15 MB more than int8
#    while the peak stage is the 0.5 B-parameter LLM either way, and any error
#    here lands straight on the waveform;
#  * the flow decoder's input and output edges, where an error is applied by
#    every one of the ten Euler steps rather than averaged over a wide matmul.
_KEEP_F16 = re.compile(
    r"^hift\.|(flow\.input_embedding$)|(proj_out)|(time_embed)|(pre_lookahead)")


def _fold_weight_norm(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fold ``weight_g``/``weight_v`` (or the new ``parametrizations``) into ``weight``."""
    out: Dict[str, np.ndarray] = {}
    handled = set()
    for k in list(sd):
        base = None
        if k.endswith(".weight_g"):
            base, gk, vk = k[: -len(".weight_g")], k, k[: -len("_g")] + "_v"
        elif k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            gk = k
            vk = base + ".parametrizations.weight.original1"
        if base is None or gk not in sd or vk not in sd:
            continue
        g = np.asarray(sd[gk], dtype=np.float32)
        v = np.asarray(sd[vk], dtype=np.float32)
        axes = tuple(range(1, v.ndim))  # weight_norm(dim=0)
        norm = np.sqrt((v * v).sum(axis=axes, keepdims=True))
        out[base + ".weight"] = (g / np.maximum(norm, 1e-12)) * v
        handled.update({gk, vk})
    for k, v in sd.items():
        if k in handled:
            continue
        out.setdefault(k, v)
    return out


# ---------------------------------------------------------------------------
# key mapping
# ---------------------------------------------------------------------------

def _map_llm(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in sd.items():
        if k.startswith("llm.model.model."):
            tail = k[len("llm.model.model."):]
            if tail.endswith("layernorm.weight") or tail == "norm.weight":
                tail = tail[: -len(".weight")]
            if tail == "embed_tokens.weight":
                tail = "embed_tokens"
            out["llm.model." + tail] = v
        elif k.startswith("llm.model.lm_head."):
            continue  # unused: CosyVoice replaces it with llm_decoder
        elif k.startswith("llm_decoder."):
            out["llm." + k] = v
        elif k == "speech_embedding.weight":
            out["llm.speech_embedding"] = v
        elif k == "llm_embedding.weight":
            out["llm.llm_embedding"] = v
        elif k.startswith("criterion_ce."):
            continue
        else:
            out["llm." + k] = v
    return out


def _load_qwen_safetensors(model_dir: str) -> Dict[str, np.ndarray]:
    for sub in ("CosyVoice-BlankEN", "."):
        d = os.path.join(model_dir, sub)
        try:
            shards = load_dir(d)
        except FileNotFoundError:
            continue
        # HF shards are keyed "model.layers.0..."; _map_llm wants the
        # "llm.model.model." prefix the CosyVoice module hierarchy produces.
        # Copy out of the mapping: these get quantised in place downstream, and
        # the file is closed right after.
        sd = _map_llm({"llm.model." + k: np.array(v, dtype=np.float32)
                       for k, v in shards.items() if k.startswith("model.")})
        shards.close()
        if sd:
            return sd
    raise FileNotFoundError(
        f"llm.pt has no Qwen weights and no safetensors shards were found under "
        f"{model_dir}")


def _map_flow(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in sd.items():
        if k == "input_embedding.weight":
            out["flow.input_embedding"] = v
        else:
            out["flow." + k] = v
    return out


_CONDNET = re.compile(r"^f0_predictor\.condnet\.(\d+)\.(weight|bias)$")


def _map_hift(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in sd.items():
        if k.startswith("generator."):
            k = k[len("generator."):]
        if k.startswith("discriminator") or k.startswith("mpd") or k.startswith("mrd"):
            continue
        m = _CONDNET.match(k)
        if m:  # nn.Sequential(conv, ELU, conv, ELU, ...) -> dense indices
            k = f"f0_predictor.condnet.{int(m.group(1)) // 2}.{m.group(2)}"
        if k in ("stft_window", "m_source.uv", "m_source.l_sin_gen.sine_waves",
                 "m_source.l_sin_gen.rand_ini"):
            continue
        out["hift." + k] = v
    return out


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _endofprompt_id(model_dir: str, default: int = 151646) -> int:
    """Look up ``<|endofprompt|>`` in the shipped tokenizer instead of assuming."""
    for sub in ("CosyVoice-BlankEN", "."):
        d = os.path.join(model_dir, sub)
        try:
            from .tokenizer import QwenTokenizer
            tok = QwenTokenizer.from_dir(d)
        except Exception:
            continue
        if "<|endofprompt|>" in tok.added:
            return int(tok.added["<|endofprompt|>"])
        if "<|endofprompt|>" in tok.vocab:
            return int(tok.vocab["<|endofprompt|>"])
    return default


def _qwen_meta(w: Dict[str, np.ndarray], config: Optional[dict]) -> dict:
    layers = 1 + max(int(m.group(1)) for k in w
                     if (m := re.match(r"llm\.model\.layers\.(\d+)\.", k)))
    hidden = int(w["llm.model.embed_tokens"].shape[1])
    kv_out = int(w["llm.model.layers.0.self_attn.k_proj.weight"].shape[0])
    inter = int(w["llm.model.layers.0.mlp.gate_proj.weight"].shape[0])
    cfg = config or {}
    head_dim = int(cfg.get("head_dim", 64))
    return {
        "hidden_size": hidden,
        "num_hidden_layers": int(cfg.get("num_hidden_layers", layers)),
        "num_attention_heads": int(cfg.get("num_attention_heads", hidden // head_dim)),
        "num_key_value_heads": int(cfg.get("num_key_value_heads", kv_out // head_dim)),
        "intermediate_size": int(cfg.get("intermediate_size", inter)),
        "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(cfg.get("rope_theta", 1000000.0)),
        "vocab_size": int(w["llm.model.embed_tokens"].shape[0]),
    }


DEFAULT_FLOW_META = {
    "token_mel_ratio": 2,
    "pre_lookahead_len": 3,
    "output_size": 80,
    "inference_cfg_rate": 0.7,
    "n_timesteps": 10,
}

DEFAULT_DIT_META = {
    "dim": 1024, "depth": 22, "heads": 16, "dim_head": 64,
    "static_chunk_size": 50, "num_decoding_left_chunks": -1,
    "conv_pos_kernel": 31, "conv_pos_groups": 16,
}

DEFAULT_HIFT_META = {
    "upsample_rates": [8, 5, 3],
    "upsample_kernel_sizes": [16, 11, 7],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "source_resblock_kernel_sizes": [7, 7, 11],
    "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "istft_n_fft": 16, "istft_hop_len": 4,
    "lrelu_slope": 0.1, "audio_limit": 0.99, "sampling_rate": 24000,
    "nb_harmonics": 8, "nsf_alpha": 0.1, "nsf_sigma": 0.003,
    "nsf_voiced_threshold": 10.0, "conv_pre_look_right": 4,
}


def _infer_dit_meta(w: Dict[str, np.ndarray]) -> dict:
    """Recover the DiT geometry from tensor shapes.

    ``dim_head`` is the one thing shapes cannot reveal — ``to_q`` is square
    whatever the head split is — so it keeps the released value (64) and the head
    count is derived from it.
    """
    meta = dict(DEFAULT_DIT_META)
    pref = "flow.decoder.estimator."
    depth = 1 + max((int(m.group(1)) for k in w
                     if (m := re.match(re.escape(pref) + r"transformer_blocks\.(\d+)\.", k))),
                    default=DEFAULT_DIT_META["depth"] - 1)
    meta["depth"] = depth
    q = w.get(pref + "transformer_blocks.0.attn.to_q.weight")
    if q is not None:
        meta["dim"] = int(q.shape[1])
        meta["heads"] = int(q.shape[0]) // meta["dim_head"]
    cp = w.get(pref + "input_embed.conv_pos_embed.conv1.0.weight")
    if cp is not None:
        meta["conv_pos_kernel"] = int(cp.shape[2])
        meta["conv_pos_groups"] = int(cp.shape[0] // cp.shape[1])
    return meta


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _choose_kind(name: str, arr: np.ndarray, profile: str) -> str:
    default_kind, llm_kind, min_numel = PROFILES[profile]
    if arr.ndim == 1 or _KEEP_F32.search(name):
        return "f32"
    if arr.size < min_numel:
        return "f32"
    if _KEEP_F16.search(name):
        return "f32" if default_kind == "f32" else "f16"
    if name.startswith("llm.model.layers.") or name == "llm.model.embed_tokens":
        return llm_kind
    return default_kind


def _as_matrix(arr: np.ndarray) -> np.ndarray:
    """Flatten a conv weight ``(out, in, k)`` to the ``(out, in * k)`` the GEMM wants."""
    return arr.reshape(arr.shape[0], -1) if arr.ndim == 3 else arr


def write_weights(tensors: Dict[str, np.ndarray], out_path: str, meta: dict,
              profile: str = "balanced", verbose: bool = True) -> None:
    stats: Dict[str, int] = {}
    with WeightWriter(out_path, meta=meta, header_reserve=4 << 20) as w:
        for name in sorted(tensors):
            arr = np.asarray(tensors[name])
            if arr.dtype == np.bool_:
                arr = arr.astype(np.float32)
            arr = arr.astype(np.float32)
            kind = _choose_kind(name, arr, profile)
            if kind in ("q8", "q4") and _is_grouped_conv(name, arr):
                kind = "f32"  # grouped convs do not go through the quantised GEMM
            if kind in ("q8", "q4"):
                w.add_quant(name, _as_matrix(arr), kind, orig_shape=arr.shape)
            elif kind == "f16":
                w.add_raw(name, arr, "f16")
            else:
                w.add_raw(name, arr, "f32")
            stats[kind] = stats.get(kind, 0) + arr.size
    if verbose:
        total = sum(stats.values())
        print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, "
              f"{total / 1e6:.1f}M params)")
        for k, v in sorted(stats.items()):
            print(f"  {k:>4}: {v / 1e6:8.2f}M params")


def _is_grouped_conv(name: str, arr: np.ndarray) -> bool:
    """The only grouped convolution in CosyVoice 3 is the DiT position embedding."""
    return arr.ndim == 3 and "conv_pos_embed" in name


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def convert(model_dir: str, out_path: str, profile: str = "balanced",
            verbose: bool = True) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}; pick one of {sorted(PROFILES)}")

    def _p(name: str) -> str:
        return os.path.join(model_dir, name)

    tensors: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {"source": os.path.basename(os.path.abspath(model_dir)),
                               "profile": profile, "sample_rate": 24000,
                               "speech_token_size": 6561}

    llm_sd = _map_llm(_fold_weight_norm(load_pt(_p("llm.pt"))))
    if "llm.model.embed_tokens" not in llm_sd:
        # Older layouts keep the Qwen body beside the checkpoint instead of
        # inside llm.pt; pull it from the safetensors shards in that case.
        llm_sd.update(_load_qwen_safetensors(model_dir))
    tensors.update(llm_sd)
    cfg_path = None
    for cand in ("CosyVoice-BlankEN/config.json", "config.json"):
        if os.path.exists(_p(cand)):
            cfg_path = _p(cand)
            break
    cfg = json.load(open(cfg_path)) if cfg_path else None
    if cfg is None and verbose:
        print("note: no Qwen config.json found, falling back to Qwen2-0.5B defaults")
    meta["qwen2"] = _qwen_meta(llm_sd, cfg)
    meta["endofprompt_id"] = _endofprompt_id(model_dir)
    flow_sd = _map_flow(_fold_weight_norm(load_pt(_p("flow.pt"))))
    tensors.update(flow_sd)
    meta["speech_token_size"] = int(flow_sd["flow.input_embedding"].shape[0])
    meta["flow"] = dict(DEFAULT_FLOW_META)
    meta["dit"] = _infer_dit_meta(flow_sd)

    hift_sd = _map_hift(_fold_weight_norm(load_pt(_p("hift.pt"))))
    tensors.update(hift_sd)
    meta["hift"] = dict(DEFAULT_HIFT_META)

    write_weights(tensors, out_path, meta, profile=profile, verbose=verbose)
