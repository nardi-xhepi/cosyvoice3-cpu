"""Generate a full-size *randomly initialised* CosyVoice 3 ``.safetensors``.

Useful for exercising the runtime — shapes, speed, memory — without the real
checkpoint.  The audio it produces is noise; only the resource behaviour is
meaningful.  Tensors are generated and written one at a time so building a
~1 B parameter file never needs more than a few hundred MB of RAM.
"""

from __future__ import annotations

import argparse
import sys
import os

import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cv3cpu.convert import (DEFAULT_DIT_META, DEFAULT_FLOW_META, DEFAULT_HIFT_META,
                            PROFILES, _choose_kind, _is_grouped_conv)  # noqa: E402
from cv3cpu.io.weights import WeightWriter  # noqa: E402

# Fun-CosyVoice3-0.5B geometry
QWEN = dict(hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
            num_key_value_heads=2, intermediate_size=4864, rms_norm_eps=1e-6,
            rope_theta=1000000.0, vocab_size=151936)
SPEECH_TOKEN_SIZE = 6561
LM_HEAD = SPEECH_TOKEN_SIZE + 200
DIT = dict(DEFAULT_DIT_META)
HIFT_BASE = 512


def _emit(w: WeightWriter, name: str, shape, profile: str, scale: float = 0.02,
          ones: bool = False) -> int:
    # crc32, not hash(): PYTHONHASHSEED randomises str hashing per process, and a
    # dummy model that changes every run makes tests flaky.
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    arr = (np.ones(shape, np.float32) if ones
           else rng.standard_normal(shape, dtype=np.float32) * scale)
    kind = _choose_kind(name, arr, profile)
    if kind in ("q8", "q4") and arr.ndim == 3 and _is_grouped_conv(name, arr, {}):
        kind = "f32"
    if kind in ("q8", "q4"):
        w.add_quant(name, arr.reshape(arr.shape[0], -1), kind, orig_shape=arr.shape)
    else:
        w.add_raw(name, arr, "f16" if kind == "f16" else "f32")
    return int(arr.size)


SMALL_QWEN = dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                  num_key_value_heads=1, intermediate_size=256, rms_norm_eps=1e-6,
                  rope_theta=1000000.0, vocab_size=2048)
SMALL_DIT = dict(DEFAULT_DIT_META, dim=128, depth=2, heads=2, dim_head=64)


def build(path: str, profile: str = "balanced", small: bool = False) -> None:
    """Write a random ``.safetensors``.  ``small`` shrinks every dimension for tests."""
    global QWEN, DIT, HIFT_BASE, LM_HEAD, SPEECH_TOKEN_SIZE
    if small:
        QWEN, DIT, HIFT_BASE = SMALL_QWEN, SMALL_DIT, 64
        SPEECH_TOKEN_SIZE = 256
        LM_HEAD = SPEECH_TOKEN_SIZE + 200
    h = QWEN["hidden_size"]
    kv = QWEN["num_key_value_heads"] * (h // QWEN["num_attention_heads"])
    inter = QWEN["intermediate_size"]
    meta = {"source": "dummy", "profile": profile, "sample_rate": 24000,
            "qwen2": QWEN, "speech_token_size": SPEECH_TOKEN_SIZE,
            "endofprompt_id": 151646 if not small else QWEN["vocab_size"] - 1,
            "flow": DEFAULT_FLOW_META, "dit": DIT, "hift": DEFAULT_HIFT_META}
    total = 0
    with WeightWriter(path, meta=meta, header_reserve=4 << 20) as w:
        # ---- LLM ----
        total += _emit(w, "llm.model.embed_tokens", (QWEN["vocab_size"], h), profile)
        for i in range(QWEN["num_hidden_layers"]):
            p = f"llm.model.layers.{i}."
            total += _emit(w, p + "self_attn.q_proj.weight", (h, h), profile)
            total += _emit(w, p + "self_attn.q_proj.bias", (h,), profile)
            total += _emit(w, p + "self_attn.k_proj.weight", (kv, h), profile)
            total += _emit(w, p + "self_attn.k_proj.bias", (kv,), profile)
            total += _emit(w, p + "self_attn.v_proj.weight", (kv, h), profile)
            total += _emit(w, p + "self_attn.v_proj.bias", (kv,), profile)
            total += _emit(w, p + "self_attn.o_proj.weight", (h, h), profile)
            total += _emit(w, p + "mlp.gate_proj.weight", (inter, h), profile)
            total += _emit(w, p + "mlp.up_proj.weight", (inter, h), profile)
            total += _emit(w, p + "mlp.down_proj.weight", (h, inter), profile)
            total += _emit(w, p + "input_layernorm", (h,), profile, ones=True)
            total += _emit(w, p + "post_attention_layernorm", (h,), profile, ones=True)
        total += _emit(w, "llm.model.norm", (h,), profile, ones=True)
        total += _emit(w, "llm.llm_decoder.weight", (LM_HEAD, h), profile)
        total += _emit(w, "llm.speech_embedding", (LM_HEAD, h), profile)

        # ---- flow ----
        d, depth = DIT["dim"], DIT["depth"]
        total += _emit(w, "flow.input_embedding", (SPEECH_TOKEN_SIZE, 80), profile)
        total += _emit(w, "flow.spk_embed_affine_layer.weight", (80, 192), profile)
        total += _emit(w, "flow.spk_embed_affine_layer.bias", (80,), profile)
        total += _emit(w, "flow.pre_lookahead_layer.conv1.weight", (1024, 80, 4), profile)
        total += _emit(w, "flow.pre_lookahead_layer.conv1.bias", (1024,), profile)
        total += _emit(w, "flow.pre_lookahead_layer.conv2.weight", (80, 1024, 3), profile)
        total += _emit(w, "flow.pre_lookahead_layer.conv2.bias", (80,), profile)
        e = "flow.decoder.estimator."
        total += _emit(w, e + "time_embed.time_mlp.0.weight", (d, 256), profile)
        total += _emit(w, e + "time_embed.time_mlp.0.bias", (d,), profile)
        total += _emit(w, e + "time_embed.time_mlp.2.weight", (d, d), profile)
        total += _emit(w, e + "time_embed.time_mlp.2.bias", (d,), profile)
        total += _emit(w, e + "input_embed.proj.weight", (d, 320), profile)
        total += _emit(w, e + "input_embed.proj.bias", (d,), profile)
        for c in ("conv1", "conv2"):
            total += _emit(w, e + f"input_embed.conv_pos_embed.{c}.0.weight",
                           (d, d // DIT["conv_pos_groups"], DIT["conv_pos_kernel"]), profile)
            total += _emit(w, e + f"input_embed.conv_pos_embed.{c}.0.bias", (d,), profile)
        for i in range(depth):
            b = e + f"transformer_blocks.{i}."
            total += _emit(w, b + "attn_norm.linear.weight", (6 * d, d), profile)
            total += _emit(w, b + "attn_norm.linear.bias", (6 * d,), profile)
            for n in ("to_q", "to_k", "to_v"):
                total += _emit(w, b + f"attn.{n}.weight", (d, d), profile)
                total += _emit(w, b + f"attn.{n}.bias", (d,), profile)
            total += _emit(w, b + "attn.to_out.0.weight", (d, d), profile)
            total += _emit(w, b + "attn.to_out.0.bias", (d,), profile)
            total += _emit(w, b + "ff.ff.0.0.weight", (2 * d, d), profile)
            total += _emit(w, b + "ff.ff.0.0.bias", (2 * d,), profile)
            total += _emit(w, b + "ff.ff.2.weight", (d, 2 * d), profile)
            total += _emit(w, b + "ff.ff.2.bias", (d,), profile)
        total += _emit(w, e + "norm_out.linear.weight", (2 * d, d), profile)
        total += _emit(w, e + "norm_out.linear.bias", (2 * d,), profile)
        total += _emit(w, e + "proj_out.weight", (80, d), profile)
        total += _emit(w, e + "proj_out.bias", (80,), profile)

        # ---- hift ----
        m = DEFAULT_HIFT_META
        total += _emit(w, "hift.conv_pre.weight", (HIFT_BASE, 80, m["conv_pre_look_right"] + 1), profile)
        total += _emit(w, "hift.conv_pre.bias", (HIFT_BASE,), profile)
        chans = []
        for i, (u, k) in enumerate(zip(m["upsample_rates"], m["upsample_kernel_sizes"])):
            cin, cout = HIFT_BASE // (2 ** i), HIFT_BASE // (2 ** (i + 1))
            chans.append(cout)
            total += _emit(w, f"hift.ups.{i}.weight", (cout, cin, k), profile)
            total += _emit(w, f"hift.ups.{i}.bias", (cout,), profile)
        down = [1] + m["upsample_rates"][::-1][:-1]
        cum = list(np.cumprod(down))[::-1]
        for i, u in enumerate(cum):
            u = int(u)
            ch = chans[i]
            ks = 1 if u == 1 else u * 2
            total += _emit(w, f"hift.source_downs.{i}.weight", (ch, m["istft_n_fft"] + 2, ks), profile)
            total += _emit(w, f"hift.source_downs.{i}.bias", (ch,), profile)
            total += _emit_resblock(w, f"hift.source_resblocks.{i}", ch,
                                    m["source_resblock_kernel_sizes"][i],
                                    m["source_resblock_dilation_sizes"][i], profile)
        for i in range(len(m["upsample_rates"])):
            for j, (k, dil) in enumerate(zip(m["resblock_kernel_sizes"],
                                             m["resblock_dilation_sizes"])):
                total += _emit_resblock(w, f"hift.resblocks.{i * 3 + j}", chans[i], k, dil, profile)
        total += _emit(w, "hift.conv_post.weight", (m["istft_n_fft"] + 2, chans[-1], 7), profile)
        total += _emit(w, "hift.conv_post.bias", (m["istft_n_fft"] + 2,), profile)
        total += _emit(w, "hift.m_source.l_linear.weight", (1, m["nb_harmonics"] + 1), profile)
        total += _emit(w, "hift.m_source.l_linear.bias", (1,), profile)
        for i in range(5):
            cin = 80 if i == 0 else 512
            k = 4 if i == 0 else 3
            total += _emit(w, f"hift.f0_predictor.condnet.{i}.weight", (512, cin, k), profile)
            total += _emit(w, f"hift.f0_predictor.condnet.{i}.bias", (512,), profile)
        total += _emit(w, "hift.f0_predictor.classifier.weight", (1, 512), profile)
        total += _emit(w, "hift.f0_predictor.classifier.bias", (1,), profile)
    print(f"{path}: {total / 1e6:.1f}M params, {os.path.getsize(path) / 1e6:.1f} MB "
          f"(profile={profile})")


def _emit_resblock(w, base, ch, k, dilations, profile) -> int:
    n = 0
    for j, d in enumerate(dilations):
        n += _emit(w, f"{base}.convs1.{j}.weight", (ch, ch, k), profile)
        n += _emit(w, f"{base}.convs1.{j}.bias", (ch,), profile)
        n += _emit(w, f"{base}.convs2.{j}.weight", (ch, ch, k), profile)
        n += _emit(w, f"{base}.convs2.{j}.bias", (ch,), profile)
        n += _emit(w, f"{base}.activations1.{j}.alpha", (ch,), profile, ones=True)
        n += _emit(w, f"{base}.activations2.{j}.alpha", (ch,), profile, ones=True)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--profile", default="balanced", choices=sorted(PROFILES))
    ap.add_argument("--small", action="store_true",
                    help="tiny stand-in with the same topology, for tests")
    a = ap.parse_args()
    build(a.out, a.profile, small=a.small)
