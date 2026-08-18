"""``python -m cv3cpu`` — convert, enroll, synthesise, inspect, benchmark."""

from __future__ import annotations

import argparse
import sys
import time


def _cmd_convert(args) -> int:
    from .convert import convert
    convert(args.model_dir, args.out, profile=args.profile)
    return 0


def _cmd_enroll(args) -> int:
    from .enroll import enroll
    pack = enroll(args.wav, args.text, args.model_dir, name=args.name)
    pack.save(args.out)
    print(f"wrote {args.out}: {pack}")
    return 0


def _cmd_info(args) -> int:
    import json

    from .io.weights import WeightFile
    f = WeightFile(args.weights)
    print(json.dumps(f.meta, indent=2, sort_keys=True))
    by_kind = {}
    for name in f.keys():
        e = f.entry(name)
        by_kind.setdefault(e["kind"], [0, 0])
        by_kind[e["kind"]][0] += 1
        by_kind[e["kind"]][1] += e["nbytes"] + e.get("sc_nbytes", 0)
    print("\ntensors:")
    for kind, (count, nbytes) in sorted(by_kind.items()):
        print(f"  {kind:>4}: {count:4d} tensors, {nbytes / 1e6:8.1f} MB")
    for stage in ("llm.", "flow.", "hift."):
        nb = sum(f.entry(n)["nbytes"] + f.entry(n).get("sc_nbytes", 0)
                 for n in f.keys() if n.startswith(stage))
        print(f"  stage {stage:<6} {nb / 1e6:8.1f} MB")
    print(f"  total        {f.total_bytes() / 1e6:8.1f} MB")
    f.close()
    return 0


def _cmd_tts(args) -> int:
    from .memory import snapshot
    from .models.llm import SamplingConfig
    from .pipeline import CosyVoice3CPU
    from .voice import VoicePack

    voice = VoicePack.load(args.voice)
    tts = CosyVoice3CPU(args.weights, tokenizer_dir=args.tokenizer, seed=args.seed,
                        track_memory=args.report_memory,
                        release_stages=not args.keep_stages,
                        hift_chunk=args.hift_chunk)
    sampling = SamplingConfig(top_p=args.top_p, top_k=args.top_k, seed=args.seed)

    def progress(stage, i, n):
        if not args.quiet:
            print(f"\r  {stage} {i}/{n}", end="", file=sys.stderr, flush=True)

    t0 = time.time()
    result = tts.synthesize(args.text, voice, split=not args.no_split,
                            sampling=sampling, on_progress=progress)
    dt = time.time() - t0
    if not args.quiet:
        print(file=sys.stderr)
    tts.save(result, args.out)
    tts.close()
    rtf = dt / max(result.duration, 1e-6)
    print(f"wrote {args.out}: {result.duration:.2f}s audio in {dt:.1f}s (RTF {rtf:.2f}), "
          f"{len(result.segments)} segment(s)")
    if args.report_memory:
        print(result.memory_report)
        snap = snapshot()
        print(f"peak RSS {snap['peak_rss_mb']:.1f} MB "
              f"(anonymous now {snap['anon_mb']:.1f} MB)")
    return 0


def _cmd_bench(args) -> int:
    """Time each stage on synthetic input; useful without a reference voice."""
    import numpy as np

    from .memory import Tracker, snapshot
    from .models.flow import Flow
    from .models.hift import HiFTGenerator
    from .models.llm import CosyVoice3LM
    from .io.weights import WeightFile

    f = WeightFile(args.weights)
    tracker = Tracker(True)
    rng = np.random.default_rng(0)
    n_tok = args.tokens
    vocab = int(f.meta.get("speech_token_size", 6561))
    tokens = rng.integers(0, vocab, size=n_tok).tolist()
    prompt_tokens = rng.integers(0, vocab, size=50).tolist()
    prompt_mel = rng.normal(size=(100, 80)).astype(np.float32)
    emb = rng.normal(size=192).astype(np.float32)

    tracker.mark("start")
    lm = CosyVoice3LM(f, "llm.")
    hidden = lm.cfg.hidden_size
    cache = lm.body.new_cache()
    t0 = time.time()
    lm.body.forward(rng.normal(size=(args.prefill, hidden)).astype(np.float32) * 0.1, cache)
    prefill_dt = time.time() - t0
    t0 = time.time()
    for _ in range(args.steps):
        lm.body.forward(rng.normal(size=(1, hidden)).astype(np.float32) * 0.1, cache)
    decode_dt = time.time() - t0
    tracker.mark("llm")
    del lm, cache
    f.evict("llm.")
    tracker.mark("llm released")

    flow = Flow(f, "flow.")
    t0 = time.time()
    mel = flow.inference(tokens, prompt_tokens, prompt_mel, emb)
    flow_dt = time.time() - t0
    tracker.mark("flow")
    del flow
    f.evict("flow.")
    tracker.mark("flow released")

    hift = HiFTGenerator(f, "hift.")
    t0 = time.time()
    wav = hift.inference(mel, chunk=args.hift_chunk)
    hift_dt = time.time() - t0
    tracker.mark("hift")
    f.close()

    audio_s = len(wav) / 24000
    print(f"llm prefill {args.prefill} tokens: {prefill_dt:.2f}s")
    print(f"llm decode {args.steps} steps: {decode_dt:.2f}s "
          f"({args.steps / max(decode_dt, 1e-9):.1f} tok/s)")
    print(f"flow {n_tok} tokens -> {mel.shape[1]} mel frames: {flow_dt:.1f}s")
    print(f"hift -> {audio_s:.2f}s audio: {hift_dt:.1f}s")
    print(f"flow+hift RTF {(flow_dt + hift_dt) / max(audio_s, 1e-9):.2f}")
    print(tracker.report())
    snap = snapshot()
    print(f"peak RSS {snap['peak_rss_mb']:.1f} MB "
          f"(anonymous now {snap['anon_mb']:.1f} MB; the rest is the mapped .safetensors, "
          f"which the kernel can reclaim)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser("cv3cpu", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="checkpoint directory -> .safetensors")
    c.add_argument("model_dir")
    c.add_argument("out")
    c.add_argument("--profile", default="balanced",
                   choices=["balanced", "tiny", "quality", "f32"])
    c.set_defaults(fn=_cmd_convert)

    c = sub.add_parser("enroll", help="reference wav -> voice pack")
    c.add_argument("wav")
    c.add_argument("out")
    c.add_argument("--text", required=True, help="transcript of the reference audio")
    c.add_argument("--model-dir", required=True, help="dir holding the ONNX encoders")
    c.add_argument("--name", default=None)
    c.set_defaults(fn=_cmd_enroll)

    c = sub.add_parser("tts", help="synthesise speech")
    c.add_argument("text")
    c.add_argument("--weights", required=True)
    c.add_argument("--voice", required=True)
    c.add_argument("--tokenizer", required=True,
                   help="the checkpoint's CosyVoice-BlankEN directory")
    c.add_argument("--out", default="out.wav")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--top-p", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=25)
    c.add_argument("--no-split", action="store_true")
    c.add_argument("--keep-stages", action="store_true",
                   help="do not release a stage's weights when it finishes")
    c.add_argument("--hift-chunk", type=int, default=250)
    c.add_argument("--report-memory", action="store_true")
    c.add_argument("--quiet", action="store_true")
    c.set_defaults(fn=_cmd_tts)

    c = sub.add_parser("info", help="show what is inside a .safetensors")
    c.add_argument("weights")
    c.set_defaults(fn=_cmd_info)

    c = sub.add_parser("bench", help="time and measure each stage on synthetic input")
    c.add_argument("weights")
    c.add_argument("--tokens", type=int, default=250)
    c.add_argument("--prefill", type=int, default=100)
    c.add_argument("--steps", type=int, default=20)
    c.add_argument("--hift-chunk", type=int, default=250)
    c.set_defaults(fn=_cmd_bench)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
