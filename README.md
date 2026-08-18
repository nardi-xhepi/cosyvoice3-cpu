# cv3cpu — CosyVoice 3 on CPU, in NumPy, inside 2 GB

A from-scratch inference runtime for [Fun-CosyVoice3-0.5B](https://github.com/QwenAudio/CosyVoice)
that has no PyTorch anywhere — not at run time, not at conversion time — and that
holds a ~860 M parameter TTS stack inside a 2 GB memory budget on an ordinary CPU.

`pip install numpy` is the whole dependency list for synthesis.

```python
from cv3cpu import CosyVoice3CPU, VoicePack

tts = CosyVoice3CPU("cosyvoice3-balanced.safetensors", tokenizer_dir="pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN")
out = tts.synthesize("Hello there, this is a cloned voice.", VoicePack.load("alice.npz"))
tts.save(out, "hello.wav")
```

## What it does

CosyVoice 3 is three networks in a row: a Qwen2-0.5B language model that turns text
into 25 Hz FSQ speech tokens, a 22-layer DiT flow-matching decoder that turns those
tokens into a mel spectrogram, and a causal HiFTNet vocoder that turns the mel into
24 kHz audio. All three are reimplemented here in NumPy, along with the Qwen BPE
tokenizer, the mel/fbank/whisper front ends, and PyTorch's CPU random number
generator (the flow decoder's ODE initial condition is a fixed `torch.randn` draw,
so a PyTorch-free runtime has to be able to reproduce it bit for bit — see
`cv3cpu/torch_rng.py`).

## Measured behaviour

Everything below is from `python -m cv3cpu bench` on a 4-vCPU machine, under a real
cgroup memory cap with a cold page cache, using a full-size model (859 M parameters,
correct geometry, random weights — see *Caveats*).

**Memory**, `balanced` profile, 10 s of audio:

| stage          | RSS     | anonymous |
| -------------- | ------- | --------- |
| start          |  37 MB  |  18 MB    |
| llm            | 431 MB  |  41 MB    |
| llm released   |  60 MB  |  40 MB    |
| flow           | 424 MB  |  46 MB    |
| flow released  |  67 MB  |  46 MB    |
| hift           | 158 MB  | 100 MB    |
| **peak**       | **484 MB** | **100 MB** |

Peak RSS is 484 MB against a 935 MB weight file, because the three stages never
overlap: when the LLM has finished emitting speech tokens its weights are handed
back to the kernel with `MADV_DONTNEED` before the flow decoder faults its own in.
The "anonymous" column is the part that genuinely has to fit — the rest is the
memory-mapped `.safetensors`, which is clean, file-backed and reclaimable.

That is why it keeps working well below the target:

| hard cap | result                                  |
| -------- | --------------------------------------- |
| 2 GB     | 4.9 tok/s LLM, flow+vocoder RTF 4.8     |
| 512 MB   | same speed                              |
| 256 MB   | works; LLM drops to ~2 tok/s (page churn) |
| 192 MB   | still works, same reduced LLM speed     |

**Speed**, `balanced`, per 10 s of audio on 4 shared vCPUs: LLM decode 2.4–4.9 tok/s,
flow 32–70 s, vocoder ~4 s. Those ranges are what repeated runs of the same work
produced on a contended machine — back-to-back runs varied by more than 2×, so treat
them as an order of magnitude rather than a benchmark. Memory, by contrast, was
stable to within a few MB across every run.

The shape of the cost is clearer than the absolute numbers. The flow decoder is
compute-bound: 10 Euler steps with classifier-free guidance is ~3.7 TFLOP for 10 s
of audio, and it runs near what this machine can do at all. The LLM is
bandwidth-bound: one token streams the whole quantised body, ~380 MB, from memory.

Either way this is offline throughput, not real time — expect several times slower
than real time end to end.

## Correctness

The suite runs against a small checkpoint committed under `tests/fixtures`, saved
exactly the way the release saves one, together with frozen ground truth: what the
upstream PyTorch implementation produced for a fixed set of inputs, and what
librosa / `torch.stft` / `torchaudio.compliance.kaldi` / the Unicode pre-tokenizer
regex produce for theirs. So the checks compare against the reference without
depending on it — `pytest` needs NumPy and nothing else, verified by running the
whole suite with `torch`, `transformers`, `librosa`, `tokenizers`, `regex`, `scipy`
and `onnxruntime` blocked at import.

Max relative error, float32 throughout:

| check                                              | error  |
| -------------------------------------------------- | ------ |
| Flow matching, full 10-step CFG solve vs upstream   | 2e-7   |
| HiFTNet vocoder, mel → waveform vs upstream         | 2e-6   |
| CosyVoice3LM logits, every decode step vs upstream  | 2e-7   |
| Incremental decoding vs one whole-sequence pass     | 5e-5   |
| Time-chunked vocoder vs whole-utterance decode      | 2e-6   |
| int8 (`balanced`) flow output vs the f32 conversion | 4e-5   |
| `torch.randn` / `torch.rand` replication            | float32 ulp |
| mel filterbank, mel spectrogram, whisper log-mel    | 1e-5   |
| Kaldi fbank vs `torchaudio.compliance.kaldi`        | 1e-4   |
| Qwen2 pre-tokenizer vs the real Unicode regex       | exact  |

The kernels are checked separately against naive reference implementations written
inline in `tests/test_ops.py` — direct-loop convolutions, a materialised softmax
attention, a direct DFT — so the tiling, masking and streaming in the fast paths
have something independent to disagree with. Weight-norm folding is recomputed from
the raw checkpoint, and the `.pt` reader is checked across every storage type
including bfloat16, strided views and non-zero storage offsets.

```sh
pip install -e ".[dev]"
pytest
```

`tools/make_reference_fixture.py` records how the frozen values were produced. It
needs PyTorch and a checkout of the reference implementation, and nothing else in
the project imports it — regenerating fixtures is a maintainer operation, not part
of building or testing.

Two implementation quirks of the reference are reproduced deliberately, because the
released weights were trained with them:

* The DiT applies rotary embeddings to the projection *before* splitting it into
  heads, and the rotation covers only `dim_head` channels — so only head 0 actually
  receives positional information.
* While below `min_len`, the sampler masks exactly one token id
  (`speech_token_size`), not the whole stop-token range, so generation can still end
  early. Pass `SamplingConfig(strict_min_len=True)` for the behaviour the upstream
  comment describes.

One upstream hazard worth knowing: `Qwen2Encoder.forward_one_step` passes a
length-1 attention mask alongside a longer KV cache. On the pinned
`transformers==4.51.3` that mask is ignored and the model attends to the whole
cache; on `transformers>=5` it is honoured and decoding silently attends to one
position. This runtime always attends to the whole cache, matching 4.51.3.

## Usage

### 1. Convert the checkpoint

```sh
python -m cv3cpu convert pretrained_models/Fun-CosyVoice3-0.5B cosyvoice3-balanced.safetensors
```

Reads `llm.pt`, `flow.pt` and `hift.pt` directly — `.pt` files are zip archives with
a pickle index, and `cv3cpu/io/torch_pickle.py` walks that format with an unpickler
that refuses every global except the tensor rebuild helpers. Weight-norm
parametrisations are folded, and each tensor is stored raw or block-quantised.

| profile    | file size | LLM stage | flow stage | hift stage | notes                                   |
| ---------- | --------- | --------- | ---------- | ---------- | --------------------------------------- |
| `balanced` | 935 MB    | 522 MB    | 358 MB     | 50 MB      | default: int8 for the big matmuls        |
| `tiny`     | 703 MB    | 291 MB    | 358 MB     | 50 MB      | 4-bit LLM body — smaller, but ~40 % slower in NumPy (nibble unpacking) and 4-bit costs real quality |
| `quality`  | ~1.7 GB   | —         | —          | —          | float16 throughout, no quantisation error |
| `f32`      | 3.4 GB    | —         | —          | —          | reference, for debugging                  |

Some tensors are never quantised, whatever the profile: norms, biases, Snake alphas
and the F0 predictor stay float32; the whole vocoder and the flow decoder's input
and output edges stay float16. The vocoder is 25 M parameters, so float16 costs
~15 MB more than int8 while the peak stage is the 0.5 B-parameter LLM either way —
and an error there lands straight on the waveform.

**What int8 costs, measured end to end.** Running the same tokens through a
full-size `f32` conversion and a `balanced` one — 22 DiT layers × 10 Euler steps,
then the vocoder — the mel differs by 0.34 % of its own standard deviation and the
waveform by 0.46 % (≈ −47 dB). Quantisation error does not compound through the
solver.

`python -m cv3cpu info model.safetensors` prints the per-stage breakdown.

### The weight file

It is an ordinary `.safetensors` — no custom container. A quantised weight needs
two tensors and some bookkeeping, which the format carries without extension:

* the payload sits under the tensor's own name, as `I8` (int8) or `U8` (packed
  4-bit nibbles);
* its per-block scales sit under `<name>.__scales__`, as `F16`;
* kind, block size and the weight's logical shape — a convolution's `(out, in, k)`
  before it was folded to `(out, in * k)` — live in a JSON blob under the
  `__metadata__` key `cv3cpu`, alongside the model config.

So any safetensors reader opens these files (you just see raw payload and scale
tensors), and files written by the reference implementation load here. The writer
streams: it reserves header space, appends tensors as it goes, and patches the JSON
in at close, padded with spaces so the reserved length is exact and the data
section stays gapless. That is what lets a multi-gigabyte file be written one
tensor at a time without a second pass.

### 2. Enroll a voice (once per speaker)

```sh
pip install onnxruntime
python -m cv3cpu enroll reference.wav alice.npz \
    --text "the exact transcript of reference.wav" \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B
```

This is the only step that needs `onnxruntime`, to run the two encoders that ship
with the checkpoint (`campplus.onnx` for the speaker vector,
`speech_tokenizer_v3.onnx` for the prompt's speech tokens). The result is a few
hundred kB of `.npz`, and synthesis never pays their memory cost.

### 3. Synthesise

```sh
python -m cv3cpu tts "Hello there." \
    --weights cosyvoice3-balanced.safetensors \
    --voice alice.npz \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --out hello.wav --report-memory
```

### Checking the memory claim yourself

```sh
tools/run_capped.sh 2G python -m cv3cpu bench cosyvoice3-balanced.safetensors
```

`run_capped.sh` puts the process in a cgroup with a hard limit that counts page
cache, so the mapped weights have to fit too. Drop the page cache first
(`sync; echo 3 > /proc/sys/vm/drop_caches`) or the limit is easy to fool.

### Without the real checkpoint

```sh
python tools/make_dummy_model.py dummy.safetensors --profile balanced   # 917 MB of noise
python -m cv3cpu bench dummy.safetensors
```

Same geometry and same resource behaviour as the real model; the audio is noise.

## How the memory budget is met

1. **Weights are mmapped, never loaded.** The weight file is a plain
   `.safetensors`, opened `MAP_PRIVATE` read-only. Pages are file-backed, shared
   between processes, and evictable — the kernel can reclaim them instead of
   pushing the process into swap.
2. **A weight is never float for longer than one tile.** `qmatmul` walks the output
   rows of a quantised matrix, expands one ~1 MB tile into a reusable scratch
   buffer and hands that to BLAS. Peak scratch is a megabyte no matter how big the
   weight is. 1 MB was measured ~1.8× faster than an L3-sized tile: the widening
   cast is bound on its output, so the smaller round trip wins.
3. **Stages are strictly sequential and released.** Peak is `max(stage)`, not the
   sum.
4. **Nothing scales with `T²` or with the length of the utterance.** Attention tiles
   over queries; convolutions are im2col'd in time tiles; the vocoder decodes the
   conv stack in windows with left context *and* the 4-frame lookahead `conv_pre`
   needs, which makes chunked decoding bit-identical to whole-utterance decoding
   rather than an approximation.
5. **The embedding table is gathered, not materialised.** 151936 × 896 stays on disk;
   one row per token is dequantised.

## Caveats

* **Not yet run against the released weights.** Every number here comes from a
  correctly shaped model with random or fixture weights. Shapes, memory, speed and
  per-module numerical agreement with the reference implementation are all measured;
  *audio quality with the real checkpoint is not*. Converting
  `Fun-CosyVoice3-0.5B` and listening is the obvious next step.
* Quantisation error is measured numerically (0.6 % per int8 matmul, 0.46 % on the
  final waveform); its effect on *perceived* speech quality is not, for the same
  reason.
* Streaming (`bistream`) synthesis, instructed TTS and the SFT/instruct paths are not
  implemented — this is the zero-shot voice-cloning path only.
* Text normalisation (WeTextProcessing / ttsfrd) is not included. CosyVoice 3 reads
  numbers and symbols without a separate frontend, so only sentence segmentation is
  ported.
* The prompt clip must be ≤ 30 s, as upstream.
* Utterances are capped at 15000 mel frames (5 minutes) per segment by the flow
  decoder's fixed noise buffer, exactly as upstream; longer text is split into
  sentences automatically.

## Layout

```
cv3cpu/
  quant.py          block-wise int8/int4 quantisation and the dequantising GEMM
  ops.py            NumPy kernels: linear, attention, conv1d, norms, STFT/ISTFT
  torch_rng.py      bit-exact replication of PyTorch's CPU MT19937 + normal_fill
  dsp.py            mel spectrogram, Kaldi fbank, whisper log-mel, wav io, resampling
  tokenizer.py      Qwen2 byte-level BPE, including the Unicode pre-tokenizer
  text.py           sentence segmentation
  convert.py        checkpoint -> .safetensors, including weight-norm folding
  pipeline.py       the staged, release-as-you-go runtime
  enroll.py         voice packs (the only place onnxruntime appears)
  io/
    torch_pickle.py  read .pt without torch
    safetensors_io.py
    weights.py       the mmapped weight file (safetensors + a quantisation index)
  models/
    qwen2.py  llm.py  dit.py  flow.py  hift.py
tools/
  make_dummy_model.py       full-size random model for benchmarking
  run_capped.sh             run under a hard cgroup memory cap
  make_reference_fixture.py maintainer-only: regenerate tests/fixtures
tests/
  fixtures/            a small checkpoint plus frozen upstream output
  reference_values.json frozen librosa / torch / regex ground truth
  test_*.py            kernels, DSP, tokenizer, weight io, end to end
```

## Licence

Apache-2.0 (see `LICENSE` and `NOTICE`), matching upstream CosyVoice. This is an
independent reimplementation — no upstream source is vendored — but the algorithms,
tensor layouts and checkpoint key names follow the reference so the released weights
load unchanged. The model weights are not part of this project and are governed by
[their own licence](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512).
