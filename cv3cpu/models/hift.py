"""CausalHiFTGenerator: mel -> 24 kHz waveform (NSF source + iSTFT head)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import ops
from ..io.weights import WeightFile
from ..torch_rng import TorchMT19937


def causal_pad(kernel_size: int, dilation: int = 1) -> int:
    """``CausalConv1d.causal_padding`` from cosyvoice/transformer/convolution.py."""
    return int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2


class HiFTGenerator:
    def __init__(self, weights: WeightFile, prefix: str = "hift."):
        self.w = weights
        self.p = prefix
        m = weights.meta.get("hift", {})
        self.upsample_rates = list(m.get("upsample_rates", [8, 5, 3]))
        self.upsample_kernel_sizes = list(m.get("upsample_kernel_sizes", [16, 11, 7]))
        self.resblock_kernel_sizes = list(m.get("resblock_kernel_sizes", [3, 7, 11]))
        self.resblock_dilation_sizes = [list(d) for d in
                                        m.get("resblock_dilation_sizes", [[1, 3, 5]] * 3)]
        self.source_resblock_kernel_sizes = list(m.get("source_resblock_kernel_sizes", [7, 7, 11]))
        self.source_resblock_dilation_sizes = [list(d) for d in
                                               m.get("source_resblock_dilation_sizes", [[1, 3, 5]] * 3)]
        self.n_fft = int(m.get("istft_n_fft", 16))
        self.hop_len = int(m.get("istft_hop_len", 4))
        self.lrelu_slope = float(m.get("lrelu_slope", 0.1))
        self.audio_limit = float(m.get("audio_limit", 0.99))
        self.sampling_rate = int(m.get("sampling_rate", 24000))
        self.nb_harmonics = int(m.get("nb_harmonics", 8))
        self.nsf_alpha = float(m.get("nsf_alpha", 0.1))
        self.nsf_sigma = float(m.get("nsf_sigma", 0.003))
        self.nsf_voiced_threshold = float(m.get("nsf_voiced_threshold", 10.0))
        self.conv_pre_look_right = int(m.get("conv_pre_look_right", 4))
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)
        self.upsample_scale = int(np.prod(self.upsample_rates)) * self.hop_len
        self.window = ops.hann_window(self.n_fft)
        self._src_noise: Optional[np.ndarray] = None
        self.noise_seed = int(m.get("noise_seed", 0))

    def _t(self, name: str):
        return self.w.get(self.p + name)

    def _k(self, name: str) -> int:
        return int(self.w.entry(self.p + name)["shape"][-1])

    # -- F0 ---------------------------------------------------------------
    def f0_predict(self, mel: np.ndarray) -> np.ndarray:
        """``(80, T)`` mel -> ``(T,)`` F0.  Run in float64, as upstream does."""
        x = np.asarray(mel, dtype=np.float64)
        n = 5
        for i in range(n):
            w = np.asarray(self.w.get_f32(self.p + f"f0_predictor.condnet.{i}.weight"),
                           dtype=np.float64)
            b = np.asarray(self.w.get_f32(self.p + f"f0_predictor.condnet.{i}.bias"),
                           dtype=np.float64)
            k = w.shape[-1]
            pad = (0, causal_pad(k)) if i == 0 else (causal_pad(k), 0)
            x = ops.conv1d(x, w, b, pad=pad)
            x = ops.elu(x)
        cw = np.asarray(self.w.get_f32(self.p + "f0_predictor.classifier.weight"), dtype=np.float64)
        cb = np.asarray(self.w.get_f32(self.p + "f0_predictor.classifier.bias"), dtype=np.float64)
        y = x.T @ cw.T + cb
        return np.abs(y[:, 0])

    # -- NSF source -------------------------------------------------------
    def source_noise(self, n: int, harmonics: int) -> np.ndarray:
        """``SineGen2.sine_waves`` — a fixed uniform buffer, generated lazily."""
        if self._src_noise is None or self._src_noise.shape[0] < n:
            g = TorchMT19937(self.noise_seed)
            need = max(n, 1)
            self._src_noise = g.random_float32(need * harmonics).reshape(need, harmonics)
        return self._src_noise[:n]

    def set_source_noise(self, buf: np.ndarray) -> None:
        """Use a caller-supplied dither buffer instead of the seeded one.

        The tests use this to line up with the buffer the reference vocoder drew.
        """
        self._src_noise = np.asarray(buf, dtype=np.float32)

    def excitation(self, f0: np.ndarray) -> np.ndarray:
        """``(T,)`` F0 -> ``(480 * T,)`` merged harmonic source."""
        t = f0.shape[0]
        up = self.upsample_scale
        harm = self.nb_harmonics + 1
        # rad is the per-sample phase increment; the reference upsamples f0 to
        # sample rate, takes it mod 1, then immediately decimates back by `up`
        # with a linear interpolation that lands strictly inside each held
        # block -- so this is algebraically the per-frame value.
        rad = (f0[:, None] * np.arange(1, harm + 1)[None, :] / self.sampling_rate) % 1.0
        phase = np.cumsum(rad, axis=0) * (2.0 * np.pi) * up          # (T, harm)
        sines = np.sin(np.repeat(phase, up, axis=0)).astype(np.float32)  # (up*T, harm)
        sines *= self.nsf_alpha

        uv = (np.repeat(f0, up) > self.nsf_voiced_threshold).astype(np.float32)[:, None]
        noise_amp = uv * self.nsf_sigma + (1.0 - uv) * (self.nsf_alpha / 3.0)
        noise = noise_amp * self.source_noise(up * t, harm)
        sine_waves = sines * uv + noise

        merged = ops.linear(sine_waves, self._t("m_source.l_linear.weight"),
                            self._t("m_source.l_linear.bias"))
        return np.tanh(merged)[:, 0]

    # -- decoder ----------------------------------------------------------
    def _resblock(self, group: str, idx: int, x: np.ndarray,
                  kernel: int, dilations: Sequence[int]) -> np.ndarray:
        base = f"{group}.{idx}."
        for j, d in enumerate(dilations):
            a1 = self._t(base + f"activations1.{j}.alpha")
            a2 = self._t(base + f"activations2.{j}.alpha")
            xt = ops.snake(x, np.asarray(a1, dtype=np.float32))
            xt = ops.conv1d(xt, self._t(base + f"convs1.{j}.weight"),
                            self._t(base + f"convs1.{j}.bias"),
                            dilation=d, pad=(causal_pad(kernel, d), 0))
            xt = ops.snake(xt, np.asarray(a2, dtype=np.float32))
            xt = ops.conv1d(xt, self._t(base + f"convs2.{j}.weight"),
                            self._t(base + f"convs2.{j}.bias"),
                            dilation=1, pad=(causal_pad(kernel, 1), 0))
            x = xt + x
        return x

    def _spectral(self, mel: np.ndarray, s_stft: np.ndarray) -> np.ndarray:
        """Run conv_pre .. conv_post; ``(80, T)`` -> ``(n_fft + 2, 120 * T + 1)``."""
        x = ops.conv1d(mel, self._t("conv_pre.weight"), self._t("conv_pre.bias"),
                       pad=(0, causal_pad(self._k("conv_pre.weight"))))
        down_rates = [1] + self.upsample_rates[::-1][:-1]
        down_cum = list(np.cumprod(down_rates))[::-1]
        for i in range(self.num_upsamples):
            x = ops.leaky_relu(x, self.lrelu_slope)
            x = ops.interpolate_nearest(x, self.upsample_rates[i])
            k = self._k(f"ups.{i}.weight")
            x = ops.conv1d(x, self._t(f"ups.{i}.weight"), self._t(f"ups.{i}.bias"),
                           pad=(k - 1, 0))
            if i == self.num_upsamples - 1:
                x = np.concatenate([x[:, 1:2], x], axis=1)  # ReflectionPad1d((1, 0))
            u = int(down_cum[i])
            ks = self._k(f"source_downs.{i}.weight")
            if u == 1:
                si = ops.conv1d(s_stft, self._t(f"source_downs.{i}.weight"),
                                self._t(f"source_downs.{i}.bias"),
                                pad=(causal_pad(ks), 0))
            else:
                si = ops.conv1d(s_stft, self._t(f"source_downs.{i}.weight"),
                                self._t(f"source_downs.{i}.bias"),
                                stride=u, pad=(u - 1, 0))
            si = self._resblock("source_resblocks", i, si,
                                self.source_resblock_kernel_sizes[i],
                                self.source_resblock_dilation_sizes[i])
            x = x + si
            acc = None
            for j in range(self.num_kernels):
                r = self._resblock("resblocks", i * self.num_kernels + j, x,
                                   self.resblock_kernel_sizes[j],
                                   self.resblock_dilation_sizes[j])
                acc = r if acc is None else acc + r
            x = acc / self.num_kernels
        x = ops.leaky_relu(x, 0.01)
        return ops.conv1d(x, self._t("conv_post.weight"), self._t("conv_post.bias"),
                          pad=(causal_pad(self._k("conv_post.weight")), 0))

    def _spectral_chunked(self, mel: np.ndarray, s_stft: np.ndarray,
                          chunk: int, context: int) -> np.ndarray:
        """Same output as :meth:`_spectral`, with memory bounded by ``chunk``.

        Every convolution in the stack is causal except ``conv_pre``, which looks
        ``conv_pre_look_right`` mel frames ahead.  So a window that carries
        ``context`` frames of history *and* that lookahead of future reproduces
        the full-utterance result exactly, once the warm-up is dropped.
        """
        up = int(np.prod(self.upsample_rates))
        t = mel.shape[1]
        look = causal_pad(self._k("conv_pre.weight"))
        pieces = []
        for m0 in range(0, t, chunk):
            m1 = min(t, m0 + chunk)
            c0 = max(0, m0 - context)
            c1 = min(t, m1 + look)
            sub = self._spectral(mel[:, c0:c1], s_stft[:, c0 * up: c1 * up + 1])
            # `sub` holds 1 + (c1 - c0) * up frames; index 0 is the extra frame
            # the reflection pad prepends, and mel frame c0 + i lands at 1 + i*up.
            lead = (m0 - c0) * up
            take = (m1 - m0) * up
            pieces.append(sub[:, : 1 + take] if m0 == 0 else
                          sub[:, 1 + lead: 1 + lead + take])
        return np.concatenate(pieces, axis=1)

    def decode(self, mel: np.ndarray, source: np.ndarray,
               chunk: Optional[int] = None, context: int = 64) -> np.ndarray:
        spec = ops.stft(source, self.n_fft, self.hop_len, self.window)
        s_stft = np.concatenate([spec.real, spec.imag], axis=0).astype(np.float32)
        if chunk is None or mel.shape[1] <= chunk:
            x = self._spectral(mel, s_stft)
        else:
            x = self._spectral_chunked(mel, s_stft, chunk, context)
        half = self.n_fft // 2 + 1
        # upstream: torch.exp(x), then clip(max=1e2) inside _istft
        magnitude = np.clip(np.exp(np.minimum(x[:half], 80.0)), None, 1e2)
        phase = np.sin(x[half:])
        y = ops.istft((magnitude * np.cos(phase) + 1j * magnitude * np.sin(phase)),
                      self.n_fft, self.hop_len, self.window)
        return np.clip(y, -self.audio_limit, self.audio_limit)

    def inference(self, mel: np.ndarray, chunk: Optional[int] = 250,
                  context: int = 64) -> np.ndarray:
        """``(80, T)`` mel -> ``(480 * T,)`` waveform."""
        mel = np.ascontiguousarray(mel, dtype=np.float32)
        f0 = self.f0_predict(mel)
        source = self.excitation(f0.astype(np.float32))
        return self.decode(mel, source, chunk=chunk, context=context)
