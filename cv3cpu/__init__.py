"""PyTorch-free, low-memory CosyVoice 3 inference on CPU.

    from cv3cpu import CosyVoice3CPU, VoicePack

    tts = CosyVoice3CPU("cosyvoice3-balanced.safetensors", tokenizer_dir="…/CosyVoice-BlankEN")
    out = tts.synthesize("Hello there.", VoicePack.load("alice.npz"))
    tts.save(out, "hello.wav")

Only NumPy is required at synthesis time.  ``onnxruntime`` is needed once per
speaker, to enroll a voice pack.
"""

__version__ = "0.1.0"

__all__ = ["CosyVoice3CPU", "VoicePack", "SynthesisResult", "SamplingConfig"]


def __getattr__(name):
    if name in ("CosyVoice3CPU", "SynthesisResult"):
        from .pipeline import CosyVoice3CPU, SynthesisResult
        return {"CosyVoice3CPU": CosyVoice3CPU, "SynthesisResult": SynthesisResult}[name]
    if name == "VoicePack":
        from .voice import VoicePack
        return VoicePack
    if name == "SamplingConfig":
        from .models.llm import SamplingConfig
        return SamplingConfig
    raise AttributeError(name)
