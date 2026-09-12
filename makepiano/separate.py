"""Source separation with Demucs: keep only the piano / accompaniment before transcribing."""
from __future__ import annotations

from pathlib import Path

STEMS = ("none", "piano", "other", "no_vocals")


def separate(wav: Path, out_wav: Path, stem: str = "other", device: str | None = None, log=print) -> Path:
    """Write a WAV containing only the requested stem.

    piano     - htdemucs_6s "piano" stem (piano only; the 6-source model is less polished)
    other     - htdemucs "other" stem (everything except vocals/drums/bass; usually best for piano transcription)
    no_vocals - all stems except vocals
    """
    if stem == "none":
        return wav
    if stem not in STEMS:
        raise ValueError(f"stem must be one of {STEMS}")
    import torch
    from demucs.api import Separator, save_audio

    model = "htdemucs_6s" if stem == "piano" else "htdemucs"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[separate] demucs {model} on {device} -> {stem} (first run downloads the model)")
    sep = Separator(model=model, device=device, progress=False)
    _, stems = sep.separate_audio_file(str(wav))
    if stem == "no_vocals":
        mix = sum(t for name, t in stems.items() if name != "vocals")
    else:
        mix = stems[stem]
    save_audio(mix, str(out_wav), samplerate=sep.samplerate)
    log(f"[separate] wrote {out_wav.name}")
    return out_wav
