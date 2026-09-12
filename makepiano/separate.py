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


STEM_LABELS = {"none": "分離なし", "other": "伴奏のみ（歌・ドラム・ベース除去）", "piano": "ピアノのみ", "no_vocals": "歌だけ除去"}


def separate_all(wav: Path, out_dir: Path, stems: list[str], device: str | None = None, log=print) -> dict[str, Path]:
    """Produce every requested stem WAV with as few model runs as possible. Returns stem -> wav path
    ("none" maps to the input mix)."""
    import torch
    from demucs.api import Separator, save_audio

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    if "none" in stems:
        result["none"] = wav
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if {"other", "no_vocals"} & set(stems):
        log(f"[separate] demucs htdemucs on {device} (first run downloads the model)")
        sep = Separator(model="htdemucs", device=device, progress=False)
        _, parts = sep.separate_audio_file(str(wav))
        if "other" in stems:
            result["other"] = out_dir / "other.wav"
            save_audio(parts["other"], str(result["other"]), samplerate=sep.samplerate)
        if "no_vocals" in stems:
            result["no_vocals"] = out_dir / "no_vocals.wav"
            save_audio(sum(t for n, t in parts.items() if n != "vocals"), str(result["no_vocals"]), samplerate=sep.samplerate)
        del sep, parts
    if "piano" in stems:
        log(f"[separate] demucs htdemucs_6s on {device} -> piano")
        sep = Separator(model="htdemucs_6s", device=device, progress=False)
        _, parts = sep.separate_audio_file(str(wav))
        result["piano"] = out_dir / "piano.wav"
        save_audio(parts["piano"], str(result["piano"]), samplerate=sep.samplerate)
    log(f"[separate] stems ready: {', '.join(result)}")
    return result
