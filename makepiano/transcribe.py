"""Audio -> MIDI using ByteDance's piano transcription model."""
from __future__ import annotations

from pathlib import Path

import torch
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate


CHECKPOINT_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
CHECKPOINT = Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"


def ensure_checkpoint(log=print) -> Path:
    """The upstream package shells out to wget, which macOS lacks. Download with urllib instead."""
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 100_000_000:
        return CHECKPOINT
    import shutil
    import subprocess
    import urllib.request
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".part")
    log("[transcribe] downloading model checkpoint (~172 MB, first run only)")
    if shutil.which("curl"):
        # Zenodo drops connections often: resume (-C -) and retry.
        for _ in range(20):
            r = subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-all-errors", "-sS",
                                "-o", str(tmp), CHECKPOINT_URL])
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 170_000_000:
                break
        else:
            raise RuntimeError("Failed to download the transcription model checkpoint")
    else:
        urllib.request.urlretrieve(CHECKPOINT_URL, tmp)
    tmp.rename(CHECKPOINT)
    return CHECKPOINT


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # MPS is not reliably supported by this model's ops


def transcribe_to_midi(wav: Path, midi_out: Path, device: str | None = None, log=print) -> Path:
    device = pick_device(device)
    log(f"[transcribe] loading audio {wav.name}")
    audio, _ = librosa.load(str(wav), sr=sample_rate, mono=True)
    log(f"[transcribe] running model on {device} ({len(audio) / sample_rate:.1f}s of audio)")
    ckpt = ensure_checkpoint(log=log)
    model = PianoTranscription(device=device, checkpoint_path=str(ckpt))
    model.transcribe(audio, str(midi_out))
    log(f"[transcribe] wrote {midi_out.name}")
    return midi_out
