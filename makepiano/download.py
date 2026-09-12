"""Download audio from a YouTube URL using yt-dlp and convert to WAV."""
from __future__ import annotations

import re
from pathlib import Path

import yt_dlp


def sanitize(name: str) -> str:
    name = re.sub(r"[#%&+]+", "", name)                  # break URLs when served by the web UI
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip(" _")  # illegal in file names
    return name[:80] or "untitled"


def _local_to_wav(src: Path, workdir: Path, log=print) -> tuple[Path, str]:
    """Convert a local audio/video file (screen recording, mp4, m4a, ...) with ffmpeg."""
    import subprocess
    wav = workdir / "source.wav"
    log(f"[input] local file {src.name}")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-500:]}")
    return wav, src.stem


def download_audio(url: str, workdir: Path, log=print) -> tuple[Path, str]:
    """Return (path to wav, video title). `url` may also be a path to a local media file."""
    workdir.mkdir(parents=True, exist_ok=True)
    local = Path(url).expanduser()
    if local.exists() and local.is_file():
        return _local_to_wav(local, workdir, log=log)
    out_tmpl = str(workdir / "source.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
        # 44.1 kHz stereo keeps full band for source separation; transcription resamples to 16 kHz itself.
        "postprocessor_args": {"ffmpegextractaudio": ["-ac", "2", "-ar", "44100"]},
    }
    log(f"[download] {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    title = info.get("title") or "untitled"
    wav = workdir / "source.wav"
    if not wav.exists():
        raise RuntimeError("yt-dlp did not produce source.wav")
    log(f"[download] title={title!r} duration={info.get('duration')}s")
    return wav, title
