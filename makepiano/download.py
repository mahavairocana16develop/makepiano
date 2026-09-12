"""Download audio from a YouTube URL using yt-dlp and convert to WAV."""
from __future__ import annotations

import re
from pathlib import Path

import yt_dlp


def sanitize(name: str) -> str:
    name = re.sub(r"[#%&+]+", "", name)                  # break URLs when served by the web UI
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip(" _")  # illegal in file names
    return name[:80] or "untitled"


def _to_wav(src: Path, wav: Path) -> None:
    import subprocess
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-500:]}")


def download_audio(url: str, cache_dir: Path, log=print) -> tuple[Path, str, str]:
    """Return (wav path inside cache_dir, title, source id). Cached by YouTube video id (or local file
    identity), so the same video is only fetched once."""
    import hashlib
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = Path(url).expanduser()
    if local.exists() and local.is_file():
        st = local.stat()
        sid = "local-" + hashlib.sha1(f"{local.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
        wav = cache_dir / f"{sid}.wav"
        if wav.exists():
            log(f"[input] local file {local.name} (cached)")
        else:
            log(f"[input] local file {local.name}")
            _to_wav(local, wav)
        return wav, local.stem, sid

    probe = {"quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(probe) as ydl:
        info = ydl.extract_info(url, download=False)
    sid = info.get("id") or hashlib.sha1(url.encode()).hexdigest()[:12]
    title = info.get("title") or "untitled"
    wav = cache_dir / f"{sid}.wav"
    if wav.exists() and wav.stat().st_size > 1000:
        log(f"[download] {url} -> cached ({wav.name}), skipping download")
        return wav, title, sid
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{sid}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        # 44.1 kHz stereo keeps full band for source separation; transcription resamples to 16 kHz itself.
        "postprocessor_args": {"ffmpegextractaudio": ["-ac", "2", "-ar", "44100"]},
    }
    log(f"[download] {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not wav.exists():
        raise RuntimeError("yt-dlp did not produce a wav file")
    log(f"[download] title={title!r} duration={info.get('duration')}s")
    return wav, title, sid
