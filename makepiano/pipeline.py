"""End-to-end: YouTube URL -> MIDI / MusicXML / SVG / PDF."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .download import download_audio, sanitize
from .render import render_svgs, svgs_to_pdf
from .score import ScoreOptions, midi_to_musicxml
from .separate import separate
from .transcribe import transcribe_to_midi


def _web_audio(src: Path, dst: Path, log=print) -> Path | None:
    """Compress a WAV to AAC for in-browser playback."""
    import subprocess
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "128k", str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[audio] ffmpeg failed: {r.stderr.strip()[-300:]}")
        return None
    return dst


def _engrave_levels(workdir: Path, wav: Path, midi: Path, opts: ScoreOptions, log) -> dict[str, dict]:
    """Engrave every requested difficulty level into workdir/<level>/."""
    levels = LEVELS if opts.level == "both" else (opts.level,)
    out: dict[str, dict] = {}
    for lv in levels:
        log(f"[score] === {lv} ===")
        d = workdir / lv
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        xml, svgs, pdf = _engrave(d, wav, midi, replace(opts, level=lv), log)
        out[lv] = {"dir": d, "musicxml": xml, "svgs": svgs, "pdf": pdf}
    return out


def _engrave(workdir: Path, wav: Path, midi: Path, opts: ScoreOptions, log) -> tuple[Path, list[Path], Path | None]:
    """Score + SVG/PDF + playback.json (timemap in seconds, notes) for one level directory."""
    import json

    import numpy as np

    xml, timing = midi_to_musicxml(midi, wav, workdir / "score.musicxml", opts, log=log)
    tm_path = workdir / "timemap.json"
    svgs = render_svgs(xml, workdir / "svg", log=log, timemap_out=tm_path)
    pdf = svgs_to_pdf(svgs, workdir / "score.pdf", log=log)
    # verovio timemap is in score quarter-notes; convert to audio seconds through the tracked beat grid
    bt = np.array(timing["beat_times"])
    idx = np.arange(len(bt))
    timemap = json.loads(tm_path.read_text())
    for ent in timemap:
        ent["t"] = round(float(np.interp(ent["qstamp"] + timing["bar_start_beat"], idx, bt)), 4)
    tm_path.unlink()
    (workdir / "playback.json").write_text(json.dumps({
        "bpm": timing["bpm"], "notes": timing["notes"], "raw_notes": timing["raw_notes"], "split": timing["split"],
        "bpm_score": round(timing["bpm"]),
        "timemap": [{"t": e["t"], "q": e["qstamp"], "on": e.get("on", []), "off": e.get("off", [])} for e in timemap],
    }), encoding="utf-8")
    return xml, svgs, pdf


LEVELS = ("original", "beginner")


@dataclass
class Result:
    title: str
    workdir: Path
    midi: Path
    musicxml: Path          # of the first level generated (original unless only beginner was requested)
    svgs: list[Path]
    pdf: Path | None
    seconds: float
    log: list[str] = field(default_factory=list)
    levels: dict[str, dict] = field(default_factory=dict)  # level -> {"dir", "musicxml", "svgs", "pdf"}

    def to_json(self) -> dict:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))


def run(url: str, out_root: Path, opts: ScoreOptions | None = None, device: str | None = None,
        keep_audio: bool = True, stem: str = "none", log=print) -> Result:
    opts = opts or ScoreOptions()
    t0 = time.time()
    lines: list[str] = []

    def _log(msg: str):
        lines.append(msg)
        log(msg)

    tmp = out_root / "_tmp"
    wav, title = download_audio(url, tmp, log=_log)
    if opts.title == "Untitled":
        opts.title = title
    workdir = out_root / sanitize(title)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    if stem != "none":
        wav = separate(wav, workdir / f"stem_{stem}.wav", stem=stem, device=device, log=_log)
    midi = transcribe_to_midi(wav, workdir / "transcription.mid", device=device, log=_log)
    levels = _engrave_levels(workdir, wav, midi, opts, _log)
    first = next(iter(levels.values()))
    xml, svgs, pdf = first["musicxml"], first["svgs"], first["pdf"]
    _web_audio(tmp / "source.wav", workdir / "audio_original.m4a", log=_log)
    if stem != "none":
        _web_audio(wav, workdir / "audio_stem.m4a", log=_log)

    if keep_audio:
        if stem != "none":
            shutil.move(str(tmp / "source.wav"), workdir / "source_mix.wav")
            wav.rename(workdir / "source.wav")  # rescore must use the audio that was actually transcribed
        else:
            shutil.move(str(tmp / "source.wav"), workdir / "source.wav")
    elif stem != "none":
        wav.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    dt = time.time() - t0
    _log(f"[done] {dt:.1f}s -> {workdir}")
    return Result(title, workdir, midi, xml, svgs, pdf, dt, lines, levels)


def rescore(workdir: Path, opts: ScoreOptions, log=print) -> Result:
    """Re-run only the scoring/rendering stages on an existing job directory."""
    t0 = time.time()
    lines: list[str] = []

    def _log(msg: str):
        lines.append(msg)
        log(msg)

    midi, wav = workdir / "transcription.mid", workdir / "source.wav"
    if not midi.exists() or not wav.exists():
        raise FileNotFoundError(f"{workdir} needs transcription.mid and source.wav (run without --no-keep-audio)")
    if opts.title == "Untitled":
        opts.title = workdir.name
    levels = _engrave_levels(workdir, wav, midi, opts, _log)
    first = next(iter(levels.values()))
    xml, svgs, pdf = first["musicxml"], first["svgs"], first["pdf"]
    for stale in ("score.musicxml", "score.pdf", "playback.json"):  # pre-levels layout
        (workdir / stale).unlink(missing_ok=True)
    shutil.rmtree(workdir / "svg", ignore_errors=True)
    if not (workdir / "audio_original.m4a").exists():
        mix = workdir / "source_mix.wav"
        _web_audio(mix if mix.exists() else wav, workdir / "audio_original.m4a", log=_log)
        if mix.exists():
            _web_audio(wav, workdir / "audio_stem.m4a", log=_log)
    return Result(opts.title, workdir, midi, xml, svgs, pdf, time.time() - t0, lines, levels)
