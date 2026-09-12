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
from .separate import STEMS, separate_all
from .transcribe import transcribe_to_midi


def _loudness_db(wav: Path) -> float | None:
    """RMS level (dBFS) of the audible part of a file; used to match synth playback volume."""
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
        rms = rms[rms > 1e-4]
        return round(float(20 * np.log10(np.sqrt(np.mean(rms ** 2)) + 1e-9)), 2) if len(rms) else None
    except Exception:  # noqa: BLE001
        return None


def _web_audio(src: Path, dst: Path, log=print) -> Path | None:
    """Compress a WAV to AAC for in-browser playback."""
    import subprocess
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "128k", str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[audio] ffmpeg failed: {r.stderr.strip()[-300:]}")
        return None
    return dst


def _engrave_levels(workdir: Path, wav: Path, midi: Path, opts: ScoreOptions, log, loudness_db: float | None = None) -> dict[str, dict]:
    """Engrave every requested difficulty level into workdir/<level>/."""
    levels = LEVELS if opts.level == "both" else (opts.level,)
    out: dict[str, dict] = {}
    for lv in levels:
        log(f"[score] === {lv} ===")
        d = workdir / lv
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        xml, svgs, pdf = _engrave(d, wav, midi, replace(opts, level=lv), log, loudness_db)
        out[lv] = {"dir": d, "musicxml": xml, "svgs": svgs, "pdf": pdf}
    return out


def _engrave(workdir: Path, wav: Path, midi: Path, opts: ScoreOptions, log, loudness_db: float | None = None) -> tuple[Path, list[Path], Path | None]:
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
        "loudness_db": loudness_db if loudness_db is not None else _loudness_db(wav),  # of the original mix
        "beats_per_bar": timing["beats_per_bar"],
        "bpm_score": round(timing["bpm"]),
        "timemap": [{"t": e["t"], "q": e["qstamp"], "on": e.get("on", []), "off": e.get("off", [])} for e in timemap],
    }), encoding="utf-8")
    return xml, svgs, pdf


LEVELS = ("original", "intermediate", "beginner")


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
    levels: dict[str, dict] = field(default_factory=dict)  # level -> {"dir", "musicxml", "svgs", "pdf"} (first stem)
    stems: dict[str, dict] = field(default_factory=dict)   # stem -> {"dir", "midi", "levels"}

    def to_json(self) -> dict:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))


def _process_stem(sdir: Path, wav: Path, opts: ScoreOptions, device, log, loudness_db: float | None = None) -> dict:
    """Transcribe one stem's audio and engrave all levels into sdir."""
    sdir.mkdir(parents=True, exist_ok=True)
    midi = transcribe_to_midi(wav, sdir / "transcription.mid", device=device, log=log)
    levels = _engrave_levels(sdir, wav, midi, opts, log, loudness_db)
    return {"dir": sdir, "midi": midi, "levels": levels}


def run(url: str, out_root: Path, opts: ScoreOptions | None = None, device: str | None = None,
        keep_audio: bool = True, stem: str | list[str] = "all", log=print) -> Result:
    """stem: "all", one stem name, or a list of stem names (none / other / piano / no_vocals)."""
    opts = opts or ScoreOptions()
    stems = list(STEMS) if stem == "all" else ([stem] if isinstance(stem, str) else list(stem))
    stems = [s_ for s_ in STEMS if s_ in stems] or ["none"]
    t0 = time.time()
    lines: list[str] = []

    def _log(msg: str):
        lines.append(msg)
        log(msg)

    tmp = out_root / "_tmp"
    mix, title = download_audio(url, tmp, log=_log)
    if opts.title == "Untitled":
        opts.title = title
    workdir = out_root / sanitize(title)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.move(str(mix), workdir / "source_mix.wav")
    mix = workdir / "source_mix.wav"
    _web_audio(mix, workdir / "audio_original.m4a", log=_log)
    loudness = _loudness_db(mix)  # synth playback is levelled against the original recording

    stem_wavs = separate_all(mix, workdir / "stems", stems, device=device, log=_log) if stems != ["none"] else {"none": mix}
    results: dict[str, dict] = {}
    for st in stems:
        wav = stem_wavs.get(st)
        if wav is None:
            continue
        _log(f"[stem] ===== {st} =====")
        sdir = workdir / "stems" / st
        try:
            results[st] = _process_stem(sdir, wav, opts, device, _log, loudness)
        except Exception as e:  # noqa: BLE001 - one bad stem should not sink the job
            _log(f"[stem] {st} failed: {e}")
            continue
        if st == "none":
            (sdir / "source.wav").symlink_to(Path("..") / ".." / "source_mix.wav")  # rescore needs audio
        else:
            _web_audio(wav, sdir / "audio.m4a", log=_log)
            if keep_audio:
                shutil.move(str(wav), sdir / "source.wav")
            else:
                wav.unlink(missing_ok=True)
    if not results:
        raise RuntimeError("Every stem failed; see log")
    if not keep_audio:
        mix.unlink(missing_ok=True)
        (workdir / "stems" / "none" / "source.wav").unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    dt = time.time() - t0
    _log(f"[done] {dt:.1f}s -> {workdir}")
    first = next(iter(results.values()))
    lv = next(iter(first["levels"].values()))
    return Result(title, workdir, first["midi"], lv["musicxml"], lv["svgs"], lv["pdf"], dt, lines, first["levels"], results)


def rescore(workdir: Path, opts: ScoreOptions, log=print) -> Result:
    """Re-run only the scoring/rendering stages on an existing job directory."""
    t0 = time.time()
    lines: list[str] = []

    def _log(msg: str):
        lines.append(msg)
        log(msg)

    if opts.title == "Untitled":
        opts.title = workdir.name
    stems_dir = workdir / "stems"
    mix = workdir / "source_mix.wav"
    loudness = _loudness_db(mix) if mix.exists() else None
    if stems_dir.is_dir():  # current layout: stems/<stem>/{source.wav, transcription.mid, <level>/}
        results: dict[str, dict] = {}
        for sdir in sorted(d for d in stems_dir.iterdir() if (d / "transcription.mid").exists()):
            wav = sdir / "source.wav"
            if not wav.exists():
                _log(f"[stem] {sdir.name}: no source.wav, skipped")
                continue
            _log(f"[stem] ===== {sdir.name} =====")
            results[sdir.name] = {"dir": sdir, "midi": sdir / "transcription.mid",
                                  "levels": _engrave_levels(sdir, wav, sdir / "transcription.mid", opts, _log, loudness)}
        first = next(iter(results.values()))
        lv = next(iter(first["levels"].values()))
        return Result(opts.title, workdir, first["midi"], lv["musicxml"], lv["svgs"], lv["pdf"], time.time() - t0, lines,
                      first["levels"], results)
    midi, wav = workdir / "transcription.mid", workdir / "source.wav"
    if not midi.exists() or not wav.exists():
        raise FileNotFoundError(f"{workdir} needs transcription.mid and source.wav (run without --no-keep-audio)")
    levels = _engrave_levels(workdir, wav, midi, opts, _log, loudness)
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
