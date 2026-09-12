"""End-to-end: YouTube URL -> MIDI / MusicXML / SVG / PDF."""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .download import download_audio, sanitize
from .render import render_svgs, svgs_to_pdf
from .score import ScoreOptions, midi_to_musicxml
from .progress import Tracker
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


def _process_stem(sdir: Path, wav: Path, opts: ScoreOptions, device, log, loudness_db: float | None = None,
                  refine: bool = True, tracker: Tracker | None = None) -> dict:
    """Transcribe one stem's audio (unless already transcribed) and engrave all levels into sdir."""
    sdir.mkdir(parents=True, exist_ok=True)
    midi = sdir / "transcription.mid"
    if midi.exists():
        log(f"[transcribe] reusing {sdir.name}/transcription.mid")
    else:
        if tracker:
            tracker.start("transcribe:" + sdir.name)
        transcribe_to_midi(wav, midi, device=device, log=log, refine=refine, tracker=tracker, step_key=sdir.name)
    if tracker:
        tracker.start("engrave:" + sdir.name)
    levels = _engrave_levels(sdir, wav, midi, opts, log, loudness_db)
    return {"dir": sdir, "midi": midi, "levels": levels}


def find_job(out_root: Path, source_id: str) -> Path | None:
    """Existing job directory for this video/file, if any (matched through job.json)."""
    for jf in out_root.glob("*/job.json"):
        try:
            if json.loads(jf.read_text()).get("source_id") == source_id:
                return jf.parent
        except Exception:  # noqa: BLE001
            continue
    return None


def run(url: str, out_root: Path, opts: ScoreOptions | None = None, device: str | None = None,
        keep_audio: bool = True, stem: str | list[str] = "all", force: bool = False, refine: bool = True, log=print,
        progress=None) -> Result:
    """stem: "all", one stem name, or a list of stem names (none / other / piano / no_vocals).
    Downloads are cached by video id and an existing job for the same video is extended in place
    (separation and transcription are reused, scores are always re-engraved) unless force=True."""
    opts = opts or ScoreOptions()
    stems = list(STEMS) if stem == "all" else ([stem] if isinstance(stem, str) else list(stem))
    stems = [s_ for s_ in STEMS if s_ in stems] or ["none"]
    t0 = time.time()
    lines: list[str] = []

    def _log(msg: str):
        lines.append(msg)
        log(msg)

    tracker = Tracker(progress)
    tracker.add("download", "音声を取得", 8)
    tracker.start("download")
    cached, title, source_id = download_audio(url, out_root / "_cache", log=_log)
    import librosa
    dur = float(librosa.get_duration(path=str(cached)))
    if opts.title == "Untitled":
        opts.title = title
    workdir = find_job(out_root, source_id)
    if workdir is not None and force:
        _log(f"[job] --force: discarding existing job {workdir.name}")
        shutil.rmtree(workdir)
        workdir = None
    if workdir is None:
        workdir = out_root / sanitize(title)
        if workdir.exists():  # a folder from an older layout with the same title
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
    else:
        _log(f"[job] reusing existing job {workdir.name}")
    mix = workdir / "source_mix.wav"
    if not mix.exists():
        try:
            os.link(cached, mix)  # same file, no extra space
        except OSError:
            shutil.copy(cached, mix)
    if not (workdir / "audio_original.m4a").exists():
        _web_audio(mix, workdir / "audio_original.m4a", log=_log)
    loudness = _loudness_db(mix)  # synth playback is levelled against the original recording

    have = {st for st in stems if (workdir / "stems" / st / "transcription.mid").exists()}
    need_sep = [st for st in stems if st != "none" and st not in have]
    if have:
        _log(f"[stem] already transcribed: {', '.join(sorted(have))}")
    # Plan the remaining work (seconds per second of audio, measured on Apple Silicon CPU).
    n_levels = 3 if opts.level == "both" else 1
    if {"other", "no_vocals"} & set(need_sep):
        tracker.add("separate", "音源分離（歌・ドラム・ベース）", 0.35 * dur + 2)
    if "piano" in need_sep:
        tracker.add("separate", "音源分離（ピアノ）", 0.35 * dur + 2)
    labels = {"none": "分離なし", "other": "伴奏のみ", "piano": "ピアノのみ", "no_vocals": "歌だけ除去"}
    for st in stems:
        if st not in have:
            tracker.add("transcribe:" + st, f"採譜（{labels[st]}）", 0.2 * dur + 3)
            if refine:
                tracker.add("refine:" + st, f"合成比較で最適化（{labels[st]}）", 0.12 * dur + 2)
        tracker.add("engrave:" + st, f"楽譜化（{labels[st]}）", 1.5 * n_levels + 1)
    if need_sep:
        tracker.start("separate")
    stem_wavs = separate_all(mix, workdir / "stems", need_sep, device=device, log=_log,
                             progress=lambda: tracker.start("separate")) if need_sep else {}
    stem_wavs["none"] = mix
    results: dict[str, dict] = {}
    for st in stems:
        sdir = workdir / "stems" / st
        wav = sdir / "source.wav" if st in have else stem_wavs.get(st)
        if wav is None or not wav.exists():
            _log(f"[stem] {st}: no audio, skipped")
            continue
        _log(f"[stem] ===== {st} =====")
        try:
            results[st] = _process_stem(sdir, wav, opts, device, _log, loudness, refine=refine, tracker=tracker)
        except Exception as e:  # noqa: BLE001 - one bad stem should not sink the job
            _log(f"[stem] {st} failed: {e}")
            continue
        if st in have:
            continue
        if st == "none":
            link = sdir / "source.wav"
            if not link.exists():
                link.symlink_to(Path("..") / ".." / "source_mix.wav")  # rescore needs audio
        else:
            _web_audio(wav, sdir / "audio.m4a", log=_log)
            if keep_audio:
                shutil.move(str(wav), sdir / "source.wav")
            else:
                wav.unlink(missing_ok=True)
    if not results:
        raise RuntimeError("Every stem failed; see log")
    (workdir / "job.json").write_text(json.dumps({
        "source_id": source_id, "url": url, "title": title, "stems": sorted(set(have) | set(results)),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if not keep_audio:
        mix.unlink(missing_ok=True)
        (workdir / "stems" / "none" / "source.wav").unlink(missing_ok=True)
    tracker.finish()
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
