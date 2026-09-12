"""CLI:
  uv run makepiano <youtube-url> [options]        download + transcribe + engrave
  uv run makepiano rescore <output-dir> [options] re-engrave an existing job with different options
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import rescore, run
from .score import ScoreOptions


def _add_score_opts(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--bpm", type=float, default=None, help="fixed tempo instead of automatic beat tracking")
    ap.add_argument("--grid", type=int, default=4, help="subdivisions per beat: 2=8th, 4=16th (default), 3=triplets")
    ap.add_argument("--split", type=int, default=60, help="MIDI pitch splitting hands (default 60 = middle C)")
    ap.add_argument("--min-velocity", type=int, default=25, help="drop notes quieter than this (0-127)")
    ap.add_argument("--max-pitch", type=int, default=96, help="drop notes above this MIDI pitch (default 96 = C7)")
    ap.add_argument("--min-note-ms", type=int, default=60, help="drop notes shorter than this (transcription noise)")
    ap.add_argument("--no-legato", action="store_true", help="write measured note lengths instead of holding to the next chord")
    ap.add_argument("--level", default="both", choices=["both", "original", "beginner"],
                    help="which arrangements to engrave (default both); beginner = melody + chord-root bass")
    ap.add_argument("--no-chords", action="store_true", help="do not add chord symbols")
    ap.add_argument("--beat-offset", type=int, default=0, help="shift bar lines by N beats")
    ap.add_argument("--beats-per-bar", type=int, default=4, help="time signature numerator (x/4)")
    ap.add_argument("--title", default="Untitled", help="score title (default: video title)")


def _opts(a) -> ScoreOptions:
    return ScoreOptions(grid=a.grid, split_pitch=a.split, min_velocity=a.min_velocity, max_pitch=a.max_pitch, min_note_ms=a.min_note_ms,
                        fixed_bpm=a.bpm, beat_offset=a.beat_offset, beats_per_bar=a.beats_per_bar,
                        legato=not a.no_legato, chords=not a.no_chords, level=a.level, title=a.title)


def _report(r) -> None:
    print(f"\nMIDI:     {r.midi}")
    for lv, d in r.levels.items():
        print(f"[{lv}]\n  MusicXML: {d['musicxml']}\n  SVG:      {d['svgs'][0].parent} ({len(d['svgs'])} pages)\n  PDF:      {d['pdf']}")


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "rescore":
        ap = argparse.ArgumentParser(prog="makepiano rescore", description="re-engrave an existing job directory")
        ap.add_argument("dir", help="job directory containing transcription.mid and source.wav")
        _add_score_opts(ap)
        a = ap.parse_args(argv[1:])
        _report(rescore(Path(a.dir), _opts(a)))
        return 0

    ap = argparse.ArgumentParser(prog="makepiano", description="YouTube URL -> piano sheet music")
    ap.add_argument("url", help="YouTube URL (any URL yt-dlp supports) or a local audio/video file")
    ap.add_argument("-o", "--out", default="output", help="output directory (default: ./output)")
    _add_score_opts(ap)
    ap.add_argument("--stem", default="none", choices=["none", "piano", "other", "no_vocals"],
                    help="source-separate first: piano (piano only), other (no vocals/drums/bass), no_vocals")
    ap.add_argument("--device", default=None, help="torch device: cpu / cuda")
    ap.add_argument("--no-keep-audio", action="store_true", help="delete the downloaded WAV (disables rescore)")
    a = ap.parse_args(argv)
    _report(run(a.url, Path(a.out), _opts(a), device=a.device, keep_audio=not a.no_keep_audio, stem=a.stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
