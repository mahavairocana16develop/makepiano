"""MIDI (+ audio for beat tracking) -> two-hand piano score (music21 / MusicXML)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import librosa
import mido
import numpy as np
from music21 import chord, clef, harmony, key, layout, metadata, meter, note, pitch, stream


@dataclass
class NoteEvent:
    pitch: int
    start: float  # seconds
    end: float
    velocity: int


@dataclass
class ScoreOptions:
    grid: int = 4            # subdivisions per beat (4 = 16th notes)
    split_pitch: int = 60    # notes below this go to the left hand
    min_velocity: int = 25   # drop very quiet (probably spurious) notes
    max_pitch: int = 96      # drop notes above this (C7): usually vocal/cymbal bleed
    min_note_ms: int = 60    # drop blips shorter than this (transcription noise)
    fixed_bpm: float | None = None  # skip beat tracking and use this tempo
    beat_offset: int = 0     # shift downbeat by N beats (fixes bar alignment)
    beats_per_bar: int = 4
    legato: bool = True      # hold each note/chord until the next onset in the same hand ("pop" style)
    max_hold_beats: float = 2.0  # ...but never stretch a note by more than this
    chords: bool = True      # add chord symbols above the treble staff
    level: str = "original"  # "original" | "beginner" (melody + chord-root bass, 8th-note grid)
    title: str = "Untitled"


# ---------------------------------------------------------------- MIDI reading

def read_midi_notes(midi_path: Path) -> list[NoteEvent]:
    mid = mido.MidiFile(str(midi_path))
    events: list[NoteEvent] = []
    active: dict[int, tuple[float, int]] = {}
    t = 0.0
    for msg in mid:  # iterating a MidiFile yields absolute-time-aware messages (delta in seconds)
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            if msg.note in active:
                s, v = active.pop(msg.note)
                events.append(NoteEvent(msg.note, s, t, v))
    for p, (s, v) in active.items():
        events.append(NoteEvent(p, s, s + 0.5, v))
    events.sort(key=lambda e: (e.start, e.pitch))
    return events


def read_pedal(midi_path: Path) -> list[tuple[float, bool]]:
    """Sustain-pedal (CC64) changes as (time_s, is_down)."""
    mid = mido.MidiFile(str(midi_path))
    out: list[tuple[float, bool]] = []
    t = 0.0
    for msg in mid:
        t += msg.time
        if msg.type == "control_change" and msg.control == 64:
            down = msg.value >= 64
            if not out or out[-1][1] != down:
                out.append((t, down))
    return out


def apply_pedal(events: list[NoteEvent], pedal: list[tuple[float, bool]], max_extend: float = 4.0) -> list[NoteEvent]:
    """Extend each note to the next pedal release if the pedal is down when the key is released."""
    if not pedal:
        return events
    times = np.array([t for t, _ in pedal])
    out = []
    for e in events:
        i = int(np.searchsorted(times, e.end, side="right")) - 1
        end = e.end
        if i >= 0 and pedal[i][1]:  # pedal down at key release
            j = i + 1
            while j < len(pedal) and pedal[j][1]:
                j += 1
            release = pedal[j][0] if j < len(pedal) else e.end + max_extend
            end = min(max(e.end, release), e.end + max_extend)
        out.append(NoteEvent(e.pitch, e.start, end, e.velocity))
    return out


def filter_events(events: list[NoteEvent], opts: "ScoreOptions") -> list[NoteEvent]:
    return [e for e in events if e.velocity >= opts.min_velocity and e.pitch <= opts.max_pitch
            and (e.end - e.start) * 1000 >= opts.min_note_ms]


# ---------------------------------------------------------------- beat tracking

def track_beats(wav: Path, fixed_bpm: float | None, duration_hint: float, log=print) -> tuple[np.ndarray, float]:
    """Return (beat_times in seconds, bpm). beat_times is a strictly increasing grid."""
    if fixed_bpm:
        bpm = fixed_bpm
        beat_times = np.arange(0, duration_hint + 4 * 60 / bpm, 60 / bpm)
        return beat_times, bpm
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", start_bpm=100)
    bpm = float(np.atleast_1d(bpm)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # Beat trackers often lock onto 2x/4x the perceived tempo: fold into a sane range.
    while bpm > 150 and len(beat_times) > 8:
        beat_times = beat_times[::2]
        bpm /= 2
    while bpm < 55:
        bpm *= 2
        beat_times = np.sort(np.concatenate([beat_times, (beat_times[:-1] + beat_times[1:]) / 2]))
    if len(beat_times) < 4:
        bpm = bpm or 100.0
        beat_times = np.arange(0, duration_hint + 4 * 60 / bpm, 60 / bpm)
        log(f"[beats] beat tracking failed, falling back to fixed {bpm:.0f} bpm")
        return beat_times, bpm
    period = 60 / bpm
    # Extend the grid backwards to 0 and forwards past the last note.
    pre = np.arange(beat_times[0] - period, -period / 2, -period)[::-1]
    post = np.arange(beat_times[-1] + period, duration_hint + 4 * period, period)
    beat_times = np.concatenate([pre, beat_times, post])
    beat_times = beat_times[beat_times >= -1e-6]
    log(f"[beats] estimated {bpm:.1f} bpm, {len(beat_times)} beats")
    return beat_times, bpm


def seconds_to_beats(times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    idx = np.arange(len(beat_times), dtype=float)
    return np.interp(times, beat_times, idx)


# ---------------------------------------------------------------- score building

_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")


def _strip_emoji(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _EMOJI_RE.sub("", text)).strip()


def _respell(pt: pitch.Pitch, k: key.Key) -> None:
    """Prefer flats in flat keys and sharps in sharp keys; keep diatonic spellings."""
    if pt.accidental is None or pt.accidental.alter == 0:
        return
    prefer_flats = k.sharps < 0
    is_sharp = pt.accidental.alter > 0
    if (prefer_flats and is_sharp) or (not prefer_flats and not is_sharp):
        pt.getEnharmonic(inPlace=True)
    if pt.accidental is not None:
        pt.accidental.displayStatus = None  # let makeAccidentals decide against the key signature


_CHORD_TEMPLATES: list[tuple[str, tuple[int, ...]]] = [
    ("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10)), ("m7", (0, 3, 7, 10)), ("maj7", (0, 4, 7, 11)),
    ("dim", (0, 3, 6)), ("m7b5", (0, 3, 6, 10)), ("sus4", (0, 5, 7)), ("aug", (0, 4, 8)), ("6", (0, 4, 7, 9)),
]
_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NAMES_FLAT = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


def _chord_symbols(notes: list[tuple[float, float, list[int]]], bars: int, beats_per_bar: int, prefer_flats: bool,
                   window: float = 2.0) -> list[tuple[float, str]]:
    """Pick the best-matching chord for each half-bar window from pitch-class weights (duration-weighted,
    bass note emphasised). notes: (offset_beats, duration_beats, midis). Emit a symbol only when it changes."""
    out: list[tuple[float, str]] = []
    last = None
    total = bars * beats_per_bar
    w0 = 0.0
    while w0 < total:
        w1 = w0 + window
        weights = np.zeros(12)
        bass = None
        for off, ql, midis in notes:
            ov = min(off + ql, w1) - max(off, w0)
            if ov <= 0:
                continue
            for m in midis:
                weights[m % 12] += ov
                if bass is None or m < bass[0]:
                    bass = (m, ov)
        if weights.sum() < 0.5:
            w0 = w1
            continue
        if bass is not None:
            weights[bass[0] % 12] += 1.0
        best = None
        for root in range(12):
            for name, tpl in _CHORD_TEMPLATES:
                pcs = {(root + i) % 12 for i in tpl}
                inside = sum(weights[pc] for pc in pcs)
                outside = weights.sum() - inside
                score_ = inside - 0.7 * outside - 0.15 * len(tpl)  # prefer simpler chords when tied
                if bass is not None and bass[0] % 12 == root:
                    score_ += 0.5
                if best is None or score_ > best[0]:
                    best = (score_, root, name, inside / weights.sum())
        _, root, name, cover = best
        if cover >= 0.6:
            fig = (_NAMES_FLAT if prefer_flats else _NAMES_SHARP)[root] + name
            if fig != last:
                out.append((w0, fig))
                last = fig
        w0 = w1
    return out


_ROOT_PC = {n: i for i, n in enumerate(_NAMES_SHARP)} | {n: i for i, n in enumerate(_NAMES_FLAT)}


def _root_pc(fig: str) -> int:
    return _ROOT_PC[fig[:2]] if len(fig) > 1 and fig[1] in "#-" else _ROOT_PC[fig[0]]


Event = tuple[Fraction, Fraction, list[int], int]  # onset_beats, duration_beats, midis, velocity


def _estimate_key(notes: list[tuple[float, float, list[int]]], log=print) -> key.Key:
    st = stream.Stream()
    for off, dur, midis in notes:
        for m in midis:
            n = note.Note(pitch.Pitch(midi=m))
            n.quarterLength = dur
            st.insert(off, n)
    try:
        k = st.analyze("key")
        log(f"[score] key: {k.name}")
    except Exception:  # noqa: BLE001
        k = key.Key("C")
    if k.sharps >= 6:  # F# / C# major etc.: prefer the enharmonic flat key (Gb / Db)
        k = key.Key(k.tonic.getEnharmonic(), k.mode)
        log(f"[score] respelled key as {k.name}")
    return k


def _simplify_beginner(evs: dict[str, list[Event]], symbols: list[tuple[float, str]], bars: int,
                       beats_per_bar: int, window: float = 2.0) -> dict[str, list[Event]]:
    """Beginner arrangement: right hand = top-line melody only, left hand = chord root every half bar."""
    # Right hand: keep the highest note of each chord; fold anything above C6 down an octave.
    rh: list[Event] = []
    for on, dur, midis, vel in evs["rh"]:
        m = max(midis)
        while m > 84:
            m -= 12
        rh.append((on, dur, [m], vel))
    # Left hand: root of the current chord symbol in the bass register (E2..G3), held for the window.
    lh: list[Event] = []
    sym_at = sorted(symbols)
    lowest_lh = {}
    for on, dur, midis, vel in evs["lh"]:
        w = float(on) // window
        lowest_lh[w] = min(lowest_lh.get(w, 127), min(midis))
    total = bars * beats_per_bar
    w0 = 0.0
    cur = None
    while w0 < total:
        while sym_at and sym_at[0][0] <= w0 + 1e-6:
            cur = sym_at.pop(0)[1]
        if cur is not None:
            pc = _root_pc(cur)
            m = 36 + ((pc - 0) % 12)  # C2..B2
            if m < 40:
                m += 12                # keep within E2..D#3
        elif (w0 // window) in lowest_lh:
            m = lowest_lh[w0 // window]
            while m < 40:
                m += 12
            while m > 55:
                m -= 12
        else:
            w0 += window
            continue
        lh.append((Fraction(w0).limit_denominator(8), Fraction(window).limit_denominator(8), [m], 80))
        w0 += window
    return {"rh": rh, "lh": lh}


def _beat_chunks(start: Fraction, end: Fraction) -> list[Fraction]:
    """Split [start, end) into rest durations aligned to beats: partial beat, whole beats, partial beat."""
    out: list[Fraction] = []
    pos = start
    nxt = Fraction(int(pos) + 1) if pos != int(pos) else pos
    if nxt > pos:
        out.append(min(nxt, end) - pos)
        pos = min(nxt, end)
    whole = int(end - pos)
    while whole > 0:
        take = 4 if whole >= 4 else 2 if whole >= 2 else 1
        # avoid a 2-beat rest that straddles an odd beat boundary (e.g. beats 2-3 in 4/4)
        if take == 2 and int(pos) % 2 == 1:
            take = 1
        out.append(Fraction(take))
        pos += take
        whole -= take
    if end > pos:
        out.append(end - pos)
    return out


def _fill_rests(m: stream.Measure, beats_per_bar: int) -> None:
    """Fill silence in a measure with rests whose durations sit on the beat grid."""
    bar_len = Fraction(beats_per_bar)
    events = sorted(((Fraction(n.offset).limit_denominator(64), Fraction(n.quarterLength).limit_denominator(64))
                     for n in m.notesAndRests), key=lambda x: x[0])
    pos = Fraction(0)
    gaps: list[tuple[Fraction, Fraction]] = []
    for off, ql in events:
        if off > pos:
            gaps.append((pos, off))
        pos = max(pos, off + ql)
    if pos < bar_len:
        gaps.append((pos, bar_len))
    if not events:
        r = note.Rest()
        r.quarterLength = float(bar_len)
        m.insert(0, r)
        return
    for g0, g1 in gaps:
        at = g0
        for d in _beat_chunks(g0, g1):
            r = note.Rest()
            r.quarterLength = float(d)
            m.insert(float(at), r)
            at += d

def _quantize(x: float, grid: int) -> Fraction:
    return Fraction(round(x * grid), grid)


def build_score(events: list[NoteEvent], beat_times: np.ndarray, bpm: float, opts: ScoreOptions, log=print
                ) -> tuple[stream.Score, int, list[tuple[float, float, int, int]]]:
    """Returns (score, bar_start_beat, score_notes). bar_start_beat is the beat index of the score's first
    bar within beat_times; score_notes are the notated events as (start_beat, end_beat, pitch, velocity)."""
    g = opts.grid
    events = filter_events(events, opts)
    if not events:
        raise RuntimeError("No notes left after filtering; lower --min-velocity")
    starts = seconds_to_beats(np.array([e.start for e in events]), beat_times)
    ends = seconds_to_beats(np.array([e.end for e in events]), beat_times)

    # Drop leading silence: start the score at the bar containing the first note.
    first_beat = int(np.floor(starts.min())) if len(starts) else 0
    bar_start = (first_beat // opts.beats_per_bar) * opts.beats_per_bar + opts.beat_offset
    starts = starts - bar_start
    ends = ends - bar_start

    hands: dict[str, dict[Fraction, list[tuple[int, Fraction, int]]]] = {"rh": {}, "lh": {}}
    for e, s_, en in zip(events, starts, ends):
        qs = _quantize(float(s_), g)
        qe = max(_quantize(float(en), g), qs + Fraction(1, g))
        if qs < 0:
            continue
        hand = "rh" if e.pitch >= opts.split_pitch else "lh"
        hands[hand].setdefault(qs, []).append((e.pitch, qe - qs, e.velocity))

    # Collapse each onset into one note/chord event with a single duration (monophonic-chord stream per hand).
    evs: dict[str, list[Event]] = {"rh": [], "lh": []}
    for hand in ("rh", "lh"):
        onsets = sorted(hands[hand])
        for i, on in enumerate(onsets):
            group = hands[hand][on]
            gap = onsets[i + 1] - on if i + 1 < len(onsets) else None
            if opts.legato:
                # Pop-piano notation: hold the chord until the next one, like a pedalled performance.
                dur = max(d for _, d, _ in group)
                if gap is not None and gap <= Fraction(opts.max_hold_beats).limit_denominator(16):
                    dur = gap
            else:
                dur = min(d for _, d, _ in group)
            if gap is not None:
                dur = min(dur, gap)  # avoid overlaps
            dur = max(dur, Fraction(1, g))
            evs[hand].append((on, dur, sorted({pc for pc, _, _ in group}), int(np.mean([v for _, _, v in group]))))

    end_beat = max((float(on + dur) for h in evs.values() for on, dur, _, _ in h), default=0.0)
    bars = int(np.ceil(end_beat / opts.beats_per_bar)) or 1
    all_notes = [(float(on), float(dur), midis) for h in evs.values() for on, dur, midis, _ in h]
    # Chord detection needs a key preference for spelling; estimate the key from pitch classes first.
    k = _estimate_key(all_notes, log)
    symbols = _chord_symbols(all_notes, bars, opts.beats_per_bar, prefer_flats=k.sharps < 0) if (opts.chords or opts.level == "beginner") else []

    if opts.level == "beginner":
        evs = _simplify_beginner(evs, symbols, bars, opts.beats_per_bar)
        log(f"[score] beginner arrangement: {len(evs['rh'])} melody notes, {len(evs['lh'])} bass notes")

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    title = _strip_emoji(opts.title) + (" (初級)" if opts.level == "beginner" else "")
    score.metadata.title = title
    # verovio only draws <movement-title>, so the tempo rides along on the title line.
    score.metadata.movementName = f"{title}   \u2669 \u2248 {round(bpm)}"
    score.metadata.composer = "makepiano"

    score_notes: list[tuple[float, float, int, int]] = []
    parts = []
    for hand, cl in (("rh", clef.TrebleClef()), ("lh", clef.BassClef())):
        p = stream.Part(id=hand)
        p.partName = ""
        p.partAbbreviation = ""
        p.insert(0, cl)
        p.insert(0, meter.TimeSignature(f"{opts.beats_per_bar}/4"))
        for on, dur, midis, vel in evs[hand]:
            score_notes.extend((float(on + bar_start), float(on + bar_start + dur), m, vel) for m in midis)
            pitches = [pitch.Pitch(midi=m) for m in midis]  # note.Note(int) adds explicit naturals
            obj = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            obj.quarterLength = float(dur)
            p.insert(float(on), obj)
        parts.append(p)

    # Key signature, then respell accidentals to match (flats in flat keys).
    for p in parts:
        p.insert(0, key.KeySignature(k.sharps))
        for n in p.notes:
            for pt in n.pitches:
                _respell(pt, k)

    if not opts.chords:
        symbols = []
    for p in parts:
        p.makeMeasures(inPlace=True)
        while len(p.getElementsByClass(stream.Measure)) < bars:
            m = stream.Measure(number=len(p.getElementsByClass(stream.Measure)) + 1)
            p.append(m)
        p.makeTies(inPlace=True)
        for m in p.getElementsByClass(stream.Measure):
            _fill_rests(m, opts.beats_per_bar)
        # Show accidentals relative to the key signature (and cancel with naturals), measure by measure.
        p.makeAccidentals(inPlace=True, overrideStatus=True, cautionaryNotImmediateRepeat=False)

    if symbols:
        measures = list(parts[0].getElementsByClass(stream.Measure))
        for off, fig in symbols:
            cs = harmony.ChordSymbol(fig)
            cs.writeAsChord = False
            mi = min(int(off // opts.beats_per_bar), len(measures) - 1)
            measures[mi].insert(off - mi * opts.beats_per_bar, cs)
        log(f"[score] {len(symbols)} chord symbols")

    for p in parts:
        score.insert(0, p)
    score.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    log(f"[score] {bars} bars, {sum(len(p.flatten().notes) for p in parts)} note/chord events")
    return score, bar_start, score_notes


def midi_to_musicxml(midi_path: Path, wav_path: Path, xml_out: Path, opts: ScoreOptions, log=print) -> tuple[Path, dict]:
    """Returns (musicxml path, timing dict for playback sync)."""
    events = read_midi_notes(midi_path)
    if not events:
        raise RuntimeError("No notes were transcribed from the audio.")
    if opts.level == "beginner" and opts.grid > 2:
        opts.grid = 2  # 8th notes are enough for a beginner arrangement
    duration = max(e.end for e in events)
    beat_times, bpm = track_beats(wav_path, opts.fixed_bpm, duration, log=log)
    score, bar_start, score_notes = build_score(events, beat_times, bpm, opts, log=log)
    score.write("musicxml", fp=str(xml_out))
    log(f"[score] wrote {xml_out.name}")
    idx = np.arange(len(beat_times))
    sec = lambda b: round(float(np.interp(b, idx, beat_times)), 3)  # noqa: E731
    timing = {
        "bpm": bpm,
        "beat_times": [round(float(t), 4) for t in beat_times],
        "bar_start_beat": int(bar_start),
        "split": opts.split_pitch,
        # notated events (quantised, legato) in audio seconds: what the piano view / synth play, like the sheet
        # [start_s, end_s, pitch, velocity, start_beat, end_beat] (beats relative to the score's first bar)
        "notes": [[sec(b0), sec(b1), m, v, round(b0 - bar_start, 4), round(b1 - bar_start, 4)] for b0, b1, m, v in score_notes],
        # model output as performed (same filtering as the score, ends extended while the sustain pedal is down):
        # what the synth plays in "performance" mode
        "raw_notes": [[round(e.start, 3), round(e.end, 3), e.pitch, e.velocity]
                      for e in apply_pedal(filter_events(events, opts), read_pedal(midi_path))],
    }
    return xml_out, timing
