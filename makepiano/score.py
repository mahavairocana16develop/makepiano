"""MIDI (+ audio for beat tracking) -> two-hand piano score (music21 / MusicXML)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import librosa
import mido
import numpy as np
from music21 import articulations, chord, clef, duration as m21duration, dynamics, expressions, harmony, key, layout, metadata, meter, note, pitch, stream


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
    level: str = "both"      # "both" (=all) | "original" | "intermediate" | "beginner"
    max_notes: int = 5       # per hand, at once
    max_span: int = 12       # semitones a hand may stretch (12 = octave; 14 for large hands)
    pedal_marks: bool = True  # notate the sustain pedal from the transcription
    dynamics_marks: bool = True  # pp..ff from performed velocities
    short_notes: bool = True  # keep clearly short (staccato-like) notes short instead of holding to the next chord
    triplets: str = "16th"   # "off" | "8th" (allow 8th-note triplets per beat) | "16th" (also 16th-note triplets)
    fingering: bool = True   # automatic fingering (numbers in the score, hands in the piano view)
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


def _fit_hands(evs: dict[str, list[Event]], max_notes: int, max_span: int, snap=None) -> tuple[dict[str, list[Event]], int, int]:
    """Make every chord physically playable: at most max_notes per hand within max_span semitones.
    Right hand keeps the melody (top) and hands surplus low notes to the left hand at the same onset;
    the left hand keeps the bass (bottom) and drops what still does not fit. Returns (evs, moved, dropped)."""
    moved = dropped = 0
    lh: dict[Fraction, list] = {on: [on, dur, list(midis), vel] for on, dur, midis, vel in evs["lh"]}
    rh_out: list[Event] = []
    for on, dur, midis, vel in evs["rh"]:
        m = sorted(set(midis))
        surplus = []
        while m and (len(m) > max_notes or m[-1] - m[0] > max_span):
            surplus.append(m.pop(0))
        rh_out.append((on, dur, m, vel))
        if surplus:
            moved += len(surplus)
            if on in lh:
                lh[on][2] = sorted(set(lh[on][2]) | set(surplus))
            else:
                lh[on] = [on, dur, sorted(surplus), vel]
    lh_out: list[Event] = []
    onsets = sorted(lh)
    for i, on in enumerate(onsets):
        _, dur, midis, vel = lh[on]
        m = sorted(set(midis))
        while m and (len(m) > max_notes or m[-1] - m[0] > max_span):
            m.pop()
            dropped += 1
        if i + 1 < len(onsets):
            dur = min(dur, onsets[i + 1] - on)  # a moved chord may have created a new onset in between
            if snap is not None:
                dur = snap(on, dur)
        lh_out.append((on, dur, m, vel))
    return {"rh": rh_out, "lh": lh_out}, moved, dropped


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


_TEMPLATE_BY_NAME = dict(_CHORD_TEMPLATES)


def _chord_pcs(fig: str) -> set[int]:
    root = _root_pc(fig)
    name = fig[2:] if len(fig) > 1 and fig[1] in "#-" else fig[1:]
    return {(root + i) % 12 for i in _TEMPLATE_BY_NAME.get(name, (0, 4, 7))}


def _bass_root(fig: str) -> int:
    m = 36 + _root_pc(fig)
    return m + 12 if m < 40 else m  # E2..D#3


def _simplify_intermediate(evs: dict[str, list[Event]], symbols: list[tuple[float, str]], bars: int,
                           beats_per_bar: int, window: float = 2.0) -> dict[str, list[Event]]:
    """Intermediate arrangement: melody plus up to two chord tones on the beats; root-fifth alternating bass."""
    sym_at = sorted(symbols)
    # chord in effect at any beat position
    def chord_at(b: float) -> str | None:
        cur = None
        for off, fig in sym_at:
            if off <= b + 1e-6:
                cur = fig
            else:
                break
        return cur
    rh: list[Event] = []
    for on, dur, midis, vel in evs["rh"]:
        m = max(midis)
        while m > 84:
            m -= 12
        notes = [m]
        fig = chord_at(float(on))
        if fig is not None and float(on) == int(float(on)):  # harmony only on the beat, keeps it playable
            pcs = _chord_pcs(fig) - {m % 12}
            tones = [x for x in range(m - 1, m - 12, -1) if x % 12 in pcs][:2]
            notes = sorted(tones) + [m]
        rh.append((on, dur, notes, vel))
    lh: list[Event] = []
    total = bars * beats_per_bar
    b = 0.0
    while b < total:
        fig = chord_at(b)
        if fig is None:
            b += 1
            continue
        root = _bass_root(fig)
        fifth = root + 7 if root + 7 <= 55 else root - 5
        pitch_ = root if int(b) % 2 == 0 else fifth
        lh.append((Fraction(int(b)), Fraction(1), [pitch_], 80))
        b += 1
    return {"rh": rh, "lh": lh}


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


def _notatable(d: Fraction) -> bool:
    """One note head: plain / dotted values, or a plain triplet (3:2, 6:4) member."""
    dur = m21duration.Duration(float(d))
    if dur.type == "complex":
        return False
    return all(t.numberNotesActual in (3, 6) and t.numberNotesNormal in (2, 4) for t in dur.tuplets)


def _beat_grids(starts: np.ndarray, base: int, triplets: str) -> dict[int, int]:
    """Per beat, the subdivision (base, 3 or 6) that fits the performed onsets best. Swing / triplet feels
    snap much better to thirds than to 16ths; a non-base grid must beat the base grid by a clear margin."""
    grids: dict[int, int] = {}
    if triplets == "off":
        return grids
    cands = (3, 6) if triplets == "16th" else (3,)
    by_beat: dict[int, list[float]] = {}
    for s_ in starts:
        by_beat.setdefault(int(np.floor(s_)), []).append(float(s_))
    for b, xs in by_beat.items():
        if len(xs) < 2:
            continue
        def err(g):
            return sum(abs(x * g - round(x * g)) / g for x in xs)
        e_base = err(base)
        best_g, best_e = base, e_base
        for g in cands:
            e = err(g)
            if e < best_e * 0.6:
                best_g, best_e = g, e
        if best_g != base:
            grids[b] = best_g
    return grids


def build_score(events: list[NoteEvent], beat_times: np.ndarray, bpm: float, opts: ScoreOptions, log=print,
                pedal_events: list[tuple[float, bool]] | None = None
                ) -> tuple[stream.Score, int, list[tuple], list[tuple[float, float]]]:
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

    grids = _beat_grids(starts, g, opts.triplets)
    if grids:
        log(f"[score] triplet feel detected in {len(grids)} beats")

    def grid_at(x) -> int:
        return grids.get(int(np.floor(float(x))), g)

    def qz(x: float) -> Fraction:
        return _quantize(x, grid_at(x))

    def snap_dur(on: Fraction, dur: Fraction) -> Fraction:
        """Keep durations notatable: a multiple of the start beat's grid, and if the note ends inside a beat
        with a different grid, end on that beat's grid instead (never past the original end)."""
        g0 = grid_at(on)
        d = Fraction(int(dur * g0), g0)  # floor to the start grid
        d = max(d, Fraction(1, g0))
        end = on + d
        g1 = grid_at(end - Fraction(1, 1000))
        if g1 != g0 and end != int(end):
            # A duration mixing quarters and thirds would need a 12-tuplet: stop at the beat boundary
            # instead (the pedal mark carries the sustain).
            boundary = Fraction(int(end))
            if boundary > on:
                d = boundary - on
        # Still not a single notatable value (e.g. 7/6 = a beat plus a triplet 16th)? Stop at the beat line.
        if not _notatable(d):
            boundary = Fraction(int(on)) + 1
            if boundary - on >= Fraction(1, g0) and boundary < on + d:
                d = boundary - on
            if not _notatable(d):
                d = Fraction(1, g0)
        return d

    hands: dict[str, dict[Fraction, list[tuple[int, Fraction, int]]]] = {"rh": {}, "lh": {}}
    for e, s_, en in zip(events, starts, ends):
        qs = qz(float(s_))
        qe = max(qz(float(en)), qs + Fraction(1, grids.get(int(np.floor(float(s_))), g)))
        if qs < 0:
            continue
        hand = "rh" if e.pitch >= opts.split_pitch else "lh"
        hands[hand].setdefault(qs, []).append((e.pitch, qe - qs, e.velocity))

    vel_of: dict[tuple[str, Fraction, int], int] = {h: 0 for h in ()}  # (hand, onset, pitch) -> performed velocity
    for hand in ("rh", "lh"):
        for on, group in hands[hand].items():
            for pc, _, v in group:
                vel_of[(hand, on, pc)] = max(v, vel_of.get((hand, on, pc), 0))
    # Collapse each onset into one note/chord event with a single duration (monophonic-chord stream per hand).
    evs: dict[str, list[Event]] = {"rh": [], "lh": []}
    for hand in ("rh", "lh"):
        onsets = sorted(hands[hand])
        for i, on in enumerate(onsets):
            group = hands[hand][on]
            gap = onsets[i + 1] - on if i + 1 < len(onsets) else None
            if opts.legato:
                # Pop-piano notation: hold the chord until the next one, like a pedalled performance...
                actual = max(d for _, d, _ in group)
                dur = actual
                if gap is not None and gap <= Fraction(opts.max_hold_beats).limit_denominator(16):
                    dur = gap
                    # ...unless the performer clearly released early: keep short notes short (rest follows).
                    if opts.short_notes and gap >= 1 and actual <= gap * Fraction(2, 5):
                        dur = actual
            else:
                dur = min(d for _, d, _ in group)
            if gap is not None:
                dur = min(dur, gap)  # avoid overlaps
            dur = snap_dur(on, dur)
            evs[hand].append((on, dur, sorted({pc for pc, _, _ in group}), int(np.mean([v for _, _, v in group]))))

    end_beat = max((float(on + dur) for h in evs.values() for on, dur, _, _ in h), default=0.0)
    bars = int(np.ceil(end_beat / opts.beats_per_bar)) or 1
    all_notes = [(float(on), float(dur), midis) for h in evs.values() for on, dur, midis, _ in h]
    # Chord detection needs a key preference for spelling; estimate the key from pitch classes first.
    k = _estimate_key(all_notes, log)
    symbols = _chord_symbols(all_notes, bars, opts.beats_per_bar, prefer_flats=k.sharps < 0) if (opts.chords or opts.level != "original") else []

    if opts.level == "beginner":
        evs = _simplify_beginner(evs, symbols, bars, opts.beats_per_bar)
        log(f"[score] beginner arrangement: {len(evs['rh'])} melody notes, {len(evs['lh'])} bass notes")
    elif opts.level == "intermediate":
        evs = _simplify_intermediate(evs, symbols, bars, opts.beats_per_bar)
        log(f"[score] intermediate arrangement: {len(evs['rh'])} right-hand events, {len(evs['lh'])} bass notes")

    evs, moved, dropped = _fit_hands(evs, opts.max_notes, opts.max_span, snap=snap_dur)
    if moved or dropped:
        log(f"[score] playability: {moved} notes moved to the left hand, {dropped} dropped (max {opts.max_notes} notes / {opts.max_span} semitones per hand)")

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    title = _strip_emoji(opts.title) + {"beginner": " (初級)", "intermediate": " (中級)"}.get(opts.level, "")
    score.metadata.title = title
    # verovio only draws <movement-title>, so the tempo rides along on the title line.
    score.metadata.movementName = f"{title}   \u2669 \u2248 {round(bpm)}"
    score.metadata.composer = "makepiano"

    fingers: dict[str, dict[tuple[Fraction, int], int]] = {"rh": {}, "lh": {}}
    if opts.fingering:
        from .fingering import assign
        for hand in ("rh", "lh"):
            fingers[hand] = assign([(on, midis) for on, _, midis, _ in evs[hand] if midis], hand)
        log(f"[score] fingering assigned to {sum(len(v) for v in fingers.values())} notes")

    score_notes: list[tuple] = []
    parts = []
    for hand, cl in (("rh", clef.TrebleClef()), ("lh", clef.BassClef())):
        p = stream.Part(id=hand)
        p.partName = ""
        p.partAbbreviation = ""
        p.insert(0, cl)
        p.insert(0, meter.TimeSignature(f"{opts.beats_per_bar}/4"))
        for on, dur, midis, vel in evs[hand]:
            score_notes.extend((float(on + bar_start), float(on + bar_start + dur), m,
                                vel_of.get((hand, on, m), vel_of.get(("rh" if hand == "lh" else "lh", on, m), vel)),
                                hand, fingers[hand].get((on, m), 0)) for m in midis)
            pitches = [pitch.Pitch(midi=m) for m in midis]  # note.Note(int) adds explicit naturals
            obj = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            obj.quarterLength = float(dur)
            if opts.fingering:
                for m in (midis if len(midis) <= 3 else [midis[-1] if hand == "rh" else midis[0]]):  # keep chords legible
                    f = fingers[hand].get((on, m))
                    if f:
                        obj.articulations.append(articulations.Fingering(f))
            p.insert(float(on), obj)
        parts.append(p)

    # Sustain pedal from the performance, quantised to the grid, as Ped./* marks on the bass staff.
    pedal_segments: list[tuple[float, float]] = []
    if opts.pedal_marks and pedal_events:
        pb_ = seconds_to_beats(np.array([t for t, _ in pedal_events]), beat_times) - bar_start
        down = None
        for (t, is_down), b in zip(pedal_events, pb_):
            if is_down and down is None:
                down = b
            elif not is_down and down is not None:
                a, z = float(_quantize(float(down), g)), float(_quantize(float(b), g))
                if z - a >= 0.5 and a >= 0:
                    if pedal_segments and a - pedal_segments[-1][1] < 1.0 / g + 1e-6:
                        pedal_segments[-1] = (pedal_segments[-1][0], z)  # merge a tiny gap
                    else:
                        pedal_segments.append((a, z))
                down = None
        n_marks = 0
        for a, z in pedal_segments:
            for part in (parts[1], parts[0]):
                inside = [n for n in part.notes if a - 1e-6 <= n.offset < z - 1e-6]
                if inside:
                    part.insert(0, expressions.PedalMark(inside[0], inside[-1]))
                    n_marks += 1
                    break
        log(f"[score] {n_marks} pedal marks")

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

    if opts.dynamics_marks and score_notes:
        measures = list(parts[0].getElementsByClass(stream.Measure))
        levels_ = [(40, "pp"), (55, "p"), (68, "mp"), (80, "mf"), (95, "f"), (128, "ff")]
        last = None
        for mi, m in enumerate(measures):
            b0, b1 = bar_start + mi * opts.beats_per_bar, bar_start + (mi + 1) * opts.beats_per_bar
            vs = [v for s0, _, _, v, *_ in score_notes if b0 <= s0 < b1]
            if not vs:
                continue
            lvl = next(name for th, name in levels_ if np.mean(vs) < th)
            if lvl != last:
                m.insert(0, dynamics.Dynamic(lvl))
                last = lvl

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
    return score, bar_start, score_notes, pedal_segments


def midi_to_musicxml(midi_path: Path, wav_path: Path, xml_out: Path, opts: ScoreOptions, log=print) -> tuple[Path, dict]:
    """Returns (musicxml path, timing dict for playback sync)."""
    events = read_midi_notes(midi_path)
    if not events:
        raise RuntimeError("No notes were transcribed from the audio.")
    if opts.level in ("beginner", "intermediate") and opts.grid > 2:
        opts.grid = 2  # 8th notes are enough for simplified arrangements
    duration = max(e.end for e in events)
    beat_times, bpm = track_beats(wav_path, opts.fixed_bpm, duration, log=log)
    pedal_events = read_pedal(midi_path)
    score, bar_start, score_notes, pedal_segments = build_score(events, beat_times, bpm, opts, log=log, pedal_events=pedal_events)
    score.write("musicxml", fp=str(xml_out))
    log(f"[score] wrote {xml_out.name}")
    idx = np.arange(len(beat_times))
    sec = lambda b: round(float(np.interp(b, idx, beat_times)), 3)  # noqa: E731
    timing = {
        "bpm": bpm,
        "beat_times": [round(float(t), 4) for t in beat_times],
        "bar_start_beat": int(bar_start),
        "split": opts.split_pitch,
        "beats_per_bar": opts.beats_per_bar,
        # notated events (quantised, legato) in audio seconds: what the piano view / synth play, like the sheet
        # notated pedal: [[start_s, end_s, start_beat, end_beat], ...]
        "pedal": [[sec(a + bar_start), sec(z + bar_start), a, z] for a, z in pedal_segments],
        # [start_s, end_s, pitch, velocity, start_beat, end_beat, hand, finger] (beats relative to the first bar)
        "notes": [[sec(b0), sec(b1), m, v, round(b0 - bar_start, 4), round(b1 - bar_start, 4), hand, f]
                  for b0, b1, m, v, hand, f in score_notes],
        # model output as performed (same filtering as the score, ends extended while the sustain pedal is down):
        # what the synth plays in "performance" mode
        "raw_notes": [[round(e.start, 3), round(e.end, 3), e.pitch, e.velocity]
                      for e in apply_pedal(filter_events(events, opts), pedal_events)],
    }
    return xml_out, timing
