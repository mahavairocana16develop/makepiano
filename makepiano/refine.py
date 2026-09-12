"""Analysis-by-synthesis refinement of the transcription.

The model is run once; its raw frame-wise outputs are post-processed with several threshold settings,
each candidate is rendered with a sampled piano and compared against the audio that was transcribed.
The candidate whose rendering sounds most like the recording wins.
"""
from __future__ import annotations

import base64
import itertools
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import librosa
import numpy as np

SR = 22050
SF_URL = "https://gleitz.github.io/midi-js-soundfonts/MusyngKite/acoustic_grand_piano-mp3.js"
CACHE = Path.home() / ".cache" / "makepiano"
_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


# ---------------------------------------------------------------- sampled piano

def _load_samples(log=print) -> dict[int, np.ndarray]:
    """88 piano samples (mono float32 at SR) decoded from the MIDI.js soundfont, cached as .npz."""
    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / "musyngkite_piano_22050.npz"
    if npz.exists():
        d = np.load(npz)
        return {int(k): d[k] for k in d.files}
    js = CACHE / "acoustic_grand_piano-mp3.js"
    if not js.exists():
        log("[refine] downloading piano samples for rendering (2.3 MB, first run only)")
        urllib.request.urlretrieve(SF_URL, js)
    text = js.read_text(encoding="utf-8", errors="ignore")
    out: dict[int, np.ndarray] = {}
    for name, b64 in re.findall(r'"([A-G]b?\d)"\s*:\s*"data:audio/mp3;base64,([A-Za-z0-9+/=]+)"', text):
        pc = _NOTE_NAMES.index(name[:-1])
        midi = 12 * (int(name[-1]) + 1) + pc
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                           input=base64.b64decode(b64), capture_output=True)
        out[midi] = np.frombuffer(r.stdout, dtype=np.float32).copy()
    np.savez_compressed(npz, **{str(k): v for k, v in out.items()})
    return out


def render(notes: list[tuple[float, float, int, int]], pedal: list[tuple[float, bool]], samples: dict[int, np.ndarray],
           length_s: float, release: float = 0.25) -> np.ndarray:
    """Mix sampled piano notes: (start, end, pitch, velocity); ends extended while the pedal is down."""
    from .score import NoteEvent, apply_pedal
    evs = apply_pedal([NoteEvent(p, s, e, v) for s, e, p, v in notes], pedal)
    n = int((length_s + 3) * SR)
    y = np.zeros(n, dtype=np.float32)
    rel = int(release * SR)
    fade = np.exp(-np.arange(rel) / (rel / 4)).astype(np.float32)
    for e in evs:
        smp = samples.get(e.pitch)
        if smp is None:
            continue
        i0 = int(e.start * SR)
        dur = max(int((e.end - e.start) * SR), int(0.03 * SR))
        seg = smp[: dur + rel].copy()
        if len(seg) > dur:  # release tail
            tail = len(seg) - dur
            seg[dur:] *= fade[:tail]
        seg *= (e.velocity / 127.0) ** 1.2
        i1 = min(n, i0 + len(seg))
        y[i0:i1] += seg[: i1 - i0]
    return y


# ---------------------------------------------------------------- similarity

def _features(y: np.ndarray) -> dict:
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-8) * 0.1  # level-normalise
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, hop_length=512), top_db=60)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=512, units="time", backtrack=False)
    return {"mel": mel, "chroma": chroma, "onsets": onsets}


def _onset_f1(ref: np.ndarray, est: np.ndarray, tol: float = 0.06) -> float:
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used = np.zeros(len(ref), bool)
    tp = 0
    for t in est:
        d = np.abs(ref - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    p, r = tp / len(est), tp / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def similarity(ref: dict, est: dict) -> dict:
    n = min(ref["mel"].shape[1], est["mel"].shape[1])
    mel_dist = float(np.mean(np.abs(ref["mel"][:, :n] - est["mel"][:, :n])))
    a, b = ref["chroma"][:, :n], est["chroma"][:, :n]
    chroma = float(np.mean(np.sum(a * b, 0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-8)))
    onset = _onset_f1(ref["onsets"], est["onsets"])
    score = chroma + onset - mel_dist / 20.0
    return {"score": round(score, 4), "chroma": round(chroma, 3), "onset_f1": round(onset, 3), "mel_dist": round(mel_dist, 2)}


# ---------------------------------------------------------------- search

GRID = {
    "onset_threshold": [0.2, 0.3, 0.4, 0.5],
    "frame_threshold": [0.05, 0.1, 0.2, 0.3],
    "offset_threshold": [0.3],
}


def refine(output_dict: dict, audio16k: np.ndarray, frames_per_second: int, classes_num: int, log=print,
           report_path: Path | None = None, progress=None) -> tuple[list, list, dict]:
    """Return (note_events, pedal_events, chosen params) for the best-matching threshold setting."""
    from piano_transcription_inference.utilities import RegressionPostProcessor

    samples = _load_samples(log)
    ref_y = librosa.resample(audio16k.astype(np.float32), orig_sr=16000, target_sr=SR)
    length_s = len(ref_y) / SR
    ref = _features(ref_y)
    results = []
    best = None
    combos = list(itertools.product(*GRID.values()))
    log(f"[refine] comparing {len(combos)} threshold settings against the recording")
    total_cands = len(combos) + 8
    for ci, vals in enumerate(combos):
        if progress:
            progress(ci / total_cands)
        params = dict(zip(GRID.keys(), vals))
        params = {**params, "pedal_offset_threshold": 0.2}
        pp = RegressionPostProcessor(frames_per_second, classes_num=classes_num, **params)
        note_events, pedal_events = pp.output_dict_to_midi_events(output_dict)
        notes = [(e["onset_time"], e["offset_time"], e["midi_note"], e["velocity"]) for e in note_events]
        pedal = []
        for e in pedal_events:
            pedal += [(e["onset_time"], True), (e["offset_time"], False)]
        pedal.sort()
        y = render(notes, pedal, samples, length_s)[: len(ref_y)]
        sim = similarity(ref, _features(y))
        row = {**params, **sim, "notes": len(notes)}
        results.append(row)
        if best is None or sim["score"] > best[0]:
            best = (sim["score"], params, note_events, pedal_events, row)
    # Stage 2: around the best onset/frame setting, try the note-off and pedal thresholds too.
    stage2 = [dict(best[1], offset_threshold=o, pedal_offset_threshold=pd)
              for o in (0.2, 0.3, 0.5) for pd in (0.1, 0.2, 0.4) if not (o == 0.3 and pd == 0.2)]
    for si, params in enumerate(stage2):
        if progress:
            progress((len(combos) + si) / total_cands)
        pp = RegressionPostProcessor(frames_per_second, classes_num=classes_num, **params)
        note_events, pedal_events = pp.output_dict_to_midi_events(output_dict)
        notes = [(e["onset_time"], e["offset_time"], e["midi_note"], e["velocity"]) for e in note_events]
        pedal = []
        for e in pedal_events:
            pedal += [(e["onset_time"], True), (e["offset_time"], False)]
        pedal.sort()
        y = render(notes, pedal, samples, length_s)[: len(ref_y)]
        sim = similarity(ref, _features(y))
        row = {**params, **sim, "notes": len(notes)}
        results.append(row)
        if sim["score"] > best[0]:
            best = (sim["score"], params, note_events, pedal_events, row)
    default = next(r for r in results if r["onset_threshold"] == 0.3 and r["frame_threshold"] == 0.1 and r["offset_threshold"] == 0.3 and r.get("pedal_offset_threshold", 0.2) == 0.2)
    log(f"[refine] best: onset {best[1]['onset_threshold']} / frame {best[1]['frame_threshold']} / offset {best[1]['offset_threshold']} / pedal {best[1]['pedal_offset_threshold']} "
        f"(score {best[0]:.3f}, onset-F1 {best[4]['onset_f1']}, chroma {best[4]['chroma']}, {best[4]['notes']} notes) "
        f"vs default score {default['score']:.3f} ({default['notes']} notes)")
    if report_path is not None:
        report_path.write_text(json.dumps({"chosen": best[1], "default": default,
                                           "candidates": sorted(results, key=lambda r: -r["score"])}, indent=1))
    return best[2], best[3], best[1]
