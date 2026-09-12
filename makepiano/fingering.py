"""Automatic piano fingering (simplified Parncutt-style rules, solved with dynamic programming).

Input per hand: chronological events (onset, [midi pitches]). Output: finger 1..5 for every (onset, pitch).
Right hand: lower pitch -> lower finger number within a chord. Left hand: mirrored (thumb on the highest note).
"""
from __future__ import annotations

import itertools
from fractions import Fraction

# Comfortable / maximum spans in semitones for finger pairs (lower finger number first), right hand.
# (min_comfortable, max_comfortable, max_possible). Sized for an average Japanese adult hand: an octave
# (12 semitones) is the practical limit between thumb and little finger.
SPANS = {
    (1, 2): (1, 6, 8), (1, 3): (3, 8, 10), (1, 4): (5, 10, 12), (1, 5): (7, 12, 13),
    (2, 3): (1, 3, 4), (2, 4): (3, 5, 7), (2, 5): (5, 8, 10),
    (3, 4): (1, 3, 4), (3, 5): (3, 5, 7), (4, 5): (1, 3, 4),
}
BLACK = {1, 3, 6, 8, 10}


def _span_cost(f_lo: int, f_hi: int, semis: int) -> float:
    """Cost of fingers f_lo < f_hi covering `semis` semitones (f_lo on the lower key)."""
    lo, hi, mx = SPANS[(f_lo, f_hi)]
    if semis < 0:  # crossed: higher finger on the lower key (thumb-under is the usual case)
        if f_lo == 1:
            return 2.0 + max(0, -semis - 5) * 0.8   # thumb under, ok up to a 4th
        return 6.0 + (-semis) * 1.0
    if lo <= semis <= hi:
        return 0.0
    if semis > mx:
        return 6.0 + (semis - mx) * 1.5
    return (lo - semis) * 0.8 if semis < lo else (semis - hi) * 1.2


def _chord_cost(fingers: tuple[int, ...], pitches: list[int], hand: str) -> float:
    c = 0.0
    for (fa, pa), (fb, pb) in zip(zip(fingers, pitches), list(zip(fingers, pitches))[1:]):
        semis = pb - pa if hand == "rh" else pa - pb
        f_lo, f_hi = (fa, fb) if fa < fb else (fb, fa)
        c += _span_cost(f_lo, f_hi, semis if fa < fb else -semis)
    for f, p in zip(fingers, pitches):
        if f == 1 and p % 12 in BLACK:
            c += 0.7
        if f == 5 and p % 12 in BLACK:
            c += 0.3
    return c


def _transition_cost(prev: tuple[tuple[int, int], ...], cur: tuple[tuple[int, int], ...], hand: str, gap: float) -> float:
    """prev/cur: tuples of (finger, pitch). Cost of moving from one to the next."""
    c = 0.0
    for fp, pp in prev:
        for fc, pc in cur:
            if fp == fc:
                if pp != pc:
                    c += 3.0 if gap < 1 else 1.2  # same finger, different key (jump)
                continue
            semis = pc - pp if hand == "rh" else pp - pc
            if fc > fp:
                c += _span_cost(fp, fc, semis)
            else:
                c += _span_cost(fc, fp, -semis)
    return c / max(1, len(prev) * len(cur)) * 1.5


def assign(events: list[tuple[Fraction, list[int]]], hand: str) -> dict[tuple[Fraction, int], int]:
    """Viterbi over per-event finger combinations. events: chronological (onset, sorted pitches)."""
    if not events:
        return {}
    cands: list[list[tuple[tuple[int, int], ...]]] = []
    for on, pitches in events:
        pitches = sorted(set(pitches))
        k = min(len(pitches), 5)
        ps = pitches[-k:] if hand == "rh" else pitches[:k]  # (should already fit; keep the melody / bass side)
        opts = []
        for combo in itertools.combinations(range(1, 6), k):
            # right hand: ascending pitch -> ascending finger; left hand: ascending pitch -> descending finger
            fingers = combo if hand == "rh" else tuple(reversed(combo))
            opts.append(tuple(zip(fingers, ps)))
        cands.append(opts)
    INF = float("inf")
    cost = [[_chord_cost(tuple(f for f, _ in o), [p for _, p in o], hand) for o in cands[0]]]
    back: list[list[int]] = [[-1] * len(cands[0])]
    for i in range(1, len(events)):
        gap = float(events[i][0] - events[i - 1][0])
        row, bk = [], []
        for o in cands[i]:
            local = _chord_cost(tuple(f for f, _ in o), [p for _, p in o], hand)
            best, arg = INF, -1
            for j, po in enumerate(cands[i - 1]):
                c = cost[i - 1][j] + _transition_cost(po, o, hand, gap)
                if c < best:
                    best, arg = c, j
            row.append(best + local)
            bk.append(arg)
        cost.append(row)
        back.append(bk)
    # backtrack
    j = min(range(len(cost[-1])), key=lambda x: cost[-1][x])
    out: dict[tuple[Fraction, int], int] = {}
    for i in range(len(events) - 1, -1, -1):
        for f, p in cands[i][j]:
            out[(events[i][0], p)] = f
        j = back[i][j]
        if j < 0:
            break
    return out
