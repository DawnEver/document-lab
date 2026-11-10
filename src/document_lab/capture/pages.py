"""Where a reader's pages meet in a stitched image: the separator rows that recur at one page pitch.

A reader in continuous scroll draws a separator between pages -- a grey gap, a rule, a dashed line -- and every
separator is drawn alike, so its rows recur, pixel for pixel, at the page pitch. :func:`page_rhythm` finds that chain:
rows whose content (not the page's background) recurs at least :data:`LEAST_SEPARATORS` times, equally spaced to a
row, from the first page on to the image's end. Rows that recur at the scroll step instead (a viewer border the
stitcher let through) do not start at the top -- the steps grow from one notch before they are even -- so they do
not form such a chain. Of the chains that remain, the longest pitch is the page (a page holds its own rules at a
shorter one), and of rows at that pitch the one least like the page's blank row is the separator (a running header
rule spans the text, a separator the whole width); the cut is at its first row, so each page starts with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ['LEAST_SEPARATORS', 'Rhythm', 'page_rhythm']

LEAST_SEPARATORS: Final = 3  # separators that make a chain: two are a coincidence, three a pitch
_SLACK: Final = 1  # rows a separator may sit off the pitch: frames placed to a fraction of a row
_LEAST_PITCH: Final = 64  # rows: no page is shorter
_GREY_STEP: Final = 8  # grey levels quantised together, so a re-rendered row still matches


@dataclass(frozen=True)
class Rhythm:
    """Pages ``pitch`` rows tall (fractional: the mean spacing), the first separator ``offset`` rows down."""

    pitch: float
    offset: int

    def cuts(self, start: int, stop: int) -> list[int]:
        """The page cuts (rows where a page starts) in ``(start, stop]``."""
        out, k = [], max(0, int((start - self.offset) // self.pitch))
        while (cut := self.offset + round(k * self.pitch)) <= stop:
            if cut > start:
                out.append(cut)
            k += 1
        return out


def _row_keys(grey: np.ndarray) -> np.ndarray:
    weights = np.random.default_rng(0).integers(1, 1 << 62, grey.shape[1], dtype=np.uint64)
    return (grey.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def _chain(rows: np.ndarray, height: int) -> float | None:
    """The pitch of ``rows`` (sorted run starts) if they are one equally spaced chain from the top to the end."""
    if len(rows) < LEAST_SEPARATORS:
        return None
    gaps = np.diff(rows)
    pitch = float(gaps.mean())
    if pitch < _LEAST_PITCH or np.abs(gaps - pitch).max() > _SLACK:
        return None
    if rows[0] > pitch + _SLACK or rows[-1] < height - pitch - _SLACK:
        return None  # not from the first page, or stops before the end: not the page's own rhythm
    return pitch


def page_rhythm(image: np.ndarray) -> Rhythm | None:
    """The page rhythm of ``image`` (``(h, w, 3)`` uint8), or None when no separator chain is in it."""
    grey = (image[..., :3].astype(np.uint16).sum(axis=2) // (3 * _GREY_STEP)).astype(np.uint8)
    keys = _row_keys(grey)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    common = int(counts.argmax())  # the page's blank row: what every other row is told apart from
    blank = grey[int(np.argmax(inverse == common))]
    found: list[tuple[float, float, int]] = []  # (pitch, share unlike the background, offset)
    for index in np.flatnonzero(counts >= LEAST_SEPARATORS):
        if index == common:
            continue
        rows = np.flatnonzero(inverse == index)
        starts = rows[np.concatenate(([True], np.diff(rows) > 1))]  # a band of equal rows is one separator
        pitch = _chain(starts, len(image))
        if pitch is not None:
            found.append((pitch, float((grey[starts[0]] != blank).mean()), int(starts[0])))
    if not found:
        return None
    longest = max(pitch for pitch, _, _ in found)
    # the row least like the blank page row, then the first of its band
    pitch, _, offset = max((f for f in found if f[0] >= longest - _SLACK), key=lambda f: (f[1], -f[2]))
    return Rhythm(pitch, offset)
