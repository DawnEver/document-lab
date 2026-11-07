"""Overlapping viewport frames of one scrolled view, stitched into the one tall image they show.

A frame is an ``(height, width, 3)`` uint8 array. Between two frames most of the view moves by one offset; what does
not move with it is an OVERLAY -- a sticky header or footer, a floating button, a scrollbar thumb, a blinking caret.

Matching. Each row is cut into :data:`BINS` column bins per channel (a cell: its mean). Cells equal at the same place
in both frames say nothing about the scroll (an overlay, or a uniform background) and are left out; the offset is the
one at which the remaining cells agree -- exactly, or within ``tolerance`` grey levels -- all but an ``outliers``
share of them (a thumb or a caret). An offset that is not unique, or an overlap with nothing left to compare, is
refused -- never guessed. ``top`` and ``bottom`` rows are declared chrome, left out of the match.

Composing. Each frame adds the rows it newly shows, so by default an overlay low in the view (a floating button, a
sticky footer) recurs down the image, as the frames showed it. With ``drop_overlays``, a pixel that stays put while
the view moves is an overlay pixel, and where a later frame shows the same spot of the page uncovered, that frame's
pixel replaces it; a spot no frame shows uncovered keeps the overlay (a sticky header at the very top, a footer at
the end).

:class:`Stitcher` holds only the last frame and the image so far, so a capture of any length costs the image it
produces, not every frame grabbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['BINS', 'AmbiguousOverlap', 'Match', 'NoOverlap', 'Stitcher', 'Unstitchable', 'offset', 'stitch']

BINS: Final = 64
_EDGE: Final = 1 << 16  # a rank: class * _EDGE + the row's distance from the view's middle
CLEAR, OVERLAID, CHROME = 0, 1, 2  # a view's class: never proven an overlay, proven one, declared chrome
_LEAST_CELLS: Final = BINS  # fewest comparable cells (one row's worth) that can tell offsets apart


class Unstitchable(ValueError):
    """Two frames whose relative scroll cannot be determined."""


class NoOverlap(Unstitchable):
    """No offset leaves enough shared rows: the scroll moved (almost) a whole view, or the view changed."""


class AmbiguousOverlap(Unstitchable):
    """More than one offset fits, or the shared rows hold nothing that moved and could tell offsets apart."""


@dataclass(frozen=True)
class Match:
    """How frames are matched.

    ``top``/``bottom``: chrome rows left out of the match. ``min_overlap``: fewest shared content rows (default a
    quarter of the content). ``tolerance``: grey levels a cell may differ by and still agree. ``outliers``: the share
    of compared cells that may disagree at the true offset (a scrollbar thumb, a caret, an animation).
    """

    top: int = 0
    bottom: int = 0
    min_overlap: int | None = None
    tolerance: float = 0
    outliers: float = 0.05


def _content(frame: np.ndarray, match: Match) -> np.ndarray:
    return frame[match.top : len(frame) - match.bottom]


def _cells(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row cut into at most :data:`BINS` column bins.

    Per channel the bin's mean, ``(rows, bins * 3)``, and whether the bin is blank -- one value across it and the
    same as its neighbours on the row -- ``(rows, bins)``.
    """
    pixels = frame.astype(np.float32)
    edges = np.linspace(0, pixels.shape[1], min(BINS, pixels.shape[1]) + 1).astype(int)[:-1]
    means = np.add.reduceat(pixels, edges, axis=1) / np.diff([*edges, pixels.shape[1]])[None, :, None]
    blank = (np.maximum.reduceat(frame, edges, axis=1) == np.minimum.reduceat(frame, edges, axis=1)).all(axis=2)
    same = (means[:, 1:] == means[:, :-1]).all(axis=2)  # a bin and its right neighbour
    blank[:, 1:] &= same
    blank[:, :-1] &= same
    return means.reshape(len(frame), -1), blank


@dataclass(frozen=True)
class _Cells:
    """Two frames' cells side by side: their means, their blanks, and which cells changed in place."""

    a: np.ndarray
    b: np.ndarray
    blank_a: np.ndarray
    blank_b: np.ndarray
    moving: np.ndarray

    def shares(self, offsets: np.ndarray, tolerance: float, stride: int) -> tuple[np.ndarray, np.ndarray]:
        """The share of compared cells that disagree at each of ``offsets``, and how many, over every ``stride``-th row.

        A cell is evidence only where it changed in place on both sides of the pair -- so an overlay, which stays put,
        is never held against the true offset -- and is not blank on both, so blank against blank (most of a sparse
        page) is counted for no offset.
        """
        rows, bins = self.moving.shape
        shares, counts = np.ones(len(offsets)), np.zeros(len(offsets), dtype=int)
        for i, d in enumerate(offsets):
            a, b = slice(d, None, stride), slice(0, rows - d, stride)
            compared = self.moving[a] & self.moving[b] & ~(self.blank_a[a] & self.blank_b[b])
            count = int(compared.sum())
            if count:
                differ = (np.abs(self.a[a] - self.b[b]) > tolerance).reshape(-1, bins, 3).any(axis=2)
                shares[i], counts[i] = (differ & compared).sum() / count, count
        return shares, counts


_STRIDE: Final = 8  # rows a first pass over every offset looks at; only offsets it cannot rule out are looked at whole
_SCREEN: Final = 0.5  # the share a first pass may show and still be looked at whole


def _disagreement(ca: np.ndarray, cb: np.ndarray, least: int, match: Match) -> tuple[np.ndarray, np.ndarray]:
    """For each offset ``0 .. rows-least``: the share of compared cells that disagree, and how many were compared.

    Share 0 is the share of cells that changed in place at all. An offset a pass over every :data:`_STRIDE`-th row
    rules out keeps share 1 and that pass's count, scaled to every row.
    """
    (pa, blank_a), (pb, blank_b) = _cells(ca), _cells(cb)
    rows, bins = blank_a.shape
    moving = (np.abs(pa - pb) > match.tolerance).reshape(rows, bins, 3).any(axis=2)
    cells = _Cells(pa, pb, blank_a, blank_b, moving)
    shares, counts = np.ones(rows - least + 1), np.zeros(rows - least + 1, dtype=int)
    shares[0], counts[0] = moving.mean(), moving.size
    offsets = np.arange(1, rows - least + 1)
    rough, seen = cells.shares(offsets, match.tolerance, _STRIDE)
    counts[offsets] = seen * _STRIDE  # the evidence an offset ruled out had, roughly
    keep = offsets[rough <= max(_SCREEN, match.outliers)]
    shares[keep], counts[keep] = cells.shares(keep, match.tolerance, 1)
    return shares, counts


def _repeats(cells: np.ndarray, least: int, tolerance: float) -> bool:
    """Does the view equal itself shifted by some offset (a blank or a repeating pattern)?"""
    rows = len(cells)
    return any((np.abs(cells[d:] - cells[: rows - d]) <= tolerance).all() for d in range(1, rows - least + 1))


def offset(a: np.ndarray, b: np.ndarray, match: Match = Match()) -> int:  # noqa: B008 -- frozen
    """Rows the view scrolled from frame ``a`` to frame ``b`` (0: it did not move).

    Raises :class:`NoOverlap` or :class:`AmbiguousOverlap` rather than return an offset that may be wrong.
    """
    if a.shape != b.shape:
        msg = f'frames differ in shape: {a.shape} != {b.shape}'
        raise ValueError(msg)
    ca, cb = _content(a, match), _content(b, match)
    rows = len(ca)
    least = max(1, rows // 4) if match.min_overlap is None else match.min_overlap
    shares, counts = _disagreement(ca, cb, least, match)
    usable = np.flatnonzero(counts[1:] >= _LEAST_CELLS) + 1
    fits = usable[shares[usable] <= match.outliers]
    if not len(fits):
        if shares[0] <= match.outliers:  # all but an outlier share is unchanged: no scroll (a caret, an animation)
            if _repeats(_cells(ca)[0], least, match.tolerance):
                msg = 'the view is blank or repeats, so whether it scrolled cannot be told: scroll past it'
                raise AmbiguousOverlap(msg)
            return 0
        if not len(usable):
            msg = 'the shared rows hold nothing that moved, so any offset fits: scroll by less, or past the blank'
            raise AmbiguousOverlap(msg)
        msg = f'no offset leaves {least} shared rows that agree: scroll by less than {rows - least} rows per step'
        raise NoOverlap(msg)
    best = int(fits[np.argmin(shares[fits])])
    rivals = [int(d) for d in fits if abs(d - best) > 1]
    if rivals:
        msg = f'{1 + len(rivals)} offsets fit ({[best, *rivals][:5]}...): the overlap repeats; scroll by another step'
        raise AmbiguousOverlap(msg)
    return best


def _proven(a: np.ndarray, b: np.ndarray, moved: int, known: np.ndarray) -> np.ndarray:
    """The screen pixels proven an overlay by the pair ``a``, ``b`` that moved by ``moved`` rows.

    A content pixel always equals its counterpart ``moved`` rows away, so a pixel still in place that does not is an
    overlay -- or the blank beside one, whose counterpart that overlay is (the rank in :class:`Stitcher` sorts those
    out). A still pixel equal to a proven counterpart is proven too, so a uniform overlay taller than the scroll,
    which a short scroll shifts onto itself, is proven whole. ``known``, the pair before's proof, holds while still.
    """
    rows = len(a)
    still = (a == b).all(axis=2)
    up = still[: rows - moved] & (b[: rows - moved] == a[moved:]).all(axis=2)  # b's pixel equals a's below it
    down = still[moved:] & (a[moved:] == b[: rows - moved]).all(axis=2)  # a's pixel equals b's above it
    proven = known & still
    proven[moved:] |= still[moved:] & ~down
    proven[: rows - moved] |= still[: rows - moved] & ~up
    while True:
        spread = proven.copy()
        spread[: rows - moved] |= up & proven[moved:]
        spread[moved:] |= down & proven[: rows - moved]
        if np.array_equal(spread, proven):
            return proven
        proven = spread


class Stitcher:
    """The tall image so far, from frames added in scroll order.

    Each frame adds the rows it newly shows. With ``drop_overlays`` each frame also bids for every spot of the page it
    shows, ranked by its screen pixel there: never proven an overlay before proven one before declared chrome, then
    nearer the view's middle before nearer its edges, where sticky bars and floating buttons keep. A spot is held by
    the lowest rank that has shown it; a spot only overlays have shown keeps one.
    """

    def __init__(self, first: np.ndarray, match: Match = Match(), *, drop_overlays: bool = False) -> None:  # noqa: B008
        """Start from ``first``; the declared chrome rows of every frame are covered from the start."""
        self.match, self.drop_overlays, self.last = match, drop_overlays, first
        rows, width = first.shape[:2]
        self._chrome = np.zeros((rows, width), dtype=bool)
        self._chrome[: match.top] = self._chrome[rows - match.bottom :] = True
        self._proven = np.zeros((rows, width), dtype=bool)  # per screen pixel, as of the last frame
        self._edge = np.abs(np.arange(rows) - rows // 2)[:, None].astype(np.uint32)
        self._canvas = np.empty((4 * rows, width, 3), dtype=np.uint8)
        self._rank = np.zeros((4 * rows, width), dtype=np.uint32)  # the rank of the view each spot holds
        self._canvas[:rows], self._rank[:rows] = first, self._ranks()
        self._top, self._end = 0, rows

    def _ranks(self) -> np.ndarray:
        kind = np.where(self._chrome, CHROME, np.where(self._proven, OVERLAID, CLEAR)).astype(np.uint32)
        return kind * _EDGE + self._edge

    @property
    def height(self) -> int:
        """Rows of the image so far."""
        return self._end

    def _room(self, end: int) -> None:
        if end > len(self._canvas):
            more = max(end, 2 * len(self._canvas)) - len(self._canvas)
            width = self._canvas.shape[1]
            self._canvas = np.concatenate([self._canvas, np.empty((more, width, 3), np.uint8)])
            self._rank = np.concatenate([self._rank, np.zeros((more, width), np.uint32)])

    def add(self, frame: np.ndarray) -> int:
        """Place ``frame`` after the last one; the rows it scrolled (0: it adds nothing)."""
        moved = offset(self.last, frame, self.match)
        if not moved:
            return 0
        rows = len(frame)
        if self.drop_overlays:
            self._proven = _proven(self.last, frame, moved, self._proven)
            placed = slice(self._top, self._top + rows)
            came = (self._canvas[placed] == self.last).all(axis=2)  # spots the last frame's view holds: re-ranked
            self._rank[placed] = np.where(came, np.maximum(self._rank[placed], self._ranks()), self._rank[placed])
        self._top += moved
        top, end = self._top, self._top + rows
        self._room(end)
        rank = self._ranks()
        fresh = np.zeros((rows, frame.shape[1]), dtype=bool)
        fresh[self._end - top :] = True  # rows below the image so far
        held = self._rank[top:end]
        # without drop_overlays nothing is proven: only a frame's chrome yields its spot, to the next frame's content
        better = rank < held if self.drop_overlays else rank // _EDGE < held // _EDGE
        take = fresh | better
        self._canvas[top:end][take], held[take] = frame[take], rank[take]
        self._end, self.last = end, frame
        return moved

    def image(self) -> np.ndarray:
        """The stitched image."""
        return self._canvas[: self._end].copy()


def stitch(frames: Sequence[np.ndarray], match: Match = Match(), *, drop_overlays: bool = False) -> np.ndarray:  # noqa: B008
    """The tall image ``frames`` show in order; repeated frames add nothing."""
    if not frames:
        msg = 'no frames to stitch'
        raise ValueError(msg)
    stitcher = Stitcher(frames[0], match, drop_overlays=drop_overlays)
    for frame in frames[1:]:
        stitcher.add(frame)
    return stitcher.image()
