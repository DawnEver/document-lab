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

:class:`Stitcher` holds only the last frame and the rows a later frame may still change: frames only move down, so
every row above the last frame's top is final, and :meth:`Stitcher.drain` hands those over. A capture of any length
costs about one view of memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    'BINS',
    'AmbiguousOverlap',
    'Match',
    'NoOverlap',
    'Stitcher',
    'Unstitchable',
    'drift',
    'locate',
    'offset',
    'shifted',
    'stitch',
]

BINS: Final = 64
_EDGE: Final = 1 << 16  # a rank: class * _EDGE + the row's distance from the view's middle
CLEAR, OVERLAID, CHROME = 0, 1, 2  # a view's class: never proven an overlay, proven one, declared chrome
_LEAST_CELLS: Final = BINS  # fewest comparable cells (one row's worth) that can tell offsets apart


class Unstitchable(ValueError):
    """Two frames whose relative scroll cannot be determined."""


class NoOverlap(Unstitchable):
    """No offset leaves enough shared rows: the scroll moved (almost) a whole view, or the view changed.

    ``error``: the mean grey error of the best offset found, when there was one -- far above the tolerance, the two
    frames share nothing at all (a reader turning whole pages).
    """

    def __init__(self, message: str, error: float | None = None) -> None:
        """``message`` and the best offset's ``error``."""
        super().__init__(message)
        self.error = error


class AmbiguousOverlap(Unstitchable):
    """More than one offset fits, or the shared rows hold nothing that moved and could tell offsets apart."""


@dataclass(frozen=True)
class Match:
    """How frames are matched.

    ``top``/``bottom``: chrome rows left out of the match. ``min_overlap``: fewest shared content rows (default a
    quarter of the content). ``tolerance``: the mean grey error the true offset may show -- a view an application
    re-renders at a fractional scroll (175% scaling) never matches exactly. ``outliers``: the share of compared cells
    left out of that mean, the worst ones (a scrollbar thumb, a caret, an overlay's edge).
    """

    top: int = 0
    bottom: int = 0
    min_overlap: int | None = None
    tolerance: float = 8.0
    outliers: float = 0.05


def _content(frame: np.ndarray, match: Match) -> np.ndarray:
    return frame[match.top : len(frame) - match.bottom]


_RECENT: list[tuple[tuple[object, ...], np.ndarray, tuple[np.ndarray, np.ndarray]]] = []  # the last frames' cells


def _cells(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_cut`, kept for the last two frames: consecutive pairs share a frame, which is cut once.

    A frame is known by the pixels it views (address, shape, strides), not by the array object: the matcher is
    handed a fresh view of the same frame each time. The frame is held with its cells, so its memory is not reused.
    """
    key = (frame.__array_interface__['data'][0], frame.shape, frame.strides)
    for seen, _, cut in _RECENT:
        if seen == key:
            return cut
    cut = _cut(frame)
    _RECENT[:] = [*_RECENT[-1:], (key, frame, cut)]
    return cut


def _cut(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row cut into at most :data:`BINS` column bins.

    Per channel the bin's mean, ``(rows, bins * 3)``, and whether the bin is blank -- one value across it and the
    same as its neighbours on the row -- ``(rows, bins)``.
    """
    width = frame.shape[1]
    edges = np.linspace(0, width, min(BINS, width) + 1).astype(int)[:-1]
    sums = np.add.reduceat(frame, edges, axis=1, dtype=np.uint32)  # summed as integers: no float copy of the frame
    means = sums.astype(np.float32) / np.diff([*edges, width]).astype(np.float32)[None, :, None]
    blank = (np.maximum.reduceat(frame, edges, axis=1) == np.minimum.reduceat(frame, edges, axis=1)).all(axis=2)
    same = (means[:, 1:] == means[:, :-1]).all(axis=2)  # a bin and its right neighbour
    blank[:, 1:] &= same
    blank[:, :-1] &= same
    return means.reshape(len(frame), -1), blank


_CANDIDATES: Final = 15  # offsets the squared-difference ranking hands on to be judged in full
_NEAR: Final = 2  # offsets this close to the best are its neighbours: a fractional scroll makes them nearly as good
_UNIQUE: Final = 2.0  # every offset but a neighbour must err at least this many times the best
_FLOOR: Final = 1.0  # grey levels: errors closer than this are not told apart
_STILL: Final = 1.0  # grey levels a cell may change in place and still count as unchanged
_EXACT: Final = 0.25  # grey levels of error under which an offset is a whole-row match, not refined
_DENSE: Final = 0.3  # an offset is judged only if it compares this share of the evidence the densest offset does


def _evidence(moving_a: np.ndarray, moving_b: np.ndarray, blank_a: np.ndarray, blank_b: np.ndarray) -> np.ndarray:
    """Cells that can tell offsets apart.

    Changed in place on both sides of the pair -- an overlay, which stays put, is never held against the true offset
    -- and not blank on both, which agrees with any offset.
    """
    return moving_a & moving_b & ~(blank_a & blank_b)


def _trimmed(errors: np.ndarray, kept: np.ndarray, outliers: float) -> tuple[float, int]:
    """The mean of the ``kept`` errors with the worst ``outliers`` share left out, and how many were kept."""
    chosen = errors[kept]
    count = len(chosen)
    if not count:
        return float('inf'), 0
    take = max(1, int(np.ceil(count * (1 - outliers))))
    if take < count:
        chosen = np.partition(chosen, take - 1)[:take]  # the take smallest, in linear time
    return float(chosen.mean()), count


def _squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every offset ``d``, the mean squared difference of rows ``a[d:]`` and ``b[:-d]`` (``(rows, cells)`` each).

    Over every row and cell at once: the cross term is a cross-correlation along the rows, computed by FFT, so ranking
    all offsets costs a few milliseconds and misses no row -- sparse content (a few lines on a blank page) included.
    """
    rows = len(a)
    size = 1 << (2 * rows - 1).bit_length()
    cross = np.fft.irfft(np.fft.rfft(a, size, axis=0) * np.conj(np.fft.rfft(b, size, axis=0)), size, axis=0)
    cross = cross[:rows].sum(axis=1)  # cross[d] = sum over y, cells of a[y + d] * b[y]
    a2 = np.cumsum((a * a).sum(axis=1)[::-1])[::-1]  # a2[d] = sum of a[d:] squared
    b2 = np.cumsum((b * b).sum(axis=1))  # b2[n - 1] = sum of b[:n] squared
    overlap = rows - np.arange(rows)
    return (a2 + b2[overlap - 1] - 2 * cross) / (overlap * a.shape[1])


class _Pair:
    """Two frames' cells, laid side by side once, to judge any offset between them."""

    def __init__(self, ca: np.ndarray, cb: np.ndarray, outliers: float) -> None:
        (pa, self.blank_a), (pb, self.blank_b) = _cells(ca), _cells(cb)
        self.rows, bins = self.blank_a.shape
        self.a, self.b = pa.reshape(self.rows, bins, 3), pb.reshape(self.rows, bins, 3)
        self.moving = np.abs(self.a - self.b).max(axis=2) > _STILL
        self.outliers = outliers

    def judge(self, d: int) -> tuple[float, int]:
        """The trimmed mean grey error at offset ``d``, over the cells that carry evidence, and their count."""
        n = self.rows - d
        kept = _evidence(self.moving[d:], self.moving[:n], self.blank_a[d:], self.blank_b[:n])
        return _trimmed(np.abs(self.a[d:] - self.b[:n]).mean(axis=2), kept, self.outliers)

    def candidates(self, least: int) -> dict[int, tuple[float, int]]:
        """The :data:`_CANDIDATES` offsets :func:`_squared` ranks best, each judged in full."""
        offsets = np.arange(1, self.rows - least + 1)
        if not len(offsets):
            return {}
        ranked = _squared(self.a.mean(axis=2), self.b.mean(axis=2))[offsets]
        return {int(d): self.judge(int(d)) for d in offsets[np.argsort(ranked)[:_CANDIDATES]]}


def _repeats(cells: np.ndarray, least: int) -> bool:
    """Does the view equal itself shifted by some offset (a blank or a repeating pattern)?"""
    rows = len(cells)
    return any((np.abs(cells[d:] - cells[: rows - d]) <= _STILL).all() for d in range(1, rows - least + 1))


def _unmoved(cells: np.ndarray, least: int) -> float:
    """0 -- the view did not scroll -- unless it is blank or repeats, when whether it scrolled cannot be told."""
    if _repeats(cells, least):
        msg = 'the view is blank or repeats, so whether it scrolled cannot be told: scroll past it'
        raise AmbiguousOverlap(msg)
    return 0.0


def offset(a: np.ndarray, b: np.ndarray, match: Match = Match()) -> int:  # noqa: B008 -- frozen
    """Rows the view scrolled from frame ``a`` to frame ``b`` (0: it did not move), to the nearest row.

    :func:`_place` says how many to the fraction of a row.
    """
    return round(_place(a, b, match))


def _place(a: np.ndarray, b: np.ndarray, match: Match) -> float:
    """Rows the view scrolled from frame ``a`` to frame ``b``, to a fraction of a row (0: it did not move).

    At 175% scaling a scroll is a fraction of a row: the whole-row offset that fits best is refined by the parabola
    through its error and its two neighbours', so a stitcher that adds the fractions up never drifts.

    The offset at which the frames are most alike: its trimmed mean error at most ``match.tolerance`` grey levels,
    and every offset but its neighbours erring :data:`_UNIQUE` times as much. Raises :class:`NoOverlap` or
    :class:`AmbiguousOverlap` rather than return an offset that may be wrong.
    """
    if a.shape != b.shape:
        msg = f'frames differ in shape: {a.shape} != {b.shape}'
        raise ValueError(msg)
    ca, cb = _content(a, match), _content(b, match)
    rows = len(ca)
    least = max(1, rows // 6) if match.min_overlap is None else match.min_overlap
    (pa, _), (pb, _) = _cells(ca), _cells(cb)
    changed = np.abs(pa - pb).reshape(rows, -1, 3).max(axis=2) > _STILL
    if not changed.any():
        return _unmoved(pa, least)
    pair = _Pair(ca, cb, match.outliers)
    judged = pair.candidates(least)
    densest = max((count / (rows - d) for d, (_, count) in judged.items()), default=0.0)
    found = {
        d: (error, count)
        for d, (error, count) in judged.items()
        if count >= _LEAST_CELLS and count / (rows - d) >= _DENSE * densest
    }
    best = min(found, key=lambda d: found[d][0]) if found else None
    if best is None or found[best][0] > match.tolerance:
        # Nothing explains the change: a small one is something moving inside a still view (a caret, an animation);
        # a sparse view that really scrolled is explained by its offset above, so it never reaches here.
        if changed.mean() <= match.outliers:
            return _unmoved(pa, least)
        if best is None:
            msg = 'the shared rows hold nothing that moved, so any offset fits: scroll by less, or past the blank'
            raise AmbiguousOverlap(msg)
        msg = f'no offset leaves {least} shared rows alike (best errs {found[best][0]:.1f} grey levels): scroll by less'
        raise NoOverlap(msg, found[best][0])
    error = found[best][0]
    bar = max(_UNIQUE * error, error + _FLOOR)
    low = {d for d, (e, _) in found.items() if e < bar}

    def valley(d: int) -> bool:  # every offset from best to d fits: one flat valley, not a repetition
        step = 1 if d > best else -1
        return all(x in low for x in range(best, d + step, step))

    rivals = sorted(d for d in low if abs(d - best) > _NEAR and not valley(d))
    if rivals:
        msg = f'{1 + len(rivals)} offsets fit ({[best, *rivals][:5]}...): the overlap repeats; scroll by another step'
        raise AmbiguousOverlap(msg)
    lower, upper = (found.get(d) or pair.judge(d) for d in (best - 1, best + 1))
    curve = lower[0] + upper[0] - 2 * error
    exact = error <= _EXACT  # a perfect match is a whole-row scroll: nothing fractional to estimate
    if exact or best <= 1 or not (np.isfinite(lower[0]) and np.isfinite(upper[0])) or curve <= 0:
        return float(best)
    return best + min(max(0.5 * (lower[0] - upper[0]) / curve, -0.5), 0.5)


def _squared_in(tall: np.ndarray, view: np.ndarray, least: int) -> np.ndarray:
    """For every position ``p`` of ``view`` in ``tall``, the mean squared difference of the rows they share.

    Both are ``(rows, cells)``; ``view`` overlaps ``tall`` by at least ``least`` rows. By FFT, like :func:`_squared`.
    """
    rows, height = len(view), len(tall)
    size = 1 << (height + rows).bit_length()
    cross = np.fft.irfft(np.fft.rfft(tall, size, axis=0) * np.conj(np.fft.rfft(view, size, axis=0)), size, axis=0)
    cross = cross[: height - least + 1].sum(axis=1)  # cross[p] = sum over y, cells of tall[p + y] * view[y]
    positions = np.arange(height - least + 1)
    shared = np.minimum(rows, height - positions)
    tall2 = np.concatenate([[0.0], np.cumsum((tall * tall).sum(axis=1))])
    view2 = np.concatenate([[0.0], np.cumsum((view * view).sum(axis=1))])
    return (tall2[positions + shared] - tall2[positions] + view2[shared] - 2 * cross) / (shared * tall.shape[1])


def locate(tall: np.ndarray, frame: np.ndarray, match: Match = Match()) -> int:  # noqa: B008 -- frozen
    """The row of ``tall`` -- the image so far -- at which ``frame`` starts.

    Where a view that lost the last frame is found again: ``frame`` shares at least the least overlap with ``tall``.
    Refused, never guessed, like :func:`offset`.
    """
    view = _content(frame, match)
    rows = len(view)
    least = max(1, rows // 6) if match.min_overlap is None else match.min_overlap
    if len(tall) < least:
        msg = 'too little captured yet to find the view in'
        raise NoOverlap(msg)
    (pt, blank_t), (pv, blank_v) = _cut(tall), _cells(view)
    bins = blank_t.shape[1]
    fine_t, fine_v = pt.reshape(len(tall), bins, 3), pv.reshape(rows, bins, 3)
    ranked = _squared_in(fine_t.mean(axis=2), fine_v.mean(axis=2), least)
    found = {}
    for p in (int(p) for p in np.argsort(ranked)[:_CANDIDATES]):
        shared = min(rows, len(tall) - p)
        kept = ~(blank_t[p : p + shared] & blank_v[:shared])
        found[p] = _trimmed(np.abs(fine_t[p : p + shared] - fine_v[:shared]).mean(axis=2), kept, match.outliers)
    found = {p: e for p, e in found.items() if e[1] >= _LEAST_CELLS}
    if not found:
        msg = 'the view shows nothing that tells where it is'
        raise AmbiguousOverlap(msg)
    best = min(found, key=lambda p: found[p][0])
    error = found[best][0]
    if error > match.tolerance:
        msg = f'the view is not in what was captured (best errs {error:.1f} grey levels)'
        raise NoOverlap(msg, error)
    bar = max(_UNIQUE * error, error + _FLOOR)
    low = {p for p, (e, _) in found.items() if e < bar}

    def valley(p: int) -> bool:
        step = 1 if p > best else -1
        return all(x in low for x in range(best, p + step, step))

    if any(abs(p - best) > _NEAR and not valley(p) for p in low):
        msg = 'the view fits more than one place in what was captured'
        raise AmbiguousOverlap(msg)
    return best - match.top


_SIDEWAYS: Final = 8  # sideways shifts tried, best first by the columns' ink: a moved page is among the first


def shifted(image: np.ndarray, dx: int) -> np.ndarray:
    """``image`` with its content ``dx`` columns further left (right when negative); the edge column fills in."""
    if dx > 0:
        return np.concatenate([image[:, dx:], np.repeat(image[:, -1:], dx, axis=1)], axis=1)
    if dx < 0:
        return np.concatenate([np.repeat(image[:, :1], -dx, axis=1), image[:, :dx]], axis=1)
    return image


def _ink(image: np.ndarray) -> np.ndarray:
    """Each column's ink: how far it is from the image's commonest grey, summed down the rows."""
    grey = image.mean(axis=2)
    paper = np.median(grey)
    return np.abs(grey - paper).sum(axis=0)


def drift(tall: np.ndarray, frame: np.ndarray, match: Match = Match(), reach: int | None = None) -> tuple[int, int]:  # noqa: B008
    """Where ``frame`` starts in ``tall`` and how many columns its content sits further right: ``(row, dx)``.

    A page that moved sideways between two captures -- a reader reopened, its side bar or zoom margin changed --
    shows the same rows ``dx`` columns over. The shifts within ``reach`` (default a quarter of the width) are ranked by
    how well the columns' ink lines up -- the margins of text tell -- and each in turn is checked as strictly as
    :func:`locate` checks a view: the first that places ``frame`` uniquely is the answer, 0 tried first. None
    placing it is refused, never guessed.
    """
    width = frame.shape[1]
    reach = width // 4 if reach is None else reach
    ta, fa = _ink(tall), _ink(frame)
    ta, fa = ta / max(ta.sum(), 1.0), fa / max(fa.sum(), 1.0)

    def misfit(dx: int) -> float:
        lo, hi = max(0, dx), min(width, width + dx)
        return float(np.abs(fa[lo:hi] - ta[lo - dx : hi - dx]).sum())

    ranked = sorted(range(-reach, reach + 1), key=misfit)[:_SIDEWAYS]
    refusals: list[Unstitchable] = []
    for dx in [0, *(d for d in ranked if d)]:
        try:
            return locate(tall, shifted(frame, dx), match), dx
        except Unstitchable as error:
            refusals.append(error)
    raise min(refusals, key=_closeness)


def _closeness(refused: Unstitchable) -> float:
    """How near a refusal came: an ambiguous fit before any miss, then the miss that erred least."""
    if isinstance(refused, NoOverlap) and refused.error is not None:
        return refused.error
    return -1.0 if isinstance(refused, AmbiguousOverlap) else float('inf')


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

    def __init__(
        self,
        first: np.ndarray,
        match: Match = Match(),  # noqa: B008 -- frozen
        *,
        drop_overlays: bool = False,
        history: int = 0,
    ) -> None:
        """Start from ``first``. ``history``: rows above the current view kept, not drained, for :meth:`relocate`."""
        self.match, self.drop_overlays, self.last, self.history = match, drop_overlays, first, history
        rows, width = first.shape[:2]
        self._chrome = np.zeros((rows, width), dtype=bool)
        self._chrome[: match.top] = self._chrome[rows - match.bottom :] = True
        self._proven = np.zeros((rows, width), dtype=bool)  # per screen pixel, as of the last frame
        self._edge = np.abs(np.arange(rows) - rows // 2)[:, None].astype(np.uint32)
        self._canvas = np.empty((4 * rows, width, 3), dtype=np.uint8)
        self._rank = np.zeros((4 * rows, width), dtype=np.uint32)  # the rank of the view each spot holds
        self._canvas[:rows], self._rank[:rows] = first, self._ranks()
        self._top, self._end = 0, rows  # in buffer rows; the buffer starts at image row self._drained
        self._at = 0.0  # the last frame's exact top, to a fraction of a row: self._top is it rounded
        self._drained = 0

    def _ranks(self) -> np.ndarray:
        kind = np.where(self._chrome, CHROME, np.where(self._proven, OVERLAID, CLEAR)).astype(np.uint32)
        return kind * _EDGE + self._edge

    @property
    def height(self) -> int:
        """Rows of the image so far, drained or held."""
        return self._drained + self._end

    @property
    def held(self) -> int:
        """Rows held in memory, not yet drained."""
        return self._end

    def drain(self, *, final: bool = False) -> np.ndarray:
        """The rows no later frame can change -- all of them when ``final`` -- handed over and forgotten."""
        cut = self._end if final else max(self._top - self.history, 0)
        rows = self._canvas[:cut].copy()
        self._canvas[: self._end - cut] = self._canvas[cut : self._end]
        self._rank[: self._end - cut] = self._rank[cut : self._end]
        self._top, self._end, self._drained = max(self._top - cut, 0), self._end - cut, self._drained + cut
        self._at = max(self._at - cut, 0.0)
        return rows

    def _room(self, end: int) -> None:
        if end > len(self._canvas):
            more = max(end, 2 * len(self._canvas)) - len(self._canvas)
            width = self._canvas.shape[1]
            self._canvas = np.concatenate([self._canvas, np.empty((more, width, 3), np.uint8)])
            self._rank = np.concatenate([self._rank, np.zeros((more, width), np.uint32)])

    def add(self, frame: np.ndarray) -> int:
        """Place ``frame`` after the last one; the rows it scrolled (0: it adds nothing)."""
        exact = self._at + _place(self.last, frame, self.match)
        moved = round(exact) - self._top
        if not moved:
            return 0  # less than a row so far: the next frame is measured from the same last one, nothing lost
        self._at = exact
        rows = len(frame)
        if not self.drop_overlays:
            return self._append(frame, moved)
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
        take = fresh | (rank < held)
        self._canvas[top:end][take], held[take] = frame[take], rank[take]
        self._end, self.last = end, frame
        return moved

    def _append(self, frame: np.ndarray, moved: int) -> int:
        """Without ``drop_overlays``, rows only: what ``frame`` newly shows goes below.

        The last frame's bottom chrome -- a footer that does not scroll -- gives way to the content ``frame`` shows
        there; nothing else is ranked or replaced.
        """
        rows, bottom = len(frame), self.match.bottom
        self._top += moved
        top, end = self._top, self._top + rows
        self._room(end)
        start = self._end - bottom - top  # the frame row at the last frame's footer, or the first row it newly shows
        self._canvas[top + start : end] = frame[start:]
        self._end, self.last = end, frame
        return moved

    def relocate(self, frame: np.ndarray) -> int:
        """Find ``frame`` anywhere in the rows held (the last ``history`` rows and the view) and go on from there.

        A view that lost the last frame -- scrolled too far, or back -- picks up wherever it overlaps what was
        captured; the rows it newly shows go below. How many it added (0: it was all captured).
        """
        top = locate(self._canvas[: self._end], frame, self.match)
        rows = len(frame)
        if top < 0 or top + rows - self.match.bottom <= 0:
            msg = 'the view is above what is held'
            raise NoOverlap(msg)
        self._top, self.last, self._at = top, frame, float(top)
        end = top + rows
        added = max(end - self._end, 0)
        if added:
            self._room(end)
            start = self._end - self.match.bottom - top
            self._canvas[top + start : end] = frame[start:]
            self._rank[top + start : end] = self._ranks()[start:]
            self._end = end
        self._proven[:] = False  # what was proven of the last frame's screen says nothing across a jump
        return added

    def image(self) -> np.ndarray:
        """The rows held, as one image: the whole stitched image while nothing has been drained."""
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
