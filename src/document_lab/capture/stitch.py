"""Overlapping viewport frames of one scrolled view, stitched into the one tall image they show.

A frame is an ``(height, width, 3)`` uint8 array. ``top`` and ``bottom`` rows of every frame are fixed chrome (a
header, a toolbar, a status bar) that does not scroll: they are left out of the match and kept once in the result.
The scroll between two frames is the one offset at which the rest of the views agree: exactly by default, or within
``tolerance`` -- the mean absolute grey-level difference (0-255) of the shared rows, each row averaged into
:data:`BINS` column bins -- so a blinking caret, a hover highlight or anti-aliasing jitter does not break the match.
An offset that is not unique, or an overlap with no row that could tell offsets apart, is refused -- never guessed.

:class:`Stitcher` holds only the last frame and the rows already placed, so a capture of any length costs the image
it produces, not every frame grabbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['BINS', 'AmbiguousOverlap', 'Match', 'NoOverlap', 'Stitcher', 'Unstitchable', 'offset', 'stitch']

BINS: Final = 64
_CHUNK: Final = 128  # rows of the row-distance matrix computed at once: bounds the memory of a tolerant match


class Unstitchable(ValueError):
    """Two frames whose relative scroll cannot be determined."""


class NoOverlap(Unstitchable):
    """No offset leaves enough shared rows: the scroll moved (almost) a whole view, or the view changed."""


class AmbiguousOverlap(Unstitchable):
    """More than one offset fits, or the shared rows are all uniform (blank) and fit any offset."""


@dataclass(frozen=True)
class Match:
    """How frames are matched. ``min_overlap``: fewest shared content rows (default a quarter of the content)."""

    top: int = 0
    bottom: int = 0
    min_overlap: int | None = None
    tolerance: float = 0


def _content(frame: np.ndarray, match: Match) -> np.ndarray:
    return frame[match.top : len(frame) - match.bottom]


def _signatures(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row of ``a`` and ``b`` as an integer, equal exactly where the rows are equal."""
    rows = np.concatenate([a, b]).reshape(len(a) + len(b), -1)
    _, ids = np.unique(rows, axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    return ids[: len(a)], ids[len(a) :]


def _profile(frame: np.ndarray) -> np.ndarray:
    """Each row's grey level averaged into at most :data:`BINS` column bins."""
    grey = frame.astype(np.float32).mean(axis=2)
    edges = np.linspace(0, grey.shape[1], min(BINS, grey.shape[1]) + 1).astype(int)
    return np.add.reduceat(grey, edges[:-1], axis=1) / np.diff(edges)


def _costs(ca: np.ndarray, cb: np.ndarray, offsets: int) -> np.ndarray:
    """Mean profile difference of the shared rows at each offset ``0 .. offsets-1``."""
    pa, pb = _profile(ca), _profile(cb)
    height = len(pa)
    distance = np.empty((height, height), dtype=np.float32)  # distance[i, j]: row i of a against row j of b
    for start in range(0, height, _CHUNK):
        distance[start : start + _CHUNK] = np.abs(pa[start : start + _CHUNK, None, :] - pb[None, :, :]).mean(axis=2)
    return np.array([distance.diagonal(-d).mean() for d in range(offsets)])


def _fits(ca: np.ndarray, cb: np.ndarray, least: int, tolerance: float) -> list[int]:
    height = len(ca)
    if not tolerance:
        sa, sb = _signatures(ca, cb)
        return [d for d in range(height - least + 1) if np.array_equal(sa[d:], sb[: height - d])]
    cost = _costs(ca, cb, height - least + 1)
    if not len(cost) or cost.min() > tolerance:
        return []
    best = int(cost.argmin())
    rivals = [int(d) for d in np.flatnonzero(cost <= tolerance) if abs(d - best) > 1]
    return [best, *rivals]


def offset(a: np.ndarray, b: np.ndarray, match: Match = Match()) -> int:  # noqa: B008 -- frozen
    """Rows the view scrolled from frame ``a`` to frame ``b`` (0: it did not move).

    Raises :class:`NoOverlap` or :class:`AmbiguousOverlap` rather than return an offset that may be wrong.
    """
    if a.shape != b.shape:
        msg = f'frames differ in shape: {a.shape} != {b.shape}'
        raise ValueError(msg)
    ca, cb = _content(a, match), _content(b, match)
    height = len(ca)
    least = max(1, height // 4) if match.min_overlap is None else match.min_overlap
    fits = _fits(ca, cb, least, match.tolerance)
    if not fits:
        msg = f'no offset leaves {least} shared rows: scroll by less than {height - least} rows per step'
        raise NoOverlap(msg)
    if len(fits) > 1:
        msg = f'{len(fits)} offsets fit ({fits[:5]}...): the overlap repeats; scroll by a different step'
        raise AmbiguousOverlap(msg)
    flat = ca[fits[0] :].reshape(height - fits[0], -1)
    if not (flat != flat[:, :1]).any():
        msg = 'the shared rows are all uniform, so any offset fits: scroll by less, or past the blank'
        raise AmbiguousOverlap(msg)
    return fits[0]


@dataclass
class Stitcher:
    """The tall image so far: frames are added in scroll order, each contributing the rows it newly shows."""

    first: np.ndarray
    match: Match = Match()
    last: np.ndarray = field(init=False)
    _rows: list[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        """Start from the first frame without its bottom chrome (the last frame's is added at the end)."""
        self.last = self.first
        self._rows = [self.first[: len(self.first) - self.match.bottom]]

    @property
    def height(self) -> int:
        """Rows of the image so far."""
        return sum(len(r) for r in self._rows) + self.match.bottom

    def add(self, frame: np.ndarray) -> int:
        """Place ``frame`` after the last one; the rows it scrolled (0: it adds nothing)."""
        moved = offset(self.last, frame, self.match)
        if moved:
            end = len(frame) - self.match.bottom
            self._rows.append(frame[end - moved : end])
            self.last = frame
        return moved

    def image(self) -> np.ndarray:
        """The stitched image, the last frame's bottom chrome below it."""
        return np.concatenate([*self._rows, self.last[len(self.last) - self.match.bottom :]])


def stitch(frames: Sequence[np.ndarray], match: Match = Match()) -> np.ndarray:  # noqa: B008 -- frozen
    """The tall image ``frames`` show in order; repeated frames add nothing."""
    if not frames:
        msg = 'no frames to stitch'
        raise ValueError(msg)
    stitcher = Stitcher(frames[0], match)
    for frame in frames[1:]:
        stitcher.add(frame)
    return stitcher.image()
