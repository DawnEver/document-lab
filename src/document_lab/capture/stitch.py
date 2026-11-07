"""Overlapping viewport frames of one scrolled view, stitched into the one tall image they show.

A frame is an ``(height, width, 3)`` uint8 array. ``top`` and ``bottom`` rows of every frame are fixed chrome (a
header, a toolbar, a status bar) that does not scroll: they are left out of the match and kept once in the result.
The scroll between two frames is the one offset at which the rest of the views agree: exactly by default, or within
``tolerance`` -- the mean absolute grey-level difference (0-255) of the shared rows, each row averaged into
:data:`BINS` column bins -- so a blinking caret, a hover highlight or anti-aliasing jitter does not break the match.
An offset that is not unique, or an overlap with no row that could tell offsets apart, is refused -- never guessed.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['BINS', 'AmbiguousOverlap', 'NoOverlap', 'Unstitchable', 'assemble', 'offset', 'stitch']


BINS: Final = 64


class Unstitchable(ValueError):
    """Two frames whose relative scroll cannot be determined."""


class NoOverlap(Unstitchable):
    """No offset leaves enough shared rows: the scroll moved (almost) a whole view, or the view changed."""


class AmbiguousOverlap(Unstitchable):
    """More than one offset fits, or the shared rows are all uniform (blank) and fit any offset."""


def _content(frame: np.ndarray, top: int, bottom: int) -> np.ndarray:
    return frame[top : len(frame) - bottom]


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


def _fits(ca: np.ndarray, cb: np.ndarray, least: int, tolerance: float) -> list[int]:
    height = len(ca)
    if not tolerance:
        sa, sb = _signatures(ca, cb)
        return [d for d in range(height - least + 1) if np.array_equal(sa[d:], sb[: height - d])]
    pa, pb = _profile(ca), _profile(cb)
    cost = np.array([np.abs(pa[d:] - pb[: height - d]).mean() for d in range(height - least + 1)])
    if not len(cost) or cost.min() > tolerance:
        return []
    best = int(cost.argmin())
    rivals = [d for d in np.flatnonzero(cost <= tolerance) if abs(d - best) > 1]
    return [best, *rivals]


def offset(
    a: np.ndarray,
    b: np.ndarray,
    *,
    top: int = 0,
    bottom: int = 0,
    min_overlap: int | None = None,
    tolerance: float = 0,
) -> int:
    """Rows the view scrolled from frame ``a`` to frame ``b`` (0: it did not move).

    ``min_overlap`` is the fewest content rows the two views must share (default: a quarter of the content
    height); ``tolerance`` see the module. Raises :class:`NoOverlap` or :class:`AmbiguousOverlap` rather than
    return an offset that may be wrong.
    """
    if a.shape != b.shape:
        msg = f'frames differ in shape: {a.shape} != {b.shape}'
        raise ValueError(msg)
    ca, cb = _content(a, top, bottom), _content(b, top, bottom)
    height = len(ca)
    least = max(1, height // 4) if min_overlap is None else min_overlap
    fits = _fits(ca, cb, least, tolerance)
    if not fits:
        msg = f'no offset leaves {least} shared rows: scroll by less than {height - least} rows per step'
        raise NoOverlap(msg)
    if len(fits) > 1:
        msg = f'{len(fits)} offsets fit ({fits[:5]}...): the overlap repeats; scroll by a different step'
        raise AmbiguousOverlap(msg)
    shared = ca[fits[0] :]
    flat = shared.reshape(len(shared), -1)
    if not (flat != flat[:, :1]).any():
        msg = 'the shared rows are all uniform, so any offset fits: scroll by less, or past the blank'
        raise AmbiguousOverlap(msg)
    return fits[0]


def assemble(frames: Sequence[np.ndarray], offsets: Sequence[int], *, top: int = 0, bottom: int = 0) -> np.ndarray:
    """The tall image of ``frames`` whose successive scrolls are ``offsets`` (one fewer than the frames)."""
    if not frames or len(offsets) != len(frames) - 1:
        msg = f'{len(frames)} frames need {max(len(frames) - 1, 0)} offsets, got {len(offsets)}'
        raise ValueError(msg)
    first, last = frames[0], frames[-1]
    height = len(first) - top - bottom
    parts = [first[: len(first) - bottom]]
    parts += [_content(b, top, bottom)[height - d :] for b, d in zip(frames[1:], offsets, strict=True) if d]
    parts.append(last[len(last) - bottom :])
    return np.concatenate(parts)


def stitch(
    frames: Sequence[np.ndarray],
    *,
    top: int = 0,
    bottom: int = 0,
    min_overlap: int | None = None,
    tolerance: float = 0,
) -> np.ndarray:
    """The tall image ``frames`` show in order; repeated frames add nothing. See :func:`offset` for the knobs."""
    if not frames:
        msg = 'no frames to stitch'
        raise ValueError(msg)
    knobs = {'top': top, 'bottom': bottom, 'min_overlap': min_overlap, 'tolerance': tolerance}
    return assemble(frames, [offset(a, b, **knobs) for a, b in pairwise(frames)], top=top, bottom=bottom)
