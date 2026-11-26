"""Where a page's picture holds DRAWN ink -- a table's rules, a figure's lines -- apart from its text.

`document_lab.regions` tells regions by their words; a table drawn in two columns, or a figure with no caption, has
too few words to be told so. Its ink is what tells it, and the page's picture is where the ink is: the caller that
holds the picture (`document_lab.per_page`) asks :func:`drawings` and hands the boxes to
:func:`document_lab.regions.regions`, so regions never reads pixels.

A pixel darker than :data:`DARK` is ink. The text's ink is wiped first (the caller's text boxes: OCR'd words, a
formula read as noise among them), then ink within ``reach`` points (the caller's: a page's scale is its own) is
one drawing. Each drawing is a box in
points, whatever its size, with what it is drawn with (:class:`Drawing`): its rules, the ink off them, its longest
stroke. Which drawings are regions, and of what kind, is `document_lab.regions`' judgment, not this module's.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.ir import BBox

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['DARK', 'RULE_SHARE', 'Drawing', 'drawings']

#: A pixel darker than this grey level (0 black .. 255 white) is ink.
DARK: Final = 128
#: Drawings are found on a grid of this many cells to ``reach`` -- coarse enough to be quick on a page of millions of
#: pixels -- and measured on their own pixels.
REACH_CELLS: Final = 3
#: A rule runs across at least this share of its drawing.
RULE_SHARE: Final = 0.5


@dataclass(frozen=True)
class Drawing:
    """A drawing: its ``box`` (points), its ``rules``, the ``loose`` ink off them and its longest straight ``stroke``.

    ``rules``: the horizontal and vertical lines across at least :data:`RULE_SHARE` of it; ``loose``: the larger side
    (points) of its largest ink off every rule -- a table's words are small, a plot's curve is not; ``stroke``: points,
    across or down -- a glyph's is short, a drawn line's long.
    """

    box: BBox
    rules: tuple[int, int]
    loose: float
    stroke: float


def drawings(pixels: np.ndarray, scale: float, text: Sequence[BBox], reach: float) -> list[Drawing]:
    """The boxes (points) of the drawn ink in ``pixels``, top to bottom; the ink in the ``text`` boxes left out.

    ``pixels``: rows top-down, grey, RGB or RGBA, at ``scale`` pixels per point; ``text``: boxes in points; ink
    within ``reach`` points of other ink is one drawing.
    """
    grey = pixels if pixels.ndim == 2 else pixels[..., :3].mean(axis=2)  # noqa: PLR2004 -- rows by columns
    dark = grey < DARK
    for box in text:  # the text's ink, wiped pixel by pixel
        dark[
            int(box.y0 * scale) : math.ceil(box.y1 * scale) + 1, int(box.x0 * scale) : math.ceil(box.x1 * scale) + 1
        ] = False
    # Drawings are found on a coarse grid (quick on millions of pixels) and measured on their own pixels (exact).
    step = max(1, round(reach / REACH_CELLS * scale))
    rows, cols = -(-dark.shape[0] // step), -(-dark.shape[1] // step)
    padded = np.zeros((rows * step, cols * step), dtype=bool)
    padded[: dark.shape[0], : dark.shape[1]] = dark
    grid = padded.reshape(rows, step, cols, step).any(axis=(1, 3))
    reach_cells = max(1, round(reach * scale / step))
    grown = grid.copy()
    for shift in range(1, reach_cells + 1):  # ink within reach joins: grow it by reach both ways
        grown[shift:] |= grid[:-shift]
        grown[:-shift] |= grid[shift:]
    wide = grown.copy()
    for shift in range(1, reach_cells + 1):
        wide[:, shift:] |= grown[:, :-shift]
        wide[:, :-shift] |= grown[:, shift:]
    out = []
    for cells in _components(wide):
        rs, cs = (np.array(v) for v in zip(*cells, strict=True))
        r0, c0 = rs.min(), cs.min()
        mask = np.zeros((rs.max() - r0 + 1, cs.max() - c0 + 1), dtype=bool)
        mask[rs - r0, cs - c0] = True
        top, left = r0 * step, c0 * step
        cut = padded[top : top + mask.shape[0] * step, left : left + mask.shape[1] * step]
        own = cut & np.repeat(np.repeat(mask, step, axis=0), step, axis=1)  # its own ink, not the reach round it
        ys, xs = np.nonzero(own)
        if not ys.size:
            continue
        own = own[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        y0, x0 = top + ys.min(), left + xs.min()
        box = BBox(x0 / scale, y0 / scale, (x0 + own.shape[1]) / scale, (y0 + own.shape[0]) / scale)
        rules, loose = _rules(own)
        out.append(Drawing(box, rules, loose / scale, _longest(own) / scale))
    return sorted(out, key=lambda d: (d.box.y0, d.box.x0))


def _rules(own: np.ndarray) -> tuple[tuple[int, int], int]:
    """The horizontal and vertical rules ``own`` (one drawing's ink) is drawn with, and its largest piece off them.

    A rule is a run of rows (columns) each inked across :data:`RULE_SHARE` of the drawing. The piece is the larger
    side, in pixels, of the largest connected ink off every rule (0 when there is none): a table's words, a plot's
    curve.
    """
    rows = own.sum(axis=1) >= RULE_SHARE * own.shape[1]
    cols = own.sum(axis=0) >= RULE_SHARE * own.shape[0]
    off = own & ~(rows[:, None] | cols[None, :])
    loose = 0
    for cells in _components(off):
        rs, cs = [r for r, _ in cells], [c for _, c in cells]
        loose = max(loose, max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
    return (_runs(rows), _runs(cols)), loose


def _longest(own: np.ndarray) -> int:
    """The longest run of inked cells along a row or a column of ``own``."""
    best = 0
    for grid in (own, own.T):
        padded = np.pad(grid, ((0, 0), (1, 1))).astype(np.int8)
        edges = np.diff(padded, axis=1)
        for row in range(grid.shape[0]):
            starts, ends = np.flatnonzero(edges[row] == 1), np.flatnonzero(edges[row] == -1)
            if starts.size:
                best = max(best, int((ends - starts).max()))
    return best


def _runs(flags: np.ndarray) -> int:
    """How many runs of True ``flags`` holds: neighboring inked rows are one rule."""
    return int(np.count_nonzero(flags[1:] & ~flags[:-1]) + bool(flags[0])) if flags.size else 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """The 4-connected components of ``mask``'s True cells."""
    seen = np.zeros_like(mask)
    out = []
    for start in zip(*np.nonzero(mask), strict=True):
        if seen[start]:
            continue
        seen[start] = True
        cells, todo = [], deque([start])
        while todo:
            r, c = todo.popleft()
            cells.append((int(r), int(c)))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        out.append(cells)
    return out
