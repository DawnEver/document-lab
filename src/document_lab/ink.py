"""Where a page's picture holds DRAWN ink -- a table's rules, a figure's lines -- apart from its text.

`document_lab.regions` tells regions by their words; a table drawn in two columns, or a figure with no caption, has
too few words to be told so. Its ink is what tells it, and the page's picture is where the ink is: the caller that
holds the picture (`document_lab.per_page`) asks :func:`drawings` and hands the boxes to
:func:`document_lab.regions.regions`, so regions never reads pixels.

A pixel darker than :data:`DARK` is ink. The text's ink is wiped first (the caller's text boxes: OCR'd words, a
formula read as noise among them), then ink within ``reach`` points (the caller's: a page's scale is its own) is
one drawing. Each drawing is a box in
points, whatever its size, with what it is drawn with (:class:`Drawing`): its rules, the ink off them, its longest
stroke. Which drawings are regions, and of what kind, is `document_lab.regions`' judgement, not this module's.
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
#: The grid ink is judged on, in points a cell: finer than any rule's spacing, coarse enough to be quick.
CELL: Final = 1.0
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
    step = max(1, round(CELL * scale))
    rows, cols = -(-grey.shape[0] // step), -(-grey.shape[1] // step)
    dark = np.zeros((rows * step, cols * step), dtype=bool)
    dark[: grey.shape[0], : grey.shape[1]] = grey < DARK
    grid = dark.reshape(rows, step, cols, step).any(axis=(1, 3))
    cell = step / scale  # points a cell
    for box in text:
        grid[int(box.y0 / cell) : math.ceil(box.y1 / cell) + 1, int(box.x0 / cell) : math.ceil(box.x1 / cell) + 1] = (
            False
        )
    reach_cells = max(1, round(reach / cell))
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
        held = [(r, c) for r, c in cells if grid[r, c]]  # its own ink, not the reach grown round it
        if not held:
            continue
        rs, cs = [r for r, _ in held], [c for _, c in held]
        own = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
        own[np.array(rs) - min(rs), np.array(cs) - min(cs)] = True
        box = BBox(min(cs) * cell, min(rs) * cell, (max(cs) + 1) * cell, (max(rs) + 1) * cell)
        rules, loose = _rules(own)
        out.append(Drawing(box, rules, loose * cell, _longest(own) * cell))
    return sorted(out, key=lambda d: (d.box.y0, d.box.x0))


def _rules(own: np.ndarray) -> tuple[tuple[int, int], int]:
    """The horizontal and vertical rules ``own`` (one drawing's ink) is drawn with, and its largest piece off them.

    A rule is a run of rows (columns) each inked across :data:`RULE_SHARE` of the drawing. The piece is the larger
    side, in cells, of the largest connected ink off every rule (0 when there is none): a table's words, a plot's
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
    """How many runs of True ``flags`` holds: neighbouring inked rows are one rule."""
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
