"""A page's ruled tables, found from its rules and text alone: a grid of rules is a table of cells, each with its box.

- Rules: each level or upright side of a stroked path, and each filled shape no thicker than :data:`THIN` points (a
  rule drawn as a bar), in a color that shows on white. A wider fill is shading: no rule, and it breaks no table.
  Along a line, gaps no wider than :data:`TOLERANCE` points are no gaps.
- Grids: rules that touch are one grid; its rules within :data:`TOLERANCE` of one another are one line (a double rule
  too), and between its lines stand its slots.
- Cells: slots no rule parts are one cell. A cell spanning slots repeats its text and its own box in each slot it
  covers, as :mod:`document_lab.html` repeats a spanning cell, so every row reads the same width.
- A cell's text: the spans whose center stands in it, as lines (:func:`document_lab.layout.lines`) top to bottom.

A grid is a table only when nothing about it is a guess: each slot side is ruled end to end or not at all, its outline
is closed and no rule runs on past it -- a side no rule closes is closed where EVERY rule across it ends (a table
open at its sides), never where only some do -- each cell is a rectangle with no rule inside it, it has two rows and
two columns or more, no drawing (a curve, a slant) runs from cell to cell, no text crosses a rule and two cells or
more hold text (a label and its value). Else :func:`grids` names it a :class:`NotATable`, with the reason.

A scanned page has no vector rules, only a picture and, once OCR'd, words. Its rules are handed over as boxes found
in its pixels (``rulings``, :func:`document_lab.ink.rulings`); this module reads no pixels. They join the page's own
rules and the same grid makes the same table, its words the OCR's. A scan fed askew is read in the frame its skew
turns back: its words and paths are turned with the rules, and each box found is turned forth to where the page shows
it. OCR's own reading of the picture is weighed too: a rule lying within one word's box is that word's ink, and a
word reading only strokes that a rule runs through is the rule read as text.

A table printed sideways -- a landscape table on a portrait page -- is read in the frame its text stands upright in:
a grid whose text runs mostly (by characters) along one baseline direction other than left to right is assembled with
the page turned a quarter (or half) turn to set that direction level, so its rows read as its rows, and each box
found is turned back to where the page shows it.

A table ruled by its rows alone -- booktabs' top rule, a rule under its header, a bottom rule, nothing down -- is a
stack: :data:`STACK_RULES` level rules or more of one extent, :data:`STACK_LENGTH` text sizes long at least (each end
within the tolerance), touching no rule down. Its text, the spans whose center stands between its first and last rule,
makes its cells by white space, its columns those of its rows below its second rule (:func:`aligned`), each cell's box
its text's. A stack is a table only when what lies in it can be nothing else: no rule runs down it and nothing else is
drawn in it (a figure framed by rules: a curve, a frame, shading -- a ground under the whole stack aside), no image lies
in it, no text crosses its rules, its text makes two rows of two columns, every row holds a short cell
(:data:`SHORT_CELL` words at most: a code, a value -- prose set in two columns has none) and it is no list (two columns,
every cell of the first below the header a bullet or an enumerator). Else :func:`grids` names it a :class:`NotATable`.
Of a stack whose rules frame two tables or more (one under the other, a caption between), each is the longest run of its
rules, from the top, that is one. A stack is read as the page stands, its rules level: one printed sideways is not
found.

A table set without rules -- columns parted by white space alone -- is no grid here, scanned or not; its cells are
read by :func:`aligned` where a caller knows where it stands (`document_lab.regions`).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.ir import BBox, Page, Table, spaced, union
from document_lab.layout import UP, Line, lines, reading_way, upright

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from document_lab.ink import Rulings
    from document_lab.ir import Provenance, TextSpan, VectorPath

__all__ = [
    'COLUMN_GAP',
    'SHORT_CELL',
    'STACK_LENGTH',
    'STACK_RULES',
    'STROKES',
    'THIN',
    'TOLERANCE',
    'WORD_INK',
    'NotATable',
    'aligned',
    'grids',
    'tables',
]

#: Points within which a grid's rules are one line, a gap along a rule is no gap, and a rule meets another (PyMuPDF's
#: snap and join tolerance: a double rule is one line).
TOLERANCE: Final = 3.0
#: A filled shape this many points thick or less, and longer than that, is a rule.
THIN: Final = 2.0
#: A side whose ends differ by this many points or less across is level (or upright).
LEVEL: Final = 0.5
#: A color whose every RGB component is at least this shows nothing on white (a tint or ink: at most 1 - this).
WHITE: Final = 0.95
#: A picture's rule within this many points of one word's box all round is that word's ink (the tops of ``mm``).
WORD_INK: Final = 1.0
#: Characters a picture's rule reads as, OCR'd: a word of these alone that a rule runs through is that rule.
STROKES: Final = frozenset('|!Il1[]_-\N{EN DASH}\N{EM DASH}')
#: Touch tests between level and upright rules are made this many level rules at a time (memory, not results).
_CHUNK: Final = 512
#: A table's columns set by white space stand at least this many of its text sizes apart: wider than a word space, as
#: no cell's words are.
COLUMN_GAP: Final = 0.5
#: Level rules of one extent, at least this many stacked and none down, rule a table by its rows (booktabs)...
STACK_RULES: Final = 3
#: ...each at least this many of its text sizes long: a formula's fraction bars, stacked alike, are shorter.
STACK_LENGTH: Final = 6.0
#: A table's row holds a short cell -- at most this many words: a code, a value, a dash -- where prose holds none.
SHORT_CELL: Final = 3
#: A list's marker before each item: a bullet or a dash, or an enumerator closed by its stop (``1.``, ``a)``, ``(iv)``,
#: ``2:``); a bare number is a table's index.
_MARKER: Final = re.compile(
    r'^(?:[\N{BULLET}\N{TRIANGULAR BULLET}\N{WHITE BULLET}\N{EN DASH}\N{EM DASH}*-]'
    r'|\(?(?:\d+|[A-Za-z]|[ivxIVX]+)[.):])$'
)

_Segment = tuple[float, float, float]  # (position, low end, high end)


@dataclass(frozen=True, slots=True)
class NotATable:
    """A grid of rules that is no table, and why (a single box, a plot's frame, a broken grid)."""

    bbox: BBox
    reason: str


def tables(page: Page, *, rulings: Rulings | None = None, tolerance: float = TOLERANCE) -> tuple[Table, ...]:
    """The ruled tables of ``page``, top to bottom, each cited as ``table[i]`` and each cell by its own box.

    ``rulings``: the rules of the page's picture (a scan's, :func:`document_lab.ink.rulings`), as :func:`grids` takes
    them.
    """
    return tuple(g for g in grids(page, rulings=rulings, tolerance=tolerance) if isinstance(g, Table))


def grids(page: Page, *, rulings: Rulings | None = None, tolerance: float = TOLERANCE) -> tuple[Table | NotATable, ...]:
    """Every grid of rules on ``page``, top to bottom: a :class:`Table`, or a :class:`NotATable` saying why not.

    ``rulings``: the rules of the page's picture (:func:`document_lab.ink.rulings`), boxes in the frame its skew
    turns back. They join the page's own rules (each box level when wider than high) -- but for one lying within
    :data:`WORD_INK` of a single word's box, which is that word's ink (the tops of ``mm`` side by side) -- and the
    page is read in that frame: its text and paths turned back about its center by the skew, each box found turned
    forth -- its center where the page shows it, its size as in the frame. A word reading only :data:`STROKES` that
    one of these rules runs through is that rule as OCR read it, and no cell's text.
    """
    skew = 0.0 if rulings is None or rulings.skew is None else rulings.skew
    seen = _turned_page(page, -skew) if skew else page
    level, upright, drawings = _rules(seen.paths)
    words = [s.provenance.bbox for s in seen.spans if s.text.strip() and s.provenance.bbox is not None]
    pictured = [
        b for b in (() if rulings is None else rulings.rules) if not any(_within(b, w, WORD_INK) for w in words)
    ]
    if pictured:
        seen = dataclasses.replace(seen, spans=tuple(s for s in seen.spans if not _reads_as_rule(s, pictured)))
    for box in pictured:
        if box.width >= box.height:
            level.append(((box.y0 + box.y1) / 2, box.x0, box.x1))
        else:
            upright.append(((box.x0 + box.x1) / 2, box.y0, box.y1))
    found: list[Table | NotATable] = []
    frames: dict[int, tuple[Page, list[VectorPath]]] = {0: (seen, drawings)}
    downs = _merge(upright, tolerance)
    free: list[_Segment] = []  # level rules touching no rule down: a stack's, if any
    for touching_h, touching_v in _connected(_merge(level, tolerance), downs, tolerance):
        if not touching_v:
            free += touching_h
            continue
        h, v = _snap(touching_h, tolerance), _snap(touching_v, tolerance)
        if len({s[0] for s in h}) > 1 and len({s[0] for s in v}) > 1:
            box = BBox(min(s[1] for s in h), min(s[1] for s in v), max(s[2] for s in h), max(s[2] for s in v))
            turn = _upright_turn(seen, box)
            if turn not in frames:
                turned = _quartered_page(seen, turn)
                frames[turn] = (turned, _rules(turned.paths)[2])
            framed, framed_drawings = frames[turn]
            h, v = _quartered_rules(h, v, turn, seen)
            grid = _grid(framed, h, v, framed_drawings, tolerance)
            found.append(_quartered_grid(grid, -turn, seen) if turn else grid)
    found += _stacks(seen, free, downs, tolerance)
    found.sort(key=lambda g: (_bbox(g).y0, _bbox(g).x0))
    numbered: list[Table | NotATable] = []
    for grid in found:
        if isinstance(grid, Table):
            index = sum(isinstance(g, Table) for g in numbered)
            grid = Table(grid.rows, grid.cell_bboxes, grid.provenance.at(locator=f'table[{index}]'))  # noqa: PLW2901
        numbered.append(_turned_grid(grid, skew, page) if skew else grid)
    return tuple(numbered)


def _upright_turn(page: Page, box: BBox) -> int:
    """The quarter turns, in degrees clockwise, that set level the baseline most of the text in ``box`` runs along.

    Most: the way the text whose center stands in ``box`` reads (`document_lab.layout.reading_way`: more of its
    characters than any other, vertical writing upright); else 0.
    """
    inside = (s for s in page.spans if box.x0 <= s.center[0] <= box.x1 and box.y0 <= s.center[1] <= box.y1)
    way = reading_way(inside)
    return 0 if way is None else round(-math.degrees(math.atan2(way[1], way[0]))) % 360


def _quarter(x: float, y: float, turn: int, page: Page) -> tuple[float, float]:
    """The point turned ``turn`` degrees clockwise (a multiple of 90) about the page's center, exactly."""
    cx, cy = page.width / 2, page.height / 2
    dx, dy = x - cx, y - cy
    return {0: (x, y), 90: (cx - dy, cy + dx), 180: (cx - dx, cy - dy), 270: (cx + dy, cy - dx)}[turn % 360]


def _quartered_box(box: BBox, turn: int, page: Page) -> BBox:
    """``box`` turned ``turn`` degrees clockwise (a multiple of 90) about the page's center."""
    (ax, ay), (bx, by) = _quarter(box.x0, box.y0, turn, page), _quarter(box.x1, box.y1, turn, page)
    return BBox(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def _quartered_page(page: Page, turn: int) -> Page:
    """``page`` with its spans (boxes and baselines) and paths turned ``turn`` degrees clockwise about its center."""
    sin, cos = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}[turn]
    spans = tuple(
        dataclasses.replace(
            s,
            provenance=s.provenance.at(bbox=_quartered_box(s.provenance.bbox, turn, page)),
            direction=(s.direction[0] * cos - s.direction[1] * sin, s.direction[0] * sin + s.direction[1] * cos),
        )
        for s in page.spans
        if s.provenance.bbox is not None
    )
    paths = tuple(
        dataclasses.replace(
            p,
            points=tuple(_quarter(float(x), float(y), turn, page) for x, y in p.points),
            provenance=p.provenance.at(bbox=_quartered_box(p.provenance.bbox, turn, page)),
        )
        if p.provenance.bbox is not None
        else p
        for p in page.paths
    )
    return dataclasses.replace(page, spans=spans, paths=paths)


def _quartered_rules(
    h: list[_Segment], v: list[_Segment], turn: int, page: Page
) -> tuple[list[_Segment], list[_Segment]]:
    """A grid's level and upright rules turned ``turn`` degrees clockwise about the page's center.

    Level and upright still: a quarter turn trades one for the other.
    """
    if not turn:
        return h, v
    boxes = [BBox(lo, at, hi, at) for at, lo, hi in h] + [BBox(at, lo, at, hi) for at, lo, hi in v]
    level, upright = [], []
    for box in (_quartered_box(b, turn, page) for b in boxes):
        if box.height == 0:
            level.append((box.y0, box.x0, box.x1))
        else:
            upright.append((box.x0, box.y0, box.y1))
    return level, upright


def _quartered_grid(grid: Table | NotATable, turn: int, page: Page) -> Table | NotATable:
    """``grid``'s boxes turned ``turn`` degrees clockwise about the page's center: from its text's frame to the page."""
    if isinstance(grid, NotATable):
        return NotATable(_quartered_box(grid.bbox, turn, page), grid.reason)
    cells = None
    if grid.cell_bboxes is not None:
        cells = tuple(
            tuple(None if b is None else _quartered_box(b, turn, page) for b in row) for row in grid.cell_bboxes
        )
    box = grid.provenance.bbox
    here = grid.provenance.at(bbox=None if box is None else _quartered_box(box, turn, page))
    return Table(grid.rows, cells, here)


def _reads_as_rule(span: TextSpan, rules: list[BBox]) -> bool:
    """Whether ``span`` reads only :data:`STROKES` and one of ``rules`` runs through its box (along it, across it)."""
    own = span.provenance.bbox
    if own is None or not span.text.strip() or not set(span.text.strip()) <= STROKES:
        return False
    for rule in rules:
        x, y = (rule.x0 + rule.x1) / 2, (rule.y0 + rule.y1) / 2
        level = rule.width >= rule.height
        if (
            (own.y0 <= y <= own.y1 and rule.x0 < own.x1 and own.x0 < rule.x1)
            if level
            else (own.x0 <= x <= own.x1 and rule.y0 < own.y1 and own.y0 < rule.y1)
        ):
            return True
    return False


def _within(box: BBox, outer: BBox, margin: float) -> bool:
    """Whether ``box`` lies inside ``outer`` grown by ``margin`` on every side."""
    return (
        box.x0 >= outer.x0 - margin
        and box.y0 >= outer.y0 - margin
        and box.x1 <= outer.x1 + margin
        and box.y1 <= outer.y1 + margin
    )


def _turn(x: float, y: float, degrees: float, page: Page) -> tuple[float, float]:
    """The point turned by ``degrees`` (clockwise as seen, y down) about the page's center."""
    angle = math.radians(degrees)
    cx, cy = page.width / 2, page.height / 2
    dx, dy = x - cx, y - cy
    return cx + dx * math.cos(angle) - dy * math.sin(angle), cy + dx * math.sin(angle) + dy * math.cos(angle)


def _turned_box(box: BBox, degrees: float, page: Page) -> BBox:
    """``box`` moved as its center turns by ``degrees`` about the page's center, its size kept."""
    x, y = _turn((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, degrees, page)
    return BBox(x - box.width / 2, y - box.height / 2, x + box.width / 2, y + box.height / 2)


def _turned_page(page: Page, degrees: float) -> Page:
    """``page`` with its spans' boxes and its paths' points turned by ``degrees`` about its center."""
    spans = tuple(
        dataclasses.replace(s, provenance=s.provenance.at(bbox=_turned_box(s.provenance.bbox, degrees, page)))
        for s in page.spans
        if s.provenance.bbox is not None
    )
    paths = tuple(
        dataclasses.replace(
            p,
            points=tuple(_turn(float(x), float(y), degrees, page) for x, y in p.points),
            provenance=p.provenance.at(bbox=_turned_box(p.provenance.bbox, degrees, page)),
        )
        if p.provenance.bbox is not None
        else p
        for p in page.paths
    )
    return dataclasses.replace(page, spans=spans, paths=paths)


def _turned_grid(grid: Table | NotATable, degrees: float, page: Page) -> Table | NotATable:
    """``grid``'s boxes turned forth by ``degrees``: as the page shows them."""
    if isinstance(grid, NotATable):
        return NotATable(_turned_box(grid.bbox, degrees, page), grid.reason)
    cells = None
    if grid.cell_bboxes is not None:
        cells = tuple(
            tuple(None if b is None else _turned_box(b, degrees, page) for b in row) for row in grid.cell_bboxes
        )
    box = grid.provenance.bbox
    return Table(grid.rows, cells, grid.provenance.at(bbox=None if box is None else _turned_box(box, degrees, page)))


def _bbox(grid: Table | NotATable) -> BBox:
    return grid.bbox if isinstance(grid, NotATable) else grid.provenance.bbox


def _shows(color: tuple[float, ...] | None) -> bool:
    """Is the paint seen on white? None paints nothing; a pattern (no components) is seen, as its color is unknown."""
    if color is None:
        return False
    if not color:
        return True
    return min(color) < WHITE if len(color) == 3 else max(color) > 1 - WHITE  # noqa: PLR2004 -- RGB


def _rules(paths: Sequence[VectorPath]) -> tuple[list[_Segment], list[_Segment], list[VectorPath]]:
    """Level rules, upright rules, and the drawings that are no rules (a side neither level nor upright)."""
    level: list[_Segment] = []
    upright: list[_Segment] = []
    drawings: list[VectorPath] = []
    for path in paths:
        if not (_shows(path.stroke) or _shows(path.fill)):
            continue
        sides = list(itertools.pairwise(_outline(path)))
        if any(abs(q[0] - p[0]) > LEVEL and abs(q[1] - p[1]) > LEVEL for p, q in sides):
            drawings.append(path)
            continue
        for is_level, segment in _path_rules(path, sides):
            (level if is_level else upright).append(segment)
    return level, upright, drawings


def _outline(path: VectorPath) -> list[tuple[float, float]]:
    """The path's points in order, its first again at the end when it is closed (read by iteration: any sequence)."""
    points = [(float(x), float(y)) for x, y in path.points]
    return points + points[:1] if path.closed else points


def _path_rules(
    path: VectorPath, sides: list[tuple[tuple[float, float], tuple[float, float]]]
) -> Iterator[tuple[bool, _Segment]]:
    """A path's rules, each level (True) or upright: its stroked sides, and its fill when it is a bar."""
    if _shows(path.stroke):
        for (x0, y0), (x1, y1) in sides:
            if abs(y1 - y0) <= LEVEL and abs(x1 - x0) >= abs(y1 - y0):
                yield True, ((y0 + y1) / 2, min(x0, x1), max(x0, x1))
            elif abs(x1 - x0) <= LEVEL:
                yield False, ((x0 + x1) / 2, min(y0, y1), max(y0, y1))
    box = path.provenance.bbox
    if _shows(path.fill) and box.height <= THIN < box.width:
        yield True, ((box.y0 + box.y1) / 2, box.x0, box.x1)
    elif _shows(path.fill) and box.width <= THIN < box.height:
        yield False, ((box.x0 + box.x1) / 2, box.y0, box.y1)


def _merge(segments: list[_Segment], tolerance: float) -> list[_Segment]:
    """The page's rules along each line: rules within :data:`LEVEL` of one position, gaps no wider than ``tolerance``.

    The window is bounded, so a dense drawing elsewhere on the page never chains two lines into one; a joined rule
    no longer than ``tolerance`` is no rule (a marker's side, the end of a crossing stroke).
    """
    merged: list[_Segment] = []
    segments = sorted(segments)
    start = 0
    for end in range(1, len(segments) + 1):
        if end < len(segments) and segments[end][0] - segments[start][0] <= LEVEL:
            continue
        line = segments[start:end]
        at = sum(s[0] for s in line) / len(line)
        merged += [(at, lo, hi) for lo, hi in _join([s[1:] for s in line], tolerance) if hi - lo > tolerance]
        start = end
    return merged


def _snap(segments: list[_Segment], tolerance: float) -> list[_Segment]:
    """One grid's rules on its lines, along each line the rules joined.

    Positions no further than ``tolerance`` apart, chained, are one line (a double rule, a rule drawn twice), set at
    their mean. Only the grid's own rules take part, so nothing elsewhere on the page moves its lines.
    """
    at = sorted({s[0] for s in segments})
    line: dict[float, float] = {}
    start = 0
    for end in range(1, len(at) + 1):
        if end < len(at) and at[end] - at[end - 1] <= tolerance:
            continue
        line |= dict.fromkeys(at[start:end], sum(at[start:end]) / (end - start))
        start = end
    runs: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for position, lo, hi in segments:
        runs[line[position]].append((lo, hi))
    return [(position, lo, hi) for position, found in runs.items() for lo, hi in _join(found, tolerance)]


def _join(runs: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Runs along one line, those overlapping or no further than ``tolerance`` apart joined."""
    joined: list[list[float]] = []
    for lo, hi in sorted(runs):
        if joined and lo <= joined[-1][1] + tolerance:
            joined[-1][1] = max(joined[-1][1], hi)
        else:
            joined.append([lo, hi])
    return [(lo, hi) for lo, hi in joined]


def _connected(
    level: list[_Segment], upright: list[_Segment], tolerance: float
) -> Iterator[tuple[list[_Segment], list[_Segment]]]:
    """The groups of rules that touch, each as its level and its upright rules (a level rule touching none: alone)."""
    if not upright:
        yield from (([segment], []) for segment in level)
        return
    h = np.array(level)
    v = np.array(upright)
    parent = list(range(len(h) + len(v)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(h), _CHUNK):
        part = h[start : start + _CHUNK]
        touch = (
            (v[None, :, 0] >= part[:, None, 1] - tolerance)
            & (v[None, :, 0] <= part[:, None, 2] + tolerance)
            & (part[:, None, 0] >= v[None, :, 1] - tolerance)
            & (part[:, None, 0] <= v[None, :, 2] + tolerance)
        )
        for i, j in zip(*np.nonzero(touch), strict=True):
            parent[root(start + int(i))] = root(len(h) + int(j))
    groups: dict[int, tuple[list[_Segment], list[_Segment]]] = defaultdict(lambda: ([], []))
    for i, segment in enumerate(level):
        groups[root(i)][0].append(segment)
    for j, segment in enumerate(upright):
        groups[root(len(h) + j)][1].append(segment)
    yield from (g for g in groups.values() if g[0])


def _ruled(runs: list[tuple[float, float]], lo: float, hi: float, tolerance: float) -> bool | None:
    """Is the slot side ``lo``..``hi`` ruled end to end (True), not at all (False), or in part (None)?"""
    covered = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in runs)
    if covered >= hi - lo - tolerance:
        return True
    return False if covered <= tolerance else None


class _Broken(Exception):
    """A grid that is no table; its argument says why."""


@dataclass
class _Slots:
    """A grid's lines, and per slot side whether it is ruled.

    ``across[i][c]``: line ``ys[i]`` over column ``c``; ``down[r][k]``: line ``xs[k]`` down row ``r``.
    """

    xs: list[float]
    ys: list[float]
    across: list[list[bool]]
    down: list[list[bool]]


def _common_end(ends: list[float], outer: float, side: int, tolerance: float) -> float | None:
    """The line closing an open side: where EVERY rule across it ends, past the outermost line.

    ``side``: -1 before the outermost line, 1 after it. None when a rule closes the side, or the rules end apart.
    """
    if max(ends) - min(ends) > tolerance:
        return None
    end = sum(ends) / len(ends)
    return end if side * (end - outer) > tolerance else None


_Runs = dict[float, list[tuple[float, float]]]  # a line's position: the runs ruled along it


def _lines(h: list[_Segment], v: list[_Segment], tolerance: float) -> tuple[_Runs, _Runs]:
    """The grid's upright and level lines, each with its runs; an open side closed where every rule across it ends.

    Raises :class:`_Broken` when a rule runs on past the outline.
    """
    at_x: _Runs = defaultdict(list)
    at_y: _Runs = defaultdict(list)
    for x, lo, hi in v:
        at_x[x].append((lo, hi))
    for y, lo, hi in h:
        at_y[y].append((lo, hi))
    x0, x1, y0, y1 = min(at_x), max(at_x), min(at_y), max(at_y)
    left, right = _common_end([s[1] for s in h], x0, -1, tolerance), _common_end([s[2] for s in h], x1, 1, tolerance)
    top, bottom = _common_end([s[1] for s in v], y0, -1, tolerance), _common_end([s[2] for s in v], y1, 1, tolerance)
    x0, x1 = x0 if left is None else left, x1 if right is None else right
    y0, y1 = y0 if top is None else top, y1 if bottom is None else bottom
    for x in (left, right):
        if x is not None:
            at_x[x].append((y0, y1))
    for y in (top, bottom):
        if y is not None:
            at_y[y].append((x0, x1))
    if any(lo < x0 - tolerance or hi > x1 + tolerance for _, lo, hi in h) or any(
        lo < y0 - tolerance or hi > y1 + tolerance for _, lo, hi in v
    ):
        msg = 'a rule runs past the outline (an open side, a tick mark)'
        raise _Broken(msg)
    return at_x, at_y


def _slots(h: list[_Segment], v: list[_Segment], tolerance: float) -> _Slots:
    """The grid's lines and ruled slot sides; a line no slot side is ruled on is no line (a stub too short to part)."""
    at_x, at_y = _lines(h, v, tolerance)
    xs, ys = sorted(at_x), sorted(at_y)
    while True:
        across = [[_ruled(at_y[y], xs[c], xs[c + 1], tolerance) for c in range(len(xs) - 1)] for y in ys]
        down = [[_ruled(at_x[x], ys[r], ys[r + 1], tolerance) for x in xs] for r in range(len(ys) - 1)]
        if any(side is None for line in across + down for side in line):
            msg = 'a rule stops inside a cell'
            raise _Broken(msg)
        if not (all(across[0]) and all(across[-1]) and all(row[0] and row[-1] for row in down)):
            msg = 'its outline is not closed'
            raise _Broken(msg)
        empty_y = [i for i in range(1, len(ys) - 1) if not any(across[i])]
        empty_x = [k for k in range(1, len(xs) - 1) if not any(row[k] for row in down)]
        if not empty_x and not empty_y:
            return _Slots(xs, ys, across, down)
        ys = [y for i, y in enumerate(ys) if i not in empty_y]
        xs = [x for k, x in enumerate(xs) if k not in empty_x]


def _cells(slots: _Slots) -> tuple[list[list[int]], list[tuple[int, int, int, int]]]:
    """Each slot's cell, and each cell's (first row, first column, last row, last column)."""
    rows, cols = len(slots.ys) - 1, len(slots.xs) - 1
    cell_of = [[-1] * cols for _ in range(rows)]
    spans: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if cell_of[r][c] >= 0:
                continue
            index, stack, members = len(spans), [(r, c)], []
            cell_of[r][c] = index
            while stack:
                i, j = stack.pop()
                members.append((i, j))
                for ni, nj, open_ in (
                    (i, j + 1, j + 1 < cols and not slots.down[i][j + 1]),
                    (i, j - 1, j > 0 and not slots.down[i][j]),
                    (i + 1, j, i + 1 < rows and not slots.across[i + 1][j]),
                    (i - 1, j, i > 0 and not slots.across[i][j]),
                ):
                    if open_ and cell_of[ni][nj] < 0:
                        cell_of[ni][nj] = index
                        stack.append((ni, nj))
            r0, r1 = min(m[0] for m in members), max(m[0] for m in members)
            c0, c1 = min(m[1] for m in members), max(m[1] for m in members)
            if len(members) != (r1 - r0 + 1) * (c1 - c0 + 1):
                msg = 'a cell is not a rectangle'
                raise _Broken(msg)
            inner = any(slots.down[i][j] for i in range(r0, r1 + 1) for j in range(c0 + 1, c1 + 1)) or any(
                slots.across[i][j] for i in range(r0 + 1, r1 + 1) for j in range(c0, c1 + 1)
            )
            if inner:
                msg = 'a rule stops inside a cell'
                raise _Broken(msg)
            spans.append((r0, c0, r1, c1))
    return cell_of, spans


def _slot(slots: _Slots, x: float, y: float) -> tuple[int, int] | None:
    """The slot the point stands in; None outside the grid."""
    c, r = bisect_right(slots.xs, x) - 1, bisect_right(slots.ys, y) - 1
    return (r, c) if 0 <= r < len(slots.ys) - 1 and 0 <= c < len(slots.xs) - 1 else None


def _crosses(path: VectorPath, slots: _Slots, cell_of: list[list[int]]) -> bool:
    """Does the drawing run from one cell into another (a curve across a plot's grid)?

    Each side is cut where it meets the grid's lines; each piece's middle stands in one slot, so a side through a
    corner is caught too. Leaving the grid ends a run: a leader from outside may point into one cell.
    """
    last = None
    for (x0, y0), (x1, y1) in itertools.pairwise(_outline(path)):
        cuts = {0.0, 1.0}
        for line, a, b in [(x, x0, x1) for x in slots.xs] + [(y, y0, y1) for y in slots.ys]:
            if a != b and 0 < (t := (line - a) / (b - a)) < 1:
                cuts.add(t)
        for t0, t1 in itertools.pairwise(sorted(cuts)):
            t = (t0 + t1) / 2
            slot = _slot(slots, x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            here = None if slot is None else cell_of[slot[0]][slot[1]]
            if here is not None and last is not None and here != last:
                return True
            last = here
    return False


def _grid(
    page: Page, h: list[_Segment], v: list[_Segment], drawings: list[VectorPath], tolerance: float
) -> Table | NotATable:
    try:
        return _table(page, _slots(h, v, tolerance), drawings, tolerance)
    except _Broken as broken:
        rules = [BBox(lo, y, hi, y) for y, lo, hi in h] + [BBox(x, lo, x, hi) for x, lo, hi in v]
        return NotATable(union(rules), str(broken))


def _table(page: Page, slots: _Slots, drawings: list[VectorPath], tolerance: float) -> Table:
    """The grid's table, its slots ruled whole; raises :class:`_Broken` naming what makes it none."""
    box = BBox(slots.xs[0], slots.ys[0], slots.xs[-1], slots.ys[-1])
    cell_of, spans = _cells(slots)
    rows, cols = len(slots.ys) - 1, len(slots.xs) - 1
    if rows == 1 or cols == 1:
        msg = 'a single box' if rows == cols else 'one row' if rows == 1 else 'one column'
        raise _Broken(msg)
    near = [d for d in drawings if _overlap(d.provenance.bbox, box)]
    if any(_crosses(path, slots, cell_of) for path in near):
        msg = 'a drawing crosses its rules (a figure)'
        raise _Broken(msg)
    boxes = [BBox(slots.xs[c0], slots.ys[r0], slots.xs[c1 + 1], slots.ys[r1 + 1]) for r0, c0, r1, c1 in spans]
    texts = _texts(page, slots, cell_of, boxes, tolerance)
    if sum(bool(t) for t in texts) < 2:  # noqa: PLR2004 -- a label and what it labels
        msg = 'fewer than two cells hold text (a drawn grid, a plot)'
        raise _Broken(msg)
    here: Provenance = page.spans[0].provenance.at(bbox=box)  # two cells hold text: the page has spans
    return Table(
        tuple(tuple(texts[cell_of[r][c]] for c in range(cols)) for r in range(rows)),
        tuple(tuple(boxes[cell_of[r][c]] for c in range(cols)) for r in range(rows)),
        here,
    )


def _overlap(a: BBox, b: BBox) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _texts(page: Page, slots: _Slots, cell_of: list[list[int]], boxes: list[BBox], tolerance: float) -> list[str]:
    """Each cell's text: its spans as lines, top to bottom; a span reaching over a rule breaks the grid."""
    held: list[list[TextSpan]] = [[] for _ in boxes]
    for span in page.spans:
        if not spaced(span.text).strip():
            continue
        slot = _slot(slots, *span.center)
        if slot is None:
            continue
        index = cell_of[slot[0]][slot[1]]
        cell, own = boxes[index], span.provenance.bbox
        reach = own.height / 2
        if (
            own.x0 < cell.x0 - tolerance
            or own.x1 > cell.x1 + tolerance
            or own.y0 < cell.y0 - reach
            or own.y1 > cell.y1 + reach
        ):
            msg = 'text crosses a rule'
            raise _Broken(msg)
        held[index].append(span)
    return [_text(page, spans) for spans in held]


def _text(page: Page, spans: list[TextSpan]) -> str:
    found = lines(Page(page.number, page.width, page.height, spans=tuple(spans)), weighted=True)
    return ' '.join(line.text for line in sorted(found, key=lambda ln: (ln.y0, ln.x0)))


def aligned(spans: Sequence[TextSpan], *, body: Sequence[TextSpan] | None = None) -> Table | None:
    """The table ``spans`` set by white space alone: columns apart by white space, rows by baseline.

    Its text size is the spans' median size. A column is a run across ``body`` -- the spans that part the columns: a
    ruled table's rows below its header, whose cells may span columns; by default all ``spans`` -- that some span
    covers, apart from the next by :data:`COLUMN_GAP` text sizes that no span of it crosses; a span stands in the column
    it overlaps most (a header wider than its column, or over two), else the nearest. A row is spans whose middles lie
    within half a text size. A row with nothing in the first column carries on the cells of the row above (a value
    wrapped onto a second line). A cell's text is its spans as a line (`document_lab.layout`), its box theirs (None when
    empty). Spans without text, or written vertically, take no part. None when the text does not make two rows of two
    columns.
    """
    spans = [s for s in spans if spaced(s.text).strip() and not s.vertical]
    if not spans:
        return None
    size = statistics.median(s.size for s in spans)
    columns = _runs(
        [s for s in body if spaced(s.text).strip() and not s.vertical] if body else spans, COLUMN_GAP * size
    )
    rows: list[list[list[list[TextSpan]]]] = []
    for band in _baselines(spans, size / 2):
        row: list[list[TextSpan]] = [[] for _ in columns]
        for s in band:
            row[_column(s.provenance.bbox, columns)].append(s)
        if rows and not row[0]:  # no first cell: the row above, carried on
            for cell, more in zip(rows[-1], row, strict=True):
                cell.append(more)
        else:
            rows.append([[cell] for cell in row])
    if len(rows) < 2 or len(columns) < 2:  # noqa: PLR2004 -- a label and its value
        return None

    def text(cell: list[list[TextSpan]]) -> str:
        return ' '.join(Line(line, '', UP).text for line in cell if line)

    def bbox(cell: list[list[TextSpan]]) -> BBox | None:
        boxes = [s.provenance.bbox for line in cell for s in line]
        return union(boxes) if boxes else None

    return Table(
        tuple(tuple(text(cell) for cell in row) for row in rows),
        tuple(tuple(bbox(cell) for cell in row) for row in rows),
        spans[0].provenance.at(bbox=union([s.provenance.bbox for s in spans])),
    )


def _column(box: BBox, columns: list[tuple[float, float]]) -> int:
    """The column ``box`` overlaps most across, else the nearest to it."""
    overlap = [min(box.x1, x1) - max(box.x0, x0) for x0, x1 in columns]
    return max(range(len(columns)), key=lambda i: overlap[i])


def _runs(spans: list[TextSpan], gap: float) -> list[tuple[float, float]]:
    """The runs across the page ``spans`` cover, left to right, each apart from the next by ``gap`` points or more."""
    runs: list[list[float]] = []
    for s in sorted(spans, key=lambda s: s.provenance.bbox.x0):
        b = s.provenance.bbox
        if runs and b.x0 - runs[-1][1] < gap:
            runs[-1][1] = max(runs[-1][1], b.x1)
        else:
            runs.append([b.x0, b.x1])
    return [(x0, x1) for x0, x1 in runs]


def _baselines(spans: list[TextSpan], reach: float) -> list[list[TextSpan]]:
    """``spans`` in bands top to bottom: each span within ``reach`` points down of its band's first middle."""
    bands: list[list[TextSpan]] = []
    for s in sorted(spans, key=lambda s: s.center[1]):
        if bands and s.center[1] - bands[-1][0].center[1] <= reach:
            bands[-1].append(s)
        else:
            bands.append([s])
    return bands


def _stacks(page: Page, free: list[_Segment], downs: list[_Segment], tolerance: float) -> list[Table | NotATable]:
    """The tables ruled by their rows alone among ``free`` (level rules touching no rule down), or why a stack is none.

    Rules of one extent are one stack; of its rules, from the top, the longest run that is a table (:func:`_stack`)
    is one, and the search goes on below it. A stack no run of which is a table is named, with why the whole is none.
    """
    found: list[Table | NotATable] = []
    for rules in _extents(free, tolerance):
        here: list[Table] = []
        why = ''
        start = 0
        while start <= len(rules) - STACK_RULES:
            for end in range(len(rules), start + STACK_RULES - 1, -1):
                try:
                    here.append(_stack(page, rules[start:end], downs, tolerance))
                except _Broken as broken:
                    why = why or str(broken)  # the whole stack's reason: tried first
                    continue
                start = end
                break
            else:
                start += 1
        found += here or [NotATable(_stack_box(rules), why)]
    return found


def _extents(free: list[_Segment], tolerance: float) -> list[list[_Segment]]:
    """``free`` level rules in stacks of one extent, top to bottom, those of :data:`STACK_RULES` rules or more.

    One extent: each end within ``tolerance`` of the stack's first rule's. Rules of a stack within ``tolerance`` of one
    another across are one (a double rule).
    """
    stacks: list[list[_Segment]] = []
    for rule in sorted(free, key=lambda r: (r[1], r[2], r[0])):
        same = next((k for k in stacks if max(abs(k[0][1] - rule[1]), abs(k[0][2] - rule[2])) <= tolerance), None)
        if same is None:
            stacks.append([rule])
        else:
            same.append(rule)
    snapped = (sorted(_snap(k, tolerance)) for k in stacks)
    return [k for k in snapped if len(k) >= STACK_RULES]


def _stack_box(rules: list[_Segment]) -> BBox:
    """The box from the first of stacked ``rules`` (top to bottom) to the last, across their extent."""
    return BBox(min(r[1] for r in rules), rules[0][0], max(r[2] for r in rules), rules[-1][0])


def _stack(page: Page, rules: list[_Segment], downs: list[_Segment], tolerance: float) -> Table:
    """The table ruled by stacked level ``rules``, top to bottom; raises :class:`_Broken` naming what makes it none."""
    box = _stack_box(rules)
    inner = BBox(box.x0 + tolerance, box.y0 + tolerance, box.x1 - tolerance, box.y1 - tolerance)
    if any(inner.x0 < x < inner.x1 and lo < inner.y1 and inner.y0 < hi for x, lo, hi in downs):
        msg = 'a rule runs down it (a figure, a frame)'
        raise _Broken(msg)
    drawn = (p.provenance.bbox for p in page.paths if _shows(p.stroke) or _shows(p.fill))
    # a shape holding the whole stack is its ground (a page's color, a panel behind it), nothing drawn in it
    if any(b is not None and b.height > THIN and _overlap(b, inner) and not _within(box, b, 0.0) for b in drawn):
        msg = 'something else is drawn in it (a figure, a frame, shading)'
        raise _Broken(msg)
    if any(f.provenance.bbox is not None and _within(f.provenance.bbox, box, tolerance) for f in page.figures):
        msg = 'an image lies in it (a figure)'
        raise _Broken(msg)
    spans = [
        s
        for s in page.spans
        if spaced(s.text).strip() and upright(s) and box.x0 <= s.center[0] <= box.x1 and box.y0 < s.center[1] < box.y1
    ]
    if spans and box.width < STACK_LENGTH * statistics.median(s.size for s in spans):
        msg = 'its rules are short (fraction bars, stacked)'
        raise _Broken(msg)
    for s in spans:
        own = s.provenance.bbox
        if any(own.y0 + own.height / 4 < at < own.y1 - own.height / 4 for at, _, _ in rules):
            msg = 'text crosses a rule'
            raise _Broken(msg)
    table = aligned(spans, body=[s for s in spans if s.center[1] > rules[1][0]])  # below the header's rule
    if table is None:
        msg = 'its text makes no two rows of two columns'
        raise _Broken(msg)
    if not all(any(t and len(t.split()) <= SHORT_CELL for t in row) for row in table.rows):
        msg = 'a row holds no short cell (prose, not cells)'
        raise _Broken(msg)
    if len(table.rows[0]) == 2 and all(_MARKER.match(row[0]) for row in table.rows[1:]):  # noqa: PLR2004 -- marker, item
        msg = 'a list (a marker before each item)'
        raise _Broken(msg)
    return Table(table.rows, table.cell_bboxes, table.provenance.at(bbox=box))
