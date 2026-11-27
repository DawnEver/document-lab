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
its text's. Its rules part its rows -- and so does a rule in its body under some of its columns, no stack of its own --
so no row runs across one: a band whose lines stand on common levels is rows of their own (a group's label beside
them repeated in each); a band below its header whose cells wrap centered -- a short cell half a line off the lines
of a wrapped one -- is one row, and proves every band of its table one row (a rule under every row, as IEEE sets one).
A stack is a table only when what lies in it can be nothing else: no rule runs down it and nothing else is drawn in it
(a figure framed by rules: a curve, a frame, shading -- a ground under the whole stack aside), no image lies in it, no
text crosses its rules, its text makes two rows of two columns, every row holds a short cell (:data:`SHORT_CELL` words
at most: a code, a value -- prose set in two columns has none) and it is no list (two columns, every cell of the first
below the header a bullet or an enumerator). Else :func:`grids` names it a :class:`NotATable`. Of a stack whose rules
frame two tables or more (one under the other, a caption between), each is the longest run of its rules, from the top,
that is one. A stack printed sideways -- its rules upright on the page, its text running along them -- is read in the
frame its text stands upright in, as a grid printed sideways is.

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
from document_lab.layout import Line, lines, upright

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

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
    'WORD_SPACES',
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
#: Points within which two coordinates computed apart are one: float rounding, no measure of the page.
_ROUNDING: Final = 1e-6
#: A table's columns set by white space stand this many of its text sizes apart, or :data:`WORD_SPACES` of its own word
#: spaces where that is less: wider than a word space, as no cell's words are. Measured on 147 tables ruled by their
#: rows (111 papers), the white between words of one cell reaches 0.42 text sizes and between columns starts at 0.59
#: but for one pair at 0.44, its table's word spaces 0.21.
COLUMN_GAP: Final = 0.5
#: ...or this many of the table's widest word space -- the white between two words of one span that a word of another
#: row stands across, so inside a cell -- where it states one: on the same tables a word space aligned down the rows
#: stands 1.1 of it at most, the closest columns 2.07.
WORD_SPACES: Final = 1.5
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
    seen = _pictured(seen, () if rulings is None else rulings.rules, level, upright)
    found: list[Table | NotATable] = []
    frames: dict[int, tuple[Page, list[VectorPath]]] = {0: (seen, drawings)}

    def frame(turn: int) -> tuple[Page, list[VectorPath]]:
        """The page turned ``turn`` degrees clockwise, and its drawings: where text running that way stands upright."""
        if turn not in frames:
            turned = _quartered_page(seen, turn)
            frames[turn] = (turned, _rules(turned.paths)[2])
        return frames[turn]

    levels, downs = _merge(level, tolerance), _merge(upright, tolerance)
    free: list[_Segment] = []  # level rules touching no rule down: a stack's, if any
    crossed: set[_Segment] = set()  # rules down touching a level one
    for touching_h, touching_v in _connected(levels, downs, tolerance):
        crossed.update(touching_v)
        if not touching_v:
            free += touching_h
            continue
        h, v = _snap(touching_h, tolerance), _snap(touching_v, tolerance)
        if len({s[0] for s in h}) > 1 and len({s[0] for s in v}) > 1:
            box = BBox(min(s[1] for s in h), min(s[1] for s in v), max(s[2] for s in h), max(s[2] for s in v))
            turn = _upright_turn(seen, box)
            framed, framed_drawings = frame(turn)
            h, v = _quartered_rules(h, v, turn, seen)
            grid = _grid(framed, h, v, framed_drawings, tolerance)
            found.append(_quartered_grid(grid, -turn, seen) if turn else grid)
    free_down = [v for v in downs if v not in crossed]  # rules down touching no level rule: a stack printed sideways
    found += _framed_stacks(seen, frame, (levels, downs), (free, free_down), tolerance)
    found.sort(key=lambda g: (_bbox(g).y0, _bbox(g).x0))
    numbered: list[Table | NotATable] = []
    for grid in found:
        if isinstance(grid, Table):
            index = sum(isinstance(g, Table) for g in numbered)
            grid = Table(grid.rows, grid.cell_bboxes, grid.provenance.at(locator=f'table[{index}]'))  # noqa: PLW2901
        numbered.append(_turned_grid(grid, skew, page) if skew else grid)
    return tuple(numbered)


def _pictured(page: Page, boxes: Sequence[BBox], level: list[_Segment], upright: list[_Segment]) -> Page:
    """``page`` without the words that read one of a picture's rules ``boxes`` as text, the rules joined to the rules.

    Each box joins ``level`` when wider than high, else ``upright`` -- but for one within :data:`WORD_INK` of one
    word's box, which is that word's ink.
    """
    words = [s.provenance.bbox for s in page.spans if s.text.strip() and s.provenance.bbox is not None]
    pictured = [b for b in boxes if not any(_within(b, w, WORD_INK) for w in words)]
    for box in pictured:
        if box.width >= box.height:
            level.append(((box.y0 + box.y1) / 2, box.x0, box.x1))
        else:
            upright.append(((box.x0 + box.x1) / 2, box.y0, box.y1))
    if not pictured:
        return page
    return dataclasses.replace(page, spans=tuple(s for s in page.spans if not _reads_as_rule(s, pictured)))


def _framed_stacks(
    page: Page,
    frame: Callable[[int], tuple[Page, list[VectorPath]]],
    rules: tuple[list[_Segment], list[_Segment]],
    free: tuple[list[_Segment], list[_Segment]],
    tolerance: float,
) -> list[Table | NotATable]:
    """The page's stacks (:func:`_stacks`), each read in the frame its text stands upright in, its boxes turned back.

    ``rules``: the page's level and upright rules; ``free``: those touching no rule across them. A stack of level rules
    is read as the page stands (or upside down, its text so); one of upright rules whose text runs along them -- a
    stack printed sideways -- a quarter turn round, in ``frame(turn)``, as a grid printed sideways is.
    """
    (levels, downs), (free_level, free_down) = rules, free
    stacks: dict[int, list[list[_Segment]]] = defaultdict(list)
    for held in _extents(free_level, tolerance):
        stacks[180 if _upright_turn(page, _stack_box(held)) == 180 else 0].append(held)  # noqa: PLR2004 -- upside down
    for held in _extents(free_down, tolerance):
        box = BBox(held[0][0], min(r[1] for r in held), held[-1][0], max(r[2] for r in held))
        if (turn := _upright_turn(page, box)) in {90, 270}:
            stacks[turn].append(held)
    found: list[Table | NotATable] = []
    for turn, held in stacks.items():
        level_now = [sorted(_quartered_rules(*((r, []) if turn % 180 == 0 else ([], r)), turn, page)[0]) for r in held]
        read = _stacks(
            frame(turn)[0],
            level_now,
            _quartered_rules(free_level, free_down, turn, page)[0],
            _quartered_rules(levels, downs, turn, page)[1],
            tolerance,
        )
        found += [_quartered_grid(g, -turn, page) for g in read] if turn else read
    return found


def _upright_turn(page: Page, box: BBox) -> int:
    """The quarter turns, in degrees clockwise, that set level the baseline most of the text in ``box`` runs along.

    Most: more than half its characters run along one direction (each span's, to the nearest quarter turn); else 0.
    """
    counts: dict[int, int] = defaultdict(int)
    for span in page.spans:
        x, y = span.center
        if span.text.strip() and box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1:
            dx, dy = span.direction
            counts[round(math.degrees(math.atan2(dy, dx)) / 90) % 4 * 90] += len(span.text.strip())
    if not counts:
        return 0
    angle, most = max(counts.items(), key=lambda item: item[1])
    return (360 - angle) % 360 if most * 2 > sum(counts.values()) else 0


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
    """``page`` with its spans (boxes and baselines), paths and figures turned ``turn`` degrees clockwise.

    The turn is about the page's center.
    """
    sin, cos = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}[turn]
    spans = tuple(
        dataclasses.replace(
            s,
            provenance=s.provenance.at(bbox=_quartered_box(s.provenance.bbox, turn, page)),
            words=tuple(_quartered_box(w, turn, page) for w in s.words),
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
    figures = tuple(
        dataclasses.replace(f, provenance=f.provenance.at(bbox=_quartered_box(f.provenance.bbox, turn, page)))
        if f.provenance.bbox is not None
        else f
        for f in page.figures
    )
    return dataclasses.replace(page, spans=spans, paths=paths, figures=figures)


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
    """``page`` with its spans' and their words' boxes and its paths' points turned by ``degrees`` about its center."""
    spans = tuple(
        dataclasses.replace(
            s,
            provenance=s.provenance.at(bbox=_turned_box(s.provenance.bbox, degrees, page)),
            words=tuple(_turned_box(w, degrees, page) for w in s.words),
        )
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


def aligned(
    spans: Sequence[TextSpan], *, body: Sequence[TextSpan] | None = None, rules: Sequence[float] = ()
) -> Table | None:
    """The table ``spans`` set by white space alone: columns apart by white space, rows by baseline.

    Its text size is the spans' median size. A column is a run across ``body`` -- the spans that part the columns: a
    ruled table's rows below its header, whose cells may span columns; by default all ``spans`` -- that some word of
    them covers (a span's words where its source states them, else the span), apart from the next by
    :data:`COLUMN_GAP` text sizes -- or :data:`WORD_SPACES` of its widest word space, where that is less: the white
    between two words of one span that a word of another row stands across -- that no word of it crosses; a span
    stands in the column it overlaps most (a header wider than its column, or over two), else the nearest. A span
    whose words stand in two columns, the gap between two of them holding the gap between the columns whole, is
    parted there: two columns set closer than its source's word break.

    ``rules``: the heights of the level rules across it, top to bottom (a stack's), the first band -- above the
    second -- its header. No row runs across one: each band between two is read alone. Without them the spans are one
    band. A band's lines are each column's spans whose middles lie within half a text size (and the spans continuing a
    word of them); lines of two columns whose middles lie within a quarter text size stand on one level. A band is
    rows of their own when two columns or more hold a line on each of the same levels, two or more, and every other
    column holds lines only on those levels or is a label set across them -- left of those columns, one line or lines
    off the levels, centered on them, repeated in each row as a merged cell is --, the first column one of them, a
    label or empty. A band whose lines stand staggered otherwise -- two of them more than a quarter text size and less
    than three quarters apart, a short cell centered beside the lines of a wrapped one --, its first column holding one
    line at most, is one row whose cells wrap; one such band below the header, of two bands or more there, proves that
    its rules part every row (the table is ruled by its rows): each band of the table is one row. Any other band is
    read by baseline: a row is spans whose middles lie within half a text size, and a row with nothing in the first
    column carries on the cells of the row above in its band (a value wrapped onto a second line).

    A cell's text is its lines (`document_lab.layout`), top to bottom, its box theirs (None when empty). Spans without
    text, or written vertically, take no part. None when the text does not make two rows of two columns.
    """
    spans = [s for s in spans if spaced(s.text).strip() and not s.vertical]
    if not spans:
        return None
    size = statistics.median(s.size for s in spans)
    parting = [s for s in body if spaced(s.text).strip() and not s.vertical] if body else spans
    columns = _runs([w for s in parting for w in _words(s)], _column_gap(parting, size))
    spans = [part for s in spans for part in _parted(s, columns)]
    bands = _bands(spans, rules)
    read = [_Band.of(band, columns, size) for band in bands]
    ruled = len(read) > 2 and any(band.wraps for band in read[1:])  # noqa: PLR2004 -- a header and two bands
    rows = [row for band in read for row in ((band.whole,) if ruled else band.rows)]
    if len(rows) < 2 or len(columns) < 2:  # noqa: PLR2004 -- a label and its value
        return None

    def text(cell: list[list[TextSpan]]) -> str:
        return ' '.join(Line(line, '').text for line in cell if line)

    def bbox(cell: list[list[TextSpan]]) -> BBox | None:
        boxes = [s.provenance.bbox for line in cell for s in line]
        return union(boxes) if boxes else None

    return Table(
        tuple(tuple(text(cell) for cell in row) for row in rows),
        tuple(tuple(bbox(cell) for cell in row) for row in rows),
        spans[0].provenance.at(bbox=union([s.provenance.bbox for s in spans])),
    )


type _Row = list[list[list[TextSpan]]]  # a row: each column's cell, as its lines of spans


def _bands(spans: list[TextSpan], rules: Sequence[float]) -> list[list[TextSpan]]:
    """``spans`` parted by ``rules`` (heights), top to bottom: each band the spans whose middles stand between two."""
    cuts = sorted(rules)
    bands: dict[int, list[TextSpan]] = defaultdict(list)
    for s in spans:
        bands[bisect_right(cuts, s.center[1])].append(s)
    return [bands[k] for k in sorted(bands)]


@dataclass(frozen=True, slots=True)
class _Band:
    """A band of a table between its rules, read.

    ``rows``: its rows read alone; ``whole``: the band as one row; ``wraps``: it is one row whose cells wrap, so its
    rules part every row of its table.
    """

    rows: list[_Row]
    whole: _Row
    wraps: bool

    @classmethod
    def of(cls, spans: list[TextSpan], columns: list[tuple[float, float]], size: float) -> _Band:
        cells: list[list[TextSpan]] = [[] for _ in columns]
        for s in spans:
            cells[_column(s.provenance.bbox, columns)].append(s)
        lines = [_cell_lines(cell, size) for cell in cells]
        whole = [list(cell) for cell in lines]
        if max(len(cell) for cell in lines) <= 1:
            return cls([whole], whole, wraps=False)
        rows = _on_levels(lines, size)
        if rows is not None:
            return cls(rows, whole, wraps=False)
        if len(lines[0]) <= 1 and _staggered(lines, size):  # one first cell at most, centered: one row
            return cls([whole], whole, wraps=True)
        return cls(_carried(spans, columns, size), whole, wraps=False)


def _cell_lines(spans: list[TextSpan], size: float) -> list[list[TextSpan]]:
    """One cell's spans as lines, top to bottom: middles within half ``size``, and the spans continuing a word."""
    found = _baselines(spans, size / 2)
    words = [{s.unspaced for s in line if s.unspaced is not None} for line in found]
    joined: list[list[TextSpan]] = []
    held: list[set[int]] = []
    for line, word in zip(found, words, strict=True):
        into = next((i for i, w in enumerate(held) if w & word), None)
        if into is None:
            joined.append(list(line))
            held.append(set(word))
        else:
            joined[into] += line
            held[into] |= word
    return joined


def _staggered(lines: list[list[list[TextSpan]]], size: float) -> bool:
    """Do two of the band's lines stand off each other's level by less than a line?

    More than a quarter ``size`` apart and less than three quarters: a short cell centered beside the lines of a
    wrapped one stands half a line off them, as no two lines of aligned rows (a line apart) or of one level do.
    """
    middles = sorted(_middle(line) for cell in lines for line in cell)
    return any(size / 4 < b - a < 3 * size / 4 for a, b in itertools.pairwise(middles))


def _middle(line: list[TextSpan]) -> float:
    """Where ``line`` stands down the page: the middle of its span of most characters (not a sub- or superscript)."""
    return max(line, key=lambda s: len(spaced(s.text).strip())).center[1]


def _on_levels(lines: list[list[list[TextSpan]]], size: float) -> list[_Row] | None:
    """The band's rows, when its columns' ``lines`` stand on levels: None when they make no rows of their own.

    Rows of their own: two columns or more hold a line on each of the same levels (within a quarter ``size``), two or
    more, and every other column holds lines only on those levels, or is a label -- left of those columns, one line or
    lines off the levels, centered on them -- repeated in each row; the first column, whose lines start rows, is one
    of those columns, a label or empty.
    """
    count = max(len(cell) for cell in lines)
    middles = [[_middle(line) for line in cell] for cell in lines]
    full = [c for c, cell in enumerate(middles) if len(cell) == count]
    if len(full) < 2:  # noqa: PLR2004 -- two columns
        return None
    level = middles[full[0]]
    if any(abs(a - b) > size / 4 for c in full for a, b in zip(middles[c], level, strict=True)):
        return None
    center = (level[0] + level[-1]) / 2
    rows: list[_Row] = [[[] for _ in lines] for _ in level]
    for c, cell in enumerate(lines):
        if not cell:
            continue
        off = any(min(abs(at - middle) for at in level) > size / 4 for middle in middles[c])
        centered = abs((middles[c][0] + middles[c][-1]) / 2 - center) <= size / 4
        if c < full[0] and len(cell) < count and (len(cell) == 1 or off) and centered:
            for row in rows:
                row[c] = cell
            continue
        if c == 0 and len(cell) < count:  # the first column starts fewer rows: its rows' other cells wrap
            return None
        for line, middle in zip(cell, middles[c], strict=True):
            i = min(range(count), key=lambda i: abs(level[i] - middle))
            if abs(level[i] - middle) > size / 4:
                return None
            rows[i][c].append(line)
    return rows


def _carried(spans: list[TextSpan], columns: list[tuple[float, float]], size: float) -> list[_Row]:
    """A band's rows by baseline: a row with nothing in the first column carries on the row above (in the band)."""
    rows: list[_Row] = []
    for band in _baselines(spans, size / 2):
        row: list[list[TextSpan]] = [[] for _ in columns]
        for s in band:
            row[_column(s.provenance.bbox, columns)].append(s)
        if rows and not row[0]:  # no first cell: the row above, carried on
            for cell, more in zip(rows[-1], row, strict=True):
                cell.append(more)
        else:
            rows.append([[cell] for cell in row])
    return rows


def _column(box: BBox, columns: list[tuple[float, float]]) -> int:
    """The column ``box`` overlaps most across, else the nearest to it."""
    overlap = [min(box.x1, x1) - max(box.x0, x0) for x0, x1 in columns]
    return max(range(len(columns)), key=lambda i: overlap[i])


def _words(span: TextSpan) -> tuple[BBox, ...]:
    """``span``'s words' boxes where its source states them, word for word with its text; else its own box."""
    return span.words if len(span.words) == len(span.text.split()) else (span.provenance.bbox,)


def _column_gap(spans: list[TextSpan], size: float) -> float:
    """The white that parts ``spans``' columns: :data:`COLUMN_GAP` ``size``, or less, their own word spaces wider.

    :data:`WORD_SPACES` of their widest word space, where that is less: the white between two words of one span that
    a word of another row stands across (inside a cell, then).
    """
    words = [(s, _words(s)) for s in spans]
    spaces = [
        b.x0 - a.x1
        for s, own in words
        for a, b in itertools.pairwise(own)
        if b.x0 > a.x1
        and any(
            abs(o.center[1] - s.center[1]) > size / 2 and any(w.x0 <= a.x1 and b.x0 <= w.x1 for w in theirs)
            for o, theirs in words
        )
    ]
    return min(COLUMN_GAP * size, WORD_SPACES * max(spaces)) if spaces else COLUMN_GAP * size


def _parted(span: TextSpan, columns: list[tuple[float, float]]) -> list[TextSpan]:
    """``span`` parted where the gap between two of its words holds the gap between two ``columns`` whole."""
    words = _words(span)
    if len(words) < 2:  # noqa: PLR2004 -- two words to part
        return [span]
    at = [_column(w, columns) for w in words]
    cuts = [
        i
        for i in range(1, len(words))
        if at[i] != at[i - 1]
        and words[i - 1].x1 <= columns[at[i - 1]][1] + _ROUNDING
        and words[i].x0 >= columns[at[i]][0] - _ROUNDING
    ]
    if not cuts:
        return [span]
    texts = span.text.split()
    return [
        dataclasses.replace(
            span,
            text=' '.join(texts[a:b]),
            provenance=span.provenance.at(bbox=union(list(words[a:b]))),
            words=words[a:b],
        )
        for a, b in itertools.pairwise([0, *cuts, len(words)])
    ]


def _runs(boxes: list[BBox], gap: float) -> list[tuple[float, float]]:
    """The runs across the page ``boxes`` cover, left to right, each apart from the next by ``gap`` points or more."""
    runs: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: b.x0):
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


def _stacks(
    page: Page, stacks: list[list[_Segment]], free: list[_Segment], downs: list[_Segment], tolerance: float
) -> list[Table | NotATable]:
    """The tables ruled by their rows alone among ``stacks``, or why a stack is none.

    ``stacks``: level rules of one extent each (:func:`_extents`), top to bottom; ``free``: the level rules touching no
    rule down, ``downs``: the rules down. The widest stack first; of its rules, from the top, the longest run that is a
    table (:func:`_stack`) is one, and the search goes on below it. A stack no run of which is a table is named, with
    why the whole is none; one inside a table found is none of its own (its rules part that table's rows).
    """
    found: list[Table | NotATable] = []
    held: list[BBox] = []
    for rules in sorted(stacks, key=lambda k: k[0][1] - k[0][2]):
        if any(_within(_stack_box(rules), box, tolerance) for box in held):
            continue
        here: list[Table] = []
        why = ''
        start = 0
        while start <= len(rules) - STACK_RULES:
            for end in range(len(rules), start + STACK_RULES - 1, -1):
                try:
                    here.append(_stack(page, rules[start:end], free, downs, tolerance))
                except _Broken as broken:
                    why = why or str(broken)  # the whole stack's reason: tried first
                    continue
                start = end
                break
            else:
                start += 1
        held += [t.provenance.bbox for t in here]
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


def _stack(page: Page, rules: list[_Segment], free: list[_Segment], downs: list[_Segment], tolerance: float) -> Table:
    """The table ruled by stacked level ``rules``, top to bottom; raises :class:`_Broken` naming what makes it none.

    Its rows are parted by its rules and by those of ``free`` in its body -- below its header's rule, within its
    extent, :data:`STACK_LENGTH` text sizes long at least, no text across them: a rule under some of its columns.
    """
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
    size = statistics.median(s.size for s in spans) if spans else 0.0
    if box.width < STACK_LENGTH * size:
        msg = 'its rules are short (fraction bars, stacked)'
        raise _Broken(msg)
    if any(_crosses_text(at, spans) for at, _, _ in rules):
        msg = 'text crosses a rule'
        raise _Broken(msg)
    body = [s for s in spans if s.center[1] > rules[1][0]]  # below the header's rule
    cuts = {
        at
        for at, lo, hi in free
        if rules[1][0] + tolerance < at < rules[-1][0] - tolerance
        and box.x0 - tolerance <= lo
        and hi <= box.x1 + tolerance
        and hi - lo >= STACK_LENGTH * size
        and not _crosses_text(at, spans)
    }
    table = aligned(spans, body=body, rules=sorted({at for at, _, _ in rules} | cuts))
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


def _crosses_text(at: float, spans: list[TextSpan]) -> bool:
    """Does a level rule at height ``at`` run through the text of one of ``spans`` (its middle half, not its edge)?"""
    return any(
        s.provenance.bbox.y0 + s.provenance.bbox.height / 4 < at < s.provenance.bbox.y1 - s.provenance.bbox.height / 4
        for s in spans
    )
