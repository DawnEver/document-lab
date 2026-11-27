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

A scanned page's ruled table has no vector rules: its rules are pixels. :func:`rulings` turns them into what
`document_lab.tables` reads a vector page's rules as -- level and upright boxes in points -- and measures the page's
skew from them (:class:`Rulings`): a sheet fed askew is never turned pixel by pixel, its rules' geometry is turned
back instead, and the table turns its words with them. Here pixels end; from the boxes on, a scan's table is
assembled exactly as a vector page's is. Every threshold is an option of :func:`rulings`, its default measured on
synthetic scans (``tests/unit/test_tables.py``, the ``ocr`` part) and on scanned standards.

A page's skew is told by its lines -- of text and rules alike -- as a whole: :func:`skew` finds the angle along which
its ink lines up sharpest (`document_lab.ocr` turns the picture it hands Tesseract back by it), or says no angle
does (a blank page, a photograph), never a guessed 0. :func:`rulings` alone cannot tell it on a page of prose: a
line of words, read along a row, runs on like a rule.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.ir import BBox

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    'BRIDGE',
    'COVER',
    'DARK',
    'GAP',
    'MIN_LENGTH',
    'RULE_SHARE',
    'RUN',
    'SKEW_LIMIT',
    'THICKNESS',
    'Drawing',
    'Rulings',
    'drawings',
    'rulings',
    'skew',
]

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


#: A run of ink along a pixel row (column) at least this many points long seeds a rule; a glyph's strokes are
#: shorter. A rule slanted by the scan still runs this far within its own thickness.
RUN: Final = 3.0
#: A rule is no thicker than this many points: a thicker band of ink is a filled shape, a block of text, a picture.
THICKNESS: Final = 3.0
#: Gaps of white along a row (column) no wider than this many points are ink (specks the scan dropped from a rule).
BRIDGE: Final = 0.75
#: Pieces along one line no further apart than this many points are one rule (a rule broken by the scan).
GAP: Final = 3.0
#: A rule is at least this share of the page's shorter side long, its pieces joined.
MIN_LENGTH: Final = 0.02
#: A rule's pieces cover at least this share of its length: the tops of a word's letters, side by side, do not.
COVER: Final = 0.9
#: Degrees from level (upright) within which a line may be a rule; the page's skew is measured within them.
SKEW_LIMIT: Final = 3.0
#: Rows (columns) read together as one: their darkness summed is ink when it reaches a dark pixel's. A thin rule
#: drawn between two rows (grey in both), or a speck of white the scan dropped in it, breaks none of its runs, and a
#: slanted rule's steps overlap.
_SPREAD: Final = 1
#: Pixels two pieces of one rule may stand apart across beyond their half thicknesses (each piece's fit).
_FIT: Final = 1.5


@dataclass(frozen=True)
class Rulings:
    """A page picture's rules: ``rules`` level and upright boxes (points), in the frame the page's ``skew`` turns back.

    ``skew``: degrees the picture's rules stand turned, clockwise as seen (a scan fed askew); the rules are given
    as the page would show them turned back by it about its center -- the pixels themselves are never turned. None
    when the picture has no rule to tell it by (then ``rules`` is empty). Each rule is a box as long as the rule and
    as thick: level when wider than high.
    """

    skew: float | None
    rules: tuple[BBox, ...]


@dataclass(frozen=True)
class _Line:
    """A connected run of ink along one direction, as found in pixels: from ``(a0, b0)`` to ``(a1, b1)``.

    ``a`` along the line, ``b`` across it (pixels); ``thickness`` its mean extent across (pixels).
    """

    a0: float
    b0: float
    a1: float
    b1: float
    thickness: float

    @property
    def slope(self) -> float:
        return (self.b1 - self.b0) / (self.a1 - self.a0) if self.a1 > self.a0 else 0.0

    @property
    def length(self) -> float:
        return self.a1 - self.a0


_Piece = tuple[float, float, float, float, float]  # turned back: (across at a0, across at a1, a0, a1, thickness)


def rulings(
    pixels: np.ndarray,
    scale: float,
    *,
    dark: int = DARK,
    run: float = RUN,
    thickness: float = THICKNESS,
    bridge: float = BRIDGE,
    gap: float = GAP,
    min_length: float = MIN_LENGTH,
    cover: float = COVER,
    skew_limit: float = SKEW_LIMIT,
) -> Rulings:
    """The horizontal and vertical rules drawn in ``pixels`` (rows top-down, grey, RGB or RGBA, ``scale`` px/pt).

    A pixel darker than ``dark`` is ink; so are two rows' pixels whose darkness together reaches it (a thin rule
    between them), and gaps no wider than ``bridge`` points along a row (a column) between ink. Each run of ink at
    least ``run`` points along a row (a column) seeds a line; runs touching across rows are one line, fitted by its
    pixels' least squares (its slant) and measured across (its thickness), and run on along its fit through ink in
    its band (its ends: a stretch before a break, a slanted rule's step; never a speck). A line no thicker than
    ``thickness`` and within ``skew_limit`` degrees of level (upright) is a rule's piece; the page's skew is the
    length-weighted median slant of the pieces at least ``min_length`` (share of the page's shorter side) long, so a
    picture whose only lines are dashed has none. Turned back by it about the page's center, pieces overlapping across
    (each within its half thickness of the other's line) and no more than ``gap`` points apart along are one rule; a
    double rule is two, which a table snaps to one line. A rule shorter than ``min_length``, or whose pieces cover
    less than ``cover`` of its length (a word's letters side by side), is none; so is a piece that, turned back, still
    slants by more than ``thickness`` end to end (a drawn diagonal). A glyph's stem longer than ``min_length`` is a
    rule here; `document_lab.tables` finds it within its word's box.
    """
    if pixels.ndim == 2:  # noqa: PLR2004 -- rows by columns
        darkness = 255 - pixels.astype(np.int16)
    else:
        darkness = 255 - (pixels[..., :3].sum(axis=2, dtype=np.int16) // 3)
    height, width = darkness.shape
    shortest = min_length * min(height, width)
    limit = math.tan(math.radians(skew_limit))
    across = thickness * scale + _SPREAD
    found = {'dark': dark, 'run': run * scale, 'bridge': round(bridge * scale)}
    level = [ln for ln in _lines(darkness, **found) if ln.thickness <= across and abs(ln.slope) <= limit]
    upright = [ln for ln in _lines(darkness.T, **found) if ln.thickness <= across and abs(ln.slope) <= limit]
    # Clockwise as seen (y down): a level rule falls to the right, an upright one leans to the left.
    slants = [(math.degrees(math.atan(ln.slope)), ln.length) for ln in level if ln.length >= shortest]
    slants += [(-math.degrees(math.atan(ln.slope)), ln.length) for ln in upright if ln.length >= shortest]
    if not slants:
        return Rulings(None, ())
    skew = _weighted_median(slants)
    turn = math.radians(skew)
    joined = {'across': across, 'gap': gap * scale, 'shortest': shortest, 'cover': cover, 'scale': scale}
    rules = [
        *_joined(_turned(level, -turn, (width / 2, height / 2)), upright=False, **joined),
        *_joined(_turned(upright, turn, (height / 2, width / 2)), upright=True, **joined),
    ]
    return Rulings(skew, tuple(sorted(rules, key=lambda b: (b.y0, b.x0))))


def _weighted_median(values: list[tuple[float, float]]) -> float:
    """The median of ``(value, weight)`` pairs: the value at half their total weight."""
    ordered = sorted(values)
    half, total = sum(w for _, w in ordered) / 2, 0.0
    for value, weight in ordered:
        total += weight
        if total >= half:
            return value
    return ordered[-1][0]


def _lines(darkness: np.ndarray, *, dark: int, run: float, bridge: int) -> list[_Line]:
    """The lines of ink along ``darkness``' rows (255 less grey): runs at least ``run`` pixels long, joined across rows.

    Each row is read with :data:`_SPREAD` rows below it, their darkness summed: ink where it reaches ``dark``'s; gaps
    no wider than ``bridge`` pixels between ink are ink. Runs touching across rows are one line, fitted by its
    pixels' least squares, from its first column to its last.
    """
    summed = darkness.copy()
    for shift in range(1, _SPREAD + 1):
        summed[:-shift] += darkness[shift:]
    spread = summed > 255 - dark
    if bridge:  # closed along the rows: grown by ``bridge`` to the right, then shrunk back from the left
        grown = spread.copy()
        for shift in range(1, bridge + 1):
            grown[:, shift:] |= spread[:, :-shift]
        spread = grown.copy()
        for shift in range(1, bridge + 1):
            spread[:, :-shift] &= grown[:, shift:]
    width = spread.shape[1]
    edges = np.diff(np.pad(spread, ((0, 0), (1, 1))).view(np.int8), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)  # row by row, as the starts: each run's end
    keep = ends - starts >= run
    rows, starts, ends = rows[keep], starts[keep], ends[keep]
    if not rows.size:
        return []
    labels = _touching(rows, starts, ends, width + 2)
    count = int(labels.max()) + 1
    n = (ends - starts).astype(np.float64)
    sx = n * (starts + ends - 1) / 2
    sxx = _squares(ends - 1) - _squares(starts - 1)
    y = rows.astype(np.float64)
    total = np.bincount(labels, n, count)
    mx, my = np.bincount(labels, sx, count) / total, np.bincount(labels, n * y, count) / total
    vxx = np.bincount(labels, sxx, count) / total - mx**2
    cxy = np.bincount(labels, sx * y, count) / total - mx * my
    first = np.full(count, width, dtype=np.int64)
    last = np.zeros(count, dtype=np.int64)
    np.minimum.at(first, labels, starts)
    np.maximum.at(last, labels, ends)
    out = []
    for i in range(count):
        slope = cxy[i] / vxx[i] if vxx[i] > 0 else 0.0
        a0, a1 = float(first[i]), float(last[i])
        b0, b1 = my[i] + slope * (a0 - mx[i]), my[i] + slope * (a1 - mx[i])
        across = total[i] / (a1 - a0) * math.cos(math.atan(slope))
        thick = max(float(across) - _SPREAD, 1.0)
        middle = 0.5 + _SPREAD / 2  # a pixel's middle, less the rows read with each row
        line = _Line(a0, float(b0) + middle, a1, float(b1) + middle, thick)
        out.append(_extended(line, spread, bridge))
    return out


def _extended(line: _Line, ink: np.ndarray, bridge: int) -> _Line:
    """``line`` run on along its own fit, each way, through stretches of ink in its band, ``bridge`` columns apart.

    The runs that seed a line are long; its ends may be shorter -- a rule's last pixels before a break, where a
    slanted rule steps to the next row -- which only its own band tells from the ink around it. A stretch narrower
    than ``bridge`` + 1 columns (a speck) runs it on no further.
    """
    height, width = ink.shape
    slope, half = line.slope, line.thickness / 2 + _SPREAD
    ends = [line.a0, line.a1]
    for side, step in ((0, -1), (1, 1)):
        column = int(line.a1) if side else int(line.a0) - 1
        stretch = missed = 0
        while 0 <= column < width and missed <= bridge:
            middle = line.b0 + slope * (column + 0.5 - line.a0) - 0.5
            low, high = max(int(middle - half + 0.5), 0), min(int(middle + half + 0.5) + 1, height)
            if low < high and ink[low:high, column].any():
                stretch, missed = stretch + 1, 0
                if stretch > bridge:
                    ends[side] = float(column + 1 if side else column)
            else:
                stretch, missed = 0, missed + 1
            column += step
    if ends == [line.a0, line.a1]:
        return line
    a0, a1 = ends
    return _Line(a0, line.b0 + slope * (a0 - line.a0), a1, line.b0 + slope * (a1 - line.a0), line.thickness)


def _squares(k: np.ndarray) -> np.ndarray:
    """The sum of ``x * x`` for ``x`` from 0 to ``k``."""
    k = k.astype(np.float64)
    return k * (k + 1) * (2 * k + 1) / 6


def _touching(rows: np.ndarray, starts: np.ndarray, ends: np.ndarray, stride: int) -> np.ndarray:
    """Each run's line: runs (in row order, each row's left to right) touching a run of the row above share it.

    Runs touch when their columns overlap or meet at a corner (a one-pixel line slanted by a step).
    """
    low, high = rows * stride + starts, rows * stride + ends  # one ordered key over every row
    above = (rows - 1) * stride
    first = np.searchsorted(high, above + starts, side='left')  # the first run above ending at or past its start
    last = np.searchsorted(low, above + ends + 1, side='left') - 1  # the last run above starting by its end
    count = np.where((first <= last) & (rows > 0), last - first + 1, 0)
    mine = np.repeat(np.arange(rows.size), count)
    theirs = np.repeat(first, count) + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    labels = np.arange(rows.size)
    while True:  # each run takes the least label it touches, until none changes
        before = labels.copy()
        least = np.minimum(labels[mine], labels[theirs])
        np.minimum.at(labels, mine, least)
        np.minimum.at(labels, theirs, least)
        labels = labels[labels]
        if np.array_equal(labels, before):
            break
    return np.unique(labels, return_inverse=True)[1]


def _turned(lines: list[_Line], angle: float, center: tuple[float, float]) -> list[_Piece]:
    """Each line turned by ``angle`` (radians) about ``center`` (along, across), as a :data:`_Piece`."""
    cos, sin = math.cos(angle), math.sin(angle)
    ca, cb = center
    out = []
    for ln in lines:
        (a0, b0), (a1, b1) = (
            (ca + (a - ca) * cos - (b - cb) * sin, cb + (a - ca) * sin + (b - cb) * cos)
            for a, b in ((ln.a0, ln.b0), (ln.a1, ln.b1))
        )
        out.append((b0, b1, min(a0, a1), max(a0, a1), ln.thickness))
    return out


def _joined(
    pieces: list[_Piece], *, across: float, gap: float, shortest: float, cover: float, scale: float, upright: bool
) -> list[BBox]:
    """The rules of turned-back ``pieces`` (pixels): those overlapping across and within ``gap`` along joined.

    Two pieces are of one rule when each stands within its half thickness (and :data:`_FIT`) of the other's line and
    no more than ``gap`` from it along; a piece still slanting by more than ``across`` end to end is none. A rule
    shorter than ``shortest``, or covered by its pieces less than ``cover`` of its length, is none. Each rule is a
    box in points, at its own pieces' length-weighted position and as thick as their length-weighted thickness.
    """
    straight = sorted(((b0 + b1) / 2, a0, a1, t) for b0, b1, a0, a1, t in pieces if abs(b1 - b0) <= across)
    rules = _grouped(straight, across, gap)
    out = []
    for rule in rules:
        lo, hi = min(p[1] for p in rule), max(p[2] for p in rule)
        inked, reach = 0.0, lo
        for _, a0, a1, _ in sorted(rule, key=lambda p: p[1]):  # their union's length along the rule
            inked += max(0.0, a1 - max(a0, reach))
            reach = max(reach, a1)
        if hi - lo < shortest or inked < cover * (hi - lo):
            continue
        weights = [a1 - a0 for _, a0, a1, _ in rule]
        at = sum(p[0] * w for p, w in zip(rule, weights, strict=True)) / sum(weights)
        thick = sum(p[3] * w for p, w in zip(rule, weights, strict=True)) / sum(weights)
        b0, b1 = (at - thick / 2) / scale, (at + thick / 2) / scale
        out.append(BBox(b0, lo / scale, b1, hi / scale) if upright else BBox(lo / scale, b0, hi / scale, b1))
    return out


def _grouped(
    straight: list[tuple[float, float, float, float]], across: float, gap: float
) -> list[list[tuple[float, float, float, float]]]:
    """``straight`` pieces, ``(at, a0, a1, thickness)`` sorted by ``at``, grouped into rules.

    Two pieces are of one rule when near across (within their half thicknesses and :data:`_FIT`) and along (no more
    than ``gap`` apart); so are pieces chained so.
    """
    parent = list(range(len(straight)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (at, lo, hi, thick) in enumerate(straight):
        for j in range(i + 1, len(straight)):
            other_at, other_lo, other_hi, other_thick = straight[j]
            if other_at - at > across + _FIT:  # sorted across: none further on is near
                break
            near = other_at - at <= (thick + other_thick) / 2 + _FIT
            if near and other_lo <= hi + gap and lo <= other_hi + gap:
                parent[root(j)] = root(i)
    rules: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for i, piece in enumerate(straight):
        rules[root(i)].append(piece)
    return list(rules.values())


#: Degrees either way a page's skew is looked for among its text lines.
SKEW_RANGE: Final = 10.0
#: Degrees between the angles tried first; the best is refined to a tenth of this.
SKEW_STEP: Final = 0.2
#: Text lines tell a skew only when their ink, projected across the best angle, is this much sharper than across the
#: median angle tried (squared profile, less one): noise, a photograph or a page of a few words tell none.
LINES_CONTRAST: Final = 0.25
#: Dark pixels a projection weighs at most: a page's ink is sampled evenly down to this many.
_PROJECTED: Final = 400_000


def skew(
    pixels: np.ndarray,
    *,
    dark: int = DARK,
    search: float = SKEW_RANGE,
    step: float = SKEW_STEP,
    contrast: float = LINES_CONTRAST,
) -> float | None:
    """The degrees the picture ``pixels`` (rows top-down, grey, RGB or RGBA) stands turned, clockwise as seen.

    Its dark pixels (below ``dark``) are projected along every angle within ``search`` degrees, ``step`` apart, then a
    tenth of that round the best -- along rows for level lines, along columns for lines set sideways, both at once --
    and the angle whose projections are sharpest is the skew. None when it is not ``contrast`` sharper than the
    median angle's: no lines tell it.
    """
    grey = pixels if pixels.ndim == 2 else pixels[..., :3].mean(axis=2)  # noqa: PLR2004 -- rows by columns
    ys, xs = np.nonzero(grey < dark)
    if not ys.size:
        return None
    every = -(-ys.size // _PROJECTED)
    ys, xs = ys[::every].astype(np.float64), xs[::every].astype(np.float64)

    def sharpness(degrees: float) -> float:
        slope = math.tan(math.radians(degrees))
        total = 0.0
        for along in (ys - xs * slope, xs + ys * slope):  # a level line's rows; an upright line's columns
            counts = np.bincount(np.round(along - along.min()).astype(np.int64))
            total += float(np.dot(counts, counts))
        return total

    coarse = np.arange(-search, search + step / 2, step)
    scores = [sharpness(a) for a in coarse]
    best = int(np.argmax(scores))
    if scores[best] < (1 + contrast) * float(np.median(scores)):
        return None
    fine = np.linspace(coarse[best] - step, coarse[best] + step, 21)
    return round(float(max(fine, key=sharpness)), 3)
