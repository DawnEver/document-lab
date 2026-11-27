"""One document read twice -- by `document_lab.pdf` and by the previous implementation, PyMuPDF -- and the difference.

Text is compared as each page's characters, whitespace dropped and order ignored (reading order is layout's, not the
reader's): the share of the characters we state that PyMuPDF states too. A glyph no font source names reads as
U+FFFD here (never a guess) while PyMuPDF prints something for it -- its glyph id, a byte -- so those characters are
counted apart (``unknown``), out of the agreement. A symbol font's code PyMuPDF prints as the private-use code point
its ``ToUnicode`` or ``cmap`` states (U+F0xx) reads here as the symbol shown (`pdf::font`), so each private-use
character of PyMuPDF's we do not print pairs with one of ours it does not match, and both are counted apart
(``private``), out of the agreement and out of where the text stands.

Where the text stands is compared character by character, since the two cut a page's text into runs differently: a
character of one of our spans is ``located`` when PyMuPDF states the same character with its origin (the start of its
baseline) inside the span's box grown by :data:`LOCATED` points -- the nearest one to where it falls along the span,
each of PyMuPDF's taken once. A span whose characters are all located and written along a page axis is measured
(:class:`Edges`). Along its writing both state each glyph's advance from its origin, so its start and end must be its
characters'. Across, the boxes differ by design -- ours reach the font's ascent and descent as the PDF states them
(the descriptor's, the AFM's of a standard font), PyMuPDF's its font program's or its substitute's -- so the baseline
is measured instead: where PyMuPDF's cuts our box, a share of its height from the ascent edge. One font's spans share
that share whatever their size, so a span off its font's median (:meth:`Compared.baselines`) stands on a baseline
PyMuPDF's does not.

Vector paths are compared by the painted path (ours: a :class:`~document_lab.ir.VectorPath` per subpath, its
``group`` the painted path; PyMuPDF: one drawing per painting): its box, curves flattened alike (PyMuPDF's own box
holds a curve's control points), and its paint -- its fill's color and rule, its stroke's color, width and dashes.
PyMuPDF gives a path filled and stroked as one drawing or as two, so each fill and each stroke is matched apart, each
of PyMuPDF's taken once. One of ours whose fill or stroke no drawing matches is ``unmatched``; a fill or stroke of
PyMuPDF's no path of ours takes is ``unreported`` when it can show -- its box meets the page and its clip as PyMuPDF
bounds them (the reader leaves out a path its clip box misses, `pdf::interpret`) -- and is no Type 3 glyph's ink
(:class:`_Theirs`). A fill painted with a pattern (``VectorPath.pattern``) is not compared: PyMuPDF draws a clip and
the pattern's shading or cells, never one solid fill -- an uncolored tiling's tint (a TikZ hatch) included. A color
is compared when both state RGB (ours keeps a separation's tint), against PyMuPDF's color with its color management
off and on: off, both convert a device color by ISO 32000's formulas; on, both convert an ``ICCBased`` color through
its profile (the reader by moxcms, PyMuPDF by lcms2) -- and the reader converts each space the one way, so a color
matches either.

Images are compared by where they are placed. PyMuPDF lists every placement, seen or not; the reader leaves out what
its clip hides, and its clip is a box holding the true one (`pdf::interpret`), so it leaves out only what cannot show.
An image of ours PyMuPDF does not place there is ``invented``; PyMuPDF's placements we do not report are ``clipped``.
Rendered ink cannot judge a clipped one: a Form showing part of an embedded page hides images that another Form,
showing another part of the same page, draws in the very place.

PyMuPDF reads the page unrotated; its coordinates are turned to the page as displayed, as ours are. The page is read
one at a time and PyMuPDF's drawings are taken one by one (its ``callback``), so memory holds one document of ours
and one page of PyMuPDF's.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, NamedTuple

import pymupdf

from document_lab.pdf import CURVE_SAMPLES, Locked, read_pdf

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from document_lab.ir import BBox, TextSpan, VectorPath

UNKNOWN = '\N{REPLACEMENT CHARACTER}'
#: Points two placements of an image may differ by and be one.
PLACED = 0.5
#: Points a PyMuPDF character's origin may lie outside one of our spans' boxes and be located in it.
LOCATED = 1.0
#: Points a drawing's box may differ from one of our paths' and be its box.
DRAWN = 0.01
#: A color component, or a stroke width or dash length in points, two paints may differ by and be one.
PAINTED = 0.01
#: The side of the grid cells (points) characters are looked up in, by their origin.
_CHARACTER_CELL = 24.0
#: The side of the grid cells (points) drawings are looked up in, by their box's top left corner.
_DRAWING_CELL = 1.0


class Edges(NamedTuple):
    """Where a located span's box stands against its characters."""

    #: How far its start or end stands from its characters' (points).
    along: float
    #: Its font: its name and the box's extent across in font sizes (the font's ascent less its descent, to 0.01),
    #: which tells apart two fonts of one name.
    font: tuple[str, float]
    #: Where PyMuPDF's baseline cuts it across: 0 at its ascent edge, 1 at its descent edge; None when it is flat.
    baseline: float | None


@dataclass(frozen=True)
class Page:
    """One page compared: its size off PyMuPDF's (points), its characters, images and paths placed."""

    number: int
    size_off: float
    matched: int
    stated: int  # characters we state, U+FFFD not counted
    reference: int  # characters PyMuPDF states
    unknown: int  # U+FFFD we state
    private: int  # PyMuPDF's private-use characters we state as another character
    located: int  # characters we state that stand where PyMuPDF states them
    edges: tuple[Edges, ...]  # one per span located whole along a page axis
    images: int  # ours
    invented: int  # ours PyMuPDF does not place
    clipped: int  # PyMuPDF's placements we do not report
    paths: int  # our painted paths
    colored: int  # ours with a paint stated in RGB (the rest keep a tint, whose color is not compared)
    unmatched: int  # ours no drawing of PyMuPDF's matches
    unreported: int  # PyMuPDF's fills and strokes that can show and no path of ours takes

    @property
    def whole(self) -> int:
        """The larger side's characters, the unknown taken from PyMuPDF's and the private from both."""
        return max(self.stated - self.private, self.reference - self.unknown - self.private)

    @property
    def agreement(self) -> float:
        """Matched over :attr:`whole`: 1 when neither states a character."""
        return 1.0 if self.whole <= 0 else self.matched / self.whole

    @property
    def locatable(self) -> int:
        """The characters we state that PyMuPDF can state the same: the private taken out."""
        return self.stated - self.private


@dataclass(frozen=True)
class Compared:
    """A document compared: both refused it for a password (``locked``), or its pages side by side."""

    path: Path
    locked: bool
    pages: tuple[Page, ...]
    reference_pages: int

    @property
    def agreement(self) -> float:
        """The document's characters matched over all its pages' (as :attr:`Page.agreement`, summed first)."""
        whole = sum(p.whole for p in self.pages)
        return 1.0 if whole <= 0 else sum(p.matched for p in self.pages) / whole

    @property
    def unknown_share(self) -> float:
        """The share of PyMuPDF's characters we state as U+FFFD."""
        reference = sum(p.reference for p in self.pages)
        return 0.0 if reference == 0 else sum(p.unknown for p in self.pages) / reference

    @property
    def location(self) -> float:
        """The share of the :attr:`Page.locatable` characters that are located: 1 when there are none."""
        locatable = sum(p.locatable for p in self.pages)
        return 1.0 if locatable <= 0 else sum(p.located for p in self.pages) / locatable

    def baselines(self) -> list[tuple[int, float]]:
        """Each measured span's page and how far its :attr:`Edges.baseline` lies from its font's median in the document.

        A font's spans share its ascent and descent, so PyMuPDF's baseline cuts each at the same share whatever its
        size: a span off its font's share stands on another baseline than PyMuPDF's.
        """
        shares: dict[tuple[str, float], list[float]] = defaultdict(list)
        for page in self.pages:
            for e in page.edges:
                if e.baseline is not None:
                    shares[e.font].append(e.baseline)
        middle = {font: median(values) for font, values in shares.items()}
        return [
            (page.number, abs(e.baseline - middle[e.font]))
            for page in self.pages
            for e in page.edges
            if e.baseline is not None
        ]


@dataclass(frozen=True)
class Plant:
    """A wrong truth planted in PyMuPDF's reading, which the comparison must report.

    ``shift``: PyMuPDF's characters and drawings moved by ``(dx, dy)`` points. ``rise``: every second line of
    PyMuPDF's text moved by this many points towards its ascent. ``recolor``: every RGB component PyMuPDF states
    turned by half the scale (``(c + 0.5) % 1``), so no color stays.
    """

    shift: tuple[float, float] = (0.0, 0.0)
    rise: float = 0.0
    recolor: bool = False


class _Mapping(NamedTuple):
    """PyMuPDF's unrotated page coordinates to ours (the page as displayed), the plant's shift applied."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @classmethod
    def of(cls, page: pymupdf.Page, plant: Plant) -> _Mapping:
        m = page.rotation_matrix
        return cls(m.a, m.b, m.c, m.d, m.e + plant.shift[0], m.f + plant.shift[1])

    def point(self, x: float, y: float) -> tuple[float, float]:
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)

    def turn(self, dx: float, dy: float) -> tuple[float, float]:
        return (self.a * dx + self.c * dy, self.b * dx + self.d * dy)

    def box(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        (ax, ay), (bx, by) = self.point(x0, y0), self.point(x1, y1)
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


_Box = tuple[float, float, float, float]


class _Char(NamedTuple):
    """A character PyMuPDF states, in our coordinates."""

    char: str
    origin: tuple[float, float]
    center: tuple[float, float]
    #: The page axis it is written along (0: x, 1: y), None when oblique.
    axis: int | None
    #: Its extent along that axis (its advance from its origin).
    along: tuple[float, float]
    #: Its baseline across that axis, and the side its ascent lies on there (-1, +1).
    baseline: float
    up: float


def _chars(raw: dict, to: _Mapping, type3: frozenset[str], rise: float) -> tuple[list[_Char], list[_Box]]:
    """PyMuPDF's characters but white space, and the boxes of those in a Type 3 font (named ``type3``).

    ``rise``: the plant's, every second line moved by it towards its ascent.
    """
    out = []
    glyphs = []
    lines = (line for block in raw['blocks'] for line in block.get('lines', ()))
    for number, line in enumerate(lines):
        dx, dy = to.turn(*line['dir'])
        axis = 0 if abs(dx) > 0.999 else 1 if abs(dy) > 0.999 else None
        up = (dy, -dx)  # the ascent's side, y down
        lift = (up[0] * rise, up[1] * rise) if number % 2 else (0.0, 0.0)
        for span in line['spans']:
            drawn = span['font'] in type3 or span['font'].startswith('Type3 (')
            for ch in span['chars']:
                x0, y0, x1, y1 = to.box(*ch['bbox'])
                box = (x0 + lift[0], y0 + lift[1], x1 + lift[0], y1 + lift[1])
                if drawn:
                    glyphs.append(box)
                if not ch['c'].strip():
                    continue
                x, y = to.point(*ch['origin'])
                origin = (x + lift[0], y + lift[1])
                along, baseline, side = (0.0, 0.0), 0.0, 0.0
                if axis is not None:
                    along = (box[axis], box[axis + 2])
                    baseline, side = origin[1 - axis], 1.0 if up[1 - axis] > 0 else -1.0
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                out.append(_Char(ch['c'], origin, center, axis, along, baseline, side))
    return out, glyphs


def _cells(x0: float, y0: float, x1: float, y1: float, side: float) -> Iterable[tuple[int, int]]:
    for i in range(int(x0 // side), int(x1 // side) + 1):
        for j in range(int(y0 // side), int(y1 // side) + 1):
            yield i, j


def _located(spans: Sequence[TextSpan], chars: list[_Char]) -> tuple[int, list[Edges]]:
    """How many of the spans' characters are located, and the edges of each span located whole along an axis."""
    grid: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for k, ch in enumerate(chars):
        grid[(ch.char, int(ch.origin[0] // _CHARACTER_CELL), int(ch.origin[1] // _CHARACTER_CELL))].append(k)
    used = bytearray(len(chars))
    located = 0
    edges = []
    for span in spans:
        box: BBox = span.provenance.bbox
        text = [c for c in span.text if c.strip() and c != UNKNOWN]
        x0, y0, x1, y1 = box.x0 - LOCATED, box.y0 - LOCATED, box.x1 + LOCATED, box.y1 + LOCATED
        wide = box.width >= box.height
        found = []
        for i, c in enumerate(text):
            share = (i + 0.5) / len(text)
            at = (
                (box.x0 + share * box.width, (box.y0 + box.y1) / 2)
                if wide
                else ((box.x0 + box.x1) / 2, box.y0 + share * box.height)
            )
            best, nearest = -1, float('inf')
            for cell in _cells(x0, y0, x1, y1, _CHARACTER_CELL):
                for k in grid.get((c, *cell), ()):
                    ch = chars[k]
                    if used[k] or not (x0 <= ch.origin[0] <= x1 and y0 <= ch.origin[1] <= y1):
                        continue
                    far = (ch.center[0] - at[0]) ** 2 + (ch.center[1] - at[1]) ** 2
                    if far < nearest:
                        best, nearest = k, far
            if best >= 0:
                used[best] = 1
                found.append(chars[best])
        located += len(found)
        axes = {ch.axis for ch in found}
        if found and len(found) == len(text) and UNKNOWN not in span.text and len(axes) == 1 and None not in axes:
            (a,) = axes
            along = (min(ch.along[0] for ch in found), max(ch.along[1] for ch in found))
            low, high = box[1 - a], box[3 - a]
            ascent, descent = (high, low) if found[0].up > 0 else (low, high)
            baseline = sum(ch.baseline for ch in found) / len(found)
            edges.append(
                Edges(
                    along=max(abs(box[a] - along[0]), abs(box[a + 2] - along[1])),
                    font=(span.font, round((high - low) / span.size, 2) if span.size > 0 else 0.0),
                    baseline=None if high <= low or span.size <= 0 else (baseline - ascent) / (descent - ascent),
                )
            )
    return located, edges


class _Drawn(NamedTuple):
    """A painted path: its box and its paint (None where not stroked, not filled)."""

    box: tuple[float, float, float, float]
    stroke: tuple[float, ...] | None
    width: float
    dashes: tuple[float, ...]
    fill: tuple[float, ...] | None
    even_odd: bool
    #: PyMuPDF's: its stroke and fill colors with its color management on (``stroke`` and ``fill`` are with it off).
    managed: tuple[tuple[float, ...] | None, tuple[float, ...] | None] = (None, None)


def _ours(paths: Sequence[VectorPath]) -> list[_Drawn]:
    """Our painted paths, each its subpaths' box; a pattern's fill is none to compare (PyMuPDF draws no solid fill)."""
    groups: dict[int, list[VectorPath]] = {}
    for path in paths:
        groups.setdefault(path.group, []).append(path)
    out = []
    for subpaths in groups.values():
        first = subpaths[0]
        fill = None if first.pattern is not None else first.fill
        if not (fill or first.stroke):
            continue
        boxes = [p.provenance.bbox for p in subpaths]
        box = (min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))
        out.append(_Drawn(box, first.stroke, first.width, first.dashes, fill, first.even_odd))
    return out


def _flattened(items: Sequence[tuple]) -> Iterable[tuple[float, float]]:
    """The points of PyMuPDF's path items, a curve flattened as the reader flattens one (`document_lab.pdf`)."""
    for item in items:
        match item:
            case ('l', p, q):
                yield from (p, q)
            case ('re', (x0, y0, x1, y1), _):
                yield from ((x0, y0), (x1, y1))
            case ('qu', quad):
                yield from quad
            case ('c', p, c1, c2, q):
                yield p
                for i in range(1, CURVE_SAMPLES + 1):
                    t = i / CURVE_SAMPLES
                    u = 1 - t
                    w = (u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t)
                    yield (
                        w[0] * p[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * q[0],
                        w[0] * p[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * q[1],
                    )
            case _:
                msg = f'a PyMuPDF path item this comparison does not know: {item[0]!r}'
                raise ValueError(msg)


def _dashes(spec: str) -> tuple[float, ...]:
    """PyMuPDF's dash array (``'[ 3 2 ] 0'``), its phase dropped."""
    return tuple(float(v) for v in spec[spec.index('[') + 1 : spec.index(']')].split())


def _meets(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


class _Theirs:
    """PyMuPDF's drawings of a page, taken one by one: each painting's box and paint, and whether it can show.

    A drawing within :data:`LOCATED` points of a character of a Type 3 font (``glyphs``, their boxes) may be that
    glyph's ink -- PyMuPDF runs a Type 3 glyph's procedure as drawings, the reader reads the glyph as text -- so it
    can match a path of ours but is never one we must report.
    """

    def __init__(self, page: pymupdf.Page, to: _Mapping, plant: Plant, glyphs: list[_Box]) -> None:
        self.to, self.plant = to, plant
        self.page = (0.0, 0.0, page.rect.width, page.rect.height)
        self.glyphs: dict[tuple[int, int], list[_Box]] = defaultdict(list)
        for g in glyphs:
            for cell in _cells(*g, _CHARACTER_CELL):
                self.glyphs[cell].append(g)
        self.clips: list[tuple[int, _Box]] = []
        self.drawn: list[_Drawn] = []
        self.shows: list[bool] = []
        page.get_cdrawings(extended=True, callback=self.take)
        managed: list[tuple[tuple[float, ...] | None, tuple[float, ...] | None]] = []
        pymupdf.TOOLS.set_icc(on=True)  # read again, colors through their profiles
        try:
            page.get_cdrawings(extended=True, callback=lambda d: managed.extend(self._paints(d)))
        finally:
            pymupdf.TOOLS.set_icc(on=False)  # back to the comparison's own reading (`compare`)
        if len(managed) == len(self.drawn):
            self.drawn = [d._replace(managed=m) for d, m in zip(self.drawn, managed, strict=True)]

    def _glyph(self, box: _Box) -> bool:
        cell = (int(box[0] // _CHARACTER_CELL), int(box[1] // _CHARACTER_CELL))
        return any(
            g[0] - LOCATED <= box[0]
            and g[1] - LOCATED <= box[1]
            and box[2] <= g[2] + LOCATED
            and box[3] <= g[3] + LOCATED
            for g in self.glyphs.get(cell, ())
        )

    def _color(self, color: Sequence[float] | None) -> tuple[float, ...] | None:
        if color is None:
            return None
        return tuple((c + 0.5) % 1 for c in color) if self.plant.recolor else tuple(color)

    def _paints(self, drawing: dict) -> list[tuple[tuple[float, ...] | None, tuple[float, ...] | None]]:
        """A drawing's stroke and fill colors, as :meth:`take` takes it (none for a clip, a group, an empty path)."""
        kind = drawing['type']
        if kind in {'clip', 'group'} or not _flattened(drawing['items']):
            return []
        stroked, filled = 's' in kind, 'f' in kind
        return [(self._color(drawing['color']) if stroked else None, self._color(drawing['fill']) if filled else None)]

    def take(self, drawing: dict) -> None:
        level = drawing['level']
        while self.clips and self.clips[-1][0] >= level:
            self.clips.pop()
        clip = self.clips[-1][1] if self.clips else self.page
        kind = drawing['type']
        if kind in {'clip', 'group'}:
            if kind == 'clip':
                x0, y0, x1, y1 = self.to.box(*drawing['scissor'])
                clip = (max(clip[0], x0), max(clip[1], y0), min(clip[2], x1), min(clip[3], y1))
            self.clips.append((level, clip))
            return
        points = [self.to.point(*p) for p in _flattened(drawing['items'])]
        if not points:
            return
        xs, ys = zip(*points, strict=True)
        box = (min(xs), min(ys), max(xs), max(ys))
        stroked, filled = 's' in kind, 'f' in kind
        self.drawn.append(
            _Drawn(
                box,
                self._color(drawing['color']) if stroked else None,
                drawing['width'] if stroked else 0.0,
                _dashes(drawing['dashes']) if stroked else (),
                self._color(drawing['fill']) if filled else None,
                bool(drawing.get('even_odd')),
            )
        )
        shows = clip[0] <= clip[2] and clip[1] <= clip[3] and _meets(box, clip)
        self.shows.append(shows and not (self.glyphs and self._glyph(box)))


def _same_color(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> bool:
    if a is None or b is None or len(a) != 3 or len(b) != 3:
        return True
    return all(abs(x - y) <= PAINTED for x, y in zip(a, b, strict=True))


def _close(a: Sequence[float], b: Sequence[float], tolerance: float) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


def _paths(ours: list[_Drawn], theirs: _Theirs) -> tuple[int, int]:
    """Our painted paths no drawing matches, and PyMuPDF's fills and strokes that can show and none of ours takes."""
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, d in enumerate(theirs.drawn):
        grid[(int(d.box[0] // _DRAWING_CELL), int(d.box[1] // _DRAWING_CELL))].append(k)
    free = [[d.fill is not None, d.stroke is not None] for d in theirs.drawn]
    unmatched = 0
    for path in ours:
        x, y = path.box[0], path.box[1]
        candidates = sorted(
            k for cell in _cells(x - DRAWN, y - DRAWN, x + DRAWN, y + DRAWN, _DRAWING_CELL) for k in grid.get(cell, ())
        )
        # A stroke painted with a pattern (no components) PyMuPDF gives as a clip and its cells, not as a drawing.
        roles = [role for role, paint in enumerate((path.fill, path.stroke)) if paint]
        takes = [
            next((k for k in candidates if free[k][role] and _paints(path, theirs.drawn[k], role)), None)
            for role in roles
        ]
        if None in takes:
            unmatched += 1
            continue
        for k, role in zip(takes, roles, strict=True):
            free[k][role] = False
    unreported = sum(sum(roles) for roles, shows in zip(free, theirs.shows, strict=True) if shows)
    return unmatched, unreported


def _paints(ours: _Drawn, theirs: _Drawn, role: int) -> bool:
    """Whether ``theirs`` paints ``ours``'s fill (``role`` 0) or stroke (1): the same box, the same paint."""
    if not _close(ours.box, theirs.box, DRAWN):
        return False
    managed_stroke, managed_fill = theirs.managed
    if role == 0:
        return ours.even_odd == theirs.even_odd and (
            _same_color(ours.fill, theirs.fill) or (managed_fill is not None and _same_color(ours.fill, managed_fill))
        )
    return (
        (
            _same_color(ours.stroke, theirs.stroke)
            or (managed_stroke is not None and _same_color(ours.stroke, managed_stroke))
        )
        and abs(ours.width - theirs.width) <= PAINTED
        and _close(ours.dashes, theirs.dashes, PAINTED)
    )


def _private(c: str) -> bool:
    """Whether ``c`` is in the Basic Multilingual Plane's private-use area."""
    return unicodedata.category(c) == 'Co'


def _same(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(abs(x - y) <= PLACED for x, y in zip(a, b, strict=True))


def _images(ours: list, theirs: pymupdf.Page, to: _Mapping) -> dict[str, int]:
    """Our placements against PyMuPDF's, turned to the page as displayed: how many, invented and clipped."""
    unmatched = [to.box(*i['bbox']) for i in theirs.get_image_info()]
    invented = 0
    for box in (tuple(b) for b in ours):
        match = next((p for p in unmatched if _same(box, p)), None)
        if match is None:
            invented += 1
        else:
            unmatched.remove(match)
    return {'images': len(ours), 'invented': invented, 'clipped': len(unmatched)}


@contextmanager
def _color_management(*, on: bool) -> Iterator[None]:
    """PyMuPDF's color management ``on`` or off, then back to MuPDF's default (on, when built with ICC)."""
    pymupdf.TOOLS.set_icc(on=on)
    try:
        yield
    finally:
        pymupdf.TOOLS.set_icc(on=bool(pymupdf.mupdf.FZ_ENABLE_ICC))


def compare(path: Path, reference_path: Path | None = None, plant: Plant | None = None) -> Compared:
    """``path`` read by both, page by page; refused by both for a password, ``locked``.

    ``reference_path``: PyMuPDF reads this document instead; ``plant``: a wrong truth planted in what PyMuPDF reads --
    the planted controls the checks must report.
    """
    plant = plant or Plant()
    with _color_management(on=False), pymupdf.open(reference_path or path) as reference:
        if reference.needs_pass:
            try:
                read_pdf(path)
            except Locked:
                return Compared(path, locked=True, pages=(), reference_pages=reference.page_count)
            msg = f'{path.name}: PyMuPDF asks a password, the reader did not'
            raise AssertionError(msg)
        doc = read_pdf(path)
        pages = []
        for ours, theirs in zip(doc.pages, reference, strict=False):
            to = _Mapping.of(theirs, plant)
            type3 = frozenset(f[3] for f in theirs.get_fonts() if f[2] == 'Type3')
            raw = theirs.get_text('rawdict', flags=pymupdf.TEXTFLAGS_TEXT)
            chars, glyphs = _chars(raw, to, type3, plant.rise)
            mine = Counter(c for s in ours.spans for c in s.text if c.strip())
            unknown = mine.pop(UNKNOWN, 0)
            other = Counter(ch.char for ch in chars)
            matched = sum((mine & other).values())
            stated = sum(mine.values())
            private = sum(n for c, n in (other - mine).items() if _private(c))
            located, edges = _located(ours.spans, chars)
            drawn = _ours(ours.paths)
            unmatched, unreported = _paths(drawn, _Theirs(theirs, to, plant, glyphs))
            pages.append(
                Page(
                    number=ours.number,
                    size_off=max(abs(ours.width - theirs.rect.width), abs(ours.height - theirs.rect.height)),
                    matched=matched,
                    stated=stated,
                    reference=sum(other.values()),
                    unknown=unknown,
                    private=min(private, stated - matched),
                    located=located,
                    edges=tuple(edges),
                    **_images([f.provenance.bbox for f in ours.figures], theirs, to),
                    paths=len(drawn),
                    colored=sum(any(c is not None and len(c) == 3 for c in (d.fill, d.stroke)) for d in drawn),
                    unmatched=unmatched,
                    unreported=unreported,
                )
            )
        return Compared(path, locked=False, pages=tuple(pages), reference_pages=reference.page_count)
