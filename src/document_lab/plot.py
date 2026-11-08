"""Curves read off plots: a frame, axes calibrated from their tick labels, curves mapped to data.

:func:`plots` reads the VECTOR plots of a page; :func:`raster_plot` reads one plot from pixels (:class:`Raster`).
Every :class:`Curve` says which (``basis``); a raster curve carries one pixel's extent in data units per point.

A plot here is an axis-aligned rectangular frame with numeric tick labels below it (x) and left of it (y). A label
is text, or glyph outlines / pixels read by shape (:mod:`document_lab.glyphs`, which refuses rather than guesses). Each
label's position is snapped to the tick mark or grid line it labels when one is drawn, else its box centre is
used. A tick label is a number in the axis' common font size, aligned with the others on one row (x) or column
(y), whose box overlaps the frame's extent along the axis; a stray number off that row or past the frame is not.

An axis is ``linear`` or ``log10``, forced by the caller or else the one whose fit needs fewer pieces (then the
smaller misfit; two labels fit both exactly, so they read as linear unless told otherwise). The fit is PIECEWISE:
the axis is cut at every tick where the data-per-point spacing (in the scale's units) changes by more than
``BREAK_TOL``, and each piece is a least-squares line through its own ticks -- a broken axis whose scale changes at
a break reads right, and an axis whose spacing is neither linear nor log10 maps linearly between each pair of
labelled ticks. Ticks whose values are not strictly monotone along the axis are refused (``ValueError``).

Every stroked open polyline inside the frame that is not a tick, a grid line or the frame is a curve, after
fragments are joined: polylines of one style (stroke, width, dash array) whose ends meet (to ``JOIN_DIGITS``
decimals) become one, unless more than two ends of that style meet there (ambiguous, left unjoined). A short
horizontal curve (at most ``SAMPLE_MAX`` long) with a text span starting within ``SAMPLE_GAP`` right of it, at its
height, is a legend sample: it goes to ``Plot.legend`` with that text, not to ``Plot.curves``.

In :func:`plots`, a frame whose curves are a RASTER image raises :class:`RasterPlot` (a silent empty result would
read as "this plot has no curves"): read it with :func:`raster_plot` over ``pdf.read_raster``. Secondary (right/top)
axes are not read.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from document_lab.glyphs import Glyph, UnreadableGlyph, rasterise, read_number
from document_lab.ir import BBox, Page, Provenance, Raster, TextSpan, VectorPath, union
from document_lab.numbers import parse_number

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    'Axis',
    'Basis',
    'Curve',
    'LegendSample',
    'NoPlotFrame',
    'Plot',
    'RasterPlot',
    'Scale',
    'UnreadableTickLabels',
    'plots',
    'raster_plot',
]

Scale = Literal['linear', 'log10']
#: How a curve was read: off drawn paths, or traced through pixels.
Basis = Literal['vector', 'raster']

#: The smallest frame side (points) that can hold a plot.
MIN_FRAME: Final = 40.0
#: How far outside the frame (points) a tick label may sit.
LABEL_REACH: Final = 30.0
#: The longest tick mark (points).
TICK_MAX: Final = 10.0
#: How close (points) two coordinates must be to count as the same edge.
EDGE_TOL: Final = 1.5
#: Coordinates closer than this (points) count as equal along an edge.
AXIS_TOL: Final = 0.5
#: A filled rectangle no thicker than this (points) is a rule, drawn as a fill.
RULE_MAX: Final = 1.5
#: The longest drawn minus sign (points) between a ``10`` and its exponent.
DASH_MAX: Final = 5.0
#: The widest gap (points) between a drawn minus sign and its digits.
DASH_GAP: Final = 2.5
#: A span holding only one of these is a minus sign set apart from its digits.
MINUS_GLYPHS: Final = frozenset({'-', chr(0x2212), chr(0x2013), chr(0x2010)})
#: The widest gap (points) between two digit spans of one number set glyph by glyph.
GLYPH_GAP: Final = 0.3
#: A span of digits only (a glyph run's piece).
_DIGITS: Final = re.compile(r'\d+')
#: Font sizes (points) closer than this are one size.
SIZE_TOL: Final = 0.25
#: How close (points) a shifted label must land to a mark to count as on it.
SNAP_TOL: Final = 0.75
#: A grid line spans at least this fraction of the frame.
GRID_SPAN: Final = 0.9
#: The relative change in data-per-point spacing between neighbouring tick intervals that marks an axis break.
BREAK_TOL: Final = 0.1
#: Decimals (points) to which two fragment ends must agree to meet.
JOIN_DIGITS: Final = 2
#: The longest legend sample line (points).
SAMPLE_MAX: Final = 40.0
#: The widest gap (points) between a legend sample line and its label.
SAMPLE_GAP: Final = 10.0
#: The largest glyph (points, either side) an outlined tick label is set in.
GLYPH_MAX: Final = 12.0
#: A glyph below this fraction of a word's height whose bottom sits above it by the rest is raised (an exponent).
RAISED: Final = 0.85
#: The widest ink blob (points) a raster label may be: glyphs that touch at a coarse pixel are one blob.
BLOB_MAX: Final = 3 * GLYPH_MAX
#: The widest gap between two glyphs of one outlined label, as a fraction of the label's height.
WORD_GAP: Final = 0.6
#: A pixel this dark (0 white .. 1 black, the mean of R, G and B) is ink: a hairline grid at 100 dpi averages 0.2.
INK_LEVEL: Final = 0.15
#: A pixel whose R, G, B spread at least this much is coloured ink, however light.
CHROMA_MIN: Final = 0.3
#: Hue sectors (degrees) a coloured curve is told apart by; the first is centred on red.
HUE_SECTOR: Final = 30.0
#: A traced curve spans at least this share of the frame's width or height (smaller ink is text or a marker).
CURVE_SPAN: Final = 0.05
#: A pixel row or column inked across this share of the frame interior is a grid line, not a curve.
GRID_INK: Final = 0.9
#: The share of the frame an image must cover for a curve-less plot to count as a raster plot.
RASTER_COVER: Final = 0.25


class RasterPlot(ValueError):
    """The plot is a raster image: :func:`plots` reads vector plots; read its pixels with :func:`raster_plot`."""


class NoPlotFrame(ValueError):
    """A raster holds no plot frame: no long horizontal line meets a long vertical one at a bottom-left corner."""


class UnreadableTickLabels(ValueError):
    """A raster axis' tick labels cannot be read deterministically, and the caller stated no values for them."""


@dataclass(frozen=True, slots=True)
class Axis:
    """A page coordinate -> data value map: per piece ``value = s(slope * position + intercept)``, ``s`` id or ``10**``.

    The first piece also covers positions before it, the last those after it.
    """

    scale: Scale
    #: The pieces in increasing position: (position the piece starts at, slope, intercept).
    pieces: tuple[tuple[float, float, float], ...]
    #: The calibration points: (page coordinate, labelled value).
    ticks: tuple[tuple[float, float], ...]

    def to_data(self, position: np.ndarray) -> np.ndarray:
        """The data values at page coordinates ``position``."""
        p = np.asarray(position, dtype=float)
        starts, slopes, intercepts = (np.array(c) for c in zip(*self.pieces, strict=True))
        k = np.clip(np.searchsorted(starts, p, side='right') - 1, 0, len(starts) - 1)
        t = slopes[k] * p + intercepts[k]
        return 10.0**t if self.scale == 'log10' else t


@dataclass(frozen=True, slots=True, eq=False)
class Curve:
    """One digitised curve in data coordinates."""

    x: np.ndarray
    y: np.ndarray
    provenance: Provenance
    axis_scales: tuple[Scale, Scale]
    stroke: tuple[float, ...] | None
    width: float
    #: The stroke's dash array (points: on, off, ...); empty for a solid curve.
    dashes: tuple[float, ...]
    basis: Basis
    #: Per point, the extent of one pixel in data units of x and of y (raster); ``None`` for a vector curve.
    x_uncertainty: np.ndarray | None
    y_uncertainty: np.ndarray | None


@dataclass(frozen=True, slots=True)
class LegendSample:
    """A legend entry inside the frame: a short sample line in a curve's style and the text beside it."""

    label: str
    stroke: tuple[float, ...] | None
    width: float
    dashes: tuple[float, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True, eq=False)
class Plot:
    """A calibrated frame, the curves inside it and its legend samples."""

    frame: BBox
    x: Axis
    y: Axis
    curves: tuple[Curve, ...]
    legend: tuple[LegendSample, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class _Label:
    value: float
    box: BBox
    size: float


@dataclass(frozen=True, slots=True)
class _Edge:
    """An axis-parallel segment: ``at`` is its constant coordinate, ``lo``..``hi`` its extent along the other."""

    vertical: bool
    at: float
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo


def plots(page: Page, *, x_scale: Scale | None = None, y_scale: Scale | None = None) -> list[Plot]:
    """Every vector plot on ``page``; ``x_scale``/``y_scale`` force an axis scale instead of inferring it."""
    edges = _edges(page.paths)
    labels = _labels(page.spans, [e for e in edges if not e.vertical and e.length <= DASH_MAX])
    labels += _outline_labels(page.paths, [e for e in edges if not e.vertical and e.length <= DASH_MAX])
    # A grid line may stop short of the far edge (a legend sits there), so a long rule anchored on an edge counts.
    found = []
    for frame in _frames(edges):
        xs = _aligned(_one_size(_x_labels(frame, labels)), vertical=False)
        ys = _aligned(_one_size(_y_labels(frame, labels)), vertical=True)
        if len(xs) < 2 or len(ys) < 2:  # noqa: PLR2004 -- a line needs two points
            continue
        x_ticks = _snap([((lb.box.x0 + lb.box.x1) / 2, lb.value) for lb in xs], _marks(frame, edges, vertical=True))
        y_ticks = _snap([((lb.box.y0 + lb.box.y1) / 2, lb.value) for lb in ys], _marks(frame, edges, vertical=False))
        x_axis, y_axis = _fit(x_ticks, x_scale), _fit(y_ticks, y_scale)
        legend, paths = [], []
        for path in _joined(_curve_paths(frame, page.paths)):
            label = _sample_label(path, page.spans)
            if label is None:
                paths.append(path)
            else:
                legend.append(LegendSample(label, path.stroke, path.width, path.dashes, path.provenance))
        curves = tuple(
            Curve(
                x_axis.to_data(np.array([p[0] for p in path.points])),
                y_axis.to_data(np.array([p[1] for p in path.points])),
                path.provenance,
                (x_axis.scale, y_axis.scale),
                path.stroke,
                path.width,
                path.dashes,
                'vector',
                None,
                None,
            )
            for path in paths
        )
        if not curves:
            _refuse_raster(page, frame)
        found.append(Plot(frame, x_axis, y_axis, curves, tuple(legend), page.paths[0].provenance.at(bbox=frame)))
    return found


def _edges(paths: Sequence[VectorPath]) -> list[_Edge]:
    """Every axis-parallel stroked segment, and every thin filled rectangle (a rule drawn as a fill)."""
    out = []
    for path in paths:
        if path.stroke is None:
            box = path.provenance.bbox
            if path.closed and box is not None and min(box.width, box.height) <= RULE_MAX:
                if box.height > box.width:
                    out.append(_Edge(vertical=True, at=(box.x0 + box.x1) / 2, lo=box.y0, hi=box.y1))
                else:
                    out.append(_Edge(vertical=False, at=(box.y0 + box.y1) / 2, lo=box.x0, hi=box.x1))
            continue
        points = (*path.points, path.points[0]) if path.closed else path.points
        for (x0, y0), (x1, y1) in itertools.pairwise(points):
            if abs(x0 - x1) <= AXIS_TOL < abs(y0 - y1):
                out.append(_Edge(vertical=True, at=(x0 + x1) / 2, lo=min(y0, y1), hi=max(y0, y1)))
            elif abs(y0 - y1) <= AXIS_TOL < abs(x0 - x1):
                out.append(_Edge(vertical=False, at=(y0 + y1) / 2, lo=min(x0, x1), hi=max(x0, x1)))
    return out


def _frames(edges: Sequence[_Edge]) -> list[BBox]:
    """A frame per bottom-left corner: a long horizontal edge (the x axis) meeting a long vertical one (the y axis).

    A corner where several edges meet (a grid line cut short along the x axis) keeps its largest frame.
    """
    long = [e for e in edges if e.length >= MIN_FRAME]
    corners: dict[tuple[float, float], BBox] = {}
    for h in (e for e in long if not e.vertical):
        for v in (e for e in long if e.vertical):
            if abs(h.lo - v.at) <= EDGE_TOL and abs(h.at - v.hi) <= EDGE_TOL:
                box = BBox(v.at, v.lo, h.hi, h.at)
                key = (round(box.x0), round(box.y1))
                if key not in corners or box.width * box.height > corners[key].width * corners[key].height:
                    corners[key] = box
    return list(corners.values())


def _labels(spans: Sequence[TextSpan], dashes: Sequence[_Edge]) -> list[_Label]:
    """Numeric spans, with ``10`` plus a raised smaller integer span merged into a power of ten."""
    spans = _glyph_runs(spans)
    out, used = [], set()
    for i, span in enumerate(spans):
        box = span.provenance.bbox
        if box is None or i in used:
            continue
        if span.text.strip() == '10':
            found = _exponent_of(i, spans, dashes)
            if found is not None:
                j, power = found
                used.add(j)
                out.append(_Label(10.0**power, union([box, spans[j].provenance.bbox]), span.size))
                continue
        value = parse_number(span.text)
        if value is None:
            continue
        minus = _minus_before(box, spans, dashes) if value > 0 else None
        if minus is not None:
            value, box = -value, union([box, minus])
        out.append(_Label(value, box, span.size))
    return out


def _outline_labels(paths: Sequence[VectorPath], dashes: Sequence[_Edge]) -> list[_Label]:
    """Numbers set as filled glyph OUTLINES (a sheet exported with its text converted to paths).

    A glyph is a small filled shape plus the subpaths whose centres it encloses (a ``0``'s hole); a word is a row of
    glyphs no further apart than ``WORD_GAP`` of its height, and a raised smaller glyph starts a word of its own (an
    exponent). A word reading ``10`` followed by a raised integer word is that power of ten, as with text labels; a
    drawn dash before a label (or between ``10`` and its exponent) is its minus. A word the reader cannot read is no
    label.
    """
    words = [w for w in _outline_words(paths) if w[0] is not None]
    used: set[int] = set()
    out = []
    for i, (value, box) in enumerate(words):
        if i in used:
            continue
        if value == 10:  # noqa: PLR2004 -- a power-of-ten base
            power = _outline_exponent(box, words, dashes)
            if power is not None:
                j, exponent = power
                used.add(j)
                out.append(_Label(10.0**exponent, union([box, words[j][1]]), round(box.height, 1)))
                continue
        minus = _minus_before(box, (), dashes) if value > 0 else None
        signed = (value, box) if minus is None else (-value, union([box, minus]))
        out.append(_Label(signed[0], signed[1], round(box.height, 1)))
    return out


def _outline_words(paths: Sequence[VectorPath]) -> list[tuple[float | None, BBox]]:
    """Each row of outlined glyphs with the number it reads as (``None`` when it reads as none)."""
    shapes = [
        p
        for p in paths
        if p.stroke is None
        and p.provenance.bbox is not None
        and len(p.points) > 2  # noqa: PLR2004 -- a fill needs an area
        and max(p.provenance.bbox.width, p.provenance.bbox.height) <= GLYPH_MAX
    ]
    shapes.sort(key=lambda p: -p.provenance.bbox.width * p.provenance.bbox.height)
    glyphs: list[tuple[BBox, list[tuple[tuple[float, float], ...]]]] = []
    for shape in shapes:
        box = shape.provenance.bbox
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        host = next((g for g in glyphs if g[0].contains(cx, cy) and g[0] != box), None)
        if host is None:
            glyphs.append((box, [shape.points]))
        else:
            host[1].append(shape.points)
    out: list[tuple[float | None, BBox]] = []
    for word in _words([g[0] for g in glyphs]):
        members = [glyphs[i] for i in word]
        try:
            value = read_number([Glyph(rasterise(polys, tuple(box)), tuple(box)) for box, polys in members])
        except UnreadableGlyph:
            value = None
        out.append((value, union([g[0] for g in members])))
    return out


def _words(boxes: Sequence[BBox]) -> list[list[int]]:
    """Glyph boxes grouped into words, as indices left to right: one row, gaps under ``WORD_GAP`` of the height.

    A raised smaller glyph (an exponent) starts a word of its own.
    """
    words: list[list[int]] = []
    for i in sorted(range(len(boxes)), key=lambda k: boxes[k].x0):
        box = boxes[i]
        cy = (box.y0 + box.y1) / 2
        for word in words:
            wb = union([boxes[k] for k in word])
            same_row = wb.y0 <= cy <= wb.y1 or box.y0 <= (wb.y0 + wb.y1) / 2 <= box.y1
            raised = box.height < RAISED * wb.height and box.y1 < wb.y1 - (1 - RAISED) * wb.height
            if same_row and not raised and -EDGE_TOL <= box.x0 - wb.x1 <= WORD_GAP * max(wb.height, box.height):
                word.append(i)
                break
        else:
            words.append([i])
    return words


def _outline_exponent(
    base: BBox, words: Sequence[tuple[float | None, BBox]], dashes: Sequence[_Edge]
) -> tuple[int, int] | None:
    """The raised smaller integer word just right of an outlined ``10``, negative when a drawn dash is between."""
    for j, (power, box) in enumerate(words):
        if power is None or power != int(power) or box.height >= RAISED * base.height:
            continue
        if not (box.y1 < base.y1 - (1 - RAISED) * base.height and box.y1 > base.y0):
            continue
        gap = box.x0 - base.x1
        if -EDGE_TOL <= gap <= EDGE_TOL:
            return j, int(power)
        dashed = any(
            base.x1 - EDGE_TOL <= d.lo and d.hi <= box.x0 + EDGE_TOL and base.y0 <= d.at <= box.y1 for d in dashes
        )
        if 0 < gap <= DASH_MAX + EDGE_TOL and dashed and power > 0:
            return j, -int(power)
    return None


def _glyph_runs(spans: Sequence[TextSpan]) -> list[TextSpan]:
    """Digit spans abutting on one baseline in one size, joined: a typesetter may emit ``250`` as three spans.

    Every other span keeps its place in the page's order, and a run takes its leftmost piece's place: the readers
    downstream pair a ``10`` with its exponent in that order, so the join must not reorder what it does not join.
    """
    digits = [i for i, s in enumerate(spans) if s.provenance.bbox is not None and _DIGITS.fullmatch(s.text)]
    digits.sort(key=lambda i: (round(spans[i].provenance.bbox.y1), spans[i].provenance.bbox.x0))
    joined: dict[int, TextSpan] = {}
    absorbed: set[int] = set()
    head = None
    for i in digits:
        span, box = spans[i], spans[i].provenance.bbox
        prev = joined[head] if head is not None else None
        pb = prev.provenance.bbox if prev is not None else None
        if (
            prev is not None
            and abs(span.size - prev.size) <= SIZE_TOL
            and abs(box.y1 - pb.y1) <= AXIS_TOL
            and abs(box.x0 - pb.x1) <= GLYPH_GAP
        ):
            joined[head] = replace(
                prev, text=prev.text + span.text, provenance=prev.provenance.at(bbox=union([pb, box]))
            )
            absorbed.add(i)
        else:
            head = i
            joined[i] = span
    return [joined.get(i, s) for i, s in enumerate(spans) if i not in absorbed]


def _minus_before(box: BBox, spans: Sequence[TextSpan], dashes: Sequence[_Edge]) -> BBox | None:
    """The box of a minus sign ending just left of the label, within its height.

    It is a drawn short horizontal stroke, or a span of its own holding only a minus glyph (a minus set in another
    font than its digits). A drawn dash may end up to ``EDGE_TOL`` inside the label's box (a glyph box carries
    side bearing) but must START left of it.
    """
    for d in dashes:
        if d.lo < box.x0 and box.x0 - DASH_GAP <= d.hi <= box.x0 + EDGE_TOL and box.y0 < d.at < box.y1:
            return BBox(d.lo, box.y0, d.hi, box.y1)
    for span in spans:
        sb = span.provenance.bbox
        if (
            span.text.strip() in MINUS_GLYPHS
            and sb is not None
            and box.x0 - DASH_GAP <= sb.x1 <= box.x0 + EDGE_TOL
            and box.y0 < (sb.y0 + sb.y1) / 2 < box.y1
        ):
            return sb
    return None


def _one_size(labels: list[_Label]) -> list[_Label]:
    """The labels in the axis' most common font size: a unit's superscript beside the axis is not a tick label."""
    if not labels:
        return labels
    sizes = [round(lb.size, 1) for lb in labels]
    common = max(set(sizes), key=sizes.count)
    return [lb for lb, size in zip(labels, sizes, strict=True) if abs(size - common) <= SIZE_TOL]


def _exponent_of(i: int, spans: Sequence[TextSpan], dashes: Sequence[_Edge]) -> tuple[int, int] | None:
    """The raised smaller integer right after span ``i`` and its value; a drawn dash in the gap makes it negative."""
    base = spans[i]
    bb = base.provenance.bbox
    for j, other in enumerate(spans):
        ob = other.provenance.bbox
        power = parse_number(other.text)
        if j == i or ob is None or power is None or power != int(power) or other.size >= base.size * 0.9:
            continue
        raised = (ob.y0 + ob.y1) / 2 < (bb.y0 + bb.y1) / 2 and ob.y1 > bb.y0 and ob.y0 >= bb.y0 - ob.height
        if not raised:
            continue
        gap = ob.x0 - bb.x1
        if -EDGE_TOL <= gap <= EDGE_TOL:
            return j, int(power)
        minus = _minus_before(ob, spans, ())
        if 0 < gap <= DASH_MAX + EDGE_TOL and minus is not None and minus.x0 >= bb.x1 - EDGE_TOL and power > 0:
            return j, -int(power)
        dashed = any(bb.x1 - EDGE_TOL <= d.lo and d.hi <= ob.x0 + EDGE_TOL and bb.y0 <= d.at <= ob.y1 for d in dashes)
        if 0 < gap <= DASH_MAX + EDGE_TOL and dashed and power > 0:
            return j, -int(power)
    return None


def _x_labels(frame: BBox, labels: list[_Label]) -> list[_Label]:
    return [
        lb
        for lb in labels
        if frame.y1 - EDGE_TOL <= lb.box.y0 <= frame.y1 + LABEL_REACH
        and lb.box.x0 <= frame.x1
        and lb.box.x1 >= frame.x0
    ]


def _y_labels(frame: BBox, labels: list[_Label]) -> list[_Label]:
    return [
        lb
        for lb in labels
        if frame.x0 - LABEL_REACH <= lb.box.x1 <= frame.x0 + EDGE_TOL
        and lb.box.y0 <= frame.y1
        and lb.box.y1 >= frame.y0
    ]


def _aligned(labels: list[_Label], *, vertical: bool) -> list[_Label]:
    """The largest set of labels sharing one row (x: top, bottom or centre) or column (y: right, left or centre).

    A tie goes to the edge facing the axis (x: top, y: right): an axis title flush with the labels' far edge ties
    with them there, and is not one.
    """
    near, far = ('x1', 'x0') if vertical else ('y0', 'y1')
    keys = (
        lambda b: getattr(b, near),
        lambda b: getattr(b, far),
        lambda b: (getattr(b, near) + getattr(b, far)) / 2,
    )
    best: list[_Label] = []
    for key in keys:
        for anchor in labels:
            row = [lb for lb in labels if abs(key(lb.box) - key(anchor.box)) <= EDGE_TOL]
            if len(row) > len(best):
                best = row
    return best


def _marks(frame: BBox, edges: Sequence[_Edge], *, vertical: bool) -> list[float]:
    """Positions of tick marks (short, on a frame edge) and grid lines (spanning the frame) across one axis."""
    lo_edge, hi_edge, span = (frame.y0, frame.y1, frame.height) if vertical else (frame.x0, frame.x1, frame.width)
    first, last = (frame.x0, frame.x1) if vertical else (frame.y0, frame.y1)
    marks = []
    for e in edges:
        if e.vertical != vertical or not first - EDGE_TOL <= e.at <= last + EDGE_TOL:
            continue
        on_edge = any(abs(b - e.lo) <= EDGE_TOL or abs(b - e.hi) <= EDGE_TOL for b in (lo_edge, hi_edge))
        if (on_edge and e.length <= TICK_MAX) or e.length >= GRID_SPAN * span:
            marks.append(e.at)
    return marks


def _snap(ticks: list[tuple[float, float]], marks: list[float]) -> tuple[tuple[float, float], ...]:
    """Labels moved onto their marks by ONE shared offset, the one that lands the most labels on a mark.

    A label sits a constant distance from its tick (same font, same alignment), so the offset is a property of
    the axis, not of each label; snapping each to its nearest mark instead lets a dense minor grid capture it.
    A label with no mark at its shifted position keeps its own position.
    """
    positions = sorted(p for p, _ in ticks)
    spacing = min((b - a for a, b in itertools.pairwise(positions) if b > a), default=LABEL_REACH)
    reach = 0.4 * spacing
    shifts = [m - p for p in positions for m in marks if abs(m - p) <= reach]
    if not shifts:
        return tuple(sorted(ticks))

    def score(shift: float) -> tuple[int, float]:
        misses = [min(abs(m - (p + shift)) for m in marks) for p in positions]
        return (sum(miss <= SNAP_TOL for miss in misses), -sum(min(miss, reach) for miss in misses))

    shift = max(shifts, key=score)
    out = []
    for pos, value in ticks:
        near = min(marks, key=lambda m: abs(m - (pos + shift)))
        out.append((near if abs(near - (pos + shift)) <= SNAP_TOL else pos, value))
    return tuple(sorted(out))


def _fit(ticks: tuple[tuple[float, float], ...], forced: Scale | None) -> Axis:
    pos = np.array([p for p, _ in ticks])
    val = np.array([v for _, v in ticks])
    steps = np.diff(val) * np.diff(pos)
    if np.any(steps == 0) or (np.any(steps > 0) and np.any(steps < 0)):
        msg = f'tick values are not monotone along the axis: {val.tolist()} at {pos.round(1).tolist()}'
        raise ValueError(msg)
    scale = forced or _infer_scale(pos, val)
    if scale == 'log10' and np.any(val <= 0):
        msg = f'a log10 axis needs positive tick values, got {val.tolist()}'
        raise ValueError(msg)
    return Axis(scale, _pieces(pos, np.log10(val) if scale == 'log10' else val), ticks)


def _pieces(pos: np.ndarray, target: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    """Lines through runs of tick intervals of one data-per-point spacing; neighbouring pieces share a break tick."""
    rates = np.diff(target) / np.diff(pos)
    runs = [[0]]
    for i in range(1, len(rates)):
        ref = float(np.mean(rates[runs[-1]]))
        if abs(rates[i] - ref) <= BREAK_TOL * abs(ref):
            runs[-1].append(i)
        else:
            runs.append([i])
    out = []
    for run in runs:
        lo, hi = run[0], run[-1] + 2
        slope, intercept = np.polyfit(pos[lo:hi], target[lo:hi], 1)
        out.append((float(pos[lo]), float(slope), float(intercept)))
    return tuple(out)


def _infer_scale(pos: np.ndarray, val: np.ndarray) -> Scale:
    """The scale whose fit needs fewer pieces, then the one a single line fits better (two points fit both)."""
    if len(val) < 3 or np.any(val <= 0):  # noqa: PLR2004 -- two points fit both scales exactly
        return 'linear'
    log, lin = np.log10(val), val
    key_log = (len(_pieces(pos, log)), _misfit(pos, log))
    key_lin = (len(_pieces(pos, lin)), _misfit(pos, lin))
    return 'log10' if key_log < key_lin else 'linear'


def _misfit(pos: np.ndarray, target: np.ndarray) -> float:
    span = float(np.ptp(target))
    if span == 0:
        return math.inf
    coeffs = np.polyfit(pos, target, 1)
    return float(np.max(np.abs(np.polyval(coeffs, pos) - target))) / span


def _curve_paths(frame: BBox, paths: Sequence[VectorPath]) -> list[VectorPath]:
    """Stroked open paths inside the frame that are not a single tick mark, grid line or frame edge."""
    out = []
    for path in paths:
        if path.stroke is None or path.closed or len(path.points) < 2:  # noqa: PLR2004
            continue
        if not all(frame.contains(x, y, tol=EDGE_TOL) for x, y in path.points):
            continue
        rules = _edges([path])
        if len(path.points) == 2 and rules and _is_rule(rules[0], frame):  # noqa: PLR2004
            continue
        out.append(path)
    return out


def _joined(paths: Sequence[VectorPath]) -> list[VectorPath]:
    """Chains of paths of one style meeting end to end, each joined into one path in chain order.

    Two ends link only when they are the only ends of that style at that point; a third makes the point ambiguous.
    """
    ends: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for i, path in enumerate(paths):
        for end, (x, y) in ((0, path.points[0]), (1, path.points[-1])):
            ends[(path.stroke, path.width, path.dashes, round(x, JOIN_DIGITS), round(y, JOIN_DIGITS))].append((i, end))
    link = {}
    for meeting in ends.values():
        if len(meeting) == 2 and meeting[0][0] != meeting[1][0]:  # noqa: PLR2004 -- one end from each side
            link[meeting[0]], link[meeting[1]] = meeting[1], meeting[0]
    seen: set[int] = set()
    out = []
    # A chain is walked from a free end; a closed loop of fragments (no free end) from wherever it is met.
    for start in sorted(range(len(paths)), key=lambda k: (k, 0) in link and (k, 1) in link):
        if start in seen:
            continue
        i = start
        enter = 1 if (i, 0) in link and (i, 1) not in link else 0
        members, points = [], []
        while i not in seen:
            seen.add(i)
            members.append(paths[i])
            run = paths[i].points if enter == 0 else paths[i].points[::-1]
            points.extend(run if not points else run[1:])
            nxt = link.get((i, 1 - enter))
            if nxt is None:
                break
            i, enter = nxt
        first = members[0]
        if len(members) == 1:
            out.append(first)
            continue
        box = union([m.provenance.bbox for m in members if m.provenance.bbox is not None])
        out.append(
            VectorPath(tuple(points), first.stroke, first.width, first.dashes, False, first.provenance.at(bbox=box))  # noqa: FBT003
        )
    return out


def _sample_label(path: VectorPath, spans: Sequence[TextSpan]) -> str | None:
    """The text of a span starting just right of a short horizontal line, at its height: what the sample stands for."""
    box = path.provenance.bbox
    if box is None or box.width > SAMPLE_MAX or box.height > EDGE_TOL:
        return None
    y = (box.y0 + box.y1) / 2
    for span in spans:
        sb = span.provenance.bbox
        if sb is not None and box.x1 <= sb.x0 <= box.x1 + SAMPLE_GAP and sb.y0 <= y <= sb.y1:
            return span.text.strip()
    return None


def _is_rule(edge: _Edge, frame: BBox) -> bool:
    """A tick (short), a grid line across the frame, or a long one anchored on a frame edge (a legend cut it)."""
    span, first, last = (frame.height, frame.y0, frame.y1) if edge.vertical else (frame.width, frame.x0, frame.x1)
    anchored = min(abs(edge.lo - first), abs(edge.hi - last)) <= EDGE_TOL
    return edge.length <= TICK_MAX or edge.length >= GRID_SPAN * span or (anchored and edge.length >= span / 2)


def _refuse_raster(page: Page, frame: BBox) -> None:
    for figure in page.figures:
        box = figure.provenance.bbox
        if box is None:
            continue
        w = min(box.x1, frame.x1) - max(box.x0, frame.x0)
        h = min(box.y1, frame.y1) - max(box.y0, frame.y0)
        if w > 0 and h > 0 and w * h >= RASTER_COVER * frame.width * frame.height:
            msg = (
                f'page {page.number}: the plot framed at {tuple(round(v, 1) for v in frame)} is a raster image; '
                'read it with raster_plot(read_raster(...))'
            )
            raise RasterPlot(msg)


def raster_plot(
    raster: Raster,
    *,
    x_values: Sequence[float] | None = None,
    y_values: Sequence[float] | None = None,
    x_scale: Scale | None = None,
    y_scale: Scale | None = None,
) -> Plot:
    """The plot in a RASTER image: its frame, axes calibrated from its tick labels, and its curves traced.

    The frame is found as in :func:`plots`, from the image's long dark (not coloured) horizontal and vertical pixel
    runs: the largest with two or more label glyph rows below it and left of it (a drop shadow frames none). A
    tick label is a row of ink glyphs below the frame (x) or left of it (y), read by :mod:`document_lab.glyphs`; one
    the reader refuses raises :class:`UnreadableTickLabels` -- unless the caller states the axis' labelled values
    (``x_values`` left to right, ``y_values`` bottom to top), which are then paired with the label positions found
    (their counts must agree). Positions snap to tick marks and grid lines as for vector plots.

    Curves are the frame's ink less its grid lines (rows or columns inked across ``GRID_INK`` of it). Coloured ink is
    one curve per ``HUE_SECTOR`` (so two crossing curves of two colours stay two); dark ink is one curve per
    connected component spanning ``CURVE_SPAN`` of the frame (dark curves that touch are one). A curve is sampled per
    pixel column at each run's centre, or every stroke width along a run longer than two (a steep stretch); a curve
    that folds back over x (a loop) keeps both branches' samples, in column order. Each point carries one pixel's
    extent in data units as its uncertainty.
    """
    box = raster.provenance.bbox
    rgb = raster.pixels.astype(float) / 255
    rows, cols = rgb.shape[:2]
    scale = (box.width / cols, box.height / rows)  # points per pixel
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    ink = (1 - rgb.mean(axis=2) > INK_LEVEL) | (chroma > CHROMA_MIN)
    edges = _pixel_edges(ink & (chroma <= CHROMA_MIN), box, scale)  # a frame is drawn dark, not in colour
    for frame in sorted(_frames(edges), key=lambda f: -f.width * f.height):
        labels, refusals = _raster_labels(ink, 1 - rgb.mean(axis=2), frame, box, scale)
        if len(_x_labels(frame, labels)) >= 2 and len(_y_labels(frame, labels)) >= 2:  # noqa: PLR2004 -- a line
            break
    else:
        msg = (
            f'no plot frame in the raster at {tuple(round(v, 1) for v in box)}: no long lines meet at a corner '
            'with two or more labels below it and left of it'
        )
        raise NoPlotFrame(msg)
    found = {'labels': labels, 'refusals': refusals, 'edges': edges}
    x_axis = _raster_axis('x', frame, values=x_values, scale=x_scale, **found)
    y_axis = _raster_axis('y', frame, values=y_values, scale=y_scale, **found)
    masks, occluders = _curve_masks(ink, chroma, rgb, frame=frame, box=box, scale=scale)
    traced = (
        _traced(mask, occluders & ~mask, rgb=rgb, here=raster.provenance, scale=scale, axes=(x_axis, y_axis))
        for mask in masks
    )
    curves = tuple(c for c in traced if c.x.size)  # ink whose every run is half hidden yields no clean sample
    return Plot(frame, x_axis, y_axis, curves, (), raster.provenance.at(bbox=frame))


def _pixel_edges(ink: np.ndarray, box: BBox, scale: tuple[float, float]) -> list[_Edge]:
    """Every run of ink pixels along a row (horizontal) or a column (vertical), runs on neighbouring lines merged."""
    out = []
    for vertical, grid in ((False, ink), (True, ink.T)):
        step, along = (scale[0], scale[1]) if vertical else (scale[1], scale[0])
        origin, start = (box.x0, box.y0) if vertical else (box.y0, box.x0)

        def edge(lines: list[int], lo: int, hi: int, *, vertical: bool = vertical) -> _Edge:
            at = origin + (sum(lines) / len(lines) + 0.5) * step  # noqa: B023 -- read before the loop moves on
            return _Edge(vertical=vertical, at=at, lo=start + lo * along, hi=start + hi * along)  # noqa: B023

        open_runs: dict[tuple[int, int], list[int]] = {}
        for line, row in enumerate(grid):
            moved: dict[tuple[int, int], list[int]] = {}
            for lo, hi in _runs(row):
                key = next((k for k in open_runs if abs(k[0] - lo) <= 1 and abs(k[1] - hi) <= 1), None)
                moved[(lo, hi)] = [*open_runs.pop(key), line] if key is not None else [line]
            out.extend(edge(lines, lo, hi) for (lo, hi), lines in open_runs.items())
            open_runs = moved
        out.extend(edge(lines, lo, hi) for (lo, hi), lines in open_runs.items())
    return out


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """The ``[lo, hi)`` index ranges of the True runs in a 1-D boolean array."""
    padded = np.concatenate(([False], row, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(change[::2].tolist(), change[1::2].tolist(), strict=True))


def _components(mask: np.ndarray) -> np.ndarray:
    """8-connected component labels of ``mask`` (0 for background, 1.. for components)."""
    parent: list[int] = [0]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    labels = np.zeros(mask.shape, dtype=np.int64)
    previous: list[tuple[int, int, int]] = []
    for r, row in enumerate(mask):
        current = []
        for lo, hi in _runs(row):
            label = len(parent)
            parent.append(label)
            for plo, phi, plabel in previous:
                if plo <= hi and phi >= lo:  # 8-connected: [lo, hi) and [plo, phi) touch diagonally or overlap
                    a, b = find(label), find(plabel)
                    parent[max(a, b)] = min(a, b)
            labels[r, lo:hi] = label
            current.append((lo, hi, label))
        previous = current
    roots = np.array([find(i) for i in range(len(parent))])
    _, dense = np.unique(roots, return_inverse=True)
    return dense[labels]


def _raster_labels(
    ink: np.ndarray, dark: np.ndarray, frame: BBox, box: BBox, scale: tuple[float, float]
) -> tuple[list[_Label], dict[int, UnreadableGlyph]]:
    """Each row of ink glyphs outside the frame as a label: its value, or NaN with the reader's refusal by index."""
    sx, sy = scale
    xs = box.x0 + (np.arange(ink.shape[1]) + 0.5) * sx
    ys = box.y0 + (np.arange(ink.shape[0]) + 0.5) * sy
    inside = (
        (xs[None, :] >= frame.x0 - EDGE_TOL)
        & (xs[None, :] <= frame.x1 + EDGE_TOL)
        & (ys[:, None] >= frame.y0 - EDGE_TOL)
        & (ys[:, None] <= frame.y1 + EDGE_TOL)
    )
    labels = _components(ink & ~inside)
    glyphs = []
    for k in range(1, int(labels.max()) + 1):
        rr, cc = np.nonzero(labels == k)
        glyph_box = BBox(
            box.x0 + cc.min() * sx, box.y0 + rr.min() * sy, box.x0 + (cc.max() + 1) * sx, box.y0 + (rr.max() + 1) * sy
        )
        if glyph_box.height > GLYPH_MAX or glyph_box.width > BLOB_MAX:  # touching glyphs at a coarse pixel are one
            continue
        cut = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
        glyphs.append(Glyph(np.where(labels[cut] == k, dark[cut], 0.0), tuple(glyph_box)))
    out, refusals = [], {}
    for word in _words([BBox(*g.box) for g in glyphs]):
        members = [glyphs[i] for i in word]
        wb = union([BBox(*g.box) for g in members])
        try:
            value = read_number(members)
        except UnreadableGlyph as refused:
            value = math.nan
            refusals[len(out)] = refused
        out.append(_Label(value, wb, round(wb.height, 1)))
    return out, refusals


def _raster_axis(
    name: Literal['x', 'y'],
    frame: BBox,
    *,
    labels: list[_Label],
    refusals: dict[int, UnreadableGlyph],
    edges: Sequence[_Edge],
    values: Sequence[float] | None,
    scale: Scale | None,
) -> Axis:
    vertical = name == 'y'
    near = _y_labels(frame, labels) if vertical else _x_labels(frame, labels)
    row = sorted(_aligned(near, vertical=vertical), key=lambda lb: (lb.box.y0 + lb.box.y1) if vertical else lb.box.x0)
    centre = (lambda b: (b.y0 + b.y1) / 2) if vertical else (lambda b: (b.x0 + b.x1) / 2)
    if values is not None:
        ordered = row[::-1] if vertical else row  # values run bottom to top, positions top to bottom
        if len(ordered) != len(values):
            msg = f'{name} axis: {len(ordered)} labels found, {len(values)} values stated'
            raise UnreadableTickLabels(msg)
        ticks = [(centre(lb.box), float(v)) for lb, v in zip(ordered, values, strict=True)]
    else:
        ticks = [(centre(lb.box), lb.value) for lb in row if not math.isnan(lb.value)]
        if len(ticks) < 2:  # noqa: PLR2004 -- a line needs two points
            refused = next((refusals[labels.index(lb)] for lb in row if math.isnan(lb.value)), None)
            msg = (
                f'{name} axis: {len(ticks)} of {len(row)} labels read; state them '
                f'({name}_values=) or read the plot at a finer resolution'
            )
            raise UnreadableTickLabels(msg) from refused
    return _fit(_snap(ticks, _marks(frame, edges, vertical=not vertical)), scale)


def _curve_masks(
    ink: np.ndarray, chroma: np.ndarray, rgb: np.ndarray, *, frame: BBox, box: BBox, scale: tuple[float, float]
) -> tuple[list[np.ndarray], np.ndarray]:
    """One pixel mask per curve inside the frame, and every pixel that can hide a curve's ink.

    A mask is a hue sector of coloured ink or a component of dark ink; what can hide ink is any ink, and everything
    outside the frame's interior.
    """
    sx, sy = scale
    c0, c1 = math.ceil((frame.x0 + EDGE_TOL - box.x0) / sx), math.floor((frame.x1 - EDGE_TOL - box.x0) / sx)
    r0, r1 = math.ceil((frame.y0 + EDGE_TOL - box.y0) / sy), math.floor((frame.y1 - EDGE_TOL - box.y0) / sy)
    inside = np.zeros(ink.shape, dtype=bool)
    inside[r0:r1, c0:c1] = True
    mask = ink & inside
    grid_rows = mask[:, c0:c1].mean(axis=1) >= GRID_INK
    grid_cols = mask[r0:r1, :].mean(axis=0) >= GRID_INK
    mask &= ~grid_rows[:, None] & ~grid_cols[None, :]
    span_x, span_y = CURVE_SPAN * (c1 - c0), CURVE_SPAN * (r1 - r0)
    coloured = chroma > CHROMA_MIN
    sector = np.floor(((_hue(rgb) + HUE_SECTOR / 2) % 360) / HUE_SECTOR).astype(int)
    out = []
    classes = [mask & coloured & (sector == k) for k in range(round(360 / HUE_SECTOR))]
    for k, cls in enumerate([*classes, mask & ~coloured]):
        labels = _components(_bridged(cls, grid_rows, grid_cols))
        kept = []
        for label in range(1, int(labels.max()) + 1):
            rr, cc = np.nonzero(labels == label)
            if np.ptp(cc) + 1 >= span_x or np.ptp(rr) + 1 >= span_y:
                kept.append((labels == label) & cls)
        if not kept:
            continue
        if k < len(classes):
            out.append(np.logical_or.reduce(kept))
        else:
            out.extend(kept)
    return out, ink | ~inside


def _bridged(mask: np.ndarray, grid_rows: np.ndarray, grid_cols: np.ndarray) -> np.ndarray:
    """``mask`` with each removed grid line filled where ink meets it from both sides (within a pixel sideways).

    Only for telling which ink is connected: the filled pixels are never samples.
    """
    out = mask.copy()
    for grid, view in ((grid_rows, out), (grid_cols, out.T)):
        source = mask if view is out else mask.T
        for lo, hi in _runs(grid):
            if lo == 0 or hi == len(grid):
                continue
            above, below = source[lo - 1], source[hi]
            reach = [np.roll(above, s) for s in (-1, 0, 1)], [np.roll(below, s) for s in (-1, 0, 1)]
            view[lo:hi] |= (np.logical_or.reduce(reach[0]) & np.logical_or.reduce(reach[1]))[None, :]
    return out


def _hue(rgb: np.ndarray) -> np.ndarray:
    """Hue in degrees (0 red, 120 green, 240 blue) per pixel."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.degrees(np.arctan2(math.sqrt(3) * (g - b), 2 * r - g - b)) % 360


def _traced(
    mask: np.ndarray,
    hidden: np.ndarray,
    *,
    rgb: np.ndarray,
    here: Provenance,
    scale: tuple[float, float],
    axes: tuple[Axis, Axis],
) -> Curve:
    """The curve's samples, mapped through ``axes``.

    A column run that ends against ``hidden`` pixels (another curve's ink, a grid line, the frame) is skipped: part
    of its stroke may lie under them, and its centre is then not the curve's.
    """
    box = here.bbox
    sx, sy = scale
    x_axis, y_axis = axes
    rows_n = mask.shape[0]
    every = [(c, lo, hi) for c in range(mask.shape[1]) for lo, hi in _runs(mask[:, c])]
    stroke = float(np.median([hi - lo for _, lo, hi in every]))
    runs = [(c, lo, hi) for c, lo, hi in every if not (lo == 0 or hi == rows_n or hidden[lo - 1, c] or hidden[hi, c])]
    cols, rows = [], []
    for c, lo, hi in runs:
        centres = [(lo + hi - 1) / 2] if hi - lo <= 2 * stroke else np.arange(lo + stroke / 2, hi - stroke / 2, stroke)
        cols.extend([c] * len(centres))
        rows.extend(centres)
    px = box.x0 + (np.array(cols, dtype=float) + 0.5) * sx
    py = box.y0 + (np.array(rows, dtype=float) + 0.5) * sy
    rr, cc = np.nonzero(mask)
    bbox = BBox(
        box.x0 + cc.min() * sx, box.y0 + rr.min() * sy, box.x0 + (cc.max() + 1) * sx, box.y0 + (rr.max() + 1) * sy
    )
    colour = tuple(round(float(v), 2) for v in rgb[mask].mean(axis=0))
    return Curve(
        x_axis.to_data(px),
        y_axis.to_data(py),
        here.at(bbox=bbox),
        (x_axis.scale, y_axis.scale),
        colour,
        stroke * sy,
        (),
        'raster',
        np.abs(x_axis.to_data(px + sx / 2) - x_axis.to_data(px - sx / 2)),
        np.abs(y_axis.to_data(py + sy / 2) - y_axis.to_data(py - sy / 2)),
    )
