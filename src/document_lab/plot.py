"""Curves read off plots: a frame, axes calibrated from their tick labels, curves mapped to data.

:func:`plots` reads the VECTOR plots of a page; :func:`raster_plot` reads one plot from pixels (:class:`Raster`).
Every :class:`Curve` says which (``basis``); a raster curve carries one pixel's extent in data units per point.

A plot here is an axis-aligned rectangular frame (stroked, or a filled rectangle as a design tool draws it) with
numeric tick labels below it (x) and left of it (y). A label
is text, or glyph outlines / stroke-font polylines / pixels read by shape (:mod:`document_lab.glyphs`, which refuses
rather than guesses); stroke-font lettering that reads is never a curve. Each
label's position is snapped to the tick mark or grid line it labels when one is drawn, else its box center is
used. A tick label is a number in the axis' common font size, aligned with the others on one row (x) or column
(y), whose box overlaps the frame's extent along the axis; a stray number off that row or past the frame is not.

An axis is ``linear`` or ``log10``, forced by the caller or else the one whose fit needs fewer pieces (then the
smaller misfit; two labels fit both exactly, so they read as linear unless told otherwise). The fit is PIECEWISE:
the axis is cut at every tick where the data-per-point spacing (in the scale's units) changes by more than
``BREAK_TOL``, and each piece is a least-squares line through its own ticks -- a broken axis whose scale changes at
a break reads right, and an axis whose spacing is neither linear nor log10 maps linearly between each pair of
labeled ticks. Ticks whose values are not strictly monotone along the axis are refused (``ValueError``).

Every stroked open polyline inside the frame (a polyline drawn past the frame is clipped to it) that is not a tick,
a grid line or the frame is a curve, after fragments are joined: polylines of one style (stroke, width, dash
array) whose ends meet (to ``JOIN_DIGITS`` decimals) become one, unless more than two ends of that style meet
there (ambiguous, left unjoined). A short horizontal curve (at most ``SAMPLE_MAX`` long) with a text span starting
within ``SAMPLE_GAP`` right of it, at its height, is a legend sample: it goes to ``Plot.legend`` with that text,
not to ``Plot.curves``.

A data point drawn as a TEXT glyph (one of ``MARK_GLYPHS``, a filled span of its own whose center is inside the
frame) is a marker at the center of its glyph box. The reader boxes a glyph by its font's ascent and descent across
its advance, not by its ink, so a marker's ink may sit off its center by up to half that box (``Markers.size``).
Markers of one glyph and fill are split into series by the curves joining them: two markers are one series when one
curve has a vertex within half a glyph box of each; the markers no curve reaches form one series of their own (a
scatter). A white glyph is an OPEN marker (it fills an outline drawn around it), its own series by its fill. A glyph
is a legend entry, not a point, when it sits on a legend sample line (it takes that sample's label) or when no curve
reaches it and a text span starts within ``SAMPLE_GAP`` right of it, at its height (a data glyph at a curve's end
beside an annotation stays a point). Mark glyphs the reader joined into one span (set side by side on one baseline)
do not state where each stands: the page raises :class:`JoinedMarkers` rather than drop or guess them.

In :func:`plots`, a frame whose curves are a RASTER image raises :class:`RasterPlot` (a silent empty result would
read as "this plot has no curves"): read it with :func:`raster_plot` over ``pdf.read_raster``. A page on which
frames carry tick labels but none reads as a plot raises :class:`UnreadFrame` naming each. A drawn minus is a path
that is wholly one short flat stroke: a flat stretch of a curve beside a label is not one. Secondary (right/top)
axes are not read.
"""

from __future__ import annotations

import bisect
import contextlib
import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from document_lab.glyphs import (
    SANS,
    STROKE,
    Font,
    Glyph,
    UnreadableGlyph,
    has_letter,
    rasterize,
    read_number,
    read_word,
    stroke,
)
from document_lab.ir import BBox, Page, Provenance, Raster, TextSpan, VectorPath, union
from document_lab.numbers import parse_number

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    'Axis',
    'Basis',
    'Curve',
    'JoinedMarkers',
    'LegendSample',
    'Markers',
    'NoCurveOfColor',
    'NoPlotFrame',
    'Plot',
    'RasterPlot',
    'Scale',
    'UnreadFrame',
    'UnreadableTickLabels',
    'UnrecognizedFrame',
    'outline_text',
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
#: A minus candidate with another glyph-sized shape this close (points) above or below it, across it, is a stroke of
#: a word set vertically -- the l or i of a rotated axis title -- not a minus, which stands alone.
STACK_GAP: Final = 6.0
#: How far (decades) a bare mantissa digit may sit from the power labels' line to be read as a multiple of a decade.
MANTISSA_TOL: Final = 0.15
#: A span holding only one of these is a minus sign set apart from its digits (``_``: TDK SIFERRIT sheets).
MINUS_GLYPHS: Final = frozenset({'-', '_', chr(0x2212), chr(0x2013), chr(0x2010)})
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
#: The relative change in data-per-point spacing between neighboring tick intervals that marks an axis break.
BREAK_TOL: Final = 0.1
#: Decimals (points) to which two fragment ends must agree to meet.
JOIN_DIGITS: Final = 2
#: The longest legend sample line (points).
SAMPLE_MAX: Final = 40.0
#: The widest gap (points) between a legend sample line and its label.
SAMPLE_GAP: Final = 10.0
#: Text glyphs that draw a data point: filled and open circle, triangles, square, diamond, star, crosses.
MARK_GLYPHS: Final = frozenset(
    '\u25cf\u25cb\u25b2\u25b3\u25bc\u25bd\u25a0\u25a1\u25c6\u25c7\u25ce\u2605\u2606\u00d7\u2715+\uff0b'
)
#: The largest glyph (points, either side) an outlined tick label is set in.
GLYPH_MAX: Final = 12.0
#: A glyph below this fraction of a word's height whose bottom sits above it by the rest is raised (an exponent).
RAISED: Final = 0.85
#: The widest ink blob (points) a raster label may be: glyphs that touch at a coarse pixel are one blob.
BLOB_MAX: Final = 3 * GLYPH_MAX
#: The widest gap between a number and the ``min.``/``max.`` after it, as a fraction of the word's height.
BOUND_GAP: Final = 1.5
#: A glyph at most this fraction of its neighbor's height is a mark (a dot, a prime), not a letter or digit.
MARK_HEIGHT_PT: Final = 0.45
#: A fill thinner than this (points) is a sliver (an arrowhead's cut apex), never a glyph: a minus is 0.5 thick at 8.
SLIVER: Final = 0.1
#: Stroke-font fragments whose boxes come this close (points) are one glyph: glyphs of a label sit 1 pt or more apart.
STROKE_TOUCH: Final = 0.3
#: A solid wedge (an arrowhead) narrows to at most this fraction of its widest run at its tip...
WEDGE_TIP: Final = 0.35
#: ...and widens to at least this fraction at its base (:func:`_solid_triangle`).
WEDGE_BASE: Final = 0.8
#: Corners of a fill closer than this fraction of its larger side are one corner (:func:`_arrowhead`).
CORNER_MERGE: Final = 0.15
#: The widest gap between a dot and the glyph below it (an ``i``), as a fraction of that glyph's height.
DOT_GAP: Final = 0.5
#: The widest gap between two glyphs of one outlined label, as a fraction of the label's height.
WORD_GAP: Final = 0.6
#: A column of at least this many glyphs stacked tight is a turned title (:func:`_columns`), never a label.
TURNED_GLYPHS: Final = 3
#: A pixel this dark (0 white .. 1 black, the mean of R, G and B) is ink: a hairline grid at 100 dpi averages 0.2.
INK_LEVEL: Final = 0.15
#: A pixel whose R, G, B spread at least this much is colored ink, however light.
CHROMA_MIN: Final = 0.3
#: Hue sectors (degrees) a colored curve is told apart by; the first is centered on red.
HUE_SECTOR: Final = 30.0
#: A traced curve spans at least this share of the frame's width or height (smaller ink is text or a marker).
CURVE_SPAN: Final = 0.05
#: A pixel row or column inked across this share of the frame interior is a grid line, not a curve.
GRID_INK: Final = 0.9
#: A raster edge at least this share dark along its line is a frame or tick line; a curve's color fills the rest.
DARK_SHARE: Final = 0.25
#: The share of the frame an image must cover for a curve-less plot to count as a raster plot.
RASTER_COVER: Final = 0.25


class RasterPlot(ValueError):
    """The plot is a raster image: :func:`plots` reads vector plots; read its pixels with :func:`raster_plot`."""


class NoPlotFrame(ValueError):
    """A raster holds no plot frame: no long horizontal line meets a long vertical one at a bottom-left corner."""


class UnreadFrame(ValueError):
    """Every frame on a page that carries a tick label failed to read as a plot (an axis with under two labels).

    Returning no plot would look exactly like a page that holds none, so the page and its frames are named instead.
    """


class UnrecognizedFrame(ValueError):
    """A vector frame's tick-label row or column holds an outlined word the glyph reader refuses.

    Calibrating from the labels that did read would silently drop one (older SIFERRIT sheets set labels in glyphs
    the reader does not know), so the frame is refused by name instead.
    """


class JoinedMarkers(ValueError):
    """Mark glyphs inside a plot's frame were joined into one span: where each of them stands is not stated.

    The reader joins glyphs set side by side on one baseline; dropping the span would lose data points silently,
    and splitting it would guess their places.
    """


class NoCurveOfColor(ValueError):
    """A raster plot holds no curve of the color the caller asked :func:`raster_plot` for."""


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
    #: The calibration points: (page coordinate, labeled value).
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
    """One digitized curve in data coordinates."""

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
class Markers:
    """Data points drawn as one text glyph in one fill, in data coordinates, ordered by x."""

    glyph: str
    fill: tuple[float, ...]
    x: np.ndarray
    y: np.ndarray
    #: The glyphs' median box height (points), their font's descent to its ascent: a point stands at its box's
    #: center, and its ink may sit off it by up to half of this.
    size: float
    #: The text of the series' legend entry, or ``None`` when the frame holds none.
    label: str | None
    provenance: Provenance


@dataclass(frozen=True, slots=True, eq=False)
class Plot:
    """A calibrated frame, the curves and glyph-marker series inside it and its legend samples.

    :func:`raster_plot` reads curves alone: a raster plot's legend and markers are empty because they are not read.
    """

    frame: BBox
    x: Axis
    y: Axis
    curves: tuple[Curve, ...]
    legend: tuple[LegendSample, ...]
    markers: tuple[Markers, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class _Label:
    value: float
    box: BBox
    size: float
    #: Set as ``10`` plus a raised exponent, not as digits.
    power: bool = False
    #: The box the label stands at, when not all of ``box``: a power label stands where its ``10`` stands, the
    #: raised exponent set off it.
    base: BBox | None = None

    def at(self, *, vertical: bool) -> float:
        """Where along the axis the label stands: the middle of its base box (its whole box but for a power)."""
        b = self.base or self.box
        return (b.y0 + b.y1) / 2 if vertical else (b.x0 + b.x1) / 2


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


def plots(
    page: Page,
    *,
    x_scale: Scale | None = None,
    y_scale: Scale | None = None,
    include_hidden: bool = False,
    include_inkless: bool = False,
    include_in_background_color: bool = False,
) -> list[Plot]:
    """Every vector plot on ``page``; ``x_scale``/``y_scale`` force an axis scale instead of inferring it.

    ``include_hidden``: what hidden layers draw is read too; ``include_inkless``: what the colorant ``None`` paints;
    ``include_in_background_color``: the text filled in the color under it (see `document_lab.ir`).
    """
    page = page.shown(
        include_hidden=include_hidden,
        include_inkless=include_inkless,
        include_in_background_color=include_in_background_color,
    )
    edges, dashes = _edges(page.paths), _dashes(page.paths)
    frames = _frames(edges)
    labels = _labels(page.spans, dashes)
    filled, (drawn, lettering_paths) = _outline_words(_without_ticks(page.paths, frames)), _stroke_words(page.paths)
    outlined = [*filled, *drawn]
    labels += _outline_labels(outlined, dashes)
    unread = [box for value, text, box in filled if value is None and text is None]
    unread_drawn = [box for value, text, box in drawn if value is None and text is None]
    lettering = _stroked_words(page.paths)
    words = [*page.spans, *_text_spans(page, outlined)]  # a legend may be outlined
    # A grid line may stop short of the far edge (a legend sits there), so a long rule anchored on an edge counts.
    found, unread_frames, stroked = [], [], []
    for frame in frames:
        x_marks, y_marks = _marks(frame, edges, vertical=True), _marks(frame, edges, vertical=False)
        xs = _aligned(_one_size(_x_labels(frame, labels)), vertical=False)
        ys = _aligned(_one_size(_y_labels(frame, labels)), vertical=True)
        xs = _decade_mantissas(_multiplied(xs, x_marks, vertical=False), vertical=False)
        ys = _decade_mantissas(_multiplied(ys, y_marks, vertical=True), vertical=True)
        # The stroke font holds only the characters its sheets print: a stroked word (``A/m``, a lone degree sign)
        # unread for a missing character refuses the frame only when it is the size of the frame's labels.
        sizes = {lb.size for lb in (*xs, *ys)}
        sized = [b for b in unread_drawn if any(abs(round(b.height, 1) - s) <= SIZE_TOL for s in sizes)]
        _refuse_unread(page, frame, (xs, ys), [*unread, *sized], (x_marks, y_marks))
        if len(xs) < 2 or len(ys) < 2:  # noqa: PLR2004 -- a line needs two points
            box = _stroked_label(frame, lettering, (x_marks, y_marks))
            if box is not None:
                stroked.append((frame, box))
            elif xs or ys:
                unread_frames.append((frame, len(xs), len(ys)))
            continue
        x_ticks = _snap(
            [((lb.box.x0 + lb.box.x1) / 2, lb.value) for lb in xs],
            _marks(frame, edges, vertical=True),
            max(lb.box.width for lb in xs) / 2,
        )
        y_ticks = _snap(
            [(lb.at(vertical=True), lb.value) for lb in ys],
            _marks(frame, edges, vertical=False),
            max(lb.box.height for lb in ys) / 2,
        )
        x_axis, y_axis = _fit(x_ticks, x_scale), _fit(y_ticks, y_scale)
        legend, paths = [], []
        for path in _joined(_curve_paths(frame, [p for p in page.paths if id(p) not in lettering_paths])):
            label = _sample_label(path, words)
            if label is None:
                paths.append(path)
            else:
                legend.append(LegendSample(label, path.stroke, path.width, path.dashes, path.provenance))
        curves = tuple(
            Curve(
                x_axis.to_data(path.points[:, 0]),
                y_axis.to_data(path.points[:, 1]),
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
        markers = _markers(page, frame, (paths, legend, words), (x_axis, y_axis))
        if not curves and not markers:
            _refuse_raster(page, frame)
        found.append(
            Plot(frame, x_axis, y_axis, curves, tuple(legend), markers, page.paths[0].provenance.at(bbox=frame))
        )
    if not found and stroked:
        named = '; '.join(
            f'{tuple(round(v, 1) for v in frame)} (first at {tuple(round(v, 1) for v in box)})'
            for frame, box in stroked
        )
        msg = (
            f'page {page.number}: no frame reads as a plot, and these draw their tick labels as stroked lines '
            f'(stroke-font lettering), which the glyph reader does not read: {named}: an unrecognized frame'
        )
        raise UnrecognizedFrame(msg)
    if not found and unread_frames:
        named = '; '.join(
            f'{tuple(round(v, 1) for v in frame)} with {nx} x and {ny} y tick labels' for frame, nx, ny in unread_frames
        )
        msg = f'page {page.number}: no frame reads as a plot, each needs two tick labels per axis: {named}'
        raise UnreadFrame(msg)
    return found


def _dashes(paths: Sequence[VectorPath]) -> list[_Edge]:
    """Each drawn minus candidate: a path that is WHOLLY one short flat stroke (or a thin filled rule).

    A short flat segment of a longer path is part of a curve (one continuing past the frame under the PDF clip, or
    crossing the label row), never a minus: DMEGC and SIFERRIT curves read so negated a tick label.
    """
    small = _Shapes(
        [p.provenance.bbox for p in paths if p.provenance.bbox is not None and _glyph_sized(p.provenance.bbox)]
    )
    out = []
    for path in paths:
        box = path.provenance.bbox
        if box is not None and box.width <= DASH_MAX + AXIS_TOL and box.height <= RULE_MAX and not small.stacked(box):
            out.extend(e for e in _edges([path]) if not e.vertical and e.length <= DASH_MAX)
    return out


def _glyph_sized(box: BBox) -> bool:
    """Whether a shape is no bigger than a glyph of a label (a dash's length either way)."""
    return box.width <= 2 * DASH_MAX and box.height <= 2 * DASH_MAX


class _Shapes:
    """Glyph-sized shapes' boxes on a grid of ``2 * DASH_MAX`` cells: a box finds its neighbors in a few cells.

    A page may draw 100k of them (a meshed figure), in rows of thousands: no box visits every other.
    """

    #: The grid's cell side (points): a shape is at most this across, so it covers at most two cells each way.
    _CELL: Final = 2 * DASH_MAX

    def __init__(self, boxes: Sequence[BBox]) -> None:
        self._cells: dict[tuple[int, int], list[BBox]] = defaultdict(list)
        for box in boxes:
            for cell in self._covering(box.x0, box.y0, box.x1, box.y1):
                self._cells[cell].append(box)

    def _covering(self, x0: float, y0: float, x1: float, y1: float) -> Iterator[tuple[int, int]]:
        cell = self._CELL
        for i in range(math.floor(x0 / cell), math.floor(x1 / cell) + 1):
            for j in range(math.floor(y0 / cell), math.floor(y1 / cell) + 1):
                yield i, j

    def stacked(self, box: BBox) -> bool:
        """Whether another shape lies across ``box`` within ``STACK_GAP`` above or below it."""
        near = self._covering(box.x0, box.y0 - STACK_GAP, box.x1, box.y1 + STACK_GAP)
        return any(
            other != box
            and other.x0 < box.x1
            and other.x1 > box.x0
            and (0 <= box.y0 - other.y1 <= STACK_GAP or 0 <= other.y0 - box.y1 <= STACK_GAP)
            for cell in near
            for other in self._cells.get(cell, ())
        )


def _refuse_unread(
    page: Page,
    frame: BBox,
    read: tuple[Sequence[_Label], Sequence[_Label]],
    unread: Sequence[BBox],
    marks: tuple[Sequence[float], Sequence[float]],
) -> None:
    """Raise :class:`UnrecognizedFrame` when an unreadable outlined word sits among the frame's tick labels.

    Among means on the x-label row (between the read labels' top and bottom) or the y-label column (within the read
    labels' extent across); an unreadable word farther out is a title, not a label. On an axis that draws tick marks
    or grid lines (``marks``: x positions, y positions), a word with no mark within half its own extent along the
    axis labels no tick (a unit or title set between labels, as SIFERRIT Pv figures set ``kW/m3``), and is not one.
    """
    xs, ys = read
    for box in unread:
        probe = [_Label(math.nan, box, round(box.height, 1))]
        cy, cx = (box.y0 + box.y1) / 2, (box.x0 + box.x1) / 2
        on_row = xs and _x_labels(frame, probe) and min(b.box.y0 for b in xs) <= cy <= max(b.box.y1 for b in xs)
        on_col = ys and _y_labels(frame, probe) and min(b.box.x0 for b in ys) <= cx <= max(b.box.x1 for b in ys)
        on_row = on_row and _on_a_mark(cx, box.width / 2, marks[0])
        on_col = on_col and _on_a_mark(cy, box.height / 2, marks[1])
        if on_row or on_col:
            msg = (
                f'page {page.number}: the plot framed at {tuple(round(v, 1) for v in frame)} has a tick label at '
                f'{tuple(round(v, 1) for v in box)} whose glyphs read as no number or word: an unrecognized frame'
            )
            raise UnrecognizedFrame(msg)


def _stroked_label(
    frame: BBox, lettering: Sequence[BBox], marks: tuple[Sequence[float], Sequence[float]]
) -> BBox | None:
    """The first stroked word where a tick label of ``frame`` belongs, or None.

    SIFERRIT sheets (N72 p3) draw labels as stroke-font polylines; where they do not read (a character the stroke font
    lacks) a page of such frames would return no plot, exactly like a page that holds none (:func:`plots` refuses it).
    Where a label belongs is the x-label band below the frame or the y-label band left of it, on a tick mark or grid
    line as :func:`_refuse_unread` asks.
    """
    for box in lettering:
        probe = [_Label(math.nan, box, round(box.height, 1))]
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        on_row = bool(_x_labels(frame, probe)) and _on_a_mark(cx, box.width / 2, marks[0])
        on_col = bool(_y_labels(frame, probe)) and _on_a_mark(cy, box.height / 2, marks[1])
        if on_row or on_col:
            return box
    return None


def _stroked_words(paths: Sequence[VectorPath]) -> list[BBox]:
    """Each row of glyph-sized stroked polylines (three or more points: not a tick, rule or arrow shaft), as a box."""
    boxes = [
        p.provenance.bbox
        for p in paths
        if p.stroke is not None
        and p.provenance.bbox is not None
        and len(p.points) > 2  # noqa: PLR2004 -- a bend: a straight stroke is a tick, rule or shaft
        and max(p.provenance.bbox.width, p.provenance.bbox.height) <= GLYPH_MAX
    ]
    return [union([boxes[i] for i in word]) for word in _words(boxes)]


def _multiplied(labels: list[_Label], marks: Sequence[float], *, vertical: bool) -> list[_Label]:
    """An axis whose one power-of-ten label is stacked against its end label, scaled by it (a unit multiplier).

    TDK SIFERRIT sheets set ``250`` over ``10^-6`` at the top of a temperature-factor axis (N48 p3): the axis is in
    units of 10^-6, and the power labels no tick. It is a multiplier when it is the axis' only power label, its box
    abuts or overlaps (within ``EDGE_TOL`` along the axis) the first or last other label's, and it sits on no
    tick mark or grid line; every other label is then multiplied by it. Any other power label is a tick.
    """
    powers = [lb for lb in labels if lb.power]
    if len(powers) != 1 or len(labels) < 3:  # noqa: PLR2004 -- a multiplier and two ticks
        return labels

    def at(lb: _Label) -> float:
        return lb.at(vertical=vertical)

    (power,) = powers
    rest = sorted((lb for lb in labels if lb is not power), key=at)
    neighbor = min((rest[0], rest[-1]), key=lambda lb: abs(at(lb) - at(power)))
    lo, hi = ('y0', 'y1') if vertical else ('x0', 'x1')
    gap = max(getattr(power.box, lo) - getattr(neighbor.box, hi), getattr(neighbor.box, lo) - getattr(power.box, hi))
    half = (getattr(power.box, hi) - getattr(power.box, lo)) / 2
    if gap > EDGE_TOL or (marks and _on_a_mark(at(power), half, marks)):
        return labels
    return [_Label(lb.value * power.value, lb.box, lb.size) for lb in rest]


def _on_a_mark(center: float, half: float, marks: Sequence[float]) -> bool:
    """Whether a word centered at ``center`` could label a mark: one lies within ``half`` of it, or none is drawn."""
    return not marks or any(abs(m - center) <= half + SNAP_TOL for m in marks)


def _edges(paths: Sequence[VectorPath]) -> list[_Edge]:
    """Every axis-parallel stroked segment and every thin filled rectangle (a rule drawn as a fill).

    A filled rectangle large enough to frame a plot gives its four sides: a design tool draws a frame as a fill.
    """
    out = []
    for path in paths:
        if path.stroke is None:
            box = path.provenance.bbox
            if not path.closed or box is None:
                continue
            if min(box.width, box.height) <= RULE_MAX:
                if box.height > box.width:
                    out.append(_Edge(vertical=True, at=(box.x0 + box.x1) / 2, lo=box.y0, hi=box.y1))
                else:
                    out.append(_Edge(vertical=False, at=(box.y0 + box.y1) / 2, lo=box.x0, hi=box.x1))
            elif min(box.width, box.height) >= MIN_FRAME and _is_rectangle(path, box):
                out.extend(
                    (
                        _Edge(vertical=False, at=box.y0, lo=box.x0, hi=box.x1),
                        _Edge(vertical=False, at=box.y1, lo=box.x0, hi=box.x1),
                        _Edge(vertical=True, at=box.x0, lo=box.y0, hi=box.y1),
                        _Edge(vertical=True, at=box.x1, lo=box.y0, hi=box.y1),
                    )
                )
            continue
        points = _closed(path) if path.closed else path.points
        for (x0, y0), (x1, y1) in itertools.pairwise(points.tolist()):
            if abs(x0 - x1) <= AXIS_TOL < abs(y0 - y1):
                out.append(_Edge(vertical=True, at=(x0 + x1) / 2, lo=min(y0, y1), hi=max(y0, y1)))
            elif abs(y0 - y1) <= AXIS_TOL < abs(x0 - x1):
                out.append(_Edge(vertical=False, at=(y0 + y1) / 2, lo=min(x0, x1), hi=max(x0, x1)))
    return out


def _without_ticks(paths: Sequence[VectorPath], frames: Sequence[BBox]) -> list[VectorPath]:
    """``paths`` less the thin filled rectangles that are tick marks of one of ``frames`` (:func:`_tick`).

    A tick drawn as a fill is the axis's ink, never a glyph of its labels: the stub of the x axis's first tick stands
    beside the y axis's corner label, as close as its glyphs stand to each other, and read with it as a mark it made
    the ``0`` a word of no number that refused the frame.
    """
    return [
        path
        for path in paths
        if not (
            path.stroke is None
            and path.provenance.bbox is not None
            and min(path.provenance.bbox.width, path.provenance.bbox.height) <= RULE_MAX
            and any(_tick(frame, edge) for edge in _edges([path]) for frame in frames)
        )
    ]


def _is_rectangle(path: VectorPath, box: BBox) -> bool:
    """The path visits the four corners of its own box and nothing else: an axis-aligned rectangle."""
    corners = [(x, y) for x in (box.x0, box.x1) for y in (box.y0, box.y1)]
    hit = {
        next((k for k, (cx, cy) in enumerate(corners) if abs(x - cx) <= AXIS_TOL and abs(y - cy) <= AXIS_TOL), None)
        for x, y in path.points
    }
    return hit == set(range(len(corners)))


def _frames(edges: Sequence[_Edge]) -> list[BBox]:
    """A frame per bottom-left corner: a long horizontal edge (the x axis) meeting a long vertical one (the y axis).

    The two meet at their ends, or cross with each running on past the other by at most ``TICK_MAX`` (axes drawn
    through the origin, an outward tick at the corner); of the edges crossing there the lowest horizontal and the
    leftmost vertical are the axes, so a grid line just inside is not one. A corner where several edges meet (a grid
    line cut short along the x axis) keeps its largest frame.
    """
    long = [e for e in edges if e.length >= MIN_FRAME]
    flat = [e for e in long if not e.vertical]
    upright = [e for e in long if e.vertical]

    def meet(h: _Edge, v: _Edge) -> bool:
        return -EDGE_TOL <= v.at - h.lo <= TICK_MAX and -EDGE_TOL <= v.hi - h.at <= TICK_MAX and v.lo < h.at

    corners: dict[tuple[float, float], BBox] = {}
    for h in flat:
        for v in upright:
            if not meet(h, v):
                continue
            if any(meet(o, v) and o.at > h.at + EDGE_TOL for o in flat):  # a lower x axis crosses this y axis
                continue
            if any(meet(h, o) and o.at < v.at - EDGE_TOL for o in upright):  # a y axis further left crosses it
                continue
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
                out.append(_Label(10.0**power, union([box, spans[j].provenance.bbox]), span.size, power=True, base=box))
                continue
        value = parse_number(span.text)
        if value is None:
            continue
        minus = _minus_before(box, spans, dashes) if value > 0 else None
        if minus is not None:
            value, box = -value, union([box, minus])
        out.append(_Label(value, box, span.size))
    return out


def _outline_labels(outlined: Sequence[tuple[float | None, str | None, BBox]], dashes: Sequence[_Edge]) -> list[_Label]:
    """Numbers set as filled glyph OUTLINES (a sheet exported with its text converted to paths).

    A glyph is a small filled shape plus the subpaths whose centers it encloses (a ``0``'s hole); a word is a row of
    glyphs no further apart than ``WORD_GAP`` of its height, and a raised smaller glyph starts a word of its own (an
    exponent). A word reading ``10`` followed by a raised integer word is that power of ten, as with text labels; a
    drawn dash before a label (or between ``10`` and its exponent) is its minus. A word the reader cannot read is no
    label.
    """
    words = [(value, box) for value, _, box in outlined if value is not None]
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
                out.append(_Label(10.0**exponent, union([box, words[j][1]]), round(box.height, 1), power=True))
                continue
        minus = _minus_before(box, (), dashes) if value > 0 else None
        signed = (value, box) if minus is None else (-value, union([box, minus]))
        out.append(_Label(signed[0], signed[1], round(box.height, 1)))
    return out


def outline_text(page: Page, *, include_hidden: bool = False, include_inkless: bool = False) -> list[TextSpan]:
    """Each word ``page`` sets as filled glyph OUTLINES that reads as text (a unit, a grade), as a span.

    A word that spells a number is a tick label (read by :func:`plots`), not text -- unless a ``min.`` or ``max.``
    word follows it, when the two are one span (``12.5 min.``, read by :func:`document_lab.numbers.bound`); a word
    the reader refuses is absent. The span's size is the word's ink height and its font ``outline``.
    ``include_hidden``, ``include_inkless``: as for :func:`plots`.
    """
    page = page.shown(include_hidden=include_hidden, include_inkless=include_inkless)
    return _text_spans(page, _outline_words(page.paths))


def _text_spans(page: Page, outlined: Sequence[tuple[float | None, str | None, BBox]]) -> list[TextSpan]:
    """The outlined words that read as text; a number set just left of a ``min.``/``max.`` word joins it (a bound)."""
    out = []
    for value, text, box in outlined:
        if value is not None or text is None:
            continue
        if text.rstrip('.').lower() in {'min', 'max'}:
            number = next(
                (
                    (v, b)
                    for v, _, b in outlined
                    if v is not None
                    and b.y0 <= (box.y0 + box.y1) / 2 <= b.y1
                    and -EDGE_TOL <= box.x0 - b.x1 <= BOUND_GAP * box.height
                ),
                None,
            )
            if number is not None:
                text, box = f'{number[0]:g} {text}', union([number[1], box])  # noqa: PLW2901 -- the joined word
        out.append(TextSpan(text, round(box.height, 1), 'outline', page.paths[0].provenance.at(bbox=box)))
    return out


def _outline_words(paths: Sequence[VectorPath]) -> list[tuple[float | None, str | None, BBox]]:
    """Each row of outlined (filled) glyphs with what it reads as against Helvetica (:func:`_read_words`)."""
    shapes = [
        p
        for p in paths
        if p.stroke is None
        and p.provenance.bbox is not None
        and len(p.points) > 2  # noqa: PLR2004 -- a fill needs an area
        and max(p.provenance.bbox.width, p.provenance.bbox.height) <= GLYPH_MAX
        and min(p.provenance.bbox.width, p.provenance.bbox.height) >= SLIVER
    ]
    shapes.sort(key=lambda p: -p.provenance.bbox.width * p.provenance.bbox.height)
    glyphs: list[tuple[BBox, list[np.ndarray]]] = []
    # Each glyph's index on a grid of GLYPH_MAX cells (a glyph covers at most two each way): a shape's host is the
    # first glyph holding its center, found among those in the center's cell, not among every glyph of the page.
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for shape in shapes:
        box = shape.provenance.bbox
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        near = cells.get((math.floor(cx / GLYPH_MAX), math.floor(cy / GLYPH_MAX)), ())
        host = next((glyphs[k] for k in near if glyphs[k][0].contains(cx, cy) and glyphs[k][0] != box), None)
        if host is None:
            for i in range(math.floor(box.x0 / GLYPH_MAX), math.floor(box.x1 / GLYPH_MAX) + 1):
                for j in range(math.floor(box.y0 / GLYPH_MAX), math.floor(box.y1 / GLYPH_MAX) + 1):
                    cells[i, j].append(len(glyphs))
            glyphs.append((box, [shape.points]))
        else:
            host[1].append(shape.points)
    glyphs = _dotted([g for g in glyphs if not (len(g[1]) == 1 and _arrowhead(g[1][0], g[0]))])
    return _read_words([Glyph(rasterize(polys, tuple(box)), tuple(box)) for box, polys in glyphs], SANS)


def _stroke_words(paths: Sequence[VectorPath]) -> tuple[list[tuple[float | None, str | None, BBox]], set[int]]:
    """Each row of STROKE-FONT glyphs with what it reads as (as :func:`_outline_words`), and its lettering's path ids.

    The ids are of the paths drawing the words that read: lettering inside a frame (``300 mT``) is never a curve.

    TDK SIFERRIT sheets (N27, N30, N41, N72) draw each glyph as its outline stroked in hairline fragments. A glyph is
    the fragments that bend (or close), each within ``STROKE_TOUCH`` of another's box, with every straight fragment
    touching one of them (a crossbar joins the two bars of an ``H``); a straight fragment touching none is a tick or
    a rule. A glyph larger than ``GLYPH_MAX`` is a curve, and a solid wedge read as no character an arrowhead
    (:func:`_solid_triangle`). Each glyph is read against :data:`document_lab.glyphs.STROKE`.
    """
    glyphs, groups = [], []
    for group in _stroke_groups(paths):
        box = union([p.provenance.bbox for p in group])
        if max(box.width, box.height) <= GLYPH_MAX:
            drawn = [_closed(p) if p.closed else p.points for p in group]
            glyph = Glyph(stroke(drawn, max(p.width or 0.0 for p in group)), tuple(box))
            if not (_solid_triangle(glyph.ink) and not _reads(glyph)):
                glyphs.append(glyph)
                groups.append(group)
    words = _read_words(glyphs, STROKE)
    read = [box for value, text, box in words if value is not None or text is not None]
    lettering = {
        id(path)
        for glyph, group in zip(glyphs, groups, strict=True)
        if any(box.contains(glyph.box[0], glyph.box[1]) and box.contains(glyph.box[2], glyph.box[3]) for box in read)
        for path in group
    }
    return words, lettering


def _stroke_groups(paths: Sequence[VectorPath]) -> list[list[VectorPath]]:
    """The glyph-sized stroked fragments grouped as :func:`_stroke_words` describes, each group one glyph."""
    lines = [
        p
        for p in paths
        if p.stroke is not None
        and p.provenance.bbox is not None
        and max(p.provenance.bbox.width, p.provenance.bbox.height) <= GLYPH_MAX
    ]
    bent = sorted(
        (p for p in lines if p.closed or len(p.points) > 2),  # noqa: PLR2004 -- a bend
        key=lambda p: p.provenance.bbox.x0,
    )
    boxes = [p.provenance.bbox for p in bent]
    starts = [b.x0 for b in boxes]
    parent = list(range(len(bent)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def touching(box: BBox) -> list[int]:
        reach = STROKE_TOUCH
        lo = bisect.bisect_left(starts, box.x0 - reach - GLYPH_MAX)
        hi = bisect.bisect_right(starts, box.x1 + reach)
        return [
            i
            for i in range(lo, hi)
            if box.x0 - reach <= boxes[i].x1 and box.y0 - reach <= boxes[i].y1 and boxes[i].y0 <= box.y1 + reach
        ]

    members: dict[int, list[VectorPath]] = defaultdict(list)
    for i, box in enumerate(boxes):
        for j in touching(box):
            parent[root(i)] = root(j)
    for straight in (p for p in lines if not (p.closed or len(p.points) > 2)):  # noqa: PLR2004 -- a bend
        touched = touching(straight.provenance.bbox)
        for i in touched[1:]:
            parent[root(i)] = root(touched[0])
        if touched:
            members[touched[0]].append(straight)
    groups: dict[int, list[VectorPath]] = defaultdict(list)
    for i, path in enumerate(bent):
        groups[root(i)].extend([path, *members[i]])
    return list(groups.values())


def _solid_triangle(ink: np.ndarray) -> bool:
    """Whether ``ink`` is a solid wedge: an arrowhead hatched with strokes (SIFERRIT N22 p3 sets one beside ``300``).

    Solid along rows (or columns): every row's ink is one run, and the runs widen steadily from at most
    ``WEDGE_TIP`` of the widest at one end to at least ``WEDGE_BASE`` of it at the other. A ``1``, ``L`` or ``T`` is
    as solid, so only a glyph that also reads as no character is dropped as one (:func:`_stroke_words`).
    """
    for grid in (ink, ink.T):
        runs = [np.flatnonzero(row) for row in grid if row.any()]
        if len(runs) < 4 or any(run[-1] - run[0] + 1 != len(run) for run in runs):  # noqa: PLR2004 -- a wedge
            continue
        widths = np.array([len(run) for run in runs])
        steps = np.diff(widths)
        widest = widths.max()
        steady = bool((steps >= -1).all() or (steps <= 1).all())
        tip, base = sorted((widths[0], widths[-1]))
        if steady and tip <= WEDGE_TIP * widest and base >= WEDGE_BASE * widest:
            return True
    return False


def _reads(glyph: Glyph) -> bool:
    """Whether ``glyph`` alone reads as a stroke-font character, by shape."""
    try:
        read_word([glyph], STROKE)
    except UnreadableGlyph:
        return False
    return True


def _read_words(glyphs: Sequence[Glyph], font: Font) -> list[tuple[float | None, str | None, BBox]]:
    """Each row of ``glyphs`` with the number it reads as in ``font``, else the text (each ``None`` if not).

    A column of glyphs set turned a quarter (:func:`_columns`: a y-axis title reading upward) is read turned upright,
    as text; one that does not read is a title all the same, never a tick label, and is left out.
    A row that reads as neither but holds a glyph that reads clearly as a letter is a word (a unit such as ``kW/m3``
    whose ``/`` the reader lacks), never a tick label: it is left out, so it can neither refuse a frame nor be text.
    """
    out: list[tuple[float | None, str | None, BBox]] = []
    boxes = [BBox(*g.box) for g in glyphs]
    upright = set()
    for column in _columns(glyphs, font):
        upright.update(column)
        turned = [_turned(glyphs[i]) for i in column]
        with contextlib.suppress(UnreadableGlyph):
            out.append((None, read_word(turned, font), union([boxes[i] for i in column])))
    rest = [i for i in range(len(glyphs)) if i not in upright]
    for word in ([rest[k] for k in word] for word in _words([boxes[i] for i in rest])):
        shapes = [glyphs[i] for i in word]
        value: float | None = None
        text: str | None = None
        try:
            value = read_number(shapes, font)
        except UnreadableGlyph:
            with contextlib.suppress(UnreadableGlyph):
                text = read_word(shapes, font)
        if value is None and text is None and has_letter(shapes, font):
            continue
        out.append((value, text, union([BBox(*g.box) for g in shapes])))
    return out


def _columns(glyphs: Sequence[Glyph], font: Font) -> list[list[int]]:
    """Glyphs set turned a quarter, as columns of indices bottom to top: a y-axis title reading upward.

    A column is :data:`TURNED_GLYPHS` or more glyphs stacked with their middles across inside the column's extent
    across, each at most ``WORD_GAP`` of the narrowest one's extent across above the last -- as a row's glyphs sit
    along it -- and not every one reading upright as a digit (tick labels down an axis are digits; a turned word's
    letters, or its letters run together at a coarse pixel, are not).
    """
    boxes = [BBox(*g.box) for g in glyphs]
    columns: list[list[int]] = []
    for i in sorted(range(len(boxes)), key=lambda k: -boxes[k].y1):
        box = boxes[i]
        cx = (box.x0 + box.x1) / 2
        for column in columns:
            cb = union([boxes[k] for k in column])
            narrow = min(box.width, *(boxes[k].width for k in column))
            if cb.x0 <= cx <= cb.x1 and -EDGE_TOL <= cb.y0 - box.y1 <= WORD_GAP * narrow:
                column.append(i)
                break
        else:
            columns.append([i])
    return [c for c in columns if len(c) >= TURNED_GLYPHS and not all(_digit(glyphs[k], font) for k in c)]


def _digit(glyph: Glyph, font: Font) -> bool:
    try:
        read_number([glyph], font)
    except UnreadableGlyph:
        return False
    return True


def _turned(glyph: Glyph) -> Glyph:
    """``glyph`` of an upward-reading column turned upright: its ink a quarter clockwise, its box so too."""
    x0, y0, x1, y1 = glyph.box
    return Glyph(np.rot90(glyph.ink, k=-1), (-y1, x0, -y0, x1))


def _arrowhead(points: np.ndarray, box: BBox) -> bool:
    """Whether a glyph's one fill is a triangle: an arrowhead (SIFERRIT N48 p4 heads its axis-title arrow so).

    Only a fill with no hole: a ``4``'s counter is a triangle too. Corners closer than ``CORNER_MERGE`` of the
    fill's larger side are one (a head's apex is cut 0.3 points wide).
    """
    reach = CORNER_MERGE * max(box.width, box.height)
    corners: list[tuple[float, float]] = []
    for x, y in points.tolist():
        if all(math.hypot(x - cx, y - cy) > reach for cx, cy in corners):
            corners.append((x, y))
    return len(corners) == 3  # noqa: PLR2004 -- a triangle


def _dotted(
    glyphs: list[tuple[BBox, list[np.ndarray]]],
) -> list[tuple[BBox, list[np.ndarray]]]:
    """Each small mark just above a taller glyph whose extent across holds its center, joined to it (an ``i``'s dot).

    Just above is a gap of at most ``DOT_GAP`` of the glyph's height; the joined glyph is read as one shape.
    """
    marks = {
        i
        for i, (box, _) in enumerate(glyphs)
        if any(
            box.width <= other.width + EDGE_TOL
            and box.height < MARK_HEIGHT_PT * other.height
            and other.x0 <= (box.x0 + box.x1) / 2 <= other.x1
            and 0 <= other.y0 - box.y1 <= DOT_GAP * other.height
            for j, (other, _) in enumerate(glyphs)
            if j != i
        )
    }
    out = [(box, list(polys)) for i, (box, polys) in enumerate(glyphs) if i not in marks]
    for i in marks:
        box, polys = glyphs[i]
        cx = (box.x0 + box.x1) / 2
        k = min(
            (k for k, (other, _) in enumerate(out) if other.x0 <= cx <= other.x1 and other.y0 >= box.y1),
            key=lambda k: out[k][0].y0 - box.y1,
            default=None,
        )
        if k is None:  # its host was itself a mark (a colon's upper dot over the lower): keep it
            out.append((box, list(polys)))
            continue
        out[k] = (union([out[k][0], box]), [*out[k][1], *polys])
    return out


def _words(boxes: Sequence[BBox]) -> list[list[int]]:
    """Glyph boxes grouped into words, as indices left to right: one row, gaps under ``WORD_GAP`` of the height.

    A raised smaller glyph (an exponent) starts a word of its own: raised above the word's median glyph bottom. A
    raised MARK (taller than wide, at most ``MARK_HEIGHT_PT`` of the word's height: a prime in ``mu'``) does not.
    """
    words: list[list[int]] = []
    for i in sorted(range(len(boxes)), key=lambda k: boxes[k].x0):
        box = boxes[i]
        cy = (box.y0 + box.y1) / 2
        for word in words:
            wb = union([boxes[k] for k in word])
            same_row = wb.y0 <= cy <= wb.y1 or box.y0 <= (wb.y0 + wb.y1) / 2 <= box.y1
            base = float(np.median([boxes[k].y1 for k in word]))  # a descender (``g``) is not the baseline
            raised = box.height < RAISED * wb.height and box.y1 < base - (1 - RAISED) * wb.height
            prime = box.height > box.width and box.height <= MARK_HEIGHT_PT * wb.height and wb.y0 < box.y1 < wb.y1
            if (prime or (same_row and not raised)) and -EDGE_TOL <= box.x0 - wb.x1 <= WORD_GAP * max(
                wb.height, box.height
            ):
                word.append(i)
                break
        else:
            words.append([i])
    return words


def _outline_exponent(
    base: BBox, words: Sequence[tuple[float | None, BBox]], dashes: Sequence[_Edge]
) -> tuple[int, int] | None:
    """The raised smaller integer word just right of an outlined ``10``, negative when a drawn dash is between.

    Outline boxes are INK boxes, without the side bearing a text box carries: ``10`` and an exponent set a point
    apart are 2-3 points apart in ink (TDK SIFERRIT N97 p5), so the gap allowed is a word gap of the base's height.
    """
    for j, (power, box) in enumerate(words):
        if power is None or power != int(power) or box.height >= RAISED * base.height:
            continue
        if not (box.y1 < base.y1 - (1 - RAISED) * base.height and box.y1 > base.y0):
            continue
        gap = box.x0 - base.x1
        if -EDGE_TOL <= gap <= max(EDGE_TOL, WORD_GAP * base.height):
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
        minus = _minus_before(ob, spans, ())
        if 0 < gap <= DASH_MAX + EDGE_TOL and minus is not None and minus.x0 >= bb.x1 - EDGE_TOL and power > 0:
            return j, -int(power)
        dashed = any(bb.x1 - EDGE_TOL <= d.lo and d.hi <= ob.x0 + EDGE_TOL and bb.y0 <= d.at <= ob.y1 for d in dashes)
        if 0 < gap <= DASH_MAX + EDGE_TOL and dashed and power > 0:
            return j, -int(power)
        if -EDGE_TOL <= gap <= max(EDGE_TOL, WORD_GAP * base.size):  # TDK sets the exponent a space apart (N48 p4)
            return j, int(power)
    return None


def _decade_mantissas(labels: list[_Label], *, vertical: bool) -> list[_Label]:
    """A bare digit 2..9 among power-of-ten labels read as that multiple of the decade its POSITION puts it in.

    TDK SIFERRIT sheets label a log axis ``10^0 5 10^1 5 10^2``: the ``5`` is 5 x 10^n, and a decade's own label may
    be missing (N97 p4 prints no 10^3). The power labels fix log10(value) as a line in position; a digit ``m`` takes
    the ``m x 10^k`` nearest that line, and keeps its value when none is within ``MANTISSA_TOL`` decade. Applied only
    when the labels as read are not monotone along the axis and at least two are SET as powers (``10`` plus an
    exponent); every label on an axis without power labels keeps its value, and such an axis is refused as before.
    """

    def at(lb: _Label) -> float:
        return lb.at(vertical=vertical)

    ordered = sorted(labels, key=at)
    steps = np.diff([lb.value for lb in ordered])
    powers = [lb for lb in ordered if lb.power]
    if len(powers) < 2 or np.all(steps > 0) or np.all(steps < 0):  # noqa: PLR2004 -- a line needs two
        return labels
    slope, intercept = np.polyfit([at(lb) for lb in powers], [math.log10(lb.value) for lb in powers], 1)
    out = []
    for lb in ordered:
        if not lb.power and lb.value == int(lb.value) and 2 <= lb.value <= 9:  # noqa: PLR2004 -- a mantissa
            line = slope * at(lb) + intercept
            k = round(line - math.log10(lb.value))
            if abs(k + math.log10(lb.value) - line) <= MANTISSA_TOL:
                out.append(_Label(lb.value * 10.0**k, lb.box, lb.size))
                continue
        out.append(lb)
    return out


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
    """The largest set of labels sharing one row (x: top, bottom or center) or column (y: right, left or center).

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
    """Positions of tick marks (short, on a frame edge) and grid lines (spanning the frame) across one axis.

    A grid line spans THIS frame: a line as long as the frame but beside it is another plot's (N97 p3 stacks two).
    """
    lo_edge, hi_edge, span = (frame.y0, frame.y1, frame.height) if vertical else (frame.x0, frame.x1, frame.width)
    first, last = (frame.x0, frame.x1) if vertical else (frame.y0, frame.y1)
    marks = []
    for e in edges:
        if e.vertical != vertical or not first - EDGE_TOL <= e.at <= last + EDGE_TOL:
            continue
        if _tick(frame, e) or min(e.hi, hi_edge) - max(e.lo, lo_edge) >= GRID_SPAN * span:
            marks.append(e.at)
    return marks


def _tick(frame: BBox, edge: _Edge) -> bool:
    """Whether ``edge`` is a tick mark of ``frame``: at most ``TICK_MAX`` long, ending on a side it runs across.

    It runs across the axis it marks, within the frame's extent along that axis.
    """
    lo_edge, hi_edge = (frame.y0, frame.y1) if edge.vertical else (frame.x0, frame.x1)
    first, last = (frame.x0, frame.x1) if edge.vertical else (frame.y0, frame.y1)
    return (
        first - EDGE_TOL <= edge.at <= last + EDGE_TOL
        and edge.length <= TICK_MAX
        and any(abs(b - edge.lo) <= EDGE_TOL or abs(b - edge.hi) <= EDGE_TOL for b in (lo_edge, hi_edge))
    )


def _snap(
    ticks: list[tuple[float, float]], marks: list[float], half_extent: float = math.inf
) -> tuple[tuple[float, float], ...]:
    """Labels moved onto their marks by ONE shared offset, the one that lands the most labels on a mark.

    A label sits a constant distance from its tick (same font, same alignment), so the offset is a property of
    the axis, not of each label; snapping each to its nearest mark instead lets a dense minor grid capture it.
    The offset is at most ``half_extent``, half the labels' own extent along the axis: a label overlaps the tick
    it names, and a larger shift lands it on a minor line instead (N49 p5). A label with no mark at its shifted
    position keeps its own position.
    """
    positions = sorted(p for p, _ in ticks)
    spacing = min((b - a for a, b in itertools.pairwise(positions) if b > a), default=LABEL_REACH)
    reach = min(0.4 * spacing, half_extent)
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
    """Lines through runs of tick intervals of one data-per-point spacing; neighboring pieces share a break tick."""
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
    """Stroked open paths inside the frame (clipped to it) that are no tick mark, grid line or frame edge.

    A grid line is whole (:func:`_is_rule`) or cut into pieces by the labels set over it (:func:`_cut_grid`).
    """
    pieces = [
        piece
        for path in paths
        if path.stroke is not None and not path.closed and len(path.points) >= 2  # noqa: PLR2004
        for piece in _clipped(path, frame)
    ]
    straight = [(piece, rules[0]) for piece in pieces if len(piece.points) == 2 and (rules := _edges([piece]))]  # noqa: PLR2004
    rules = {id(piece) for piece, edge in straight if _is_rule(edge, frame)} | _cut_grid(frame, straight)
    return [piece for piece in pieces if id(piece) not in rules]


def _cut_grid(frame: BBox, straight: Sequence[tuple[VectorPath, _Edge]]) -> set[int]:
    """The ids of the straight pieces that draw a grid line cut by the labels set over it.

    TDK ACT1210 p5 stops every grid line at each label inside the frame: a grid line is then two or more pieces in one
    pen on one line (within ``AXIS_TOL``) that together reach both frame edges (within ``EDGE_TOL``).
    """
    out = set()
    for path, edge in straight:
        line = [
            e
            for p, e in straight
            if e.vertical == edge.vertical
            and abs(e.at - edge.at) <= AXIS_TOL
            and (p.stroke, p.width) == (path.stroke, path.width)
        ]
        first, last = (frame.y0, frame.y1) if edge.vertical else (frame.x0, frame.x1)
        if len(line) > 1 and min(e.lo for e in line) <= first + EDGE_TOL and max(e.hi for e in line) >= last - EDGE_TOL:
            out.add(id(path))
    return out


def _clipped(path: VectorPath, frame: BBox) -> list[VectorPath]:
    """The runs of ``path`` inside the frame, each cut where it crosses the frame's edge.

    A sheet draws a curve past its frame and clips it there (DMEGC): the part inside is the curve. A path wholly
    inside (to ``EDGE_TOL``) is returned as it is.
    """
    (x0, y0), (x1, y1) = path.points.min(axis=0), path.points.max(axis=0)
    if frame.contains(x0, y0, tol=EDGE_TOL) and frame.contains(x1, y1, tol=EDGE_TOL):
        return [path]
    box = frame
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for a, b in itertools.pairwise(map(tuple, path.points.tolist())):
        cut = _clip_segment(a, b, box)
        if cut is None:
            if current:
                runs.append(current)
            current = []
            continue
        start, end = cut
        if current and current[-1] == start:
            current.append(end)
        else:
            if current:
                runs.append(current)
            current = [start, end]
        if end != b:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [
        VectorPath(
            _polyline(run),
            path.stroke,
            path.width,
            path.dashes,
            closed=False,
            provenance=path.provenance.at(bbox=_points_box(run)),
        )
        for run in runs
        if len(run) >= 2 and run[0] != run[-1]  # noqa: PLR2004 -- a curve needs two points
    ]


def _clip_segment(
    a: tuple[float, float], b: tuple[float, float], box: BBox
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The part of segment ``a``-``b`` inside ``box`` (Liang-Barsky), its ends kept exact where they are inside."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, x0 - box.x0), (dx, box.x1 - x0), (-dy, y0 - box.y0), (dy, box.y1 - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
    if lo > hi:
        return None
    start = a if lo == 0 else (x0 + lo * dx, y0 + lo * dy)
    end = b if hi == 1 else (x0 + hi * dx, y0 + hi * dy)
    return start, end


def _polyline(points: Sequence[tuple[float, float]] | np.ndarray) -> np.ndarray:
    """``points`` as a :class:`VectorPath` holds them: an ``(n, 2)`` float64 array, read-only."""
    out = np.array(points, dtype=np.float64).reshape(-1, 2)
    out.flags.writeable = False
    return out


def _closed(path: VectorPath) -> np.ndarray:
    """A closed path's points with its start repeated at the end: the closing segment drawn."""
    return np.vstack((path.points, path.points[:1]))


def _points_box(points: Sequence[tuple[float, float]]) -> BBox:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _joined(paths: Sequence[VectorPath]) -> list[VectorPath]:
    """Chains of paths of one style meeting end to end, each joined into one path in chain order.

    Two ends link only when they are the only ends of that style at that point; a third makes the point ambiguous.
    """
    ends: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for i, path in enumerate(paths):
        for end, (x, y) in enumerate(path.points[[0, -1]].tolist()):
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
        members, runs = [], []
        while i not in seen:
            seen.add(i)
            members.append(paths[i])
            run = paths[i].points if enter == 0 else paths[i].points[::-1]
            runs.append(run if not runs else run[1:])
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
            VectorPath(
                _polyline(np.vstack(runs)),
                first.stroke,
                first.width,
                first.dashes,
                closed=False,
                provenance=first.provenance.at(bbox=box),
            )
        )
    return out


def _sample_label(path: VectorPath, spans: Sequence[TextSpan]) -> str | None:
    """The text of a span starting just right of a short horizontal line, at its height: what the sample stands for."""
    box = path.provenance.bbox
    if box is None or box.width > SAMPLE_MAX or box.height > EDGE_TOL:
        return None
    return _text_right_of(box, spans)


def _text_right_of(box: BBox, spans: Sequence[TextSpan]) -> str | None:
    """The text of a span (not a mark glyph) starting within ``SAMPLE_GAP`` right of ``box``, at its height."""
    y = (box.y0 + box.y1) / 2
    for span in spans:
        sb, text = span.provenance.bbox, span.text.strip()
        if (
            sb is not None
            and text not in MARK_GLYPHS
            and box.x1 <= sb.x0 <= box.x1 + SAMPLE_GAP
            and sb.y0 <= y <= sb.y1
        ):
            return text
    return None


def _markers(
    page: Page,
    frame: BBox,
    found: tuple[Sequence[VectorPath], Sequence[LegendSample], Sequence[TextSpan]],
    axes: tuple[Axis, Axis],
) -> tuple[Markers, ...]:
    """The mark glyphs inside ``frame`` as series: per glyph and fill, split by the curves joining them.

    ``found`` is the frame's curves, its legend samples and the page's words (a legend label may be outlined).
    """
    paths, legend, words = found
    groups: dict[tuple[str, tuple[float, ...]], list[TextSpan]] = defaultdict(list)
    labels: dict[tuple[str, tuple[float, ...]], str] = {}
    for span in page.spans:
        glyph, box = span.text.strip(), span.provenance.bbox
        if span.fill is None or box is None or not frame.contains(*span.center):
            continue
        if len(glyph) > 1 and set(glyph.replace(' ', '')) <= MARK_GLYPHS:
            msg = (
                f'page {page.number}: the mark glyphs {glyph!r} at {tuple(round(v, 1) for v in box)}, inside the '
                f'plot framed at {tuple(round(v, 1) for v in frame)}, were joined into one span: where each '
                'stands is not stated'
            )
            raise JoinedMarkers(msg)
        if glyph not in MARK_GLYPHS:
            continue
        key = (glyph, span.fill)
        sample = next((e.label for e in legend if e.provenance.bbox and _on(span, e.provenance.bbox)), None)
        if sample is None and not any(_near(span, path) for path in paths):
            sample = _text_right_of(box, words)
        if sample is None:
            groups[key].append(span)
        else:
            labels.setdefault(key, sample)
    out = []
    for (glyph, fill), spans in groups.items():
        for members in _series(spans, paths):
            members.sort(key=lambda m: m.center[0])
            px, py = np.array([m.center[0] for m in members]), np.array([m.center[1] for m in members])
            boxes = [m.provenance.bbox for m in members if m.provenance.bbox is not None]
            size = float(np.median([b.height for b in boxes]))
            here = members[0].provenance.at(bbox=union(boxes))
            label = labels.get((glyph, fill))
            out.append(Markers(glyph, fill, axes[0].to_data(px), axes[1].to_data(py), size, label, here))
    return tuple(out)


def _reach(span: TextSpan) -> float:
    """How far (points) a glyph's ink may sit from its box's center: half the box's height."""
    box = span.provenance.bbox
    return 0.0 if box is None else box.height / 2


def _on(span: TextSpan, box: BBox) -> bool:
    """Whether the glyph's center lies on ``box`` (a legend sample line), within its reach."""
    return box.contains(*span.center, _reach(span))


def _near(span: TextSpan, path: VectorPath) -> bool:
    """Whether ``path`` has a vertex within the glyph's reach of its center: the curve runs through the marker."""
    cx, cy = span.center
    return any(math.hypot(x - cx, y - cy) <= _reach(span) for x, y in path.points)


def _series(spans: Sequence[TextSpan], paths: Sequence[VectorPath]) -> list[list[TextSpan]]:
    """``spans`` grouped by the curves joining them; the ones no curve reaches together (a scatter)."""
    parent = list(range(len(spans)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reached: set[int] = set()
    for path in paths:
        on = [i for i, span in enumerate(spans) if _near(span, path)]
        reached.update(on)
        for i in on[1:]:
            parent[root(i)] = root(on[0])
    groups: dict[int, list[TextSpan]] = defaultdict(list)
    for i, span in enumerate(spans):
        groups[root(i) if i in reached else -1].append(span)
    return list(groups.values())


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
    color: tuple[float, float, float] | None = None,
) -> Plot:
    """The plot in a RASTER image: its frame, axes calibrated from its tick labels, and its curves traced.

    The frame is found as in :func:`plots`, from the image's long dark (not colored) horizontal and vertical pixel
    runs: of those with two or more aligned label glyph rows below it and left of it (a drop shadow frames none), the
    one whose fewer-labeled axis READS the most labels, then has the most, then the largest (a border box around the
    figure, with a caption's numbers below it, is not the plot's frame). A
    tick label is a row of ink glyphs below the frame (x) or left of it (y), read by :mod:`document_lab.glyphs`; one
    the reader refuses raises :class:`UnreadableTickLabels` -- unless the caller states the axis' labeled values
    (``x_values`` left to right, ``y_values`` bottom to top), which are then paired with the label positions found
    and checked against those that read (:func:`_stated`). Positions snap to tick marks and grid lines as for vector
    plots.

    Curves are the frame's ink less its grid lines (rows or columns inked across ``GRID_INK`` of it). Colored ink is
    one curve per ``HUE_SECTOR`` (so two crossing curves of two colors stay two); dark ink is one curve per
    connected component spanning ``CURVE_SPAN`` of the frame (dark curves that touch are one). A curve is sampled per
    pixel column at each run's center, or every stroke width along a run longer than two (a steep stretch); a curve
    that folds back over x (a loop) keeps both branches' samples, in column order. Each point carries one pixel's
    extent in data units as its uncertainty.

    ``color`` (RGB, 0..1) keeps only the curves drawn in it: those of its ``HUE_SECTOR`` when it is colored, the
    dark ones when it is not; when none is, :class:`NoCurveOfColor` is raised rather than an empty plot returned.
    """
    box = raster.provenance.bbox
    rgb = raster.pixels.astype(float) / 255
    rows, cols = rgb.shape[:2]
    scale = (box.width / cols, box.height / rows)  # points per pixel
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    ink = (1 - rgb.mean(axis=2) > INK_LEVEL) | (chroma > CHROMA_MIN)
    edges = _dark_edges(ink, ink & (chroma <= CHROMA_MIN), box, scale)
    candidates = []
    for frame in _frames(edges):
        labels, refusals = _raster_labels(ink, 1 - rgb.mean(axis=2), frame, box, scale)
        rows = [_aligned(_x_labels(frame, labels), vertical=False), _aligned(_y_labels(frame, labels), vertical=True)]
        if min(len(r) for r in rows) >= 2:  # noqa: PLR2004 -- a line needs two points
            read = min(sum(not math.isnan(lb.value) for lb in r) for r in rows)
            rank = (read, min(len(r) for r in rows), frame.width * frame.height)
            candidates.append((rank, frame, labels, refusals))
    if not candidates:
        msg = (
            f'no plot frame in the raster at {tuple(round(v, 1) for v in box)}: no long lines meet at a corner '
            'with two or more labels below it and left of it'
        )
        raise NoPlotFrame(msg)
    _, frame, labels, refusals = max(candidates, key=lambda c: c[0])
    found = {'labels': labels, 'refusals': refusals, 'edges': edges}
    x_axis = _raster_axis('x', frame, values=x_values, scale=x_scale, **found)
    y_axis = _raster_axis('y', frame, values=y_values, scale=y_scale, **found)
    masks, occluders = _curve_masks(ink, chroma, rgb, frame=frame, box=box, scale=scale)
    if color is not None:
        masks = [m for m in masks if _sector(rgb[m].mean(axis=0)) == _sector(np.asarray(color, dtype=float))]
        if not masks:
            where = tuple(round(v, 1) for v in frame)
            msg = f'no curve drawn in color {tuple(color)} inside the plot framed at {where}'
            raise NoCurveOfColor(msg)
    traced = (
        _traced(mask, occluders & ~mask, rgb=rgb, here=raster.provenance, scale=scale, axes=(x_axis, y_axis))
        for mask in masks
    )
    curves = tuple(c for c in traced if c.x.size)  # ink whose every run is half hidden yields no clean sample
    return Plot(frame, x_axis, y_axis, curves, (), (), raster.provenance.at(bbox=frame))


def _dark_edges(ink: np.ndarray, dark: np.ndarray, box: BBox, scale: tuple[float, float]) -> list[_Edge]:
    """The runs of ``ink`` (:func:`_pixel_edges`) at least :data:`DARK_SHARE` dark along their line.

    A frame and its ticks are drawn dark; a curve drawn over the x axis from the origin covers the axis' pixels with
    its color, so the axis read from dark ink alone stopped short of the corner. Read from all ink, the axis runs
    on under the curve; a curve alone (no dark under it) is still no edge.
    """
    sx, sy = scale
    out = []
    for e in _pixel_edges(ink, box, scale):
        if e.vertical:
            col = min(int((e.at - box.x0) / sx), dark.shape[1] - 1)
            line = dark[int((e.lo - box.y0) / sy) : max(int((e.hi - box.y0) / sy), int((e.lo - box.y0) / sy) + 1), col]
        else:
            row = min(int((e.at - box.y0) / sy), dark.shape[0] - 1)
            line = dark[row, int((e.lo - box.x0) / sx) : max(int((e.hi - box.x0) / sx), int((e.lo - box.x0) / sx) + 1)]
        if line.size and line.mean() >= DARK_SHARE:
            out.append(e)
    return out


def _pixel_edges(ink: np.ndarray, box: BBox, scale: tuple[float, float]) -> list[_Edge]:
    """Every run of ink pixels along a row (horizontal) or a column (vertical), runs on neighboring lines merged."""
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
    """Each row of ink glyphs outside the frame as a label: its value, or NaN with the reader's refusal by index.

    Glyphs set turned a quarter (:func:`_columns`) are an axis title, not labels.
    """
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
    boxes = [BBox(*g.box) for g in glyphs]
    title = {i for column in _columns(glyphs, SANS) for i in column}  # a turned y title in the label column is no label
    rest = [i for i in range(len(glyphs)) if i not in title]
    for word in _words([boxes[i] for i in rest]):
        members = [glyphs[rest[k]] for k in word]
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
    marks = _marks(frame, edges, vertical=not vertical)
    if values is not None:
        return _fit(_stated(name, row, values, marks), scale)
    ticks = [(lb.at(vertical=vertical), lb.value) for lb in row if not math.isnan(lb.value)]
    if len(ticks) < 2:  # noqa: PLR2004 -- a line needs two points
        refused = next((refusals[labels.index(lb)] for lb in row if math.isnan(lb.value)), None)
        msg = (
            f'{name} axis: {len(ticks)} of {len(row)} labels read; state them '
            f'({name}_values=) or read the plot at a finer resolution'
        )
        raise UnreadableTickLabels(msg) from refused
    return _fit(_snap(ticks, marks), scale)


def _stated(
    name: str, row: Sequence[_Label], values: Sequence[float], marks: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """The caller's ``values`` (left to right, bottom to top) paired with the axis' labels, snapped to ``marks``.

    There must be as many labels as values, and every label that DID read must read as its stated value: a stated
    value is checked against what the page shows, never trusted blind. Else :class:`UnreadableTickLabels` is raised.
    """
    ordered = list(row[::-1] if name == 'y' else row)  # values run bottom to top, positions top to bottom
    vertical = name == 'y'
    if len(ordered) != len(values):
        msg = f'{name} axis: {len(ordered)} labels found, {len(values)} values stated'
        raise UnreadableTickLabels(msg)
    for lb, v in zip(ordered, values, strict=True):
        if not math.isnan(lb.value) and not math.isclose(lb.value, v, rel_tol=1e-9, abs_tol=0.0):
            at = lb.at(vertical=vertical)
            msg = f'{name} axis: the label at {at:.1f} reads {lb.value:g}, the stated value is {v:g}'
            raise UnreadableTickLabels(msg)
    return _snap([(lb.at(vertical=vertical), float(v)) for lb, v in zip(ordered, values, strict=True)], marks)


def _curve_masks(
    ink: np.ndarray, chroma: np.ndarray, rgb: np.ndarray, *, frame: BBox, box: BBox, scale: tuple[float, float]
) -> tuple[list[np.ndarray], np.ndarray]:
    """One pixel mask per curve inside the frame, and every pixel that can hide a curve's ink.

    A mask is a hue sector of colored ink or a component of dark ink; what can hide ink is any ink, and everything
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
    colored = chroma > CHROMA_MIN
    sector = np.floor(((_hue(rgb) + HUE_SECTOR / 2) % 360) / HUE_SECTOR).astype(int)
    out = []
    classes = [mask & colored & (sector == k) for k in range(round(360 / HUE_SECTOR))]
    for k, cls in enumerate([*classes, mask & ~colored]):
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


def _sector(rgb: np.ndarray) -> int:
    """The ``HUE_SECTOR`` a color falls in (as :func:`_curve_masks` sorts curves), or -1 when it is not colored."""
    if float(rgb.max() - rgb.min()) <= CHROMA_MIN:
        return -1
    return int(np.floor(((float(_hue(rgb)) + HUE_SECTOR / 2) % 360) / HUE_SECTOR))


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
    of its stroke may lie under them, and its center is then not the curve's.
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
        centers = [(lo + hi - 1) / 2] if hi - lo <= 2 * stroke else np.arange(lo + stroke / 2, hi - stroke / 2, stroke)
        cols.extend([c] * len(centers))
        rows.extend(centers)
    px = box.x0 + (np.array(cols, dtype=float) + 0.5) * sx
    py = box.y0 + (np.array(rows, dtype=float) + 0.5) * sy
    rr, cc = np.nonzero(mask)
    bbox = BBox(
        box.x0 + cc.min() * sx, box.y0 + rr.min() * sy, box.x0 + (cc.max() + 1) * sx, box.y0 + (rr.max() + 1) * sy
    )
    color = tuple(round(float(v), 2) for v in rgb[mask].mean(axis=0))
    return Curve(
        x_axis.to_data(px),
        y_axis.to_data(py),
        here.at(bbox=bbox),
        (x_axis.scale, y_axis.scale),
        color,
        stroke * sy,
        (),
        'raster',
        np.abs(x_axis.to_data(px + sx / 2) - x_axis.to_data(px - sx / 2)),
        np.abs(y_axis.to_data(py + sy / 2) - y_axis.to_data(py - sy / 2)),
    )
