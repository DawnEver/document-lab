"""Curves read off VECTOR plots: a frame, axes calibrated from their tick labels, polylines mapped to data.

A plot here is an axis-aligned rectangular frame with numeric tick labels below it (x) and left of it (y). Each
label's position is snapped to the tick mark or grid line it labels when one is drawn, else its box centre is
used. An axis is ``linear`` or ``log10``, chosen by which one the labels fit better, or forced by the caller (two
labels fit both exactly, so they read as linear unless told otherwise). Every stroked open polyline inside the
frame that is not a tick, a grid line or the frame is a curve.

A plot that is a RASTER image raises :class:`RasterPlot`: digitising pixels is not implemented, and a silent
empty result would read as "this plot has no curves". Secondary (right/top) axes are not read.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from document_lab.ir import BBox, Page, Provenance, TextSpan, VectorPath, union

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['Axis', 'Curve', 'Plot', 'RasterPlot', 'Scale', 'parse_number', 'plots']

Scale = Literal['linear', 'log10']

#: Typography a tick label may use: a true minus sign and a thin-space digit separator.
MINUS: Final = chr(0x2212)
THIN_SPACE: Final = chr(0x2009)
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
#: Font sizes (points) closer than this are one size.
SIZE_TOL: Final = 0.25
#: How close (points) a shifted label must land to a mark to count as on it.
SNAP_TOL: Final = 0.75
#: A grid line spans at least this fraction of the frame.
GRID_SPAN: Final = 0.9
#: The share of the frame an image must cover for a curve-less plot to count as a raster plot.
RASTER_COVER: Final = 0.25

_NUMBER: Final = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
_SUPERSCRIPT: Final = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_POWER: Final = re.compile(r'^10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)$')
_INTEGER: Final = re.compile(r'^[+-]?\d+$')


class RasterPlot(ValueError):
    """The plot is a raster image; only vector plots are digitised."""


def parse_number(text: str) -> float | None:
    """A tick label's value: decimals, exponents, a typographic minus, ``10⁻³``; ``None`` if it is not a number."""
    clean = text.strip().replace(MINUS, '-').replace(THIN_SPACE, '')
    if _NUMBER.match(clean):
        return float(clean)
    power = _POWER.match(clean)
    if power:
        return 10.0 ** int(power.group(1).translate(_SUPERSCRIPT))
    return None


@dataclass(frozen=True, slots=True)
class Axis:
    """A page coordinate -> data value map: ``value = s(slope * position + intercept)``, ``s`` identity or ``10**``."""

    scale: Scale
    slope: float
    intercept: float
    #: The calibration points: (page coordinate, labelled value).
    ticks: tuple[tuple[float, float], ...]

    def to_data(self, position: np.ndarray) -> np.ndarray:
        """The data values at page coordinates ``position``."""
        t = self.slope * np.asarray(position, dtype=float) + self.intercept
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


@dataclass(frozen=True, slots=True, eq=False)
class Plot:
    """A calibrated frame and the curves inside it."""

    frame: BBox
    x: Axis
    y: Axis
    curves: tuple[Curve, ...]
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
    # A grid line may stop short of the far edge (a legend sits there), so a long rule anchored on an edge counts.
    found = []
    for frame in _frames(edges):
        xs = _one_size(_x_labels(frame, labels))
        ys = _one_size(_y_labels(frame, labels))
        if len(xs) < 2 or len(ys) < 2:  # noqa: PLR2004 -- a line needs two points
            continue
        x_ticks = _snap([((lb.box.x0 + lb.box.x1) / 2, lb.value) for lb in xs], _marks(frame, edges, vertical=True))
        y_ticks = _snap([((lb.box.y0 + lb.box.y1) / 2, lb.value) for lb in ys], _marks(frame, edges, vertical=False))
        x_axis, y_axis = _fit(x_ticks, x_scale), _fit(y_ticks, y_scale)
        curves = tuple(
            Curve(
                x_axis.to_data(np.array([p[0] for p in path.points])),
                y_axis.to_data(np.array([p[1] for p in path.points])),
                path.provenance,
                (x_axis.scale, y_axis.scale),
                path.stroke,
                path.width,
            )
            for path in _curve_paths(frame, page.paths)
        )
        if not curves:
            _refuse_raster(page, frame)
        found.append(Plot(frame, x_axis, y_axis, curves, page.spans[0].provenance.at(bbox=frame)))
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
        dash = _dash_before(box, dashes) if value > 0 else None
        if dash is not None:
            value, box = -value, union([box, BBox(dash.lo, box.y0, dash.hi, box.y1)])
        out.append(_Label(value, box, span.size))
    return out


def _dash_before(box: BBox, dashes: Sequence[_Edge]) -> _Edge | None:
    """A drawn minus sign: a short horizontal stroke ending just left of the label, within its height."""
    for d in dashes:
        if box.x0 - DASH_GAP <= d.hi <= box.x0 + EDGE_TOL / 3 and box.y0 < d.at < box.y1:
            return d
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
        text = other.text.strip().replace(MINUS, '-')
        if j == i or ob is None or not _INTEGER.match(text) or other.size >= base.size * 0.9:
            continue
        raised = (ob.y0 + ob.y1) / 2 < (bb.y0 + bb.y1) / 2 and ob.y1 > bb.y0 and ob.y0 >= bb.y0 - ob.height
        if not raised:
            continue
        gap = ob.x0 - bb.x1
        if -EDGE_TOL <= gap <= EDGE_TOL:
            return j, int(text)
        dashed = any(bb.x1 - EDGE_TOL <= d.lo and d.hi <= ob.x0 + EDGE_TOL and bb.y0 <= d.at <= ob.y1 for d in dashes)
        if 0 < gap <= DASH_MAX + EDGE_TOL and dashed and not text.startswith('-'):
            return j, -int(text)
    return None


def _x_labels(frame: BBox, labels: list[_Label]) -> list[_Label]:
    return [
        lb
        for lb in labels
        if frame.y1 - EDGE_TOL <= lb.box.y0 <= frame.y1 + LABEL_REACH
        and frame.x0 - LABEL_REACH / 3 <= (lb.box.x0 + lb.box.x1) / 2 <= frame.x1 + LABEL_REACH / 3
    ]


def _y_labels(frame: BBox, labels: list[_Label]) -> list[_Label]:
    return [
        lb
        for lb in labels
        if frame.x0 - LABEL_REACH <= lb.box.x1 <= frame.x0 + EDGE_TOL
        and frame.y0 - LABEL_REACH / 3 <= (lb.box.y0 + lb.box.y1) / 2 <= frame.y1 + LABEL_REACH / 3
    ]


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
    scale = forced or _infer_scale(pos, val)
    if scale == 'log10' and np.any(val <= 0):
        msg = f'a log10 axis needs positive tick values, got {val.tolist()}'
        raise ValueError(msg)
    target = np.log10(val) if scale == 'log10' else val
    slope, intercept = np.polyfit(pos, target, 1)
    return Axis(scale, float(slope), float(intercept), ticks)


def _infer_scale(pos: np.ndarray, val: np.ndarray) -> Scale:
    if len(val) < 3 or np.any(val <= 0):  # noqa: PLR2004 -- two points fit both scales exactly
        return 'linear'
    return 'log10' if _misfit(pos, np.log10(val)) < _misfit(pos, val) else 'linear'


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
                'only vector plots are digitised'
            )
            raise RasterPlot(msg)
