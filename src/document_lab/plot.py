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
    clean = text.strip().replace(MINUS, '-').replace(THIN_SPACE, '').replace(' ', '')
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


def plots(page: Page, *, x_scale: Scale | None = None, y_scale: Scale | None = None) -> list[Plot]:
    """Every vector plot on ``page``; ``x_scale``/``y_scale`` force an axis scale instead of inferring it."""
    labels = _labels(page.spans)
    found = []
    for frame in _frames(page.paths):
        xs = _x_labels(frame, labels)
        ys = _y_labels(frame, labels)
        if len(xs) < 2 or len(ys) < 2:  # noqa: PLR2004 -- a line needs two points
            continue
        x_ticks = _snap([((lb.box.x0 + lb.box.x1) / 2, lb.value) for lb in xs], _x_marks(frame, page.paths))
        y_ticks = _snap([((lb.box.y0 + lb.box.y1) / 2, lb.value) for lb in ys], _y_marks(frame, page.paths))
        x_axis, y_axis = _fit(x_ticks, x_scale), _fit(y_ticks, y_scale)
        curves = tuple(
            Curve(
                x_axis.to_data(np.array([p[0] for p in path.points])),
                y_axis.to_data(np.array([p[1] for p in path.points])),
                path.provenance,
                (x_axis.scale, y_axis.scale),
                path.stroke,
            )
            for path in _curve_paths(frame, page.paths)
        )
        if not curves:
            _refuse_raster(page, frame)
        here = Provenance(page.spans[0].provenance.sha256, page.spans[0].provenance.uri, page.number, frame)
        found.append(Plot(frame, x_axis, y_axis, curves, here))
    return found


def _labels(spans: Sequence[TextSpan]) -> list[_Label]:
    """Numeric spans, with ``10`` plus a raised smaller integer span merged into a power of ten."""
    out, used = [], set()
    for i, span in enumerate(spans):
        box = span.provenance.bbox
        if box is None or i in used:
            continue
        if span.text.strip() == '10':
            exp = _exponent_of(i, spans)
            if exp is not None:
                used.add(exp)
                power = int(spans[exp].text.strip().replace(MINUS, '-'))
                out.append(_Label(10.0**power, union([box, spans[exp].provenance.bbox])))
                continue
        value = parse_number(span.text)
        if value is not None:
            out.append(_Label(value, box))
    return out


def _exponent_of(i: int, spans: Sequence[TextSpan]) -> int | None:
    base = spans[i]
    bb = base.provenance.bbox
    for j, other in enumerate(spans):
        ob = other.provenance.bbox
        if j == i or ob is None or not _INTEGER.match(other.text.strip().replace(MINUS, '-')):
            continue
        raised = (ob.y0 + ob.y1) / 2 < (bb.y0 + bb.y1) / 2
        if other.size < base.size * 0.9 and abs(ob.x0 - bb.x1) <= EDGE_TOL and raised:
            return j
    return None


def _is_rect(path: VectorPath) -> BBox | None:
    pts = path.points[:-1] if len(path.points) == 5 and path.points[0] == path.points[-1] else path.points  # noqa: PLR2004
    if not path.closed or len(pts) != 4:  # noqa: PLR2004
        return None
    xs = sorted({round(p[0], 3) for p in pts})
    ys = sorted({round(p[1], 3) for p in pts})
    if len(xs) != 2 or len(ys) != 2:  # noqa: PLR2004
        return None
    return BBox(xs[0], ys[0], xs[1], ys[1])


def _frames(paths: Sequence[VectorPath]) -> list[BBox]:
    seen: dict[tuple[float, ...], BBox] = {}
    for path in paths:
        box = _is_rect(path)
        if box is not None and box.width >= MIN_FRAME and box.height >= MIN_FRAME:
            seen.setdefault(tuple(round(v, 1) for v in box), box)
    return list(seen.values())


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


def _segment(path: VectorPath) -> tuple[tuple[float, float], tuple[float, float]] | None:
    return (path.points[0], path.points[1]) if len(path.points) == 2 and not path.closed else None  # noqa: PLR2004


def _x_marks(frame: BBox, paths: Sequence[VectorPath]) -> list[float]:
    """X of vertical tick marks on the bottom/top edge and of vertical grid lines."""
    marks = []
    for path in paths:
        seg = _segment(path)
        if seg is None or abs(seg[0][0] - seg[1][0]) > EDGE_TOL / 3:
            continue
        lo, hi = sorted((seg[0][1], seg[1][1]))
        on_edge = any(abs(e - lo) <= EDGE_TOL or abs(e - hi) <= EDGE_TOL for e in (frame.y0, frame.y1))
        if (on_edge and hi - lo <= TICK_MAX) or hi - lo >= GRID_SPAN * frame.height:
            marks.append(seg[0][0])
    return marks


def _y_marks(frame: BBox, paths: Sequence[VectorPath]) -> list[float]:
    """Y of horizontal tick marks on the left/right edge and of horizontal grid lines."""
    marks = []
    for path in paths:
        seg = _segment(path)
        if seg is None or abs(seg[0][1] - seg[1][1]) > EDGE_TOL / 3:
            continue
        lo, hi = sorted((seg[0][0], seg[1][0]))
        on_edge = any(abs(e - lo) <= EDGE_TOL or abs(e - hi) <= EDGE_TOL for e in (frame.x0, frame.x1))
        if (on_edge and hi - lo <= TICK_MAX) or hi - lo >= GRID_SPAN * frame.width:
            marks.append(seg[0][1])
    return marks


def _snap(ticks: list[tuple[float, float]], marks: list[float]) -> tuple[tuple[float, float], ...]:
    """Each label's position moved to the nearest mark within 40% of the label spacing, when there is one."""
    positions = sorted(p for p, _ in ticks)
    spacing = min((b - a for a, b in itertools.pairwise(positions) if b > a), default=LABEL_REACH)
    out = []
    for pos, value in ticks:
        near = min(marks, key=lambda m: abs(m - pos), default=None)
        out.append((near if near is not None and abs(near - pos) <= 0.4 * spacing else pos, value))
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
    out = []
    for path in paths:
        if path.stroke is None or path.closed or len(path.points) < 2:  # noqa: PLR2004
            continue
        if not all(frame.contains(x, y, tol=EDGE_TOL) for x, y in path.points):
            continue
        seg = _segment(path)
        if seg is not None and _is_rule(seg, frame):
            continue
        out.append(path)
    return out


def _is_rule(seg: tuple[tuple[float, float], tuple[float, float]], frame: BBox) -> bool:
    """A tick mark, grid line or frame edge: an axis-parallel segment that is short or spans the frame."""
    (x0, y0), (x1, y1) = seg
    if abs(x0 - x1) <= EDGE_TOL / 3:
        length, span = abs(y1 - y0), frame.height
    elif abs(y0 - y1) <= EDGE_TOL / 3:
        length, span = abs(x1 - x0), frame.width
    else:
        return False
    return length <= TICK_MAX or length >= GRID_SPAN * span


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
