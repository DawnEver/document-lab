"""Curves read off VECTOR plots: a frame, axes calibrated from their tick labels, polylines mapped to data.

A plot here is an axis-aligned rectangular frame with numeric tick labels below it (x) and left of it (y). Each
label's position is snapped to the tick mark or grid line it labels when one is drawn, else its box centre is
used. A tick label is a number in the axis' common font size, aligned with the others on one row (x) or column
(y), whose box overlaps the frame's extent along the axis; a stray number off that row or past the frame is not.

An axis is ``linear`` or ``log10``, forced by the caller or else the one whose fit needs fewer pieces (then the
smaller misfit; two labels fit both exactly, so they read as linear unless told otherwise). The fit is PIECEWISE:
the axis is cut at every tick where the data-per-point spacing (in the scale's units) changes by more than
``BREAK_TOL``, and each piece is a least-squares line through its own ticks -- a broken axis whose scale changes at
a break reads right. Ticks whose values are not strictly monotone along the axis are refused (``ValueError``).

Every stroked open polyline inside the frame that is not a tick, a grid line or the frame is a curve, after
fragments are joined: polylines of one style (stroke, width, dash array) whose ends meet (to ``JOIN_DIGITS``
decimals) become one, unless more than two ends of that style meet there (ambiguous, left unjoined). A short
horizontal curve (at most ``SAMPLE_MAX`` long) with a text span starting within ``SAMPLE_GAP`` right of it, at its
height, is a legend sample: it goes to ``Plot.legend`` with that text, not to ``Plot.curves``.

A plot that is a RASTER image raises :class:`RasterPlot`: digitising pixels is not implemented, and a silent
empty result would read as "this plot has no curves". Secondary (right/top) axes are not read.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from document_lab.ir import BBox, Page, Provenance, TextSpan, VectorPath, union
from document_lab.numbers import parse_number

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ['Axis', 'Curve', 'LegendSample', 'Plot', 'RasterPlot', 'Scale', 'plots']

Scale = Literal['linear', 'log10']

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
#: The share of the frame an image must cover for a curve-less plot to count as a raster plot.
RASTER_COVER: Final = 0.25


class RasterPlot(ValueError):
    """The plot is a raster image; only vector plots are digitised."""


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
            )
            for path in paths
        )
        if not curves:
            _refuse_raster(page, frame)
        found.append(Plot(frame, x_axis, y_axis, curves, tuple(legend), page.spans[0].provenance.at(bbox=frame)))
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
    """The largest set of labels sharing one row (x: top, bottom or centre) or column (y: left, right or centre)."""
    lo, hi = ('x0', 'x1') if vertical else ('y0', 'y1')
    keys = (
        lambda b: getattr(b, lo),
        lambda b: getattr(b, hi),
        lambda b: (getattr(b, lo) + getattr(b, hi)) / 2,
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
                'only vector plots are digitised'
            )
            raise RasterPlot(msg)
