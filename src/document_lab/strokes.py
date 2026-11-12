"""Thick strokes in a raster: fragments, curves joined from dashes, and the dash pattern that names a curve.

A plot drawn with thick curve strokes over hairline grid and lettering reads by a morphological OPENING: a square of
side ``opening_pt`` (wider than any hairline, narrower than the strokes) keeps the strokes alone. A dashed curve
then comes out as separate fragments, which this module joins:

- ``link``: each fragment's right end goes to the fragment whose left end lies on its own extrapolated direction
  (nearest first, each end once). That joins a curve across its gaps and refuses a jump onto a curve crossing it.
- ``bands``: where curves run too close for links alone, the solid curves (one chain each) define bands, and every
  other fragment joins the band of its nearest solid curve on the stated side.
- ``split_by_dots``: a dash-dot curve's dots are smaller than the opening that reads the dashes; read again with a
  smaller opening, the dots (``dots``) trace the dash-dot curve alone, which tells it from a dashed curve two
  strokes away. ``link`` joins a dash-dot curve through its dots: a dot has no direction of its own.

Coordinates are page pt (x right, y down), so every distance here is in pt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from document_lab.plot import _components, _runs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.ir import Raster

Side = Literal['above', 'below', 'nearest']

INK_LEVEL = 128  # a pixel darker than this (0..255 grey) is ink
GAP_PT = 9.0  # longest dash gap a link bridges
OFF_PT = 1.6  # largest miss of the extrapolated end (about one stroke)
END_PT = 2.5  # the end segment a fragment's direction is fit on
DOT_PT = 2.2  # a fragment no longer than this is a dot


@dataclass(frozen=True)
class Fragment:
    """One connected piece of thick ink: its centre-line samples in page pt, and its stroke width.

    ``axis``: ``(x, y, slope)`` of its ink's principal axis through the ink centroid, when read from pixels. A short
    dash's ends lie on it: its centre-line samples bend towards the caps, its pixels do not.
    """

    x: np.ndarray
    y: np.ndarray
    stroke: float
    axis: tuple[float, float, float] | None = None

    @property
    def length(self) -> float:
        """The larger extent, in pt."""
        return float(max(np.ptp(self.x), np.ptp(self.y)))


def opening(ink: np.ndarray, k: int) -> np.ndarray:
    """The binary opening of ``ink`` by a ``k`` x ``k`` square: the union of every square that fits inside it."""
    if k <= 1:
        return ink.copy()

    def window_sum(a: np.ndarray) -> np.ndarray:
        s = np.pad(a.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]

    fits = window_sum(ink) == k * k  # fits[r, c]: the square with top-left (r, c) lies in ink
    covered = window_sum(np.pad(fits, ((k - 1, k - 1), (k - 1, k - 1)))) > 0
    return covered & ink


def fragments(
    raster: Raster, *, opening_pt: float, min_pt: float = 0.0, exclude: Sequence[tuple[float, float, float, float]] = ()
) -> list[Fragment]:
    """The thick-ink fragments of ``raster`` after an opening of side ``opening_pt``.

    ``exclude``: page-pt boxes ``(x0, y0, x1, y1)`` blanked first (a legend). A fragment shorter than ``min_pt`` is
    dropped. Each is sampled per pixel column at its run centres, per row where it is taller than wide.
    """
    x0, y0, x1, y1 = raster.provenance.bbox  # type: ignore[misc]
    rows, cols = raster.pixels.shape[:2]
    sx, sy = (x1 - x0) / cols, (y1 - y0) / rows
    ink = raster.pixels.mean(axis=2) < INK_LEVEL
    for ex0, ey0, ex1, ey1 in exclude:
        r0, r1 = max(int((ey0 - y0) / sy), 0), max(int(np.ceil((ey1 - y0) / sy)), 0)
        c0, c1 = max(int((ex0 - x0) / sx), 0), max(int(np.ceil((ex1 - x0) / sx)), 0)
        ink[r0:r1, c0:c1] = False
    labels = _components(opening(ink, round(opening_pt / ((sx + sy) / 2))))
    out = []
    for i in range(1, int(labels.max()) + 1):
        rr, cc = np.nonzero(labels == i)
        r0, c0 = rr.min(), cc.min()
        mask = labels[r0 : rr.max() + 1, c0 : cc.max() + 1] == i
        if max(mask.shape[0] * sy, mask.shape[1] * sx) < min_pt:
            continue
        rowwise = mask.shape[0] > mask.shape[1]
        xs, ys, widths = [], [], []
        for j, line in enumerate(mask if rowwise else mask.T):
            for lo, hi in _runs(line):
                centre = (lo + hi - 1) / 2 + 0.5
                if rowwise:
                    xs.append(x0 + (c0 + centre) * sx)
                    ys.append(y0 + (r0 + j + 0.5) * sy)
                    widths.append((hi - lo) * sx)
                else:
                    xs.append(x0 + (c0 + j + 0.5) * sx)
                    ys.append(y0 + (r0 + centre) * sy)
                    widths.append((hi - lo) * sy)
        px, py = x0 + (cc + 0.5) * sx, y0 + (rr + 0.5) * sy
        out.append(Fragment(np.array(xs), np.array(ys), float(np.median(widths)), _principal_axis(px, py)))
    return out


def _principal_axis(px: np.ndarray, py: np.ndarray) -> tuple[float, float, float]:
    """``(x, y, slope)``: the ink centroid and the slope of the cloud's major axis.

    A vertical axis reads as a large finite slope.
    """
    cx, cy = float(px.mean()), float(py.mean())
    cov = np.cov(np.vstack([px - cx, py - cy])) if len(px) > 1 else np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    dx, dy = vecs[:, int(np.argmax(vals))]
    return cx, cy, float(dy / dx) if abs(dx) > 1e-9 else 1e9  # noqa: PLR2004


End = tuple[float, float, float | None]


def _ends(f: Fragment, end_pt: float) -> tuple[End, End]:
    """``(x, y, slope)`` at the left and the right end.

    A dot (no longer than ``DOT_PT``) has no direction: its slope is ``None`` and both its ends sit at its centre
    height. A fragment too short to hold ``end_pt`` of line beyond a cap at each end has its ends on its ink's
    principal axis. A longer one fits a line on the ``end_pt`` of x just inside each cap (one stroke from the end) and
    takes the end on that line: a column through a sloped stroke's cap samples a corner, not the line.
    """
    o = np.argsort(f.x, kind='stable')
    x, y = f.x[o], f.y[o]
    lo, hi = float(x[0]), float(x[-1])
    if f.length <= DOT_PT:
        centre = float(np.mean(y))
        return (lo, centre, None), (hi, centre, None)
    cap = f.stroke
    if hi - lo < 2 * (cap + end_pt) and f.axis is not None:
        cx, cy, slope = f.axis
        return (lo, cy + slope * (lo - cx), slope), (hi, cy + slope * (hi - cx), slope)

    def end(sel: np.ndarray, at: float) -> End:
        if np.count_nonzero(sel) < 2 or np.ptp(x[sel]) < 0.3:  # noqa: PLR2004 -- pt: no slope in x to fit
            near = np.abs(x - at) <= cap + end_pt
            return at, float(np.mean(y[near])), 0.0
        slope, intercept = np.polyfit(x[sel], y[sel], 1)
        return at, float(intercept + slope * at), float(slope)

    left = end((x >= lo + cap) & (x <= lo + cap + end_pt), lo)
    right = end((x <= hi - cap) & (x >= hi - cap - end_pt), hi)
    return left, right


def _direction(right: float | None, left: float | None) -> float:
    """The slope a gap is crossed on: both ends' mean when they agree, else the right end's; a dot defers."""
    if right is None or left is None:
        return next((v for v in (right, left) if v is not None), 0.0)
    return 0.5 * (right + left) if abs(right - left) < 1.0 else right


def link(
    frags: Sequence[Fragment], *, gap_pt: float = GAP_PT, off_pt: float = OFF_PT, end_pt: float = END_PT
) -> list[list[int]]:
    """Chains of fragment indices, each left to right.

    A link runs from a right end to a left end within ``gap_pt``, the left end missing the right end's extrapolated
    direction by at most ``off_pt``. Within that cone the nearest end (by distance) is taken first, each end once.
    """
    ends = [_ends(f, end_pt) for f in frags]
    candidates = []
    for i, (_, (rx, ry, rs)) in enumerate(ends):
        for j, ((lx, ly, ls), _) in enumerate(ends):
            dx = lx - rx
            if i == j or not -0.5 < dx <= gap_pt:  # noqa: PLR2004 -- pt of overlap a butt joint may show
                continue
            s = _direction(rs, ls)
            miss = abs(ly - (ry + s * dx))
            if miss <= off_pt:
                candidates.append((float(np.hypot(max(dx, 0.0), miss)), i, j))
    after: dict[int, int] = {}
    before: dict[int, int] = {}
    for _, i, j in sorted(candidates):
        if i not in after and j not in before:
            after[i], before[j] = j, i
    chains = []
    for i in range(len(frags)):
        if i in before:
            continue
        chain = [i]
        while chain[-1] in after:
            chain.append(after[chain[-1]])
        chains.append(chain)
    return chains


def coverage(frags: Sequence[Fragment], chain: Sequence[int]) -> float:
    """Inked fraction of the chain's x span: about 1 for a solid curve, well below for a dashed one."""
    spans = [(frags[i].x.min(), frags[i].x.max()) for i in chain]
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    return float(sum(b - a for a, b in spans) / max(hi - lo, 1e-9))


def centre_line(
    frags: Sequence[Fragment], chain: Sequence[int], *, bin_pt: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """The chain's samples averaged per ``bin_pt`` of x: one ``(x, y)`` per bin, x increasing."""
    px = np.concatenate([frags[i].x for i in chain])
    py = np.concatenate([frags[i].y for i in chain])
    b = np.floor((px - px.min()) / bin_pt).astype(int)
    keys = np.unique(b)
    return np.array([px[b == k].mean() for k in keys]), np.array([py[b == k].mean() for k in keys])


def bands(
    frags: Sequence[Fragment],
    solids: Sequence[Sequence[int]],
    *,
    reach_pt: float = 30.0,
    side: Side = 'above',
    majority: float = 0.6,
) -> tuple[list[list[int]], list[int]]:
    """Each fragment outside ``solids`` joined to the band of its nearest solid curve.

    ``side``: where the solid lies relative to the fragment on the page -- ``above`` (smaller y), ``below`` or
    ``nearest``. Each sample votes for the nearest solid within ``reach_pt`` (over its x span, plus 3 pt); a
    fragment joins the band a ``majority`` of its samples vote for, else it is an orphan. Returns the members per
    solid (in the order given) and the orphans.
    """
    lines = [centre_line(frags, ch) for ch in solids]
    used = {i for ch in solids for i in ch}
    members: list[list[int]] = [[] for _ in solids]
    orphans = []
    tol = 0.6  # pt a fragment may overlap its solid's ink on the wrong side
    for i, f in enumerate(frags):
        if i in used:
            continue
        votes = []
        for x, y in zip(f.x, f.y, strict=True):
            best, best_d = None, reach_pt
            for k, (sx, sy) in enumerate(lines):
                if not sx.min() - 3 <= x <= sx.max() + 3:
                    continue
                d = y - float(np.interp(x, sx, sy))
                d = {'above': d, 'below': -d, 'nearest': abs(d)}[side]
                if -tol <= d < best_d:
                    best, best_d = k, d
            votes.append(best)
        hit = [v for v in votes if v is not None]
        if hit and len(hit) >= majority * len(votes):
            members[max(set(hit), key=hit.count)].append(i)
        else:
            orphans.append(i)
    return members, orphans


def split_by_dots(
    frags: Sequence[Fragment], dots: Sequence[Fragment], *, off_pt: float = 1.0, reach_pt: float = 15.0
) -> tuple[list[int], list[int]]:
    """``frags`` split into those on the dash-dot curve the ``dots`` trace, and the rest.

    The dots of one dash-dot curve, read with an opening small enough to keep them, trace that curve: the polyline
    through their centres (x increasing). A fragment whose samples lie on it -- median miss within ``off_pt`` over the
    samples within ``reach_pt`` of the dots' x span -- is the dash-dot curve's; every other fragment is not. A curve
    two strokes away is told apart where the dash lengths cannot be (both are dashes). Needs two dots or more.
    """
    if len(dots) < 2:  # noqa: PLR2004 -- a line needs two points
        msg = f'{len(dots)} dot(s): a dash-dot curve is traced by two or more'
        raise ValueError(msg)
    centres = sorted((float(d.x.mean()), float(d.y.mean())) for d in dots)
    dx, dy = np.array([c[0] for c in centres]), np.array([c[1] for c in centres])
    on, off = [], []
    for i, f in enumerate(frags):
        near = (f.x >= dx[0] - reach_pt) & (f.x <= dx[-1] + reach_pt)
        if not near.any():
            off.append(i)
            continue
        miss = float(np.median(np.abs(f.y[near] - _polyline(f.x[near], dx, dy))))
        (on if miss <= off_pt else off).append(i)
    return on, off


def _polyline(x: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """The polyline through ``(px, py)`` at ``x``, its end segments extended straight (never held flat)."""
    y = np.interp(x, px, py)
    lo, hi = x < px[0], x > px[-1]
    y[lo] = py[0] + (py[1] - py[0]) / max(px[1] - px[0], 1e-9) * (x[lo] - px[0])
    y[hi] = py[-1] + (py[-1] - py[-2]) / max(px[-1] - px[-2], 1e-9) * (x[hi] - px[-1])
    return y


def dots(frags: Sequence[Fragment], *, max_pt: float = DOT_PT) -> list[Fragment]:
    """The fragments no longer than ``max_pt``: candidate dots."""
    return [f for f in frags if f.length <= max_pt]
