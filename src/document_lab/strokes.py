"""Thick strokes in a raster: fragments, curves joined from dashes, and the dash pattern that names a curve.

A plot drawn with thick curve strokes over hairline grid and lettering reads by a morphological OPENING: a square of
side ``opening_pt`` (wider than any hairline, narrower than the strokes) keeps the strokes alone. A dashed curve
then comes out as separate fragments, which this module joins:

- ``link``: each fragment's right end goes to the fragment whose left end lies on its own extrapolated direction
  (mutual best, by miss). That joins a curve across its gaps and refuses a jump onto a curve crossing it.
- ``bands``: where curves run too close for links alone, the solid curves (one chain each) define bands, and every
  other fragment joins the band of its nearest solid curve on the stated side.
- ``attach_dots``: a dash-dot curve's dots are smaller than the opening that reads the dashes; read the band again
  with a smaller opening, and a dot sits in the gap of the dash-dot chain only, which names the pattern
  (``pattern``) and so tells two close dashed curves apart.

Coordinates are page pt (x right, y down), so every distance here is in pt.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from document_lab.plot import _components, _runs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.ir import Raster

Side = Literal['above', 'below', 'nearest']
Pattern = Literal['solid', 'dashed', 'dash-dot']

INK_LEVEL = 128  # a pixel darker than this (0..255 grey) is ink
GAP_PT = 9.0  # longest dash gap a link bridges
OFF_PT = 1.6  # largest miss of the extrapolated end (about one stroke)
END_PT = 2.5  # the end segment a fragment's direction is fit on
DOT_PT = 2.2  # a fragment no longer than this is a dot
SOLID_COVERAGE = 0.9  # inked fraction of its span above which a chain is solid


@dataclass(frozen=True)
class Fragment:
    """One connected piece of thick ink: its centre-line samples in page pt, and its stroke width."""

    x: np.ndarray
    y: np.ndarray
    stroke: float

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
        out.append(Fragment(np.array(xs), np.array(ys), float(np.median(widths))))
    return out


def _ends(f: Fragment, end_pt: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(x, y, slope)`` at the left and the right end, the slope fit on the last ``end_pt`` of x."""
    o = np.argsort(f.x, kind='stable')
    x, y = f.x[o], f.y[o]

    def slope(sel: np.ndarray) -> float:
        return 0.0 if np.ptp(x[sel]) < 0.3 else float(np.polyfit(x[sel], y[sel], 1)[0])  # noqa: PLR2004 -- pt

    lo, hi = x[0], x[-1]
    left = (float(lo), float(np.mean(y[x <= lo + 0.3])), slope(x <= lo + end_pt))
    right = (float(hi), float(np.mean(y[x >= hi - 0.3])), slope(x >= hi - end_pt))
    return left, right


def link(
    frags: Sequence[Fragment], *, gap_pt: float = GAP_PT, off_pt: float = OFF_PT, end_pt: float = END_PT
) -> list[list[int]]:
    """Chains of fragment indices, each left to right.

    A link runs from a right end to a left end within ``gap_pt``, the left end missing the right end's extrapolated
    direction by at most ``off_pt``; links are taken mutual-best by miss (plus a small gap term).
    """
    ends = [_ends(f, end_pt) for f in frags]
    candidates = []
    for i, (_, (rx, ry, rs)) in enumerate(ends):
        for j, ((lx, ly, ls), _) in enumerate(ends):
            dx = lx - rx
            if i == j or not -0.5 < dx <= gap_pt:  # noqa: PLR2004 -- pt of overlap a butt joint may show
                continue
            s = 0.5 * (rs + ls) if abs(rs - ls) < 1.0 else rs
            miss = abs(ly - (ry + s * dx))
            if miss <= off_pt:
                candidates.append((miss + 0.05 * max(dx, 0.0), i, j))
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


def attach_dots(
    frags: Sequence[Fragment], chains: Sequence[Sequence[int]], dots: Sequence[Fragment], *, off_pt: float = OFF_PT
) -> list[list[int]]:
    """Per chain, the indices of ``dots`` lying in one of its gaps and on its line (within ``off_pt``).

    The line across a gap is the straight segment from one fragment's right end to the next's left end; a dot
    claimed by two chains goes to the nearer, a dot no chain claims to none.
    """
    gaps = []
    for k, ch in enumerate(chains):
        for a, b in itertools.pairwise(ch):
            (_, right), (left, _) = _ends(frags[a], END_PT), _ends(frags[b], END_PT)
            gaps.append((k, right[0], right[1], left[0], left[1]))
    out: list[list[int]] = [[] for _ in chains]
    for d, dot in enumerate(dots):
        cx, cy = float(dot.x.mean()), float(dot.y.mean())
        best, best_miss = None, off_pt
        for k, xa, ya, xb, yb in gaps:
            if not xa - 0.5 <= cx <= xb + 0.5 or xb <= xa:
                continue
            miss = abs(cy - (ya + (yb - ya) * (cx - xa) / (xb - xa)))
            if miss <= best_miss:
                best, best_miss = k, miss
        if best is not None:
            out[best].append(d)
    return out


def pattern(frags: Sequence[Fragment], chain: Sequence[int], n_dots: int, *, min_dots: int = 2) -> Pattern:
    """The chain's line style from its ink and the dots in its gaps.

    ``solid`` when the chain inks its span, ``dash-dot`` when at least ``min_dots`` dots sit in its gaps, else
    ``dashed``.
    """
    if coverage(frags, chain) > SOLID_COVERAGE:
        return 'solid'
    return 'dash-dot' if n_dots >= min_dots else 'dashed'


def dots(frags: Sequence[Fragment], *, max_pt: float = DOT_PT) -> list[Fragment]:
    """The fragments no longer than ``max_pt``: candidate dots."""
    return [f for f in frags if f.length <= max_pt]
