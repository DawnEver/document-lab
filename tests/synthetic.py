"""Synthetic PDFs drawn with PyMuPDF so every expected value is known exactly -- the planted controls.

A plot is drawn on a frame from `FRAME`; its tick labels sit beside tick marks at known data values, and its curve
is a known function sampled at known points, so a digitizer that is wrong cannot pass by agreeing with itself.
"""

from __future__ import annotations

import functools
import itertools
import math
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import pymupdf

from document_lab.strokefont import TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from document_lab.ir import Page, Table

FRAME = pymupdf.Rect(100, 100, 400, 300)
RED = (1, 0, 0)
BLUE = (0, 0, 1)
FONT_SIZE = 8
#: A tagged PDF's page size (points).
PAGE = (400.0, 300.0)


#: The size (points) of a marker glyph, as small as the sheets' marks.
MARK_SIZE = 5
#: The PDF font name the marker glyphs are set in (PyMuPDF's CJK font, which holds the geometric shapes).
MARK_FONT = 'mark'


@dataclass
class Axis:
    ticks: Sequence[float]
    log: bool = False
    labels: Sequence[str] | None = None  # defaults to the tick values' shortest spelling
    power_labels: bool = False  # draw "10" plus a raised exponent span instead of the value
    vector_minus: bool = False  # a negative exponent's minus is a drawn dash, not a glyph
    #: A negative label's (or exponent's) minus is its own span (another font), apart from its digits.
    span_minus: bool = False
    #: The glyph a span minus is set as: ``-``, or ``_`` as TDK/EPCOS SIFERRIT sheets set it (raised, small).
    minus_glyph: str = '-'
    #: Points of space set between ``10`` and its raised exponent (TDK SIFERRIT: about one).
    exponent_gap: float = 0.0
    #: Each glyph of an x label is its own span (alternating fonts), as some typesetters emit them.
    glyph_spans: bool = False
    #: Points between a drawn minus and its digits' box; negative when the dash runs into the box.
    minus_gap: float = 1.0
    label_drop: float = 0.0  # points a y label sits below its tick, as real sheets' labels do
    at: Sequence[float] | None = None  # each tick's fraction along the frame; a broken axis changes scale at one

    def frac(self, value: float) -> float:
        if self.at is not None:
            return float(np.interp(value, self.ticks, self.at))
        lo, hi = self.ticks[0], self.ticks[-1]
        if self.log:
            return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (value - lo) / (hi - lo)


@dataclass
class PlotSpec:
    x: Axis
    y: Axis
    #: (function, sample xs, color), optionally with a PDF dash string such as '[3 2] 0'.
    curves: list[tuple[Callable[[float], float], Sequence[float], tuple[float, float, float], *tuple[str, ...]]] = (
        field(default_factory=list)
    )
    frame: pymupdf.Rect = field(default_factory=lambda: pymupdf.Rect(FRAME))
    tick_marks: bool = True
    grid: bool = False
    grid_grey: float = 0.7  # the major grid's grey level (0 black .. 1 white)
    minor_grid: bool = False  # a log axis also gets grid lines at 2..9 x each decade
    frame_as_lines: bool = False  # the frame is four separate segments, not one rectangle
    #: Points the x axis runs on left of the corner and the y axis below it (axes drawn through the origin).
    overhang: float = 0.0
    #: A rectangle drawn around the whole figure (a border box), or None.
    border: pymupdf.Rect | None = None
    grid_reach: float = 1.0  # the fraction of the frame width a horizontal grid line spans (a legend cuts it)
    #: Spans (fractions of the frame height from its top) where a label set over the grid cuts each vertical grid line.
    grid_gaps: Sequence[tuple[float, float]] = ()
    stray: Sequence[tuple[str, float, float, float]] = ()  # extra text: (text, x, y, size)
    #: A y-axis title set ROTATED 90 degrees (reading upward) with its baseline at x, starting at y: (text, x, y, size).
    y_title: tuple[str, float, float, float] | None = None
    fragments: int = 1  # each curve drawn as this many polylines meeting end to end, every other one reversed
    #: Legend entries inside the frame's top right: (label, color, PDF dash string or None), a sample line each.
    legend: Sequence[tuple[str, tuple[float, float, float], str | None]] = ()
    #: Tick labels drawn in the stroke font (:func:`stroke_text`): polylines only, no text, as TDK SIFERRIT N72 does.
    stroke_font: bool = False
    #: The frame is a filled, unstroked rectangle with grid lines inside it only, as sheets drawn in a design tool are.
    filled_frame: bool = False
    #: Data points drawn as a text glyph centered on each: (glyph, [(x, y), ...], color).
    markers: Sequence[tuple[str, Sequence[tuple[float, float]], tuple[float, float, float]]] = ()
    #: Legend entries inside the frame's top left: (glyph, label, color), the glyph with its label just right of it.
    marker_legend: Sequence[tuple[str, str, tuple[float, float, float]]] = ()


def to_page(spec: PlotSpec, x: float, y: float) -> tuple[float, float]:
    f = spec.frame
    return (f.x0 + spec.x.frac(x) * f.width, f.y1 - spec.y.frac(y) * f.height)


def _label(v: float) -> str:
    return f'{v:g}'


def _centered(page: pymupdf.Page, text: str, cx: float, cy: float, size: float = FONT_SIZE) -> None:
    width = pymupdf.get_text_length(text, fontname='helv', fontsize=size)
    page.insert_text((cx - width / 2, cy + size * 0.35), text, fontname='helv', fontsize=size)


def _power_label(
    page: pymupdf.Page,
    value: float,
    right: float,
    cy: float,
    *,
    vector_minus: bool,
    span_minus: bool = False,
    minus_glyph: str = '-',
    exponent_gap: float = 0.0,
) -> None:
    """``10`` and a raised exponent right-aligned on ``right``; a value between decades is its bare mantissa digit."""
    decade = 10.0 ** math.floor(math.log10(value) + 1e-9)
    if not math.isclose(value, decade, rel_tol=1e-9):  # a minor label, as SIFERRIT sheets print 5 x 10^n: ``5``
        text = f'{value / decade:g}'
        width = pymupdf.get_text_length(text, fontname='helv', fontsize=FONT_SIZE)
        _centered(page, text, right - width / 2, cy)
        return
    exp = str(round(math.log10(value)))
    small = FONT_SIZE * 0.6
    dash = vector_minus and exp.startswith('-')
    glyph = span_minus and exp.startswith('-')
    shown = exp[1:] if dash or glyph else exp
    base_w = pymupdf.get_text_length('10', fontname='helv', fontsize=FONT_SIZE)
    dash_w = 3.5 if dash else pymupdf.get_text_length(minus_glyph, fontname='tiro', fontsize=small) if glyph else 0.0
    exp_w = pymupdf.get_text_length(shown, fontname='helv', fontsize=small)
    x0 = right - base_w - exponent_gap - dash_w - exp_w
    page.insert_text((x0, cy + FONT_SIZE * 0.35), '10', fontname='helv', fontsize=FONT_SIZE)
    if dash:
        page.draw_line((x0 + base_w + 0.5, cy - 2.6), (x0 + base_w + 3.0, cy - 2.6), color=(0, 0, 0), width=0.4)
    if glyph:
        page.insert_text(
            (x0 + base_w + exponent_gap, cy - FONT_SIZE * 0.3), minus_glyph, fontname='tiro', fontsize=small
        )
    page.insert_text(
        (x0 + base_w + exponent_gap + dash_w, cy - FONT_SIZE * 0.3), shown, fontname='helv', fontsize=small
    )


def _glyphs(page: pymupdf.Page, text: str, cx: float, cy: float) -> None:
    """``text`` centered on ``cx``, one span per glyph: neighbors alternate font, so no reader merges them."""
    fonts = ('helv', 'cour')
    widths = [pymupdf.get_text_length(ch, fontname=fonts[i % 2], fontsize=FONT_SIZE) for i, ch in enumerate(text)]
    x = cx - sum(widths) / 2
    for i, (ch, w) in enumerate(zip(text, widths, strict=True)):
        page.insert_text((x, cy + FONT_SIZE * 0.35), ch, fontname=fonts[i % 2], fontsize=FONT_SIZE)
        x += w


#: Stroke-font digit height (points) and the gap between its glyphs, as the SIFERRIT sheets set them.
STROKE_HEIGHT = 7.0
STROKE_GAP = 1.2
#: Width of the hairline every stroke-font glyph is drawn with (points).
STROKE_WIDTH = 0.4


def _stroke_lines(char: str) -> tuple[list[list[tuple[float, float]]], tuple[float, float]]:
    """``char``'s polylines (unit height, y down from its top) and its (top, bottom) place in digit heights."""
    place, _, drawn = TEMPLATES[char]
    lines = [[tuple(float(v) / 100 for v in p.split(',')) for p in line.split()] for line in drawn.split(';')]
    return lines, place


def stroke_width(text: str, height: float = STROKE_HEIGHT) -> float:
    """The width :func:`stroke_text` draws ``text`` in."""
    widths = [height * 0.5 if c == '-' else height * 0.12 if c == '.' else _glyph_width(c, height) for c in text]
    return sum(widths) + STROKE_GAP * (len(text) - 1)


def _glyph_width(char: str, height: float) -> float:
    lines, (top, bottom) = _stroke_lines(char)
    return max(x for line in lines for x, _ in line) * height * (top - bottom)


def stroke_text(
    page: pymupdf.Page,
    text: str,
    x0: float,
    baseline: float,
    height: float = STROKE_HEIGHT,
    *,
    pieces: int = 2,
    width: float = STROKE_WIDTH,
) -> float:
    """``text`` in the stroke font from ``x0`` on ``baseline``: each glyph's polylines stroked as ``pieces`` hairlines.

    A minus is a thin closed rectangle at mid height and a point a small closed square on the baseline, as the
    SIFERRIT sheets draw them. Returns the x the text ends at.
    """
    x = x0
    for char in text:
        if char in '-.':
            w, h = (height * 0.5, height * 0.045) if char == '-' else (height * 0.12, height * 0.12)
            y = baseline - height / 2 if char == '-' else baseline - h
            page.draw_polyline(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], color=(0, 0, 0), width=width, closePath=True
            )
            x += w + STROKE_GAP
            continue
        lines, (top, bottom) = _stroke_lines(char)
        size = height * (top - bottom)
        y0 = baseline - height * top
        for line in lines:
            points = [(x + px * size, y0 + py * size) for px, py in line]
            step = max(1, math.ceil((len(points) - 1) / pieces))
            for start in range(0, max(1, len(points) - 1), step):
                piece = points[start : start + step + 1]
                if len(piece) > 1:
                    page.draw_polyline(piece, color=(0, 0, 0), width=width)
        x += _glyph_width(char, height) + STROKE_GAP
    return x - STROKE_GAP


def _stroke_power(page: pymupdf.Page, value: float, right: float, cy: float) -> None:
    """``10`` and a raised stroke-font exponent right-aligned on ``right``, as SIFERRIT N72 sets a log axis."""
    exp = str(round(math.log10(value)))
    small = STROKE_HEIGHT * 0.8
    x0 = right - stroke_width('10') - 1.5 - stroke_width(exp, small)
    end = stroke_text(page, '10', x0, cy + STROKE_HEIGHT / 2)
    stroke_text(page, exp, end + 1.5, cy + STROKE_HEIGHT / 2 - 0.4 * STROKE_HEIGHT, small)


def _minor(ticks: Sequence[float]) -> list[float]:
    return [k * 10**e for e in range(round(math.log10(ticks[0])), round(math.log10(ticks[-1]))) for k in range(2, 10)]


def _draw_frame(page: pymupdf.Page, spec: PlotSpec) -> None:
    f = spec.frame
    if spec.border is not None:
        page.draw_rect(spec.border, color=(0, 0, 0), width=0.8)
    if spec.overhang:
        o = spec.overhang
        lines = ((f.tl, f.tr), ((f.x0 - o, f.y1), f.br), (f.tl, (f.x0, f.y1 + o)), (f.tr, f.br))
        for a, b in lines:
            page.draw_line(a, b, color=(0, 0, 0), width=0.8)
    elif spec.filled_frame:
        page.draw_rect(f, color=None, fill=(0.95, 0.95, 0.95), width=0)
    elif spec.frame_as_lines:
        for a, b in ((f.tl, f.tr), (f.bl, f.br), (f.tl, f.bl), (f.tr, f.br)):
            page.draw_line(a, b, color=(0, 0, 0), width=0.8)
    else:
        page.draw_rect(f, color=(0, 0, 0), width=0.8)
    if spec.minor_grid:
        for v in _minor(spec.x.ticks) if spec.x.log else ():
            px, _ = to_page(spec, v, spec.y.ticks[0])
            page.draw_line((px, f.y0), (px, f.y1), color=(0.7, 0.7, 0.7), width=0.3)
        for v in _minor(spec.y.ticks) if spec.y.log else ():
            _, py = to_page(spec, spec.x.ticks[0], v)
            page.draw_line((f.x0, py), (f.x1, py), color=(0.7, 0.7, 0.7), width=0.3)


def _minus_span(page: pymupdf.Page, text: str, cx: float, cy: float, glyph: str = '-') -> None:
    """``text`` centered on ``cx`` with its minus (``glyph``) set in another font, so it is a span of its own."""
    digits = text[1:]
    minus_w = pymupdf.get_text_length(glyph, fontname='tiro', fontsize=FONT_SIZE)
    width = pymupdf.get_text_length(digits, fontname='helv', fontsize=FONT_SIZE)
    start = cx - (minus_w + 0.5 + width) / 2
    page.insert_text((start, cy + FONT_SIZE * 0.35), glyph, fontname='tiro', fontsize=FONT_SIZE)
    page.insert_text((start + minus_w + 0.5, cy + FONT_SIZE * 0.35), digits, fontname='helv', fontsize=FONT_SIZE)


def _x_label(page: pymupdf.Page, spec: PlotSpec, text: str, px: float) -> None:
    """A label centered on its tick; a drawn minus is part of the centered whole, as typeset labels are."""
    y = spec.frame.y1 + 10
    if spec.stroke_font:
        stroke_text(page, text, px - stroke_width(text) / 2, y + STROKE_HEIGHT / 2)
        return
    if spec.x.glyph_spans:
        _glyphs(page, text, px, y)
        return
    if spec.x.span_minus and text.startswith('-'):
        _minus_span(page, text, px, y, spec.x.minus_glyph)
        return
    if not (spec.x.vector_minus and text.startswith('-')):
        _centered(page, text, px, y)
        return
    digits = text[1:]
    width = pymupdf.get_text_length(digits, fontname='helv', fontsize=FONT_SIZE)
    lead = 2.5 + spec.x.minus_gap
    start = px - (lead + width) / 2
    page.draw_line((start, y - 0.5), (start + 2.5, y - 0.5), color=(0, 0, 0), width=0.5)
    _centered(page, digits, start + lead + width / 2, y)


def _y_axis(page: pymupdf.Page, spec: PlotSpec) -> None:
    f = spec.frame
    for i, v in enumerate(spec.y.ticks):
        _, py = to_page(spec, spec.x.ticks[0], v)
        if spec.tick_marks:
            page.draw_line((f.x0, py), (f.x0 + 4, py), color=(0, 0, 0), width=0.5)
        if spec.grid and not (spec.filled_frame and i in (0, len(spec.y.ticks) - 1)):
            page.draw_line((f.x0, py), (f.x0 + spec.grid_reach * f.width, py), color=(spec.grid_grey,) * 3, width=0.3)
        py += spec.y.label_drop
        text = spec.y.labels[i] if spec.y.labels else _label(v)
        if spec.stroke_font and spec.y.power_labels:
            _stroke_power(page, v, f.x0 - 4, py)
            continue
        if spec.stroke_font:
            stroke_text(page, text, f.x0 - 4 - stroke_width(text), py + STROKE_HEIGHT / 2)
            continue
        if spec.y.power_labels:
            _power_label(
                page,
                v,
                f.x0 - 4,
                py,
                vector_minus=spec.y.vector_minus,
                span_minus=spec.y.span_minus,
                minus_glyph=spec.y.minus_glyph,
                exponent_gap=spec.y.exponent_gap,
            )
            continue
        width = pymupdf.get_text_length(text, fontname='helv', fontsize=FONT_SIZE)
        if spec.y.span_minus and text.startswith('-'):
            _minus_span(page, text, f.x0 - 4 - width / 2, py, spec.y.minus_glyph)
            continue
        _centered(page, text, f.x0 - 4 - width / 2, py)


def draw_plot(page: pymupdf.Page, spec: PlotSpec) -> None:
    f = spec.frame
    _draw_frame(page, spec)
    for i, v in enumerate(spec.x.ticks):
        px, _ = to_page(spec, v, spec.y.ticks[0])
        if spec.tick_marks:
            page.draw_line((px, f.y1), (px, f.y1 - 4), color=(0, 0, 0), width=0.5)
        if spec.grid and not (spec.filled_frame and i in (0, len(spec.x.ticks) - 1)):
            cuts = [f.y0, *(f.y0 + k * f.height for gap in spec.grid_gaps for k in gap), f.y1]
            for top, bottom in itertools.batched(cuts, 2):
                page.draw_line((px, top), (px, bottom), color=(spec.grid_grey,) * 3, width=0.3)
        _x_label(page, spec, spec.x.labels[i] if spec.x.labels else _label(v), px)
    _y_axis(page, spec)
    for text, x, y, size in spec.stray:  # a font object sets every character Helvetica has (``Ω``), not only Latin-1
        writer = pymupdf.TextWriter(page.rect)
        writer.append((x, y), text, font=pymupdf.Font('helv'), fontsize=size)
        writer.write_text(page)
    if spec.y_title is not None:
        text, x, y, size = spec.y_title
        page.insert_text((x, y), text, fontname='helv', fontsize=size, rotate=90)
    for fn, xs, color, *dashes in spec.curves:
        points = [to_page(spec, x, fn(x)) for x in xs]
        step = math.ceil((len(points) - 1) / spec.fragments)
        for k, start in enumerate(range(0, len(points) - 1, step)):
            piece = points[start : start + step + 1]
            page.draw_polyline(piece[::-1] if k % 2 else piece, color=color, width=1.0, dashes=next(iter(dashes), None))
    for k, (text, color, dashes) in enumerate(spec.legend):
        y = spec.frame.y0 + 12 + 12 * k
        x = spec.frame.x1 - 80
        page.draw_line((x, y), (x + 20, y), color=color, width=1.0, dashes=dashes)
        page.insert_text((x + 24, y + FONT_SIZE * 0.35), text, fontname='helv', fontsize=FONT_SIZE)
    _draw_marks(page, spec)


@functools.cache
def _ink(glyph: str) -> tuple[float, float]:
    """The center of ``glyph``'s ink at `MARK_SIZE` relative to its text origin, measured on a rendered pixmap.

    Rendering measures the drawn shape without the reader, so a planted center is not the reader's own answer.
    """
    zoom, origin = 40, (10.0, 20.0)
    doc = pymupdf.open()
    page = doc.new_page(width=30, height=30)
    page.insert_font(fontname=MARK_FONT, fontbuffer=pymupdf.Font('cjk').buffer)
    page.insert_text(origin, glyph, fontname=MARK_FONT, fontsize=MARK_SIZE)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    ink = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w) < 128  # mid-gray
    rows, cols = np.nonzero(ink)
    center = ((cols.min() + cols.max() + 1) / 2 / zoom, (rows.min() + rows.max() + 1) / 2 / zoom)
    return (center[0] - origin[0], center[1] - origin[1])


def mark_box_height() -> float:
    """The height (points) of a marker glyph's font box at `MARK_SIZE`: its font's descent to its ascent."""
    font = pymupdf.Font('cjk')
    return MARK_SIZE * (font.ascender - font.descender)


def draw_mark(page: pymupdf.Page, glyph: str, cx: float, cy: float, color: tuple[float, float, float]) -> None:
    """``glyph`` set as text with its ink centered on (``cx``, ``cy``)."""
    dx, dy = _ink(glyph)
    page.insert_font(fontname=MARK_FONT, fontbuffer=pymupdf.Font('cjk').buffer)
    page.insert_text((cx - dx, cy - dy), glyph, fontname=MARK_FONT, fontsize=MARK_SIZE, color=color)


def _draw_marks(page: pymupdf.Page, spec: PlotSpec) -> None:
    for glyph, points, color in spec.markers:
        for x, y in points:
            draw_mark(page, glyph, *to_page(spec, x, y), color)
    for k, (glyph, text, color) in enumerate(spec.marker_legend):
        y = spec.frame.y0 + 12 + 12 * k
        x = spec.frame.x0 + 20
        draw_mark(page, glyph, x, y, color)
        page.insert_text((x + 5, y + FONT_SIZE * 0.35), text, fontname='helv', fontsize=FONT_SIZE)


def plot_pdf(path: Path, *specs: PlotSpec, outlined: bool = False) -> Path:
    """One page per spec; ``outlined`` sets every glyph as a filled outline (no text), as some sheets are exported."""
    doc = pymupdf.open()
    for spec in specs:
        draw_plot(doc.new_page(width=612, height=792), spec)
    if outlined:
        out = pymupdf.open()
        for page in doc:
            svg = pymupdf.open('svg', page.get_svg_image(text_as_path=True).encode())
            out.insert_pdf(pymupdf.open('pdf', svg.convert_to_pdf()))
        doc.close()
        doc = out
    doc.save(path)
    doc.close()
    return path


#: Points around the frame a rasterized plot keeps, so its tick labels are inside the image.
RASTER_MARGIN = 40.0


def raster_pdf(
    path: Path, spec: PlotSpec | None, dpi: float, margin: float = RASTER_MARGIN
) -> tuple[Path, pymupdf.Rect]:
    """A page whose only content is ``spec``'s plot rendered to pixels at ``dpi`` (blank when ``spec`` is None).

    The image keeps ``margin`` points around the frame left and below, half of it above and right. Returns the PDF and
    the image's rectangle on the page.
    """
    clip = pymupdf.Rect(FRAME.x0 - margin, FRAME.y0 - margin / 2, FRAME.x1 + margin / 2, FRAME.y1 + margin)
    with pymupdf.open() as source:
        page = source.new_page(width=612, height=792)
        if spec is not None:
            draw_plot(page, spec)
        pix = page.get_pixmap(clip=clip, dpi=dpi, alpha=False)
    doc = pymupdf.open()
    doc.new_page(width=612, height=792).insert_image(clip, pixmap=pix)
    doc.save(path)
    doc.close()
    return path, clip


def image_pdf(
    path: Path,
    stored: bytes,
    image: str,
    *,
    size: tuple[float, float],
    matrix: Sequence[float] | None = None,
    deflate: bool = True,
    rotate: int = 0,
    soft_mask: bytes | None = None,
    profile: tuple[bytes, int] | None = None,
    over: Callable[[pymupdf.Page], object] | None = None,
) -> Path:
    """A one-page PDF of ``size`` points placing one image XObject, saved to ``path``.

    ``image`` is the image dictionary's own entries (``/Width 4 /Height 2 /ColorSpace /DeviceGray ...``), ``stored``
    its stream: deflated when ``deflate``, else stored as given (a codec's ``/Filter`` stated in ``image``).
    ``matrix`` is the ``cm`` placing it (PDF space, y up; default the whole page, upright); ``rotate`` the page's
    ``/Rotate``; ``soft_mask`` the 8-bit gray samples of an ``/SMask`` its size; ``profile`` an ICC profile and its
    component count, the image's ``/ColorSpace`` an ``ICCBased`` space of it; ``over`` draws on the page after.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    if soft_mask is not None:
        mask = doc.get_new_xref()
        width, height = (re.search(rf'/{k} (\d+)', image)[1] for k in ('Width', 'Height'))  # type: ignore[index]
        entries = f'/Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8'
        doc.update_object(mask, f'<< /Type /XObject /Subtype /Image {entries} >>')
        doc.update_stream(mask, soft_mask)
        image += f' /SMask {mask} 0 R'
    if profile is not None:
        icc = doc.get_new_xref()
        doc.update_object(icc, f'<< /N {profile[1]} >>')
        doc.update_stream(icc, profile[0])
        image += f' /ColorSpace [/ICCBased {icc} 0 R]'
    xobject = doc.get_new_xref()
    doc.update_object(xobject, f'<< /Type /XObject /Subtype /Image {image} >>')
    doc.update_stream(xobject, stored, compress=deflate)
    if not deflate:  # storing it as given drops the dictionary's /Filter and its parameters: state them again
        doc.update_object(xobject, f'<< /Type /XObject /Subtype /Image {image} /Length {len(stored)} >>')
    content = doc.get_new_xref()
    doc.update_object(content, '<< >>')
    cm = ' '.join(f'{v:g}' for v in (matrix or (size[0], 0, 0, size[1], 0, 0)))
    doc.update_stream(content, f'q {cm} cm /Im1 Do Q\n'.encode())
    doc.xref_set_key(page.xref, 'Resources', f'<< /XObject << /Im1 {xobject} 0 R >> >>')
    doc.xref_set_key(page.xref, 'Contents', f'{content} 0 R')
    if rotate:
        page.set_rotation(rotate)
    if over is not None:
        over(page)
    doc.save(path)
    doc.close()
    return path


#: CIE XYZ (D65) to linear sRGB (IEC 61966-2-1).
XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
#: The Bradford cone response matrix.
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
#: The ICC profile connection space's white, D50 (ICC.1 7.2.16).
D50 = np.array([0.9642, 1.0, 0.8249])


def _to_xyz(primaries: Sequence[tuple[float, float]], white: tuple[float, float]) -> np.ndarray:
    """The matrix from linear RGB to CIE XYZ (its white's) of RGB ``primaries`` and ``white`` given as CIE xy."""
    xyz = lambda x, y: np.array([x / y, 1.0, (1 - x - y) / y])  # noqa: E731 -- a formula
    columns = np.column_stack([xyz(*p) for p in primaries])
    return columns * np.linalg.solve(columns, xyz(*white))


def rgb_in_srgb(
    samples: np.ndarray, *, primaries: Sequence[tuple[float, float]], white: tuple[float, float], gamma: float
) -> np.ndarray:
    """8-bit samples of the RGB space of ``primaries``, ``white`` and a pure ``gamma`` as 8-bit sRGB, by the formulas.

    The white must be D65's (sRGB's): no adaptation is made. The truth a color-managed reading is held to.
    """
    linear = (samples / 255.0) ** gamma @ (XYZ_TO_SRGB @ _to_xyz(primaries, white)).T
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.round(encoded * 255).astype(np.uint8)


def icc_profile(*, primaries: Sequence[tuple[float, float]], white: tuple[float, float], gamma: float) -> bytes:
    """An ICC v2 RGB display profile, matrix and TRC (ICC.1:2001-04), of ``primaries`` and ``white`` (CIE xy) and a
    pure ``gamma`` for each channel: its colorants Bradford-adapted to D50, its white point D50 as v2 states it.
    """
    to_xyz = _to_xyz(primaries, white)
    source = BRADFORD @ np.array([white[0] / white[1], 1.0, (1 - white[0] - white[1]) / white[1]])
    adapt = np.linalg.inv(BRADFORD) @ np.diag(BRADFORD @ D50 / source) @ BRADFORD
    colorants = adapt @ to_xyz

    def s15(values: Sequence[float]) -> bytes:
        return b''.join(round(v * 65536).to_bytes(4, 'big', signed=True) for v in values)

    def xyz(values: Sequence[float]) -> bytes:
        return b'XYZ ' + bytes(4) + s15(values)

    curve = b'curv' + bytes(4) + (1).to_bytes(4, 'big') + round(gamma * 256).to_bytes(2, 'big') + bytes(2)
    text = b'Synthetic RGB'
    description = b'desc' + bytes(4) + (len(text) + 1).to_bytes(4, 'big') + text + bytes(1) + bytes(8) + bytes(3 + 67)
    tags = [
        (b'desc', description),
        (b'cprt', b'text' + bytes(4) + b'No copyright\0'),
        (b'wtpt', xyz(D50)),
        (b'rXYZ', xyz(colorants[:, 0])),
        (b'gXYZ', xyz(colorants[:, 1])),
        (b'bXYZ', xyz(colorants[:, 2])),
        (b'rTRC', curve),
        (b'gTRC', curve),
        (b'bTRC', curve),
    ]
    table, data, offset = b'', b'', 128 + 4 + 12 * len(tags)
    for signature, body in tags:
        table += signature + (offset + len(data)).to_bytes(4, 'big') + len(body).to_bytes(4, 'big')
        data += body + bytes(-len(body) % 4)
    size = offset + len(data)
    header = (
        size.to_bytes(4, 'big')
        + bytes(4)
        + bytes([2, 0x10, 0, 0])
        + b'mntrRGB XYZ '
        + bytes(12)
        + b'acsp'
        + bytes(24)
        + bytes(4)  # perceptual
        + s15(D50)
        + bytes(4 + 16 + 28)
    )
    return header + len(tags).to_bytes(4, 'big') + table + data


def jpx(samples: np.ndarray, *, codestream: bool = False, profile: bytes | None = None) -> bytes:
    """``samples`` -- rows x columns (gray, or 16-bit gray as ``uint16``), or rows x columns x 2, 3 or 4 (gray and
    alpha, RGB, RGB and alpha) -- coded JPEG 2000 losslessly (the reversible 5/3 wavelet) by OpenJPEG, through
    Pillow's plugin: a JP2 file (its color space enumerated, gray or sRGB), or the bare ``codestream``, or -- with
    ``profile`` -- that codestream in a JP2 whose ``colr`` box embeds the ICC profile (T.800 I.5.3.3, method 2).
    """
    from io import BytesIO  # noqa: PLC0415 -- only the JPEG 2000 cases need it

    from PIL import Image  # noqa: PLC0415 -- Pillow is the dev extra's, the truth of these cases alone

    image = Image.fromarray(samples)  # its mode by its shape and type: L, LA, RGB, RGBA or I;16
    out = BytesIO()
    image.save(out, 'JPEG2000', irreversible=False, no_jp2=codestream or profile is not None)
    coded = out.getvalue()
    if profile is None:
        return coded

    def box(kind: bytes, body: bytes) -> bytes:
        return (8 + len(body)).to_bytes(4, 'big') + kind + body

    rows, columns = samples.shape[:2]
    components = 1 if samples.ndim == 2 else samples.shape[2]
    header = rows.to_bytes(4, 'big') + columns.to_bytes(4, 'big') + components.to_bytes(2, 'big') + bytes([7, 7, 0, 0])
    colr = bytes([2, 0, 0]) + profile
    return (
        box(b'jP  ', b'\r\n\x87\n')
        + box(b'ftyp', b'jp2 ' + bytes(4) + b'jp2 ')
        + box(b'jp2h', box(b'ihdr', header) + box(b'colr', colr))
        + box(b'jp2c', coded)
    )


def fax_g4(white: np.ndarray) -> bytes:
    """``white`` -- rows of 1 white, 0 black, as CCITT fax decodes by default -- coded Group 4 by MuPDF."""
    rows, columns = white.shape
    packed = np.packbits(white.astype(np.uint8), axis=1).tobytes()
    stride = len(packed) // rows
    coded = pymupdf.mupdf.fz_compress_ccitt_fax_g4(pymupdf.mupdf.python_buffer_data(packed), columns, rows, stride)
    return bytes(coded.fz_buffer_extract())


def jbig2_mmr(white: np.ndarray) -> bytes:
    """``white`` -- rows of 1 white, 0 black, as a ``JBIG2Decode`` image's samples read -- as an embedded JBIG2 stream
    (ITU-T T.88 Annex D.3): a page information segment and an immediate lossless generic region, MMR-coded (6.2.6:
    its data is T.6, the coding of :func:`fax_g4`, by MuPDF; a T.6 white run is a JBIG2 0, the filter's 1)."""
    rows, columns = white.shape

    def segment(number: int, kind: int, data: bytes) -> bytes:
        # 7.2: number, flags (type; a one-byte page association), no referred-to segment, page 1, data length.
        return number.to_bytes(4, 'big') + bytes([kind, 0, 1]) + len(data).to_bytes(4, 'big') + data

    size = columns.to_bytes(4, 'big') + rows.to_bytes(4, 'big')
    page = size + bytes(8) + bytes([1]) + bytes(2)  # 7.4.8: no resolution, lossless, white, OR, not striped
    region = size + bytes(8) + bytes([0]) + bytes([1])  # 7.4.1 at (0, 0), OR; 7.4.6.2: MMR
    return segment(0, 48, page) + segment(1, 38, region + fax_g4(white))


TABLE_ROWS = (('Parameter', 'Value', 'Unit'), ('alpha', '1.25', 'mm'), ('beta', '-3', 'kg'), ('gamma', '4.5e2', 's'))
TABLE_ORIGIN = (72.0, 400.0)
TABLE_CELL = (120.0, 20.0)


def table_cell_rect(row: int, col: int) -> pymupdf.Rect:
    x0, y0 = TABLE_ORIGIN
    w, h = TABLE_CELL
    return pymupdf.Rect(x0 + col * w, y0 + row * h, x0 + (col + 1) * w, y0 + (row + 1) * h)


def draw_table(page: pymupdf.Page, rows: Sequence[Sequence[str]] = TABLE_ROWS) -> None:
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            rect = table_cell_rect(r, c)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), text, fontname='helv', fontsize=9)


#: A table with merged cells: ``Parameter`` spans rows 0-1, ``Limits`` columns 1-2 -- each slot it covers repeats it.
MERGED_ROWS = (
    ('Parameter', 'Limits', 'Limits'),
    ('Parameter', 'min', 'max'),
    ('alpha', '1.0', '2.5'),
    ('beta', '-3', '4'),
)
#: ``MERGED_ROWS``' merged blocks: (first row, first column, last row, last column).
MERGED = ((0, 0, 1, 0), (0, 1, 0, 2))


def _merged_block(merged: Sequence[tuple[int, int, int, int]], r: int, c: int) -> tuple[int, int, int, int]:
    return next((b for b in merged if b[0] <= r <= b[2] and b[1] <= c <= b[3]), (r, c, r, c))


def _rule(page: pymupdf.Page, a: pymupdf.Point, b: pymupdf.Point, *, filled: bool, width: float) -> None:
    if filled:
        half = width / 2
        box = pymupdf.Rect(min(a.x, b.x) - half, min(a.y, b.y) - half, max(a.x, b.x) + half, max(a.y, b.y) + half)
        page.draw_rect(box, color=None, fill=(0, 0, 0), width=0)
    else:
        page.draw_line(a, b, color=(0, 0, 0), width=width)


def draw_ruled_table(
    page: pymupdf.Page,
    rows: Sequence[Sequence[str]] = TABLE_ROWS,
    *,
    merged: Sequence[tuple[int, int, int, int]] = (),
    filled: bool = False,
    width: float = 0.5,
) -> None:
    """``rows`` as one grid of rules, each cell side drawn once: a stroked line or (``filled``) a thin filled bar.

    No rule is drawn inside a ``merged`` block (first row, first column, last row, last column); its text is set once,
    in its top-left slot.
    """
    n_rows, n_cols = len(rows), len(rows[0])
    for r in range(n_rows + 1):
        for c in range(n_cols):
            if r in {0, n_rows} or _merged_block(merged, r - 1, c) != _merged_block(merged, r, c):
                cell = table_cell_rect(min(r, n_rows - 1), c)
                y = cell.y1 if r == n_rows else cell.y0
                _rule(page, pymupdf.Point(cell.x0, y), pymupdf.Point(cell.x1, y), filled=filled, width=width)
    for r in range(n_rows):
        for c in range(n_cols + 1):
            if c in {0, n_cols} or _merged_block(merged, r, c - 1) != _merged_block(merged, r, c):
                cell = table_cell_rect(r, min(c, n_cols - 1))
                x = cell.x1 if c == n_cols else cell.x0
                _rule(page, pymupdf.Point(x, cell.y0), pymupdf.Point(x, cell.y1), filled=filled, width=width)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            if _merged_block(merged, r, c)[:2] == (r, c) and text:
                rect = table_cell_rect(r, c)
                page.insert_text((rect.x0 + 4, rect.y1 - 6), text, fontname='helv', fontsize=9)


#: Booktabs' rules: the top and bottom rules heavier than the one under the header (0.08 em and 0.05 em at 10 pt).
TOP_RULE, MID_RULE = 0.8, 0.5


def draw_booktabs(
    page: pymupdf.Page,
    rows: Sequence[Sequence[str]] = TABLE_ROWS,
    *,
    at: tuple[float, float] = TABLE_ORIGIN,
    filled: bool = True,
    rules: Sequence[int] | None = None,
) -> pymupdf.Rect:
    """``rows`` as booktabs sets them: no rule down, a top rule, one under the header row, a bottom rule.

    Rules are thin filled bars, as pdfLaTeX draws them (or ``filled`` False, stroked lines), across every column;
    columns are parted by white space alone, each cell's text set :data:`TABLE_CELL` apart from ``at``. ``rules``:
    the row boundaries ruled (0 the top, ``len(rows)`` the bottom), by default top, under the header and bottom.
    Returns the box from the top rule to the bottom one.
    """
    x0, y0 = at
    w, h = TABLE_CELL
    right = x0 + len(rows[0]) * w
    for r in (0, 1, len(rows)) if rules is None else rules:
        thick = MID_RULE if 0 < r < len(rows) else TOP_RULE
        _rule(page, pymupdf.Point(x0, y0 + r * h), pymupdf.Point(right, y0 + r * h), filled=filled, width=thick)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            if text:
                page.insert_text((x0 + c * w + 4, y0 + (r + 1) * h - 6), text, fontname='helv', fontsize=9)
    return pymupdf.Rect(x0, y0, right, y0 + len(rows) * h)


#: A table ruled under every row whose cells wrap, each cell its lines: row 1 holds cells of one, two and three lines,
#: row 2 two cells of two lines beside short ones (alone, rows of their own to the eye), row 3 one line each.
WRAPPED_ROWS = (
    (('Ref.',), ('Domain',), ('Objective',), ('Remarks',)),
    (('[1]',), ('Stator',), ('Minimize the noise', 'and vibration'), ('Mesh dependence', 'was not studied', 'at all')),
    (('[2]',), ('Rotor',), ('Minimize torque', 'ripple'), ('Only half a pole', 'was modeled')),
    (('[3]',), ('Rotor poles',), ('Maximize torque',), ('N/A',)),
)
#: A wrapped table's columns, each this many points wide, and the step from one line of a cell to the next.
WRAPPED_WIDTHS, LINE_PITCH = (60.0, 80.0, 110.0, 110.0), 11.0


def draw_wrapped_rows(
    page: pymupdf.Page,
    rows: Sequence[Sequence[Sequence[str]]] = WRAPPED_ROWS,
    *,
    at: tuple[float, float] = TABLE_ORIGIN,
    first_ruled: bool = True,
) -> None:
    """``rows`` ruled under every row, as IEEE sets a table: each cell's lines centered down its row, as tall as its
    tallest cell, and a rule between every two rows -- under the first column too unless not ``first_ruled`` (the
    rules between rows then start at the second column, the first ruled apart: a reference beside its rows)."""
    x0, y = at
    right = x0 + sum(WRAPPED_WIDTHS)
    _rule(page, pymupdf.Point(x0, y), pymupdf.Point(right, y), filled=True, width=TOP_RULE)
    for r, row in enumerate(rows):
        height = max(len(cell) for cell in row) * LINE_PITCH + 6
        x = x0
        for cell, width in zip(row, WRAPPED_WIDTHS, strict=True):
            top = y + (height - len(cell) * LINE_PITCH) / 2
            for i, line in enumerate(cell):
                page.insert_text((x + 4, top + i * LINE_PITCH + 9), line, fontname='helv', fontsize=9)
            x += width
        y += height
        last = r == len(rows) - 1
        start = x0 if r == 0 or last or first_ruled else x0 + WRAPPED_WIDTHS[0]
        thick = TOP_RULE if last else MID_RULE
        _rule(page, pymupdf.Point(start, y), pymupdf.Point(right, y), filled=True, width=thick)
        if not (first_ruled or r == 0 or last):  # the first column's own short rule
            _rule(page, pymupdf.Point(x0, y), pymupdf.Point(x0 + 20, y), filled=True, width=MID_RULE)


#: A table ruled between groups of rows: each group's label, set beside its rows and centered down them, then its rows.
GROUPED_HEADER = ('Data', 'Split', 'P', 'R')
GROUPED_ROWS = (
    ('DI', (('DI', '0.847', '0.798'), ('AMP', '0.771', '0.677'), ('Guitar', '0.879', '0.849'))),
    ('AMP', (('DI', '0.853', '0.836'), ('AMP', '0.824', '0.781'))),
)


def draw_grouped_rows(page: pymupdf.Page, *, at: tuple[float, float] = TABLE_ORIGIN) -> None:
    """:data:`GROUPED_ROWS` as booktabs sets groups: a rule under the header and between groups, none between rows."""
    x0, y = at
    w = TABLE_CELL[0]
    right = x0 + len(GROUPED_HEADER) * w
    _rule(page, pymupdf.Point(x0, y), pymupdf.Point(right, y), filled=True, width=TOP_RULE)
    for c, text in enumerate(GROUPED_HEADER):
        page.insert_text((x0 + c * w + 4, y + 13), text, fontname='helv', fontsize=9)
    y += LINE_PITCH + 6
    for g, (label, rows) in enumerate(GROUPED_ROWS):
        _rule(page, pymupdf.Point(x0, y), pymupdf.Point(right, y), filled=True, width=MID_RULE)
        page.insert_text((x0 + 4, y + 3 + len(rows) * LINE_PITCH / 2 + 3.5), label, fontname='helv', fontsize=9)
        for r, row in enumerate(rows):
            for c, text in enumerate(row, start=1):
                page.insert_text((x0 + c * w + 4, y + 3 + r * LINE_PITCH + 9), text, fontname='helv', fontsize=9)
        y += len(rows) * LINE_PITCH + 6
        if g == len(GROUPED_ROWS) - 1:
            _rule(page, pymupdf.Point(x0, y), pymupdf.Point(right, y), filled=True, width=TOP_RULE)


def draw_sideways_booktabs(page: pymupdf.Page, rows: Sequence[Sequence[str]] = TABLE_ROWS, turn: int = -90) -> None:
    """:func:`draw_booktabs`' table turned ``turn`` degrees clockwise about :data:`SCAN_PAGE`'s center, its text with
    it: a table ruled by its rows printed sideways (-90: read bottom to top)."""
    x0, y0 = TABLE_ORIGIN
    w, h = TABLE_CELL
    right = x0 + len(rows[0]) * w
    for r in (0, 1, len(rows)):
        y = y0 + r * h
        page.draw_line(turned(x0, y, turn), turned(right, y, turn), color=(0, 0, 0), width=MID_RULE)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            at = turned(x0 + c * w + 4, y0 + (r + 1) * h - 6, turn)
            page.insert_text(at, text, fontname='helv', fontsize=9, rotate=-turn % 360)


#: A list set between rules, as booktabs rules a table: a heading, then each item after its enumerator.
LIST_ROWS = (
    ('Steps', ''),
    ('1.', 'Mount the stator on the bench'),
    ('2.', 'Couple the load machine'),
    ('3.', 'Record the winding temperature'),
)


def mixed_pdf(path: Path) -> Path:
    """Page 1: a heading, a table and a raster image with a caption; page 2: one red polyline."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), 'Hello document', fontname='helv', fontsize=14)
    draw_table(page)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (0, 128, 255))
    page.insert_image(pymupdf.Rect(72, 600, 172, 700), pixmap=pix)
    page.insert_text((72, 715), 'Figure 1. A blue square.', fontname='helv', fontsize=9)
    page2 = doc.new_page(width=612, height=792)
    page2.draw_polyline([(100, 100), (200, 150), (300, 120)], color=RED, width=1.5)
    doc.save(path)
    doc.close()
    return path


#: A run of text on a synthetic page: its baseline's left end (x, y), its text, its PyMuPDF font name and its size.
type Run = tuple[float, float, str, str, float]


def text_pdf(path: Path, pages: Sequence[Sequence[Run]], bars: Sequence[Sequence[pymupdf.Rect]] = ()) -> Path:
    """Letter pages of runs of text, each where it is placed, and filled bars (a fraction's): every box is known."""
    doc = pymupdf.open()
    for number, runs in enumerate(pages):
        page = doc.new_page(width=612, height=792)
        for x, y, text, font, size in runs:
            page.insert_text((x, y), text, fontname=font, fontsize=size)
        for bar in bars[number] if number < len(bars) else ():
            page.draw_rect(bar, color=None, fill=(0, 0, 0))
    doc.save(path)
    doc.close()
    return path


#: A word of a text layer: its text, its box as the page shows it (points), the degrees its baseline is turned clockwise
#: as seen (-90: read bottom to top) and whether it is written vertically (read down a column).
type LayerWord = tuple[str, tuple[float, float, float, float], float, bool]


def layer_pdf(path: Path, words: Sequence[LayerWord], width: float = 612, height: float = 792) -> Path:
    """A blank page with a text layer laid by our own writer, as `document_lab.ocr` lays Tesseract's words: each word
    invisible over its box, along its turn -- boxes as loose or as tight as an OCR's, every one known."""
    from document_lab import _pdf  # noqa: PLC0415 -- the writer `document_lab.ocr` lays its layers with

    with pymupdf.open() as doc:
        doc.new_page(width=width, height=height)
        reader = _pdf.open(doc.tobytes(), b'')
    laid = []
    for text, (x0, y0, x1, y1), turn, vertical in words:
        across, down = (y1 - y0, x1 - x0) if turn % 180 else (x1 - x0, y1 - y0)  # the writer turns its box about
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2  # the center, from the word's own frame
        laid.append((text, (cx - across / 2, cy - down / 2, cx + across / 2, cy + down / 2), float(turn), vertical))
    writer = _pdf.create(path)
    writer.copy_page_with_text(reader, 1, laid)
    writer.close()
    return path


#: Lines top to bottom, each ``(before, drawn, actual, after)``: ``drawn`` is shown, marked with ``/ActualText``
#: ``actual``. A drawn space stated a zero-width space, a no-break space, a word joiner; the control: ``XY`` stated
#: the letters ``fi`` of its ligature, a replacement that is no space.
ACTUAL_TEXT_LINES = (
    ('80', ' ', '\u200b', 'V'),
    ('80', ' ', '\u00a0', 'V'),
    ('80', ' ', '\u2060', 'V'),
    ('Ex', 'XY', 'fi', 'nal'),
)


def actual_text_pdf(path: Path) -> Path:
    """One page of :data:`ACTUAL_TEXT_LINES`, 30 points apart, set in Helvetica 10."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    font = doc.get_new_xref()
    doc.update_object(font, '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    doc.xref_set_key(page.xref, 'Resources', f'<< /Font << /F1 {font} 0 R >> >>')
    content = '\n'.join(
        f'BT /F1 10 Tf 20 {160 - 30 * i} Td ({before}) Tj '
        f'/Span << /ActualText <FEFF{actual.encode("utf-16-be").hex()}> >> BDC ({drawn}) Tj EMC ({after}) Tj ET'
        for i, (before, drawn, actual, after) in enumerate(ACTUAL_TEXT_LINES)
    )
    stream = doc.get_new_xref()
    doc.update_object(stream, '<< >>')
    doc.update_stream(stream, content.encode('ascii'))
    doc.xref_set_key(page.xref, 'Contents', f'{stream} 0 R')
    doc.save(path)
    doc.close()
    return path


def cid_font_pdf(
    path: Path,
    cids: Sequence[int],
    *,
    registry: str = 'Adobe',
    ordering: str = 'Japan1',
    to_unicode: Mapping[int, str] | None = None,
) -> Path:
    """One page showing ``cids`` in one line, in Ryumin-Light on ``/Identity-H`` (each two-byte code its own CID).

    The font is a CID font on the collection ``registry``-``ordering`` that embeds no program, as a Japanese PDF sets
    text in a printer's resident font; it states no ``/ToUnicode`` unless ``to_unicode`` gives one (text by code).
    """
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=100)
    font, descendant = doc.get_new_xref(), doc.get_new_xref()
    doc.update_object(
        descendant,
        f'<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Ryumin-Light /CIDSystemInfo << /Registry ({registry}) '
        f'/Ordering ({ordering}) /Supplement 7 >> /DW 1000 /FontDescriptor << /Type /FontDescriptor '
        '/FontName /Ryumin-Light /Flags 6 /FontBBox [-170 -331 1024 903] /ItalicAngle 0 /Ascent 723 /Descent -241 '
        '/CapHeight 709 /StemV 69 >> >>',
    )
    stated = ''
    if to_unicode is not None:
        cmap = doc.get_new_xref()
        doc.update_object(cmap, '<< >>')
        pairs = ' '.join(f'<{code:04X}> <{text.encode("utf-16-be").hex()}>' for code, text in to_unicode.items())
        doc.update_stream(
            cmap,
            '/CIDInit /ProcSet findresource begin 12 dict begin begincmap 1 begincodespacerange <0000> <FFFF> '
            f'endcodespacerange {len(to_unicode)} beginbfchar {pairs} endbfchar endcmap CMapName currentdict /CMap '
            'defineresource pop end end'.encode('ascii'),
        )
        stated = f' /ToUnicode {cmap} 0 R'
    doc.update_object(
        font,
        '<< /Type /Font /Subtype /Type0 /BaseFont /Ryumin-Light-Identity-H /Encoding /Identity-H '
        f'/DescendantFonts [{descendant} 0 R]{stated} >>',
    )
    doc.xref_set_key(page.xref, 'Resources', f'<< /Font << /F1 {font} 0 R >> >>')
    stream = doc.get_new_xref()
    doc.update_object(stream, '<< >>')
    doc.update_stream(stream, f'BT /F1 12 Tf 20 50 Td <{"".join(f"{c:04X}" for c in cids)}> Tj ET'.encode('ascii'))
    doc.xref_set_key(page.xref, 'Contents', f'{stream} 0 R')
    doc.save(path)
    doc.close()
    return path


def hatch_pdf(path: Path) -> Path:
    """A 200 x 100 page: a blue 5 x 5 solid fill, then a 50 x 50 hatch -- an uncolored tiling pattern (a diagonal
    stroke, as TikZ's ``north east lines``) tinted red through ``[/Pattern /DeviceRGB]``: a pattern, not a solid fill.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    pattern = doc.get_new_xref()
    doc.update_object(pattern, '<< /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 >>')
    doc.update_stream(pattern, b'0 0 m 4 4 l S')
    doc.xref_set_key(
        page.xref, 'Resources', f'<< /ColorSpace << /C [/Pattern /DeviceRGB] >> /Pattern << /H {pattern} 0 R >> >>'
    )
    stream = doc.get_new_xref()
    doc.update_object(stream, '<< >>')
    doc.update_stream(stream, b'0 0 1 rg 0 0 5 5 re f /C cs 1 0 0 /H scn 10 10 50 50 re f')
    doc.xref_set_key(page.xref, 'Contents', f'{stream} 0 R')
    doc.save(path)
    doc.close()
    return path


def hidden_layer(page: Page) -> Page:
    """``page`` with everything it draws in a hidden layer: what a layer turned off in the default view reads as."""
    return replace(
        page,
        spans=tuple(replace(s, hidden=True) for s in page.spans),
        paths=tuple(replace(p, hidden=True) for p in page.paths),
        figures=tuple(replace(f, hidden=True) for f in page.figures),
    )


def tagged_pdf(
    path: Path, contents: Sequence[str], tree: Mapping[str, str], *, size: tuple[float, float] = PAGE
) -> Path:
    """A tagged PDF, written by PyMuPDF so the reader meets another writer's file.

    One page per content stream of ``contents`` (``size`` points), with Helvetica as ``/F1`` and Helvetica-Bold as
    ``/F2`` in its resources and ``/StructParents`` its 0-based index. ``tree`` holds the structure's objects by
    name: each source text may refer to another as ``{name}`` and to page n as ``{page<n>}``; ``tree['root']`` is
    the catalog's ``/StructTreeRoot``.
    """
    doc = pymupdf.open()
    xrefs = {name: doc.get_new_xref() for name in tree}
    refs = {name: f'{xref} 0 R' for name, xref in xrefs.items()}
    for i, page in enumerate(_content_pages(doc, contents, size)):
        doc.xref_set_key(page, 'StructParents', str(i))
        refs[f'page{i + 1}'] = f'{page} 0 R'
    for name, source in tree.items():
        doc.update_object(xrefs[name], source.format(**refs))
    doc.xref_set_key(doc.pdf_catalog(), 'StructTreeRoot', refs['root'])
    doc.xref_set_key(doc.pdf_catalog(), 'MarkInfo', '<< /Marked true >>')
    doc.save(path)
    doc.close()
    return path


def content_pdf(path: Path, contents: Sequence[str], *, size: tuple[float, float] = PAGE) -> Path:
    """A PDF whose pages draw exactly ``contents``, one content stream a page (``size`` points), Helvetica as ``/F1``
    and Helvetica-Bold as ``/F2`` in its resources: for what no PyMuPDF drawing call states (caps, joins, miter
    limits, dashes, text rendering modes)."""
    doc = pymupdf.open()
    _content_pages(doc, contents, size)
    doc.save(path)
    doc.close()
    return path


def _content_pages(doc: pymupdf.Document, contents: Sequence[str], size: tuple[float, float]) -> list[int]:
    """Pages added to ``doc`` drawing ``contents``, one a page, with :func:`content_pdf`'s fonts: their xrefs."""
    fonts = []
    for base in ('Helvetica', 'Helvetica-Bold'):
        fonts.append(doc.get_new_xref())
        doc.update_object(fonts[-1], f'<< /Type /Font /Subtype /Type1 /BaseFont /{base} /Encoding /WinAnsiEncoding >>')
    pages = []
    for content in contents:
        page = doc.new_page(width=size[0], height=size[1])
        stream = doc.get_new_xref()
        doc.update_object(stream, '<< >>')
        doc.update_stream(stream, content.encode('latin-1'))
        doc.xref_set_key(page.xref, 'Contents', f'{stream} 0 R')
        doc.xref_set_key(page.xref, 'Resources', f'<< /Font << /F1 {fonts[0]} 0 R /F2 {fonts[1]} 0 R >> >>')
        pages.append(page.xref)
    return pages


def in_background_color(page: Page) -> Page:
    """``page`` with every span filled in the color under it: what text drawn white on the white page reads as."""
    return replace(page, spans=tuple(replace(s, in_background_color=True) for s in page.spans))


#: The XMP packet :func:`document_data_pdf` embeds: modified a day after the ``/Info`` dates.
XMP_PACKET = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Synthetic (XMP)</rdf:li></rdf:Alt></dc:title>
<dc:creator><rdf:Seq><rdf:li>Ada</rdf:li><rdf:li>Grace</rdf:li></rdf:Seq></dc:creator>
<xmp:ModifyDate>2024-01-03T03:04:05Z</xmp:ModifyDate>
</rdf:Description></rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>"""


def document_data_pdf(path: Path) -> Path:
    """Three pages (the third turned a quarter) stating what a document states beside its pages.

    Its outline: Introduction (page 1), Method (page 2) with Setup under it, Results (page 3), each leading to its
    heading as displayed; labels ``i``, ``A-1``, ``A-2``; ``/Info`` and an XMP packet a day newer; on page 1 a URI
    link and a link to Setup, and a text field ``name`` = ``Ada``; on page 2 a checked box ``agree`` and an attached
    ``note.txt``; on page 3 a URI link; the document embeds ``data.csv``.
    """
    doc = pymupdf.open()
    for number, heading in enumerate(('1 Introduction', '2 Method', '3 Results'), start=1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), heading, fontname='helv', fontsize=14)
        if number == 2:  # the page with a subsection
            page.insert_text((72, 400), '2.1 Setup', fontname='helv', fontsize=12)
    doc[2].set_rotation(90)

    def to(page: int, y: float) -> dict:
        return {'kind': pymupdf.LINK_GOTO, 'page': page, 'to': pymupdf.Point(72, y)}

    doc.set_toc(
        [
            [1, 'Introduction', 1, to(0, 100)],
            [1, 'Method', 2, to(1, 100)],
            [2, 'Setup', 2, to(1, 400)],
            [1, 'Results', 3, to(2, 100)],
        ]
    )
    doc.set_page_labels(
        [
            {'startpage': 0, 'prefix': '', 'style': 'r', 'firstpagenum': 1},
            {'startpage': 1, 'prefix': 'A-', 'style': 'D', 'firstpagenum': 1},
        ]
    )
    doc.set_metadata(
        {
            'title': 'Synthetic',
            'author': 'Ada',
            'creationDate': "D:20240102030405+00'00'",
            'modDate': "D:20240102030405+00'00'",
        }
    )
    doc.set_xml_metadata(XMP_PACKET)
    doc[0].insert_link(
        {'kind': pymupdf.LINK_URI, 'from': pymupdf.Rect(72, 700, 200, 720), 'uri': 'https://example.org/'}
    )
    doc[0].insert_link({**to(1, 400), 'from': pymupdf.Rect(72, 650, 200, 670)})
    doc[2].insert_link(
        {'kind': pymupdf.LINK_URI, 'from': pymupdf.Rect(72, 100, 200, 120), 'uri': 'https://example.org/3'}
    )
    text = pymupdf.Widget()
    text.field_type, text.field_name, text.field_value = pymupdf.PDF_WIDGET_TYPE_TEXT, 'name', 'Ada'
    text.rect = pymupdf.Rect(72, 200, 272, 220)
    doc[0].add_widget(text)
    box = pymupdf.Widget()
    box.field_type, box.field_name, box.field_value = pymupdf.PDF_WIDGET_TYPE_CHECKBOX, 'agree', True
    box.rect = pymupdf.Rect(72, 300, 90, 318)
    doc[1].add_widget(box)
    doc.embfile_add('data.csv', b'a,b\n1,2\n', filename='data.csv', desc='numbers')
    doc[1].add_file_annot(pymupdf.Point(300, 300), b'hi', 'note.txt')
    doc.save(path)
    doc.close()
    return path


def shown(x: float, y: float, text: str, *, size: float = 10, font: str = 'F1') -> str:
    """A content-stream text object showing ``text`` with its baseline at ``(x, y)`` in page space (y down, on a
    :data:`PAGE`-high page)."""
    escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    return f'BT /{font} {size} Tf {x} {PAGE[1] - y} Td ({escaped}) Tj ET'


def marked(tag: str, mcid: int, body: str) -> str:
    """``body`` as the marked-content sequence ``mcid`` tagged ``tag``."""
    return f'/{tag} << /MCID {mcid} >> BDC {body} EMC'


#: The page a synthetic scan is of (Letter, as the tables above draw on); it is turned about its center.
SCAN_PAGE = pymupdf.Rect(0, 0, 612, 792)


def turned(x: float, y: float, degrees: float) -> tuple[float, float]:
    """The point of :data:`SCAN_PAGE` turned ``degrees`` clockwise as seen (y down) about its center."""
    cx, cy = SCAN_PAGE.width / 2, SCAN_PAGE.height / 2
    a = math.radians(degrees)
    return cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a), cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)


def scan_image(
    *draws: Callable[[pymupdf.Page], object], dpi: float = 300, angle: float = 0.0, noise: float = 0.0, seed: int = 0
) -> np.ndarray:
    """:data:`SCAN_PAGE` with each of ``draws`` drawn on it, as a scanner gives it: grey rows at ``dpi``.

    The page is turned ``angle`` degrees clockwise (:func:`turned`), as a sheet fed askew; ``noise`` scales the
    scanner's grain (Gaussian, 25 grey levels at 1), black specks (0.2 %) and white dropouts (3 %), from ``seed``.
    Rules broken on paper are drawn by ``draws`` (white boxes over them), so they turn with the page.
    """
    with pymupdf.open() as doc:
        page = doc.new_page(width=SCAN_PAGE.width, height=SCAN_PAGE.height)
        for draw in draws:
            draw(page)
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(angle)
        pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csGRAY, alpha=False)
    whole = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
    rows, columns = round(SCAN_PAGE.height * dpi / 72), round(SCAN_PAGE.width * dpi / 72)
    top, left = (whole.shape[0] - rows) // 2, (whole.shape[1] - columns) // 2  # the turned page's center kept
    grey = whole[top : top + rows, left : left + columns].astype(np.float64)
    if noise:
        rng = np.random.default_rng(seed)
        grey += rng.normal(0, 25 * noise, grey.shape)
        grey[rng.random(grey.shape) < 0.002 * noise] = 0
        grey[rng.random(grey.shape) < 0.03 * noise] = 255
    return np.clip(grey, 0, 255).astype(np.uint8)


def scan_pdf(path: Path, grey: np.ndarray) -> Path:
    """A page of :data:`SCAN_PAGE`'s size that is the one 8-bit grey image ``grey`` (:func:`scan_image`)."""
    rows, columns = grey.shape
    image = f'/Width {columns} /Height {rows} /ColorSpace /DeviceGray /BitsPerComponent 8'
    return image_pdf(path, grey.tobytes(), image, size=(SCAN_PAGE.width, SCAN_PAGE.height))


def broken_rules(page: pymupdf.Page, cut: float = 1.5) -> None:
    """White cuts ``cut`` points long across :func:`draw_ruled_table`'s rules, away from where rules cross."""
    x0, y0 = TABLE_ORIGIN
    for r in range(len(TABLE_ROWS) + 1):
        for x in (13, 58, 88, 158, 198, 268, 308, 338):
            page.draw_rect(
                pymupdf.Rect(x0 + x, y0 + 20 * r - 1, x0 + x + cut, y0 + 20 * r + 1), color=None, fill=(1, 1, 1)
            )
    for c in range(len(TABLE_ROWS[0]) + 1):
        for y in (5, 28, 51, 70):
            page.draw_rect(
                pymupdf.Rect(x0 + 120 * c - 1, y0 + y, x0 + 120 * c + 1, y0 + y + cut), color=None, fill=(1, 1, 1)
            )


def double_border(page: pymupdf.Page, apart: float = 2.5) -> None:
    """A second rule ``apart`` points round :func:`draw_ruled_table`'s outline: a double rule."""
    outline = table_cell_rect(0, 0) | table_cell_rect(len(TABLE_ROWS) - 1, len(TABLE_ROWS[0]) - 1)
    page.draw_rect(
        pymupdf.Rect(outline.x0 - apart, outline.y0 - apart, outline.x1 + apart, outline.y1 + apart),
        color=(0, 0, 0),
        width=0.5,
    )


def table_differences(
    table: Table,
    rows: Sequence[Sequence[str]],
    *,
    merged: Sequence[tuple[int, int, int, int]] = (),
    angle: float = 0.0,
    within: float = 0.5,
) -> tuple[list[str], list[str]]:
    """How a table found on a scan differs from the one :func:`draw_ruled_table` drew: ``(structure, text)``.

    Structure: its shape, which slots share a cell (``merged``), and each cell's box -- its center where
    :func:`turned` by ``angle`` puts the drawn cell's and its size the drawn cell's, each ``within`` points. Text:
    each cell whose text is not the drawn one's.
    """
    structure: list[str] = []
    if [len(r) for r in table.rows] != [len(r) for r in rows]:
        return [f'shape {[len(r) for r in table.rows]}, drawn {[len(r) for r in rows]}'], []
    boxes = table.cell_bboxes or ()
    for r, row in enumerate(rows):
        for c in range(len(row)):
            r0, c0, r1, c1 = _merged_block(merged, r, c)
            drawn = table_cell_rect(r0, c0) | table_cell_rect(r1, c1)
            x, y = turned((drawn.x0 + drawn.x1) / 2, (drawn.y0 + drawn.y1) / 2, angle)
            box = boxes[r][c]
            if box is None:
                structure.append(f'cell ({r}, {c}) has no box')
                continue
            off = max(abs((box.x0 + box.x1) / 2 - x), abs((box.y0 + box.y1) / 2 - y))
            size = max(abs(box.width - drawn.width), abs(box.height - drawn.height))
            if off > within or size > within:
                structure.append(f'cell ({r}, {c}) box {tuple(round(v, 1) for v in box)} off by {off:.1f} pt')
    text = [
        f'cell ({r}, {c}) reads {table.rows[r][c]!r}, drawn {row[c]!r}'
        for r, row in enumerate(rows)
        for c in range(len(row))
        if table.rows[r][c] != row[c]
    ]
    return structure, text


def draw_sideways_table(page: pymupdf.Page, rows: Sequence[Sequence[str]] = TABLE_ROWS, turn: int = -90) -> None:
    """:func:`draw_ruled_table`'s grid turned ``turn`` degrees clockwise about :data:`SCAN_PAGE`'s center, its text
    with it: a landscape table printed sideways on a portrait page (-90: read bottom to top, as standards set them)."""
    n_rows, n_cols = len(rows), len(rows[0])
    outline = table_cell_rect(0, 0) | table_cell_rect(n_rows - 1, n_cols - 1)
    for y in [outline.y0 + TABLE_CELL[1] * r for r in range(n_rows + 1)]:
        page.draw_line(turned(outline.x0, y, turn), turned(outline.x1, y, turn), color=(0, 0, 0), width=0.5)
    for x in [outline.x0 + TABLE_CELL[0] * c for c in range(n_cols + 1)]:
        page.draw_line(turned(x, outline.y0, turn), turned(x, outline.y1, turn), color=(0, 0, 0), width=0.5)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            rect = table_cell_rect(r, c)
            at = turned(rect.x0 + 4, rect.y1 - 6, turn)
            page.insert_text(at, text, fontname='helv', fontsize=9, rotate=-turn % 360)


#: A landscape table's caption above it and a note below it, each ``(text, baseline origin)`` in the table's own frame
#: (:func:`draw_sideways_table`'s before it is turned): set turned with it, they read bottom to top.
LANDSCAPE_CAPTION = ('Table 10 - Magnetic properties of the sintered grades', (72.0, 385.0))
LANDSCAPE_NOTE = ('NOTE The values are typical of the grades listed above.', (72.0, 505.0))
#: The upright running head over it: two spans at the page's top, ``(text, baseline origin)``.
LANDSCAPE_HEAD = (('IEC 60404-8-1:2023', (72.0, 40.0)), ('- 24 -', (290.0, 40.0)))


def landscape_pdf(path: Path, pages: int = 1) -> Path:
    """``pages`` pages of a landscape table printed sideways (:func:`draw_sideways_table`, read bottom to top) under an
    upright running head (:data:`LANDSCAPE_HEAD`), its caption above it and a note below it in its own frame -- as a
    standard prints a landscape table on a portrait page. All in Helvetica 9: most of each page reads sideways."""
    with pymupdf.open() as doc:
        for _ in range(pages):
            page = doc.new_page(width=SCAN_PAGE.width, height=SCAN_PAGE.height)
            for text, at in LANDSCAPE_HEAD:
                page.insert_text(at, text, fontname='helv', fontsize=9)
            draw_sideways_table(page)
            for text, (x, y) in (LANDSCAPE_CAPTION, LANDSCAPE_NOTE):
                page.insert_text(turned(x, y, -90), text, fontname='helv', fontsize=9, rotate=90)
        doc.save(path)
    return path


#: A run of text on a page: its baseline origin (x, y), its text and the degrees PyMuPDF turns it counterclockwise
#: (0 upright, 90 read bottom to top, 270 read top to bottom).
type TurnedRun = tuple[float, float, str, int]


def turned_text_pdf(path: Path, runs: Sequence[TurnedRun]) -> Path:
    """One letter page of ``runs``, each set in Helvetica 9 at its origin, turned as it says: every box is known."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        for x, y, text, rotate in runs:
            page.insert_text((x, y), text, fontname='helv', fontsize=9, rotate=rotate)
        doc.save(path)
    return path
