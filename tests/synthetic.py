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
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from document_lab.ir import Page, Table

FRAME = pymupdf.Rect(100, 100, 400, 300)
RED = (1, 0, 0)
BLUE = (0, 0, 1)
FONT_SIZE = 8
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
    over: Callable[[pymupdf.Page], object] | None = None,
) -> Path:
    """A one-page PDF of ``size`` points placing one image XObject, saved to ``path``.

    ``image`` is the image dictionary's own entries (``/Width 4 /Height 2 /ColorSpace /DeviceGray ...``), ``stored``
    its stream: deflated when ``deflate``, else stored as given (a codec's ``/Filter`` stated in ``image``).
    ``matrix`` is the ``cm`` placing it (PDF space, y up; default the whole page, upright); ``rotate`` the page's
    ``/Rotate``; ``soft_mask`` the 8-bit gray samples of an ``/SMask`` its size; ``over`` draws on the page after.
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


def fax_g4(white: np.ndarray) -> bytes:
    """``white`` -- rows of 1 white, 0 black, as CCITT fax decodes by default -- coded Group 4 by MuPDF."""
    rows, columns = white.shape
    packed = np.packbits(white.astype(np.uint8), axis=1).tobytes()
    stride = len(packed) // rows
    coded = pymupdf.mupdf.fz_compress_ccitt_fax_g4(pymupdf.mupdf.python_buffer_data(packed), columns, rows, stride)
    return bytes(coded.fz_buffer_extract())


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


def hidden_layer(page: Page) -> Page:
    """``page`` with everything it draws in a hidden layer: what a layer turned off in the default view reads as."""
    return replace(
        page,
        spans=tuple(replace(s, hidden=True) for s in page.spans),
        paths=tuple(replace(p, hidden=True) for p in page.paths),
        figures=tuple(replace(f, hidden=True) for f in page.figures),
    )


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
