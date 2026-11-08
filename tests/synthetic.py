"""Synthetic PDFs drawn with PyMuPDF so every expected value is known exactly -- the planted controls.

A plot is drawn on a frame from `FRAME`; its tick labels sit beside tick marks at known data values, and its curve
is a known function sampled at known points, so a digitiser that is wrong cannot pass by agreeing with itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pymupdf

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

FRAME = pymupdf.Rect(100, 100, 400, 300)
RED = (1, 0, 0)
BLUE = (0, 0, 1)
FONT_SIZE = 8


@dataclass
class Axis:
    ticks: Sequence[float]
    log: bool = False
    labels: Sequence[str] | None = None  # defaults to the tick values' shortest spelling
    power_labels: bool = False  # draw "10" plus a raised exponent span instead of the value
    vector_minus: bool = False  # a negative exponent's minus is a drawn dash, not a glyph
    #: A negative label's (or exponent's) minus is its own span (another font), apart from its digits.
    span_minus: bool = False
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
    #: (function, sample xs, colour), optionally with a PDF dash string such as '[3 2] 0'.
    curves: list[tuple[Callable[[float], float], Sequence[float], tuple[float, float, float], *tuple[str, ...]]] = (
        field(default_factory=list)
    )
    frame: pymupdf.Rect = field(default_factory=lambda: pymupdf.Rect(FRAME))
    tick_marks: bool = True
    grid: bool = False
    minor_grid: bool = False  # a log axis also gets grid lines at 2..9 x each decade
    frame_as_lines: bool = False  # the frame is four separate segments, not one rectangle
    grid_reach: float = 1.0  # the fraction of the frame width a horizontal grid line spans (a legend cuts it)
    stray: Sequence[tuple[str, float, float, float]] = ()  # extra text: (text, x, y, size)
    fragments: int = 1  # each curve drawn as this many polylines meeting end to end, every other one reversed
    #: Legend entries inside the frame's top right: (label, colour, PDF dash string or None), a sample line each.
    legend: Sequence[tuple[str, tuple[float, float, float], str | None]] = ()


def to_page(spec: PlotSpec, x: float, y: float) -> tuple[float, float]:
    f = spec.frame
    return (f.x0 + spec.x.frac(x) * f.width, f.y1 - spec.y.frac(y) * f.height)


def _label(v: float) -> str:
    return f'{v:g}'


def _centered(page: pymupdf.Page, text: str, cx: float, cy: float, size: float = FONT_SIZE) -> None:
    width = pymupdf.get_text_length(text, fontname='helv', fontsize=size)
    page.insert_text((cx - width / 2, cy + size * 0.35), text, fontname='helv', fontsize=size)


def _power_label(
    page: pymupdf.Page, value: float, right: float, cy: float, *, vector_minus: bool, span_minus: bool = False
) -> None:
    exp = str(round(math.log10(value)))
    small = FONT_SIZE * 0.6
    dash = vector_minus and exp.startswith('-')
    glyph = span_minus and exp.startswith('-')
    shown = exp[1:] if dash or glyph else exp
    base_w = pymupdf.get_text_length('10', fontname='helv', fontsize=FONT_SIZE)
    dash_w = 3.5 if dash else pymupdf.get_text_length('-', fontname='tiro', fontsize=small) if glyph else 0.0
    exp_w = pymupdf.get_text_length(shown, fontname='helv', fontsize=small)
    x0 = right - base_w - dash_w - exp_w
    page.insert_text((x0, cy + FONT_SIZE * 0.35), '10', fontname='helv', fontsize=FONT_SIZE)
    if dash:
        page.draw_line((x0 + base_w + 0.5, cy - 2.6), (x0 + base_w + 3.0, cy - 2.6), color=(0, 0, 0), width=0.4)
    if glyph:
        page.insert_text((x0 + base_w, cy - FONT_SIZE * 0.3), '-', fontname='tiro', fontsize=small)
    page.insert_text((x0 + base_w + dash_w, cy - FONT_SIZE * 0.3), shown, fontname='helv', fontsize=small)


def _glyphs(page: pymupdf.Page, text: str, cx: float, cy: float) -> None:
    """``text`` centred on ``cx``, one span per glyph: neighbours alternate font, so no reader merges them."""
    fonts = ('helv', 'cour')
    widths = [pymupdf.get_text_length(ch, fontname=fonts[i % 2], fontsize=FONT_SIZE) for i, ch in enumerate(text)]
    x = cx - sum(widths) / 2
    for i, (ch, w) in enumerate(zip(text, widths, strict=True)):
        page.insert_text((x, cy + FONT_SIZE * 0.35), ch, fontname=fonts[i % 2], fontsize=FONT_SIZE)
        x += w


def _minor(ticks: Sequence[float]) -> list[float]:
    return [k * 10**e for e in range(round(math.log10(ticks[0])), round(math.log10(ticks[-1]))) for k in range(2, 10)]


def _draw_frame(page: pymupdf.Page, spec: PlotSpec) -> None:
    f = spec.frame
    if spec.frame_as_lines:
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


def _minus_span(page: pymupdf.Page, text: str, cx: float, cy: float) -> None:
    """``text`` centred on ``cx`` with its minus set in another font, so it is a span of its own."""
    digits = text[1:]
    minus_w = pymupdf.get_text_length('-', fontname='tiro', fontsize=FONT_SIZE)
    width = pymupdf.get_text_length(digits, fontname='helv', fontsize=FONT_SIZE)
    start = cx - (minus_w + 0.5 + width) / 2
    page.insert_text((start, cy + FONT_SIZE * 0.35), '-', fontname='tiro', fontsize=FONT_SIZE)
    page.insert_text((start + minus_w + 0.5, cy + FONT_SIZE * 0.35), digits, fontname='helv', fontsize=FONT_SIZE)


def _x_label(page: pymupdf.Page, spec: PlotSpec, text: str, px: float) -> None:
    """A label centred on its tick; a drawn minus is part of the centred whole, as typeset labels are."""
    y = spec.frame.y1 + 10
    if spec.x.glyph_spans:
        _glyphs(page, text, px, y)
        return
    if spec.x.span_minus and text.startswith('-'):
        _minus_span(page, text, px, y)
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
        if spec.grid:
            page.draw_line((f.x0, py), (f.x0 + spec.grid_reach * f.width, py), color=(0.7, 0.7, 0.7), width=0.3)
        py += spec.y.label_drop
        if spec.y.power_labels:
            _power_label(page, v, f.x0 - 4, py, vector_minus=spec.y.vector_minus, span_minus=spec.y.span_minus)
            continue
        text = spec.y.labels[i] if spec.y.labels else _label(v)
        width = pymupdf.get_text_length(text, fontname='helv', fontsize=FONT_SIZE)
        if spec.y.span_minus and text.startswith('-'):
            _minus_span(page, text, f.x0 - 4 - width / 2, py)
            continue
        _centered(page, text, f.x0 - 4 - width / 2, py)


def draw_plot(page: pymupdf.Page, spec: PlotSpec) -> None:
    f = spec.frame
    _draw_frame(page, spec)
    for i, v in enumerate(spec.x.ticks):
        px, _ = to_page(spec, v, spec.y.ticks[0])
        if spec.tick_marks:
            page.draw_line((px, f.y1), (px, f.y1 - 4), color=(0, 0, 0), width=0.5)
        if spec.grid:
            page.draw_line((px, f.y0), (px, f.y1), color=(0.7, 0.7, 0.7), width=0.3)
        _x_label(page, spec, spec.x.labels[i] if spec.x.labels else _label(v), px)
    _y_axis(page, spec)
    for text, x, y, size in spec.stray:
        page.insert_text((x, y), text, fontname='helv', fontsize=size)
    for fn, xs, colour, *dashes in spec.curves:
        points = [to_page(spec, x, fn(x)) for x in xs]
        step = math.ceil((len(points) - 1) / spec.fragments)
        for k, start in enumerate(range(0, len(points) - 1, step)):
            piece = points[start : start + step + 1]
            page.draw_polyline(
                piece[::-1] if k % 2 else piece, color=colour, width=1.0, dashes=next(iter(dashes), None)
            )
    for k, (text, colour, dashes) in enumerate(spec.legend):
        y = spec.frame.y0 + 12 + 12 * k
        x = spec.frame.x1 - 80
        page.draw_line((x, y), (x + 20, y), color=colour, width=1.0, dashes=dashes)
        page.insert_text((x + 24, y + FONT_SIZE * 0.35), text, fontname='helv', fontsize=FONT_SIZE)


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
