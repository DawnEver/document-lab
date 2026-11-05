"""Synthetic PDFs drawn with PyMuPDF so every expected value is known exactly -- the planted controls.

A plot is drawn on a frame from `FRAME`; its tick labels sit beside tick marks at known data values, and its curve
is a known function sampled at known points, so a digitiser that is wrong cannot pass by agreeing with itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

    def frac(self, value: float) -> float:
        lo, hi = self.ticks[0], self.ticks[-1]
        if self.log:
            return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return (value - lo) / (hi - lo)


@dataclass
class PlotSpec:
    x: Axis
    y: Axis
    curves: list[tuple[Callable[[float], float], Sequence[float], tuple[float, float, float]]] = field(
        default_factory=list
    )
    frame: pymupdf.Rect = field(default_factory=lambda: pymupdf.Rect(FRAME))
    tick_marks: bool = True
    grid: bool = False


def to_page(spec: PlotSpec, x: float, y: float) -> tuple[float, float]:
    f = spec.frame
    return (f.x0 + spec.x.frac(x) * f.width, f.y1 - spec.y.frac(y) * f.height)


def _label(v: float) -> str:
    return f'{v:g}'


def _centered(page: pymupdf.Page, text: str, cx: float, cy: float, size: float = FONT_SIZE) -> None:
    width = pymupdf.get_text_length(text, fontname='helv', fontsize=size)
    page.insert_text((cx - width / 2, cy + size * 0.35), text, fontname='helv', fontsize=size)


def _power_label(page: pymupdf.Page, value: float, right: float, cy: float) -> None:
    exp = str(round(math.log10(value)))
    base_w = pymupdf.get_text_length('10', fontname='helv', fontsize=FONT_SIZE)
    exp_w = pymupdf.get_text_length(exp, fontname='helv', fontsize=FONT_SIZE * 0.6)
    x0 = right - base_w - exp_w
    page.insert_text((x0, cy + FONT_SIZE * 0.35), '10', fontname='helv', fontsize=FONT_SIZE)
    page.insert_text((x0 + base_w, cy - FONT_SIZE * 0.3), exp, fontname='helv', fontsize=FONT_SIZE * 0.6)


def draw_plot(page: pymupdf.Page, spec: PlotSpec) -> None:
    f = spec.frame
    page.draw_rect(f, color=(0, 0, 0), width=0.8)
    for i, v in enumerate(spec.x.ticks):
        px, _ = to_page(spec, v, spec.y.ticks[0])
        if spec.tick_marks:
            page.draw_line((px, f.y1), (px, f.y1 - 4), color=(0, 0, 0), width=0.5)
        if spec.grid:
            page.draw_line((px, f.y0), (px, f.y1), color=(0.7, 0.7, 0.7), width=0.3)
        text = spec.x.labels[i] if spec.x.labels else _label(v)
        _centered(page, text, px, f.y1 + 10)
    for i, v in enumerate(spec.y.ticks):
        _, py = to_page(spec, spec.x.ticks[0], v)
        if spec.tick_marks:
            page.draw_line((f.x0, py), (f.x0 + 4, py), color=(0, 0, 0), width=0.5)
        if spec.grid:
            page.draw_line((f.x0, py), (f.x1, py), color=(0.7, 0.7, 0.7), width=0.3)
        if spec.y.power_labels:
            _power_label(page, v, f.x0 - 4, py)
            continue
        text = spec.y.labels[i] if spec.y.labels else _label(v)
        width = pymupdf.get_text_length(text, fontname='helv', fontsize=FONT_SIZE)
        _centered(page, text, f.x0 - 4 - width / 2, py)
    for fn, xs, colour in spec.curves:
        page.draw_polyline([to_page(spec, x, fn(x)) for x in xs], color=colour, width=1.0)


def plot_pdf(path: Path, *specs: PlotSpec) -> Path:
    doc = pymupdf.open()
    for spec in specs:
        draw_plot(doc.new_page(width=612, height=792), spec)
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
