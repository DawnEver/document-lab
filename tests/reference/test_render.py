"""The ``reference`` part, rendering: the own renderer (paths, clips, images, text from font outlines, shadings,
patterns, soft masks and transparency groups) accepted against PyMuPDF's renders of the real PDFs
``DOCUMENT_LAB_REFERENCE_PDFS`` names.

Every :data:`STEP`-th page of each document, its first among them, is drawn whole by both, text and annotations
too, at :data:`DPI`, and compared as ``tests/rendering.py`` says: a page agrees when at most :data:`PAGE_SHARE` of
the pixels either inks are off by more than its ``PIXEL_TOLERANCE`` levels after the blur. PyMuPDF draws a page two
ways, color-managed (``ICCBased`` through its profile, ``DeviceCMYK`` through a press profile) or not (every space by
the standard's formulas); the reader converts ``ICCBased`` the first way and ``DeviceCMYK`` the second
(`tests/unit/test_render.py`), so a page agrees with whichever agrees better. A page holding what the own renderer
cannot draw (``undrawn``: a glyph of a font it has no program for, a JPX or JBIG2 image, a shading or pattern that
does not read) is not compared. The planted control: each document's compared page with the most dark
ink, held to its own reference turned upside down and moved a few pixels, must be reported.

A standard font a document does not embed is drawn with PDFium's Foxit substitute where PyMuPDF draws URW's, two
designs (``tests/unit/test_render_text.py`` holds the substitution itself to PyMuPDF drawing the Foxit program):
its glyphs' edges differ, which the measurement below includes. A CJK font a document does not embed is drawn from a
fallback fetched on first use (Noto Sans or Serif CJK, ``document_lab.fonts``), where PyMuPDF draws every one with its
own Droid Sans Fallback: dense strokes of two designs differ on up to a third of their pixels (32.8 % on IEC 60205's
first page), so here the reader is handed PyMuPDF's own program for those fallbacks (:func:`_fallback`) -- the
choice of glyph and its place are compared, not the design. Measured on 2026-11-27, the 10 pages of the one local
document holding such fonts: 0.00-0.02 % off with Droid on both sides; with the fallbacks themselves 3.2-32.8 %
(sans only, 2.2-19.3 %).

Measured on 2026-11-28 over 283 local PDFs, every 10th page (774 pages; 1 holding what is not drawn -- a Chinese
font not embedded): of the 773 compared, 78.0 % have at most 0.5 % of their inked pixels off, 93.5 % at most 2 %,
99.4 % at most 5 %; the worst 14.2 % (finely striped plots sampled a fraction of a pixel apart); the 84 pages with a
font not embedded (measured 2026-11-27) 70.2 %, 91.7 % and 100 %, worst 3.8 %. The pages holding shadings,
patterns, soft masks or transparency groups (185, every page of a document up to 6) 73.0 %, 92.4 % and 98.9 %,
worst 9.4 % (D130 of task document-lab-pdf-reader).
:data:`PAGE_SHARE` guards against what is not anti-aliasing or a substitute's design: the faults this comparison
found -- CMYK JPEGs inverted twice (88 % off), a page held to the wrong color mode (50 %) -- are far above it, as are
glyphs placed a fraction of a pixel off MuPDF's grid (25-45 % on synthetic pages of text).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pymupdf
import pytest

from document_lab import _pdf, fonts
from tests.reference.test_reference import PDFS, _needs_documents
from tests.rendering import PIXEL_TOLERANCE, blurred_difference, inked

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.reference

#: The resolution both renders are compared at.
DPI = 150
#: Every this many pages of a document are compared, the first among them.
STEP = 10
#: The share of a page's compared pixels that may be off: above the anti-aliasing measured, below a fault.
PAGE_SHARE = 0.25


def _pixels(data: bytes, width: int, height: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _fallback(name: str) -> bytes:
    """PyMuPDF's own CJK program for a CJK fallback (the design compared away); any other fetched as in use."""
    if name.startswith(('noto-sans-', 'noto-serif-')) and name[-3:] in ('-sc', '-tc', '-jp', '-kr'):
        return bytes(pymupdf.Font('cjk').buffer)
    return fonts.program(name)


@lru_cache(maxsize=1)  # the document under test; never the whole corpus at once
def _document(pdf: Path) -> tuple[bytes, _pdf.Reader]:
    data = pdf.read_bytes()
    return data, _pdf.open(data, b'', fonts=_fallback)


def _reference(data: bytes, number: int, *, managed: bool) -> np.ndarray:
    """Page ``number`` of ``data`` rendered by PyMuPDF, color-managed or not."""
    pymupdf.TOOLS.set_icc(managed)
    try:
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            pix = doc[number - 1].get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB, alpha=False, annots=True)
            return _pixels(pix.samples, pix.w, pix.h).copy()
    finally:
        pymupdf.TOOLS.set_icc(True)


def _own(pdf: Path, number: int) -> np.ndarray | None:
    """Page ``number`` drawn by the own renderer; None when it holds what the own renderer cannot draw."""
    width, height, _, pixels, undrawn = _document(pdf)[1].render(
        number, None, DPI / 72, text=True, annotations=True, alpha=False
    )
    return None if undrawn else _pixels(pixels, width, height)


def off_share(pdf: Path, number: int, *, misplaced: bool = False) -> float | None:
    """The share of page ``number``'s compared pixels the own render has off from PyMuPDF's -- turned upside down and
    moved by :data:`PLANTED_SHIFT` pixels when ``misplaced``: a planted wrong reference; None when the page holds what
    the own renderer cannot draw.
    """
    own = _own(pdf, number)
    if own is None:
        return None
    shares = []
    for managed in (True, False):
        reference = _reference(_document(pdf)[0], number, managed=managed)
        if misplaced:
            reference = np.ascontiguousarray(np.roll(reference[::-1, ::-1], PLANTED_SHIFT, axis=(0, 1)))
        if reference.shape != own.shape:
            return 1.0
        keep = inked(own, reference)
        if not keep.any():
            return 0.0
        shares.append(float((blurred_difference(own, reference)[keep] > PIXEL_TOLERANCE).mean()))
    return min(shares)


def _numbers(pdf: Path) -> list[int]:
    return [n for n, _, _ in _document(pdf)[1].pages][::STEP]


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_the_own_renderer_draws_a_document_as_the_reference_does(pdf):
    _needs_documents()
    assert pdf is not None
    try:
        _document(pdf)
    except _pdf.Locked:
        return  # the reader's own part checks a locked document
    off = {n: off_share(pdf, n) for n in _numbers(pdf)}
    bad = {n: f'{share:.3f}' for n, share in off.items() if share is not None and share > PAGE_SHARE}
    assert bad == {}, f'pages off by more than {PAGE_SHARE:.0%} of their compared pixels: {bad}'


#: A level below which a pixel is dark: what the planted control's page is chosen by.
DARK = 128
#: The share of a page's pixels the own renderer must draw dark for its planted misplacement to be told apart.
PLANTED_INK = 0.01
#: Pixels (down, across) the planted reference is moved by, after it is turned upside down.
PLANTED_SHIFT = (5, 7)


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_a_misplaced_reference_is_reported(pdf):
    """The planted control: the compared page the own renderer draws most dark ink on, held to its reference turned
    upside down and moved by :data:`PLANTED_SHIFT` pixels.

    A document whose compared pages draw less than :data:`PLANTED_INK` of their pixels dark (a page of faint
    scanned paper), or that the own renderer leaves undrawn, has nothing to tell apart.
    """
    _needs_documents()
    assert pdf is not None
    try:
        _document(pdf)
    except _pdf.Locked:
        return
    darks = {}
    for number in _numbers(pdf):
        own = _own(pdf, number)
        if own is not None:
            darks[number] = float((own.min(axis=2) < DARK).mean())
    if not darks or max(darks.values()) < PLANTED_INK:
        return
    number = max(darks, key=darks.__getitem__)
    share = off_share(pdf, number, misplaced=True)
    assert share is not None
    assert share > PAGE_SHARE, f'page {number} passed as its reference misplaced ({share:.3f} off)'
