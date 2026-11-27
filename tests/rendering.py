"""How the own renderer is held to PyMuPDF's renders (the reference, D37): the same pixels at the same resolution,
compared after a 3 x 3 box blur so anti-aliasing that spreads an edge differently by a pixel is not a difference.

A render agrees with the reference when, over the pixels either inks (blank paper dilutes nothing), at most
:data:`SHARE_TOLERANCE` are off by more than :data:`PIXEL_TOLERANCE` levels -- a tenth of the range, so a mark
misplaced by a pixel, a wrong cap or join, a mask the wrong way round is off, an edge anti-aliased otherwise is not --
and the blurred difference averages at most :data:`MEAN_TOLERANCE` levels, which a color shifted across a whole
area exceeds. Measured on the synthetic cases of ``tests/unit/test_render.py`` (2026-11-27): the faithful renders
0-0.1 % of inked pixels off, means 0-1.8 levels (6.0 for thin dashed lines, all edge); the planted wrong renders
1.4-100 % off, or (a profile ignored) a mean of 17 levels.
"""

from __future__ import annotations

import re

import numpy as np
import pymupdf

from document_lab import _pdf

#: Levels a pixel's blurred difference may reach and the pixel still agree.
PIXEL_TOLERANCE = 24
#: The share of the compared pixels that may disagree.
SHARE_TOLERANCE = 0.01
#: Levels the blurred difference may average over the compared pixels.
MEAN_TOLERANCE = 8.0
#: A channel below this inks the pixel: it is compared.
INK = 250


def page_pdf(content: str, objects: dict[str, tuple[str, bytes | None]] | None = None, *, resources: str = '') -> bytes:
    """A one-page PDF (200 x 150 points) drawing ``content``.

    ``objects`` are named objects (a dictionary's source and its stream's bytes, None for no stream); a dictionary
    names another as ``{Name}``, and those named ``Im...`` or ``Fm...`` are the page's XObjects. ``resources`` adds
    entries to the page's resources. A stated ``/Filter`` is kept: the bytes are stored as given.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=150)
    objects = objects or {}
    numbers = {name: doc.get_new_xref() for name in objects}
    refs = {name: f'{number} 0 R' for name, number in numbers.items()}
    for name, (template, data) in objects.items():
        source = template.format(**refs)
        doc.update_object(numbers[name], source)
        if data is not None:
            doc.update_stream(numbers[name], data, compress=False)
            if stated := re.search(r'/Filter (/\w+|\[[^\]]*\])', source):
                doc.xref_set_key(numbers[name], 'Filter', stated[1])
    content_number = doc.get_new_xref()
    doc.update_object(content_number, '<< >>')
    doc.update_stream(content_number, content.encode('latin-1'))
    xobjects = ' '.join(f'/{name} {refs[name]}' for name in objects if name.startswith(('Im', 'Fm')))
    doc.xref_set_key(page.xref, 'Contents', f'{content_number} 0 R')
    doc.xref_set_key(page.xref, 'Resources', f'<< /XObject << {xobjects} >> {resources} >>')
    return doc.tobytes()


def reference(data: bytes, dpi: float, *, managed: bool = True) -> np.ndarray:
    """Page 1 of ``data`` rendered by PyMuPDF at ``dpi``, RGB on white; color management off unless ``managed``."""
    pymupdf.TOOLS.set_icc(managed)
    try:
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            pix = doc[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
    finally:
        pymupdf.TOOLS.set_icc(True)


def own(data: bytes, dpi: float, page: int = 1) -> tuple[np.ndarray, dict[str, int]]:
    """Page ``page`` of ``data`` drawn by the own renderer at ``dpi`` (no text), and what it did not draw."""
    width, height, channels, pixels, undrawn = _pdf.open(data, b'').render(
        page, None, dpi / 72, text=False, annotations=True, alpha=False
    )
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels).copy(), dict(undrawn)


def blurred_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each pixel's largest channel difference between ``a`` and ``b`` after a 3 x 3 box blur of each.

    The blur is linear, so it is taken of the difference -- in 16 bits, a page at 150 dpi being millions of pixels.
    """
    difference = a.astype(np.int16) - b.astype(np.int16)
    padded = np.pad(difference, ((1, 1), (1, 1), (0, 0)), mode='edge')
    h, w = difference.shape[:2]
    total = np.zeros_like(difference)
    for i in range(3):
        for j in range(3):
            total += padded[i : i + h, j : j + w]
    return np.abs(total).max(axis=2).astype(np.float32) / 9


def inked(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The pixels either image inks (any channel below :data:`INK`), widened by a pixel each way."""
    ink = (a.min(axis=2) < INK) | (b.min(axis=2) < INK)
    padded = np.pad(ink, 1)
    h, w = ink.shape
    return np.logical_or.reduce([padded[i : i + h, j : j + w] for i in range(3) for j in range(3)])


def disagreement(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> tuple[float, float]:
    """The blurred difference's mean and the share of pixels off by more than :data:`PIXEL_TOLERANCE`, over the
    pixels ``keep`` marks -- by default those either image inks (:func:`inked`), so blank paper dilutes nothing."""
    assert a.shape == b.shape, f'{a.shape} vs {b.shape}'
    difference = blurred_difference(a, b)
    kept = difference[inked(a, b) if keep is None else keep]
    if kept.size == 0:
        return 0.0, 0.0
    return float(kept.mean()), float((kept > PIXEL_TOLERANCE).mean())


def agrees(mean: float, share: float) -> bool:
    """Whether a render whose disagreement is ``(mean, share)`` agrees with its reference."""
    return mean <= MEAN_TOLERANCE and share <= SHARE_TOLERANCE
