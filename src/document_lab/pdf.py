"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space. Pixels
are the reader's only as an image's own samples (no codec is decoded): :func:`read_raster` takes a region that is
one image whole from them, and draws any other with a renderer -- PyMuPDF (the reference, the default) or
document-lab's own (``renderer='own'``: the ``render`` crate, drawing from this reader's interpreter; it draws no
text until its stage 2, so it refuses a region holding text -- D37 of task document-lab-pdf-reader).

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pymupdf

from document_lab import _pdf
from document_lab._pdf import Locked, Malformed, NotPdf, Refused, Unsupported
from document_lab.ir import BBox, Document, Figure, Page, Provenance, Raster, TextSpan, VectorPath

__all__ = [
    'CURVE_SAMPLES',
    'Locked',
    'Malformed',
    'NotPdf',
    'Refused',
    'Renderer',
    'Unsupported',
    'read_pdf',
    'read_raster',
]

#: Points per cubic Bezier segment when a curve is flattened.
CURVE_SAMPLES: Final = 8
#: The resolution (dpi) a region holding no image renders at.
VECTOR_DPI: Final = 300.0
#: How far (in pixels) a placed image's edges and sides may miss a region's and the region still be the image whole.
WHOLE_IMAGE_SLACK: Final = 0.25
#: The text rendering mode that neither fills nor strokes (an OCR layer): text that shows nothing.
INVISIBLE_TEXT: Final = 3
#: A matrix entry at most this (points) is zero: the image is square to the page.
_SQUARE: Final = 1e-6

#: What draws a region: PyMuPDF, or document-lab's own renderer (no text until its stage 2).
Renderer = Literal['pymupdf', 'own']


def read_pdf(
    path: Path | str, *, password: str = '', curve_samples: int = CURVE_SAMPLES, annotations: bool = False
) -> Document:
    """Read every page of the PDF at ``path``, opened with ``password`` when it is encrypted.

    ``curve_samples``: the points each curve of a vector path is flattened to. ``annotations``: each page's
    annotations' appearances read too, after its content, each value citing its annotation's subtype -- off by
    default, as a page's text is its content's (PyMuPDF's too). Optional content the file's default view hides, and
    what the colorant ``None`` paints, are read either way, each value flagged ``hidden`` or ``inkless`` for the
    caller to keep or drop (`document_lab.ir`).

    Raises:
        NotPdf: the file is no PDF.
        Locked: it is encrypted and ``password`` opens it for no one.
        Unsupported: it uses what the reader does not read.
        Malformed: it is not what a PDF must be.

    """
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    uri = path.resolve().as_uri()
    reader = _pdf.open(data, password.encode())
    pages = tuple(
        _page(
            reader,
            number,
            (width, height),
            Provenance(sha, uri, page=number),
            samples=curve_samples,
            annotations=annotations,
        )
        for number, width, height in reader.pages
    )
    return Document(sha256=sha, uri=uri, pages=pages, repairs=tuple(reader.repairs))


def _page(
    reader: _pdf.Reader, number: int, size: tuple[float, float], here: Provenance, *, samples: int, annotations: bool
) -> Page:
    spans, paths, images, warnings = reader.page(number, samples, annotations=annotations)
    return Page(
        number=number,
        width=size[0],
        height=size[1],
        spans=tuple(_span(row, here) for row in spans),
        paths=tuple(_path(row, here) for row in paths),
        figures=tuple(
            Figure(here.at(bbox=BBox(*box)), w, h, image, hidden=hidden, inkless=inkless, annotation=annotation)
            for box, w, h, image, _, (hidden, inkless, annotation) in images
        ),
        warnings=tuple(warnings),
    )


def _span(row: tuple, here: Provenance) -> TextSpan:
    text, font, size, bold, box, _mode, unspaced, (hidden, _, annotation) = row
    return TextSpan(text, size, font, here.at(bbox=BBox(*box)), bold, unspaced, hidden=hidden, annotation=annotation)


def _path(row: tuple, here: Provenance) -> VectorPath:
    points, closed, stroke, width, dashes, phase, fill, even_odd, pattern, group, (hidden, inkless, annotation) = row
    xs, ys = zip(*points, strict=True)
    return VectorPath(
        points=tuple(points),
        stroke=None if stroke is None else tuple(stroke),
        width=width,
        dashes=tuple(dashes),
        closed=closed,
        provenance=here.at(bbox=BBox(min(xs), min(ys), max(xs), max(ys))),
        dash_phase=phase,
        fill=None if fill is None else tuple(fill),
        even_odd=even_odd,
        pattern=pattern,
        group=group,
        hidden=hidden,
        inkless=inkless,
        annotation=annotation,
    )


def read_raster(
    path: Path | str,
    page: int,
    bbox: BBox,
    *,
    dpi: float | None = None,
    password: str = '',
    renderer: Renderer = 'pymupdf',
) -> Raster:
    """The region ``bbox`` of 1-based ``page`` as pixels, everything drawn there composited.

    ``dpi`` defaults to the finest native resolution among the images shown (an annotation's too, not one hidden or
    inkless) the region overlaps, so a pixel is a source pixel and none is invented; a region holding no image
    renders at ``VECTOR_DPI``. A region that is one image whole at its native resolution, nothing else visible on it,
    is that image's own samples where the reader holds them as the page shows them (:func:`_samples`); any other
    region is drawn by ``renderer``: PyMuPDF, or the own renderer (:func:`_drawn`).

    Raises:
        Unsupported: ``renderer='own'`` and the region holds what it does not draw yet (text, a shading, ...).

    """
    if renderer not in ('pymupdf', 'own'):
        msg = f'renderer {renderer!r}: one of pymupdf, own'
        raise ValueError(msg)
    path = Path(path)
    data = path.read_bytes()
    reader = _pdf.open(data, password.encode())
    spans, paths, images, _ = reader.page(page, 1, annotations=True)
    under = [row for row in images if not any(row[-1][:2]) and _meets(row[0], bbox)]
    if dpi is None:
        dpi = max(filter(None, map(_native, under)), default=VECTOR_DPI)
    pixels = _samples(reader, under, bbox, dpi) if _alone(spans, paths, bbox) else None
    if pixels is None and renderer == 'own':
        pixels = _drawn(reader, page, bbox, dpi, spans)
    if pixels is None:
        pixels = _rendered(data, page, bbox, dpi, password)
    here = Provenance(hashlib.sha256(data).hexdigest(), path.resolve().as_uri(), page=page, bbox=bbox)
    return Raster(pixels, here)


def _meets(box: tuple[float, float, float, float], bbox: BBox) -> bool:
    return box[0] < bbox.x1 and box[2] > bbox.x0 and box[1] < bbox.y1 and box[3] > bbox.y0


def _native(image: tuple) -> float:
    """An image row's finest resolution (dpi): its samples over its placed sides' lengths; 0 when it has no area."""
    _, width, height, _, (a, b, c, d, _, _), _ = image
    across, down = math.hypot(a, b), math.hypot(c, d)
    return 72 * max(width / across, height / down) if across and down else 0.0


def _alone(spans: list, paths: list, bbox: BBox) -> bool:
    """Whether no path that shows and no visible text (an OCR layer is invisible) is drawn on ``bbox``."""
    for points, _, _, width, *_, (hidden, inkless, _) in paths:
        if hidden or inkless:
            continue
        xs, ys = [x for x, _ in points], [y for _, y in points]
        reach = width / 2
        if _meets((min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach), bbox):
            return False
    return not _texts(spans, bbox)


def _texts(spans: list, bbox: BBox) -> int:
    """How many spans of visible text (an OCR layer is invisible, hidden content too) are drawn on ``bbox``."""
    return sum(
        not hidden and mode != INVISIBLE_TEXT and _meets(box, bbox)
        for _, _, _, _, box, mode, _, (hidden, _, _) in spans
    )


def _drawn(reader: _pdf.Reader, page: int, bbox: BBox, dpi: float, spans: list) -> np.ndarray:
    """``bbox`` drawn by the own renderer, refusing a region holding what it does not draw yet, by name and count."""
    texts = _texts(spans, bbox)
    if texts:
        msg = (
            f'page {page}: the own renderer draws no text until its stage 2, and the region holds {texts} span(s) of '
            "it; render it with renderer='pymupdf'"
        )
        raise Unsupported(msg)
    width, height, channels, pixels, undrawn = reader.render(
        page, tuple(bbox), dpi / 72, text=False, annotations=True, alpha=False
    )
    if undrawn:
        named = ', '.join(f'{what} x{count}' for what, count in undrawn)
        msg = (
            f'page {page}: the own renderer does not draw all the region holds yet: {named}; render it with '
            "renderer='pymupdf'"
        )
        raise Unsupported(msg)
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels).copy()


def _samples(reader: _pdf.Reader, under: list, bbox: BBox, dpi: float) -> np.ndarray | None:
    """The one image under ``bbox`` as RGB rows from its own samples; None unless they are what the page shows.

    They are when it is placed square to the page -- turned or mirrored by quarter turns, its samples turned alike --
    edge to edge on ``bbox`` within :data:`WHOLE_IMAGE_SLACK` of a pixel, a sample a pixel at ``dpi``, and the reader
    holds its samples as they show (``Reader.rgb``: lossless gray or RGB, or an ICC-based image through its profile).
    Anything else -- skewed, part of it, resampled, an inline image -- is the renderer's.
    """
    if len(under) != 1:
        return None
    box, width, height, number, (a, b, c, d, _, _), _ = under[0]
    turned = abs(a) <= _SQUARE and abs(d) <= _SQUARE  # its rows run down the page, its columns across
    if number is None or not (turned or (abs(b) <= _SQUARE and abs(c) <= _SQUARE)):
        return None
    across, down = (height, width) if turned else (width, height)
    slack = WHOLE_IMAGE_SLACK * 72 / dpi
    sides = (across * 72 / dpi - (bbox.x1 - bbox.x0), down * 72 / dpi - (bbox.y1 - bbox.y0))
    edges = (box[0] - bbox.x0, box[1] - bbox.y0, box[2] - bbox.x1, box[3] - bbox.y1)
    if any(abs(gap) > slack for gap in (*sides, *edges)):
        return None
    srgb = reader.rgb(number)
    if srgb is None:
        return None
    rows = np.frombuffer(srgb, dtype=np.uint8).reshape(height, width, 3)
    # A sample (row i, column j) sits at u = (j + 1/2) / width, v = 1 - (i + 1/2) / height of the image's unit
    # square, mapped to the page (y down) by x = a u + c v + e, y = b u + d v + f.
    if turned:
        rows, down_flipped, across_flipped = rows.transpose(1, 0, 2), b < 0, c > 0
    else:
        down_flipped, across_flipped = d > 0, a < 0
    return np.array(rows[:: -1 if down_flipped else 1, :: -1 if across_flipped else 1])


def _rendered(data: bytes, page: int, bbox: BBox, dpi: float, password: str) -> np.ndarray:
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        if password:
            pdf.authenticate(password)
        zoom = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pdf[page - 1].get_pixmap(matrix=zoom, clip=pymupdf.Rect(*bbox), colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
