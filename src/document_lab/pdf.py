"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space. Pixels
are not the reader's: :func:`read_raster` renders a region with PyMuPDF (D4 of task document-lab-pdf-reader).

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

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
    'Unsupported',
    'read_pdf',
    'read_raster',
]

#: Points per cubic Bezier segment when a curve is flattened.
CURVE_SAMPLES: Final = 8
#: The resolution (dpi) a region holding no image renders at.
VECTOR_DPI: Final = 300.0


def read_pdf(path: Path | str, *, password: str = '', curve_samples: int = CURVE_SAMPLES) -> Document:
    """Read every page of the PDF at ``path``, opened with ``password`` when it is encrypted.

    ``curve_samples``: the points each curve of a vector path is flattened to.

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
        _page(reader, number, (width, height), Provenance(sha, uri, page=number), curve_samples)
        for number, width, height in reader.pages
    )
    return Document(sha256=sha, uri=uri, pages=pages, repairs=tuple(reader.repairs))


def _page(reader: _pdf.Reader, number: int, size: tuple[float, float], here: Provenance, samples: int) -> Page:
    spans, paths, images, warnings = reader.page(number, samples)
    return Page(
        number=number,
        width=size[0],
        height=size[1],
        spans=tuple(_span(row, here) for row in spans),
        paths=tuple(_path(row, here) for row in paths),
        figures=tuple(Figure(here.at(bbox=BBox(*box)), w, h, image) for box, w, h, image in images),
        warnings=tuple(warnings),
    )


def _span(row: tuple, here: Provenance) -> TextSpan:
    text, font, size, bold, box, _mode, unspaced, direction = row
    return TextSpan(text, size, font, here.at(bbox=BBox(*box)), bold, unspaced=unspaced, direction=direction)


def _path(row: tuple, here: Provenance) -> VectorPath:
    points, closed, stroke, width, dashes, phase, fill, even_odd, group = row
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
        group=group,
    )


def read_raster(path: Path | str, page: int, bbox: BBox, *, dpi: float | None = None, password: str = '') -> Raster:
    """The region ``bbox`` of 1-based ``page`` as pixels, everything drawn there composited.

    ``dpi`` defaults to the finest native resolution among the images the region overlaps, so a pixel is a source
    pixel and none is invented; a region holding no image renders at ``VECTOR_DPI``.
    """
    path = Path(path)
    data = path.read_bytes()
    if dpi is None:
        _, _, images, _ = _pdf.open(data, password.encode()).page(page, 1)
        native = [
            72 * w / (box[2] - box[0])
            for box, w, _, _ in images
            if box[2] > box[0] and box[0] < bbox.x1 and box[2] > bbox.x0 and box[1] < bbox.y1 and box[3] > bbox.y0
        ]
        dpi = max(native, default=VECTOR_DPI)
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        if password:
            pdf.authenticate(password)
        zoom = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pdf[page - 1].get_pixmap(matrix=zoom, clip=pymupdf.Rect(*bbox), colorspace=pymupdf.csRGB, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
    here = Provenance(hashlib.sha256(data).hexdigest(), path.resolve().as_uri(), page=page, bbox=bbox)
    return Raster(pixels, here)
