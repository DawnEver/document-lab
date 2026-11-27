"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space. Pixels
are the reader's only as an image's own samples (no codec is decoded): :func:`read_raster` takes a region that is
one image whole from them, and renders any other with PyMuPDF (D4 of task document-lab-pdf-reader).

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

import hashlib
import math
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
#: How far (in pixels) a placed image's edges and sides may miss a region's and the region still be the image whole.
WHOLE_IMAGE_SLACK: Final = 0.25
#: The text rendering mode that neither fills nor strokes (an OCR layer): text that shows nothing.
INVISIBLE_TEXT: Final = 3
#: Gray depths whose levels spread over 0..255 exactly (255 is a multiple of each level count less one).
_GRAY_BITS: Final = frozenset({1, 2, 4, 8})
#: Families whose samples are gray or RGB as they stand. An ``ICCBased`` one is the gray or RGB of its component
#: count, its profile not applied: the crate's color model (``pdf::color::Space::Icc``) reads it so, and this follows.
#: Exact for an sRGB profile (the IEC scans measured); a wider-gamut profile would show otherwise once rendered.
_GRAY_OR_RGB: Final = frozenset({'DeviceGray', 'DeviceRGB', 'ICCBased'})


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
        figures=tuple(Figure(here.at(bbox=BBox(*box)), w, h, image) for box, w, h, image, *_ in images),
        warnings=tuple(warnings),
    )


def _span(row: tuple, here: Provenance) -> TextSpan:
    text, font, size, bold, box, mode, unspaced, vertical = row
    return TextSpan(text, size, font, here.at(bbox=BBox(*box)), bold, unspaced, mode, vertical)


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
    pixel and none is invented; a region holding no image renders at ``VECTOR_DPI``. A region that is one image whole
    at its native resolution, nothing else visible on it, is that image's own samples where the reader holds them as
    the page shows them (:func:`_samples`); any other region is rendered by PyMuPDF.
    """
    path = Path(path)
    data = path.read_bytes()
    reader = _pdf.open(data, password.encode())
    spans, paths, images, _ = reader.page(page, 1)
    under = [row for row in images if _meets(row[0], bbox)]
    if dpi is None:
        dpi = max(filter(None, map(_native, under)), default=VECTOR_DPI)
    pixels = _samples(reader, under, bbox, dpi) if _alone(spans, paths, bbox) else None
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
    """Whether no path and no visible text (an OCR layer is invisible) is drawn on ``bbox``."""
    for points, _, _, width, *_ in paths:
        xs, ys = [x for x, _ in points], [y for _, y in points]
        reach = width / 2
        if _meets((min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach), bbox):
            return False
    return not any(mode != INVISIBLE_TEXT and _meets(box, bbox) for _, _, _, _, box, mode, *_ in spans)


def _samples(reader: _pdf.Reader, under: list, bbox: BBox, dpi: float) -> np.ndarray | None:
    """The one image under ``bbox`` as RGB rows from its own samples; None unless they are what the page shows.

    They are when it is placed square to the page -- turned or mirrored by quarter turns, its samples turned alike --
    edge to edge on ``bbox`` within :data:`WHOLE_IMAGE_SLACK` of a pixel, a sample a pixel at ``dpi``, and its samples
    are what :func:`_rgb` takes. Anything else -- skewed, part of it, resampled, an inline image -- is the renderer's.
    """
    if len(under) != 1:
        return None
    box, width, height, number, _, placement = under[0]
    if number is None or placement is None:  # inline, or skewed
        return None
    turned, rows_reversed, columns_reversed = placement  # as the reader places its samples
    across, down = (height, width) if turned else (width, height)
    slack = WHOLE_IMAGE_SLACK * 72 / dpi
    sides = (across * 72 / dpi - (bbox.x1 - bbox.x0), down * 72 / dpi - (bbox.y1 - bbox.y0))
    edges = (box[0] - bbox.x0, box[1] - bbox.y0, box[2] - bbox.x1, box[3] - bbox.y1)
    if any(abs(gap) > slack for gap in (*sides, *edges)):
        return None
    rows = _rgb(reader.image(number))
    if rows is None:
        return None
    if turned:
        rows = rows.transpose(1, 0, 2)
    return np.array(rows[:: -1 if rows_reversed else 1, :: -1 if columns_reversed else 1])


def _rgb(samples: tuple) -> np.ndarray | None:
    """An image's samples as RGB rows; None unless the reader holds them as they show.

    It does for lossless gray of 1, 2, 4 or 8 bits (its levels spread over 0..255) and 8-bit RGB (:data:`_GRAY_OR_RGB`),
    unmasked, mapped as stated by default (no ``/Decode``, or ``[0 1]`` for each component), holding every row. A
    codec, Indexed, CMYK, 16 bits, another ``/Decode`` or a mask is the renderer's.
    """
    width, height, space, components, bits, data, codec, decode, masked, _ = samples
    remapped = decode is not None and decode != [0.0, 1.0] * components
    if codec is not None or masked or remapped or space not in _GRAY_OR_RGB:
        return None
    if not ((components == 1 and bits in _GRAY_BITS) or (components == 3 and bits == 8)):  # noqa: PLR2004 -- RGB
        return None
    stride = (width * components * bits + 7) // 8  # each row ends on a whole byte
    if len(data) < stride * height:
        return None
    rows = np.frombuffer(data, dtype=np.uint8, count=stride * height).reshape(height, stride)
    if bits < 8:  # noqa: PLR2004 -- whole bytes
        places = np.unpackbits(rows, axis=1).reshape(height, -1, bits)
        weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint8)
        rows = (places * weights).sum(axis=2, dtype=np.uint8)[:, :width] * np.uint8(255 // (2**bits - 1))
    pixels = rows[:, : width * components].reshape(height, width, components)
    return np.repeat(pixels, 3, axis=2) if components == 1 else pixels


def _rendered(data: bytes, page: int, bbox: BBox, dpi: float, password: str) -> np.ndarray:
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        if password:
            pdf.authenticate(password)
        zoom = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pdf[page - 1].get_pixmap(matrix=zoom, clip=pymupdf.Rect(*bbox), colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
