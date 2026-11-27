"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space. Pixels
are not the reader's: :func:`read_raster` renders a region with PyMuPDF (D4 of task document-lab-pdf-reader).

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Final, overload

import numpy as np
import pymupdf

from document_lab import _pdf
from document_lab._pdf import Locked, Malformed, NotPdf, Refused, Unsupported
from document_lab.ir import BBox, Document, Figure, Page, Provenance, Raster, TextSpan, VectorPath, file_uri

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

#: The segments each curve is flattened into by default. Measured 2026-11-26 against a flatness (the other way):
#: on 283 real PDFs (6499 pages) a flatness of 0.1 point makes 8.1M points where 8 per curve make 22.5M, and
#: bounds the error on a large arc that 8 per curve exceed (half a point on an arc a few inches across); but a
#: glyph drawn as an outline is read off a raster 32 rows high (:mod:`document_lab.glyphs`), and on 1991 such
#: shapes of the heaviest documents, rasterized against 64 per curve, 8 per curve differ in 0.14% of the pixels
#: where a flatness of 0.1, 0.05, 0.02, 0.01 point differs in 7.2%, 3.0%, 1.7%, 0.8%: a point is coarse at a
#: glyph's scale.
CURVE_SAMPLES: Final = 8
#: The resolution (dpi) a region holding no image renders at.
VECTOR_DPI: Final = 300.0


def read_pdf(
    path: Path | str, *, password: str = '', flatness: float | None = None, curve_samples: int | None = None
) -> Document:
    """The PDF at ``path``, opened with ``password`` when it is encrypted; each page read when first asked for.

    The reader reads the file whole, hashes it, and keeps it in its memory for the document's life (Python holds no
    copy): no file handle stays open, and the pages read later are those of the bytes hashed, whatever becomes of
    the file.

    Each curve of a vector path is flattened into a polyline: into as few segments as keep it within ``flatness``
    points of the curve, or into ``curve_samples`` segments each, whatever its size; neither given, into
    :data:`CURVE_SAMPLES` each.

    Raises:
        ValueError: both ``flatness`` and ``curve_samples`` are given, or one is not above 0.
        NotPdf: the file is no PDF.
        Locked: it is encrypted and ``password`` opens it for no one.
        Unsupported: it uses what the reader does not read.
        Malformed: it is not what a PDF must be.

    """
    if curve_samples is not None and flatness is not None:
        msg = f'a curve is flattened by a flatness or by curve_samples, not both: {flatness}, {curve_samples}'
        raise ValueError(msg)
    flattening = (
        {'flatness': flatness}
        if flatness is not None
        else {'curve_samples': CURVE_SAMPLES if curve_samples is None else curve_samples}
    )
    if (flatness is not None and not flatness > 0) or (curve_samples is not None and curve_samples < 1):
        msg = f'a curve is flattened within a flatness above 0, or into curve_samples of 1 or more: {flattening}'
        raise ValueError(msg)
    path = Path(path)
    reader = _pdf.open(path, password.encode())
    sha, uri = reader.sha256, file_uri(path.resolve())
    return Document(sha256=sha, uri=uri, pages=_Pages(reader, sha, uri, flattening), repairs=_Repairs(reader))


class _Repairs(Sequence[str]):
    """What the reader has repaired so far: opening the file, then reading each page read since."""

    __slots__ = ('_reader',)

    def __init__(self, reader: _pdf.Reader) -> None:
        self._reader = reader

    def __len__(self) -> int:
        return len(self._reader.repairs)

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...
    def __getitem__(self, index: int | slice) -> str | list[str]:
        return self._reader.repairs[index]

    def __repr__(self) -> str:
        return repr(self._reader.repairs)


class _Pages(Sequence[Page]):
    """A document's pages, each read when it is first asked for, then kept for the document's life.

    Kept, not read again: callers come back to pages (``per_page`` passes over a document four times, ``paper``
    goes back and forth between captions and figures), and a page read again would cost the reading again and be
    another object than the one a caller holds. A page kept is mostly arrays -- its drawing's points are one
    array, no Python object per point -- so every page kept costs no more than reading them all at once did.
    """

    __slots__ = ('_flattening', '_kept', '_reader', '_sha', '_sizes', '_uri')

    def __init__(self, reader: _pdf.Reader, sha: str, uri: str, flattening: dict[str, float]) -> None:
        self._reader, self._sha, self._uri, self._flattening = reader, sha, uri, flattening
        self._sizes: list[tuple[int, float, float]] = reader.pages
        self._kept: list[Page | None] = [None] * len(self._sizes)

    def __len__(self) -> int:
        return len(self._sizes)

    @overload
    def __getitem__(self, index: int) -> Page: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Page, ...]: ...
    def __getitem__(self, index: int | slice) -> Page | tuple[Page, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(len(self))[index])
        i = range(len(self))[index]
        page = self._kept[i]
        if page is None:
            number, width, height = self._sizes[i]
            here = Provenance(self._sha, self._uri, page=number)
            page = self._kept[i] = _page(self._reader, number, (width, height), here, self._flattening)
        return page

    def __repr__(self) -> str:
        return f'<{len(self)} pages, {sum(p is not None for p in self._kept)} read>'


def _page(
    reader: _pdf.Reader, number: int, size: tuple[float, float], here: Provenance, flattening: dict[str, float]
) -> Page:
    spans, paths, images, warnings = reader.page(number, **flattening)
    cite = partial(Provenance, here.sha256, here.uri, number)  # a box on this page: a Provenance made directly
    return Page(
        number=number,
        width=size[0],
        height=size[1],
        spans=tuple(_span(row, cite) for row in spans),
        paths=_paths(paths, cite),
        figures=tuple(Figure(cite(bbox=box), w, h, image) for box, w, h, image in images),
        warnings=tuple(warnings),
    )


def _span(row: tuple, cite: Callable[..., Provenance]) -> TextSpan:
    text, font, size, bold, box, _mode, unspaced = row
    return TextSpan(text, size, font, cite(bbox=box), bold, unspaced)


def _paths(columns: tuple, cite: Callable[..., Provenance]) -> tuple[VectorPath, ...]:
    """A page's subpaths off the reader's columns (``PathColumns`` in ``crates/pybind``): no Python object per point.

    Every subpath's points are a read-only view of one array holding the page's points; its painted path's style
    is shared by its subpaths.
    """
    points, bounds, boxes, closed, groups, styles = columns
    xy = np.frombuffer(points, dtype=np.float64).reshape(-1, 2)
    starts = np.frombuffer(bounds, dtype=np.int64).tolist()
    painted = [
        (
            None if stroke is None else tuple(stroke),
            width,
            tuple(dashes),
            phase,
            None if fill is None else tuple(fill),
            rule,
        )
        for stroke, width, dashes, phase, fill, rule in styles
    ]
    out = []
    for start, end, box, shut, group in zip(
        starts[:-1],
        starts[1:],
        np.frombuffer(boxes, dtype=np.float64).reshape(-1, 4).tolist(),
        np.frombuffer(closed, dtype=np.uint8).astype(bool).tolist(),
        np.frombuffer(groups, dtype=np.int64).tolist(),
        strict=True,
    ):
        stroke, width, dashes, phase, fill, even_odd = painted[group]
        out.append(
            VectorPath(
                points=xy[start:end],
                stroke=stroke,
                width=width,
                dashes=dashes,
                closed=shut,
                provenance=cite(bbox=box),
                dash_phase=phase,
                fill=fill,
                even_odd=even_odd,
                group=group,
            )
        )
    return tuple(out)


def read_raster(path: Path | str, page: int, bbox: BBox, *, dpi: float | None = None, password: str = '') -> Raster:
    """The region ``bbox`` of 1-based ``page`` as pixels, everything drawn there composited.

    ``dpi`` defaults to the finest native resolution among the images the region overlaps, so a pixel is a source
    pixel and none is invented; a region holding no image renders at ``VECTOR_DPI``.
    """
    path = Path(path)
    data = path.read_bytes()  # rendered and cited alike: the pixels are those of the bytes hashed
    reader = _pdf.open(data, password.encode())
    if dpi is None:
        _, _, images, _ = reader.page(page, curve_samples=1)  # its images alone
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
    here = Provenance(reader.sha256, file_uri(path.resolve()), page=page, bbox=bbox)
    return Raster(pixels, here)
