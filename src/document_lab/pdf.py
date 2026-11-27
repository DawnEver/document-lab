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
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Final, overload

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


def read_pdf(
    path: Path | str,
    *,
    password: str = '',
    flatness: float | None = None,
    curve_samples: int | None = None,
    annotations: bool = False,
) -> Document:
    """The PDF at ``path``, opened with ``password`` when it is encrypted; each page read when first asked for.

    The file is read whole here, hashed, and kept in memory for the document's life: no file handle stays open,
    and the pages read later are those of the bytes hashed, whatever becomes of the file.

    Each curve of a vector path is flattened into a polyline: into as few segments as keep it within ``flatness``
    points of the curve, or into ``curve_samples`` segments each, whatever its size; neither given, into
    :data:`CURVE_SAMPLES` each.

    ``annotations``: each page's annotations' appearances read too, after its content, each value citing its
    annotation's subtype -- off by default, as a page's text is its content's (PyMuPDF's too). Optional content the
    file's default view hides is read either way, each value flagged ``hidden`` for the caller to keep or drop.

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
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    uri = path.resolve().as_uri()
    reader = _pdf.open(data, password.encode())
    reading = {**flattening, 'annotations': annotations}
    return Document(sha256=sha, uri=uri, pages=_Pages(reader, sha, uri, reading), repairs=_Repairs(reader))


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

    __slots__ = ('_kept', '_reader', '_reading', '_sha', '_sizes', '_uri')

    def __init__(self, reader: _pdf.Reader, sha: str, uri: str, reading: dict[str, float | bool]) -> None:
        self._reader, self._sha, self._uri, self._reading = reader, sha, uri, reading
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
            page = self._kept[i] = _page(self._reader, number, (width, height), here, self._reading)
        return page

    def __repr__(self) -> str:
        return f'<{len(self)} pages, {sum(p is not None for p in self._kept)} read>'


def _page(
    reader: _pdf.Reader, number: int, size: tuple[float, float], here: Provenance, reading: dict[str, float | bool]
) -> Page:
    """Page ``number`` read as ``reading`` says: its curves' flattening, and whether its annotations are drawn."""
    spans, paths, images, warnings = reader.page(number, **reading)
    cite = partial(Provenance, here.sha256, here.uri, number)  # a box on this page: a Provenance made directly
    return Page(
        number=number,
        width=size[0],
        height=size[1],
        spans=tuple(_span(row, cite) for row in spans),
        paths=_paths(paths, cite),
        figures=tuple(
            Figure(cite(bbox=box), w, h, image, hidden=hidden, annotation=annotation)
            for box, w, h, image, _, _, (hidden, annotation) in images
        ),
        warnings=tuple(warnings),
    )


def _span(row: tuple, cite: Callable[..., Provenance]) -> TextSpan:
    text, font, size, bold, box, mode, unspaced, vertical, direction, (hidden, annotation) = row
    return TextSpan(
        text,
        size,
        font,
        cite(bbox=box),
        bold,
        unspaced=unspaced,
        render_mode=mode,
        vertical=vertical,
        direction=direction,
        hidden=hidden,
        annotation=annotation,
    )


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
            pattern,
            hidden,
            annotation,
        )
        for stroke, width, dashes, phase, fill, rule, pattern, (hidden, annotation) in styles
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
        stroke, width, dashes, phase, fill, even_odd, pattern, hidden, annotation = painted[group]
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
                pattern=pattern,
                group=group,
                hidden=hidden,
                annotation=annotation,
            )
        )
    return tuple(out)


def read_raster(path: Path | str, page: int, bbox: BBox, *, dpi: float | None = None, password: str = '') -> Raster:
    """The region ``bbox`` of 1-based ``page`` as pixels, everything drawn there composited.

    What is drawn there is what the page shows: its annotations' appearances too, nothing a hidden layer draws.
    ``dpi`` defaults to the finest native resolution among the images shown the region overlaps, so a pixel is a
    source pixel and none is invented; a region holding no image renders at ``VECTOR_DPI``. A region that is one image
    whole at its native resolution, nothing else visible on it, is that image's own samples where the reader holds
    them as the page shows them (:func:`_samples`); any other region is rendered by PyMuPDF.
    """
    path = Path(path)
    data = path.read_bytes()
    reader = _pdf.open(data, password.encode())
    spans, paths, images, _ = reader.page(page, annotations=True, curve_samples=1)
    spans, images = _shown(spans), _shown(images)
    under = [row for row in images if _meets(row[0], bbox)]
    if dpi is None:
        dpi = max(filter(None, map(_native, under)), default=VECTOR_DPI)
    pixels = _samples(reader, under, bbox, dpi) if _alone(spans, paths, bbox) else None
    if pixels is None:
        pixels = _rendered(data, page, bbox, dpi, password)
    here = Provenance(hashlib.sha256(data).hexdigest(), path.resolve().as_uri(), page=page, bbox=bbox)
    return Raster(pixels, here)


def _shown(rows: list) -> list:
    """The rows (spans or images, each ending in its layer) no hidden layer draws."""
    return [row for row in rows if not row[-1][0]]


def _meets(box: tuple[float, float, float, float], bbox: BBox) -> bool:
    return box[0] < bbox.x1 and box[2] > bbox.x0 and box[1] < bbox.y1 and box[3] > bbox.y0


def _native(image: tuple) -> float:
    """An image row's finest resolution (dpi): its samples over its placed sides' lengths; 0 when it has no area."""
    _, width, height, _, (a, b, c, d, _, _), *_ = image
    across, down = math.hypot(a, b), math.hypot(c, d)
    return 72 * max(width / across, height / down) if across and down else 0.0


def _alone(spans: list, paths: tuple, bbox: BBox) -> bool:
    """Whether no path shown and no visible text (an OCR layer is invisible) is drawn on ``bbox``.

    ``paths`` are the reader's columns (``PathColumns`` in ``crates/pybind``): each subpath's box, its stroke's
    width and its layer its painted path's. A path a hidden layer draws is not shown: it is never in the way.
    """
    _, _, boxes, _, groups, styles = paths
    widths = [width for _, width, *_ in styles]
    shown = [not hidden for *_, (hidden, _) in styles]
    for (x0, y0, x1, y1), group in zip(
        np.frombuffer(boxes, dtype=np.float64).reshape(-1, 4).tolist(),
        np.frombuffer(groups, dtype=np.int64).tolist(),
        strict=True,
    ):
        reach = widths[group] / 2
        if shown[group] and _meets((x0 - reach, y0 - reach, x1 + reach, y1 + reach), bbox):
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
    box, width, height, number, _, placement, _ = under[0]
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
