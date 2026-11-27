"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space, and its
links beside them; what the document states beside its pages -- outline, page labels, metadata, form fields, embedded
files -- on the :class:`Document`. Pixels are the reader's only as an image's own samples (no codec is decoded):
:func:`read_raster` takes a region that is one image whole from them, and draws any other with a renderer -- PyMuPDF
(the reference, the default) or document-lab's own (``renderer='own'``: the ``render`` crate, drawing from this
reader's interpreter; it draws no text until its stage 2, so it refuses a region holding text -- D37 of task
document-lab-pdf-reader); an embedded file's bytes are :func:`read_attachment`'s.

A tagged file's structure tree comes back as :attr:`~document_lab.ir.Document.structure`, each element holding the
very values it covers: the spans, paths and figures drawn under its marked content, the annotations and images it
refers to. An element's ``/ActualText`` is stated by the spans it covers, as a marked-content sequence's is: the
first in order carries it, the others the empty text, each keeping the text it draws (``drawn``).

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Final, Literal, overload

import numpy as np
import pymupdf

from document_lab import _pdf
from document_lab._pdf import Locked, Malformed, NotPdf, Refused, Unsupported
from document_lab.ir import (
    Action,
    Attachment,
    Attribute,
    BBox,
    Destination,
    Document,
    Element,
    Field,
    Figure,
    Info,
    Link,
    Mark,
    MarkedContent,
    Metadata,
    ObjectReference,
    OutlineItem,
    Page,
    Provenance,
    Raster,
    Structure,
    TextSpan,
    VectorPath,
    Widget,
    Xmp,
    file_uri,
)

__all__ = [
    'CURVE_SAMPLES',
    'Locked',
    'Malformed',
    'NotPdf',
    'Refused',
    'Renderer',
    'Unsupported',
    'read_attachment',
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
#: Where a span row and an image row (``Reader.page``) hold their layer.
_SPAN_LAYER: Final = 9
_IMAGE_LAYER: Final = 6
#: The text rendering mode that neither fills nor strokes (an OCR layer): text that shows nothing.
INVISIBLE_TEXT: Final = 3

#: What draws a region: PyMuPDF, or document-lab's own renderer (no text until its stage 2).
Renderer = Literal['pymupdf', 'own']


def read_pdf(
    path: Path | str,
    *,
    password: str = '',
    flatness: float | None = None,
    curve_samples: int | None = None,
    annotations: bool = False,
) -> Document:
    """The PDF at ``path``, opened with ``password`` when it is encrypted; each page read when first asked for.

    The reader reads the file whole, hashes it, and keeps it in its memory for the document's life (Python holds no
    copy): no file handle stays open, and the pages read later are those of the bytes hashed, whatever becomes of
    the file.

    Each curve of a vector path is flattened into a polyline: into as few segments as keep it within ``flatness``
    points of the curve, or into ``curve_samples`` segments each, whatever its size; neither given, into
    :data:`CURVE_SAMPLES` each.

    ``annotations``: each page's annotations' appearances read too, after its content, each value citing its
    annotation's subtype -- off by default, as a page's text is its content's (PyMuPDF's too). Optional content the
    file's default view hides, and what the colorant ``None`` paints, are read either way, each value flagged
    ``hidden`` or ``inkless`` for the caller to keep or drop (`document_lab.ir`). Links, the outline and the rest of
    what the document states beside its pages are read either way: they are data, not ink.

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
    reading = {**flattening, 'annotations': annotations}
    outline, labels, metadata, fields, attachments, warnings = reader.document()
    here = Provenance(sha, uri)
    pages: Sequence[Page] = _Pages(reader, sha, uri, reading)
    tree = reader.structure()
    structure = None
    if tree is not None:
        pages = _actual_texts(tree, tuple(pages))  # a tagged file's structure holds every page's values: read them now
        structure = _structure(tree, pages)
    return Document(
        sha256=sha,
        uri=uri,
        pages=pages,
        repairs=_Repairs(reader),
        structure=structure,
        outline=_outline(outline),
        page_labels=() if labels is None else tuple(labels),
        metadata=_metadata(*metadata),
        fields=tuple(_field(row, here) for row in fields),
        attachments=tuple(_attachment(row, here) for row in attachments),
        warnings=tuple(warnings),
    )


type _Drawn = TextSpan | VectorPath | Figure


def _drawn_under(pages: Sequence[Page]) -> dict[tuple[int, Mark], list[_Drawn]]:
    """Every value drawn under a mark, by page and mark, in drawing order (an annotation's own content apart)."""
    found: dict[tuple[int, Mark], list[_Drawn]] = defaultdict(list)
    for page in pages:
        for value in (*page.spans, *page.paths, *page.figures):
            if value.mark is not None and (value.annotation is None or value.mark.stream is not None):
                found[page.number, value.mark].append(value)
    return found


def _marks(tree: tuple, index: int) -> list[tuple[int, Mark]]:
    """The marked content element ``index`` of ``tree`` and those below it cover, in order."""
    elements = tree[0]
    out = []
    for kind, at, mcid, number, _ in elements[index][9]:
        if kind == 'element':
            out += _marks(tree, at)
        elif kind == 'marked':
            out.append((at, Mark(mcid, number)))
    return out


def _actual_texts(tree: tuple, pages: tuple[Page, ...]) -> tuple[Page, ...]:
    """``pages`` with each element's ``/ActualText`` stated by the spans it covers.

    The first carries it, every later one the empty text, each keeping the text it draws; an outer element's governs
    an inner one's.
    """
    elements, roots, _ = tree
    drawn = _drawn_under(pages)
    replaced: dict[int, TextSpan] = {}

    def visit(index: int) -> None:
        actual = elements[index][5]
        if actual is None:
            for kind, at, *_ in elements[index][9]:
                if kind == 'element':
                    visit(at)
            return
        spans = [v for key in _marks(tree, index) for v in drawn.get(key, ()) if isinstance(v, TextSpan)]
        for i, span in enumerate(spans):
            replaced[id(span)] = dataclasses.replace(
                span, text=actual if i == 0 else '', drawn=span.drawn if span.drawn is not None else span.text
            )

    for root in roots:
        visit(root)
    if not replaced:
        return pages
    return tuple(dataclasses.replace(page, spans=tuple(replaced.get(id(s), s) for s in page.spans)) for page in pages)


def _structure(tree: tuple, pages: tuple[Page, ...]) -> Structure:
    """The structure tree's rows as elements holding the values of ``pages`` they cover."""
    elements, roots, warnings = tree
    drawn = _drawn_under(pages)
    placed: dict[tuple[int, int], tuple[Figure, ...]] = defaultdict(tuple)
    for page in pages:
        for figure in page.figures:
            if figure.image is not None:
                placed[page.number, figure.image] += (figure,)

    def kid(row: tuple) -> Element | MarkedContent | ObjectReference:
        kind, at, mcid, number, box = row
        if kind == 'element':
            return element(at)
        if kind == 'marked':
            mark = Mark(mcid, number)
            values = drawn.get((at, mark), [])
            return MarkedContent(
                at,
                mark,
                spans=tuple(v for v in values if isinstance(v, TextSpan)),
                paths=tuple(v for v in values if isinstance(v, VectorPath)),
                figures=tuple(v for v in values if isinstance(v, Figure)),
            )
        return ObjectReference(
            number,
            page=at,
            bbox=None if box is None else BBox(*box),
            figures=placed.get((at, number), ()) if at is not None else (),
        )

    def element(index: int) -> Element:
        role, standard, title, lang, alt, actual, expansion, attributes, page, kids = elements[index]
        return Element(
            role,
            standard,
            kids=tuple(kid(row) for row in kids),
            title=title,
            lang=lang,
            alt=alt,
            actual_text=actual,
            expansion=expansion,
            attributes=tuple(Attribute(owner, name, _frozen(value)) for owner, name, value in attributes),
            page=page,
        )

    return Structure(tuple(element(i) for i in roots), tuple(warnings))


def _frozen(value: object) -> object:
    """An attribute's value with each list made a tuple: the IR is immutable."""
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_frozen(v) for v in value)
    return value


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
    links, link_warnings = reader.links(number)
    return Page(
        number=number,
        width=size[0],
        height=size[1],
        spans=tuple(_span(row, cite) for row in spans),
        paths=_paths(paths, cite),
        figures=tuple(
            Figure(
                cite(bbox=box),
                w,
                h,
                image,
                hidden=hidden,
                inkless=inkless,
                annotation=annotation,
                mark=_mark(marked),
                artifact=marked[2],
            )
            for box, w, h, image, _, _, (hidden, inkless, annotation), marked in images
        ),
        links=tuple(
            Link(here.at(bbox=BBox(*box)), _destination(destination), _action(action))
            for box, destination, action in links
        ),
        warnings=(*warnings, *link_warnings),
    )


def _mark(row: tuple[int | None, int | None, bool]) -> Mark | None:
    mcid, stream, _ = row
    return None if mcid is None else Mark(mcid, stream)


def _destination(row: tuple | None) -> Destination | None:
    if row is None:
        return None
    page, fit, x, y, zoom, box = row
    return Destination(page, fit, x, y, zoom, None if box is None else BBox(*box))


def _action(row: tuple | None) -> Action | None:
    if row is None:
        return None
    kind, destination, uri, file, name = row
    return Action(kind, _destination(destination), uri, file, name)


def _outline(rows: list[tuple]) -> tuple[OutlineItem, ...]:
    """The outline tree from its items flattened in document order, each with its depth."""

    def level(at: int, depth: int) -> tuple[tuple[OutlineItem, ...], int]:
        items = []
        while at < len(rows) and rows[at][0] == depth:
            _, title, destination, action = rows[at]
            children, at = level(at + 1, depth + 1)
            items.append(OutlineItem(title, _destination(destination), _action(action), children))
        return tuple(items), at

    return level(0, 0)[0]


#: The ``/Info`` keys :class:`Info` names, by its field.
_INFO_KEYS: Final = {
    'Title': 'title',
    'Author': 'author',
    'Subject': 'subject',
    'Keywords': 'keywords',
    'Creator': 'creator',
    'Producer': 'producer',
    'CreationDate': 'created',
    'ModDate': 'modified',
}


def _metadata(info: list[tuple[str, str]] | None, xmp: tuple | None, newer: str | None) -> Metadata:
    named = None
    if info is not None:
        known = {_INFO_KEYS[k]: v for k, v in info if k in _INFO_KEYS}
        named = Info(**known, other=tuple((k, v) for k, v in info if k not in _INFO_KEYS))
    packet = None
    if xmp is not None:
        title, creators, description, created, modified, metadata_date, tool, producer, keywords, text = xmp
        packet = Xmp(
            text, title, tuple(creators), description, created, modified, metadata_date, tool, producer, keywords
        )
    return Metadata(named, packet, newer)


def _value(value: str | list[str] | None) -> str | tuple[str, ...] | None:
    return tuple(value) if isinstance(value, list) else value


def _field(row: tuple, here: Provenance) -> Field:
    name, kind, flags, value, default, options, widgets = row
    return Field(
        name,
        kind,
        flags,
        _value(value),
        _value(default),
        tuple(options),
        tuple(
            Widget(dataclasses.replace(here, page=page, bbox=None if box is None else BBox(*box)), state, export)
            for page, box, state, export in widgets
        ),
    )


def _attachment(row: tuple, here: Provenance) -> Attachment:
    key, name, description, mime, size, created, modified, page, obj = row
    return Attachment(dataclasses.replace(here, page=page), obj, key, name, description, mime, size, created, modified)


def _span(row: tuple, cite: Callable[..., Provenance]) -> TextSpan:
    text, font, size, bold, box, mode, unspaced, vertical, direction, layer, fill, (drawn, marked) = row
    hidden, _, annotation = layer
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
        fill=None if fill is None else tuple(fill),
        drawn=drawn,
        mark=_mark(marked),
        artifact=marked[2],
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
            inkless,
            annotation,
            marked,
        )
        for stroke, width, dashes, phase, fill, rule, pattern, (hidden, inkless, annotation), marked in styles
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
        stroke, width, dashes, phase, fill, even_odd, pattern, hidden, inkless, annotation, marked = painted[group]
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
                inkless=inkless,
                annotation=annotation,
                mark=_mark(marked),
                artifact=marked[2],
            )
        )
    return tuple(out)


def read_attachment(path: Path | str, attachment: Attachment, *, password: str = '') -> bytes:
    """The bytes of ``attachment``, embedded in the PDF at ``path`` (opened with ``password``), every filter undone.

    Raises:
        ValueError: ``path`` is not the file ``attachment`` was read from (its content hash differs).

    """
    reader = _pdf.open(Path(path), password.encode())
    if reader.sha256 != attachment.provenance.sha256:
        msg = f'{path}: not the file the attachment was read from (sha256 {reader.sha256}, the attachment cites '
        msg += f'{attachment.provenance.sha256})'
        raise ValueError(msg)
    return bytes(reader.attachment(attachment.object))


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

    What is drawn there is what the page shows: its annotations' appearances too, nothing a hidden layer draws, no
    image the colorant ``None`` paints. ``dpi`` defaults to the finest native resolution among the images shown the
    region overlaps, so a pixel is a source pixel and none is invented; a region holding no image renders at
    ``VECTOR_DPI``. A region that is one image whole at its native resolution, nothing else visible on it, is that
    image's own samples where the reader holds them as the page shows them (:func:`_samples`); any other region is
    drawn by ``renderer``: PyMuPDF, or the own renderer (:func:`_drawn`).

    Raises:
        Unsupported: ``renderer='own'`` and the region holds what it does not draw yet (text, a shading, ...).

    """
    if renderer not in ('pymupdf', 'own'):
        msg = f'renderer {renderer!r}: one of pymupdf, own'
        raise ValueError(msg)
    path = Path(path)
    data = path.read_bytes()  # rendered and cited alike: the pixels are those of the bytes hashed
    reader = _pdf.open(data, password.encode())
    spans, paths, images, _ = reader.page(page, annotations=True, curve_samples=1)
    spans = [row for row in spans if not row[_SPAN_LAYER][0]]  # text is text whatever its color: hidden only
    under = [row for row in images if not any(row[_IMAGE_LAYER][:2]) and _meets(row[0], bbox)]
    if dpi is None:
        dpi = max(filter(None, map(_native, under)), default=VECTOR_DPI)
    pixels = _samples(reader, under, bbox, dpi) if _alone(spans, paths, bbox) else None
    if pixels is None and renderer == 'own':
        pixels = _drawn(reader, page, bbox, dpi, spans)
    if pixels is None:
        pixels = _rendered(data, page, bbox, dpi, password)
    here = Provenance(reader.sha256, file_uri(path.resolve()), page=page, bbox=bbox)
    return Raster(pixels, here)


def _meets(box: tuple[float, float, float, float], bbox: BBox) -> bool:
    return box[0] < bbox.x1 and box[2] > bbox.x0 and box[1] < bbox.y1 and box[3] > bbox.y0


def _native(image: tuple) -> float:
    """An image row's finest resolution (dpi): its samples over its placed sides' lengths; 0 when it has no area."""
    _, width, height, _, (a, b, c, d, _, _), *_ = image
    across, down = math.hypot(a, b), math.hypot(c, d)
    return 72 * max(width / across, height / down) if across and down else 0.0


def _alone(spans: list, paths: tuple, bbox: BBox) -> bool:
    """Whether no path that shows and no visible text (an OCR layer is invisible) is drawn on ``bbox``.

    ``paths`` are the reader's columns (``PathColumns`` in ``crates/pybind``): each subpath's box, its stroke's
    width and its layer its painted path's. A path a hidden layer draws, or the colorant ``None`` paints, does not
    show: it is never in the way.
    """
    _, _, boxes, _, groups, styles = paths
    widths = [width for _, width, *_ in styles]
    shows = [not (hidden or inkless) for *_, (hidden, inkless, _), _ in styles]
    for (x0, y0, x1, y1), group in zip(
        np.frombuffer(boxes, dtype=np.float64).reshape(-1, 4).tolist(),
        np.frombuffer(groups, dtype=np.int64).tolist(),
        strict=True,
    ):
        reach = widths[group] / 2
        if shows[group] and _meets((x0 - reach, y0 - reach, x1 + reach, y1 + reach), bbox):
            return False
    return not _texts(spans, bbox)


def _texts(spans: list, bbox: BBox) -> int:
    """How many of ``spans`` (none hidden) are visible text (an OCR layer is invisible) drawn on ``bbox``."""
    return sum(mode != INVISIBLE_TEXT and _meets(box, bbox) for _, _, _, _, box, mode, *_ in spans)


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
    box, width, height, number, _, placement, *_ = under[0]
    if number is None or placement is None:  # inline, or skewed
        return None
    turned, rows_reversed, columns_reversed = placement  # as the reader places its samples
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
    if turned:
        rows = rows.transpose(1, 0, 2)
    return np.array(rows[:: -1 if rows_reversed else 1, :: -1 if columns_reversed else 1])


def _rendered(data: bytes, page: int, bbox: BBox, dpi: float, password: str) -> np.ndarray:
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        if password:
            pdf.authenticate(password)
        zoom = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pdf[page - 1].get_pixmap(matrix=zoom, clip=pymupdf.Rect(*bbox), colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
