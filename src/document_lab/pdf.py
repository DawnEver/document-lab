"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space. Pixels
are not the reader's: :func:`read_raster` renders a region with PyMuPDF (D4 of task document-lab-pdf-reader).

A tagged file's structure tree comes back as :attr:`~document_lab.ir.Document.structure`, each element holding the
very values it covers: the spans, paths and figures drawn under its marked content, the annotations and images it
refers to. An element's ``/ActualText`` is stated by the spans it covers, as a marked-content sequence's is: the
first in order carries it, the others the empty text, each keeping the text it draws (``drawn``).

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pymupdf

from document_lab import _pdf
from document_lab._pdf import Locked, Malformed, NotPdf, Refused, Unsupported
from document_lab.ir import (
    Attribute,
    BBox,
    Document,
    Element,
    Figure,
    Mark,
    MarkedContent,
    ObjectReference,
    Page,
    Provenance,
    Raster,
    Structure,
    TextSpan,
    VectorPath,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def read_pdf(
    path: Path | str, *, password: str = '', curve_samples: int = CURVE_SAMPLES, annotations: bool = False
) -> Document:
    """Read every page of the PDF at ``path``, opened with ``password`` when it is encrypted.

    ``curve_samples``: the points each curve of a vector path is flattened to. ``annotations``: each page's
    annotations' appearances read too, after its content, each value citing its annotation's subtype -- off by
    default, as a page's text is its content's (PyMuPDF's too). Optional content the file's default view hides is
    read either way, each value flagged ``hidden`` for the caller to keep or drop.

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
    tree = reader.structure()
    if tree is None:
        return Document(sha256=sha, uri=uri, pages=pages, repairs=tuple(reader.repairs))
    pages = _actual_texts(tree, pages)
    return Document(sha256=sha, uri=uri, pages=pages, repairs=tuple(reader.repairs), structure=_structure(tree, pages))


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
            Figure(
                here.at(bbox=BBox(*box)),
                w,
                h,
                image,
                hidden=hidden,
                annotation=annotation,
                mark=_mark(marked),
                artifact=marked[2],
            )
            for box, w, h, image, (hidden, annotation), marked in images
        ),
        warnings=tuple(warnings),
    )


def _mark(row: tuple[int | None, int | None, bool]) -> Mark | None:
    mcid, stream, _ = row
    return None if mcid is None else Mark(mcid, stream)


def _span(row: tuple, here: Provenance) -> TextSpan:
    text, font, size, bold, box, _mode, unspaced, (hidden, annotation), drawn, marked = row
    return TextSpan(
        text,
        size,
        font,
        here.at(bbox=BBox(*box)),
        bold,
        unspaced,
        hidden=hidden,
        annotation=annotation,
        drawn=drawn,
        mark=_mark(marked),
        artifact=marked[2],
    )


def _path(row: tuple, here: Provenance) -> VectorPath:
    points, closed, stroke, width, dashes, phase, fill, even_odd, pattern, group, (hidden, annotation), marked = row
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
        annotation=annotation,
        mark=_mark(marked),
        artifact=marked[2],
    )


def read_raster(path: Path | str, page: int, bbox: BBox, *, dpi: float | None = None, password: str = '') -> Raster:
    """The region ``bbox`` of 1-based ``page`` as pixels, everything drawn there composited.

    ``dpi`` defaults to the finest native resolution among the images shown (an annotation's too, not one hidden) the
    region overlaps, so a pixel is a source pixel and none is invented; a region holding no image renders at
    ``VECTOR_DPI``.
    """
    path = Path(path)
    data = path.read_bytes()
    if dpi is None:
        _, _, images, _ = _pdf.open(data, password.encode()).page(page, 1, annotations=True)
        native = [
            72 * w / (box[2] - box[0])
            for box, w, _, _, (hidden, _), _ in images
            if not hidden
            and box[2] > box[0]
            and box[0] < bbox.x1
            and box[2] > bbox.x0
            and box[1] < bbox.y1
            and box[3] > bbox.y0
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
