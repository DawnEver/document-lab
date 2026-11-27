"""A PDF read into the IR by document-lab's own reader (``document_lab._pdf``, Rust).

What a page draws comes back as text spans, vector paths and image figures, each with its box in page space, and its
links beside them; what the document states beside its pages -- outline, page labels, metadata, form fields, embedded
files -- on the :class:`Document`. Pixels are not the reader's: :func:`read_raster` renders a region with PyMuPDF (D4
of task document-lab-pdf-reader); an embedded file's bytes are :func:`read_attachment`'s.

What the reader refuses raises a :class:`Refused` of its kind -- :class:`NotPdf`, :class:`Locked` (the password
opens no access), :class:`Unsupported`, :class:`Malformed` -- each a ``ValueError``.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Final

import numpy as np
import pymupdf

from document_lab import _pdf
from document_lab._pdf import Locked, Malformed, NotPdf, Refused, Unsupported
from document_lab.ir import (
    Action,
    Attachment,
    BBox,
    Destination,
    Document,
    Field,
    Figure,
    Info,
    Link,
    Metadata,
    OutlineItem,
    Page,
    Provenance,
    Raster,
    TextSpan,
    VectorPath,
    Widget,
    Xmp,
)

__all__ = [
    'CURVE_SAMPLES',
    'Locked',
    'Malformed',
    'NotPdf',
    'Refused',
    'Unsupported',
    'read_attachment',
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
    read either way, each value flagged ``hidden`` for the caller to keep or drop. Links, the outline and the rest
    of what the document states beside its pages are read either way: they are data, not ink.

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
    outline, labels, metadata, fields, attachments, warnings = reader.document()
    here = Provenance(sha, uri)
    return Document(
        sha256=sha,
        uri=uri,
        pages=pages,
        repairs=tuple(reader.repairs),
        outline=_outline(outline),
        page_labels=() if labels is None else tuple(labels),
        metadata=_metadata(*metadata),
        fields=tuple(_field(row, here) for row in fields),
        attachments=tuple(_attachment(row, here) for row in attachments),
        warnings=tuple(warnings),
    )


def _page(
    reader: _pdf.Reader, number: int, size: tuple[float, float], here: Provenance, *, samples: int, annotations: bool
) -> Page:
    spans, paths, images, warnings = reader.page(number, samples, annotations=annotations)
    links, link_warnings = reader.links(number)
    return Page(
        number=number,
        width=size[0],
        height=size[1],
        spans=tuple(_span(row, here) for row in spans),
        paths=tuple(_path(row, here) for row in paths),
        figures=tuple(
            Figure(here.at(bbox=BBox(*box)), w, h, image, hidden=hidden, annotation=annotation)
            for box, w, h, image, (hidden, annotation) in images
        ),
        links=tuple(
            Link(here.at(bbox=BBox(*box)), _destination(destination), _action(action))
            for box, destination, action in links
        ),
        warnings=(*warnings, *link_warnings),
    )


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


def _span(row: tuple, here: Provenance) -> TextSpan:
    text, font, size, bold, box, _mode, unspaced, (hidden, annotation) = row
    return TextSpan(text, size, font, here.at(bbox=BBox(*box)), bold, unspaced, hidden=hidden, annotation=annotation)


def _path(row: tuple, here: Provenance) -> VectorPath:
    points, closed, stroke, width, dashes, phase, fill, even_odd, pattern, group, (hidden, annotation) = row
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
    )


def read_attachment(path: Path | str, attachment: Attachment, *, password: str = '') -> bytes:
    """The bytes of ``attachment``, embedded in the PDF at ``path`` (opened with ``password``), every filter undone.

    Raises:
        ValueError: ``path`` is not the file ``attachment`` was read from (its content hash differs).

    """
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != attachment.provenance.sha256:
        msg = f'{path}: not the file the attachment was read from (sha256 {sha}, the attachment cites '
        msg += f'{attachment.provenance.sha256})'
        raise ValueError(msg)
    return bytes(_pdf.open(data, password.encode()).attachment(attachment.object))


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
            for box, w, _, _, (hidden, _) in images
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
