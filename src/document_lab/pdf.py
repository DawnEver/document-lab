"""A PDF read into the IR with PyMuPDF: text spans, vector paths, ruled tables and image figures, each with its box."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pymupdf

from document_lab.ir import BBox, Document, Figure, Page, Provenance, Raster, Table, TextSpan, VectorPath

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ['read_pdf', 'read_raster']

#: PyMuPDF's span flag bit for a bold font.
BOLD_FLAG: Final = 16
#: Points per cubic Bezier segment when a curve is flattened to a polyline.
BEZIER_SAMPLES: Final = 8
#: How far below an image (points) a caption may start.
CAPTION_GAP: Final = 30.0
#: The resolution (dpi) a region holding no image is rendered at.
VECTOR_DPI: Final = 300.0
_CAPTION_RE: Final = re.compile(r'^\s*(?:fig\.?|figure)\s*\d+', re.IGNORECASE)


def read_pdf(path: Path | str, *, tables: bool = True) -> Document:
    """Read every page of the PDF at ``path``; ``tables=False`` skips table detection, the slow part."""
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    uri = path.resolve().as_uri()
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        pages = tuple(_page(page, Provenance(sha, uri, page=page.number + 1), tables=tables) for page in pdf)
    return Document(sha256=sha, uri=uri, pages=pages)


def read_raster(path: Path | str, page: int, bbox: BBox, *, dpi: float | None = None) -> Raster:
    """The region ``bbox`` of 1-based ``page`` as pixels, everything drawn there composited.

    ``dpi`` defaults to the finest native resolution among the images the region overlaps, so a pixel is a source
    pixel and none is invented; a region holding no image renders at ``VECTOR_DPI``.
    """
    path = Path(path)
    data = path.read_bytes()
    clip = pymupdf.Rect(*bbox)
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        source = pdf[page - 1]
        if dpi is None:
            native = [
                72 * info['width'] / pymupdf.Rect(info['bbox']).width
                for info in source.get_image_info()
                if pymupdf.Rect(info['bbox']).intersects(clip) and pymupdf.Rect(info['bbox']).width > 0
            ]
            dpi = max(native, default=VECTOR_DPI)
        zoom = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = source.get_pixmap(matrix=zoom, clip=clip, colorspace=pymupdf.csRGB, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
    here = Provenance(hashlib.sha256(data).hexdigest(), path.resolve().as_uri(), page=page, bbox=bbox)
    return Raster(pixels, here)


def _page(page: pymupdf.Page, here: Provenance, *, tables: bool) -> Page:
    blocks = page.get_text('dict')['blocks']
    spans = tuple(_spans(blocks, here))
    return Page(
        number=here.page or 0,
        width=page.rect.width,
        height=page.rect.height,
        spans=spans,
        paths=tuple(_paths(page, here)),
        tables=tuple(_tables(page, here)) if tables else (),
        figures=tuple(_figures(page, blocks, here)),
    )


def _spans(blocks: list[dict], here: Provenance) -> Iterator[TextSpan]:
    for block in blocks:
        for line in block.get('lines', ()):
            for span in line['spans']:
                if span['text'].strip():
                    bold = bool(span['flags'] & BOLD_FLAG) or 'bold' in span['font'].lower()
                    yield TextSpan(span['text'], span['size'], span['font'], here.at(bbox=BBox(*span['bbox'])), bold)


def _bezier(p0: pymupdf.Point, c1: pymupdf.Point, c2: pymupdf.Point, p1: pymupdf.Point) -> list[tuple[float, float]]:
    out = []
    for i in range(1, BEZIER_SAMPLES + 1):
        t = i / BEZIER_SAMPLES
        u = 1 - t
        x = u**3 * p0.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t**3 * p1.x
        y = u**3 * p0.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t**3 * p1.y
        out.append((x, y))
    return out


def _paths(page: pymupdf.Page, here: Provenance) -> Iterator[VectorPath]:
    """Each drawing split into its connected polylines; a rectangle or quad is its own closed path."""
    for drawing in page.get_drawings():
        stroke = tuple(drawing['color']) if drawing.get('color') is not None else None
        width = drawing.get('width') or 0.0
        dashes = _dash_array(drawing.get('dashes'))
        runs: list[list[tuple[float, float]]] = []
        closed_shapes: list[list[tuple[float, float]]] = []
        for item in drawing['items']:
            kind = item[0]
            if kind == 're':
                r = item[1]
                closed_shapes.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
                continue
            if kind == 'qu':
                q = item[1]
                closed_shapes.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)])
                continue
            start = (item[1].x, item[1].y)
            tail = [(item[2].x, item[2].y)] if kind == 'l' else _bezier(*item[1:5])
            if not runs or runs[-1][-1] != start:
                runs.append([start])
            runs[-1].extend(tail)
        closes = bool(drawing.get('closePath'))
        for points in closed_shapes:
            yield _vector(points, stroke, width, dashes, closed=True, here=here)
        for points in runs:
            yield _vector(points, stroke, width, dashes, closed=closes and len(runs) == 1, here=here)


def _vector(
    points: list[tuple[float, float]],
    stroke: tuple[float, ...] | None,
    width: float,
    dashes: tuple[float, ...],
    *,
    closed: bool,
    here: Provenance,
) -> VectorPath:
    xs, ys = zip(*points, strict=True)
    box = BBox(min(xs), min(ys), max(xs), max(ys))
    return VectorPath(tuple(points), stroke, width, dashes, closed, here.at(bbox=box))


def _dash_array(spec: str | None) -> tuple[float, ...]:
    """The array of a PDF dash spec such as ``[ 3 2 ] 0`` (the phase is dropped); ``()`` for a solid stroke."""
    if not spec:
        return ()
    inside = spec[spec.index('[') + 1 : spec.index(']')]
    return tuple(float(v) for v in inside.split())


def _tables(page: pymupdf.Page, here: Provenance) -> Iterator[Table]:
    """Each detected table; a cell's text is read from its own box, so text and provenance are one fact.

    (``Table.extract()`` of pymupdf 1.27.2.3 reorders characters: ``1.25`` came back as ``125`` and ``.`` on two lines.)
    """
    for index, found in enumerate(page.find_tables().tables):
        boxes = tuple(tuple(BBox(*c) if c else None for c in row.cells) for row in found.rows)
        rows = tuple(tuple(_text_in(page, box) if box else '' for box in row) for row in boxes)
        yield Table(rows, boxes, here.at(bbox=BBox(*found.bbox), locator=f'table[{index}]'))


def _text_in(page: pymupdf.Page, box: BBox) -> str:
    return ' '.join(page.get_text('text', clip=pymupdf.Rect(*box)).split())


def _figures(page: pymupdf.Page, blocks: list[dict], here: Provenance) -> Iterator[Figure]:
    texts = [(BBox(*b['bbox']), _block_text(b)) for b in blocks if b['type'] == 0]
    for info in page.get_image_info():
        box = BBox(*info['bbox'])
        yield Figure(here.at(bbox=box), _caption_below(box, texts))


def _block_text(block: dict) -> str:
    return ' '.join(''.join(s['text'] for s in line['spans']) for line in block['lines']).strip()


def _caption_below(box: BBox, texts: list[tuple[BBox, str]]) -> str | None:
    """The nearest text block starting with a figure label, within `CAPTION_GAP` below the box and overlapping it."""
    below = [
        (tb.y0 - box.y1, text)
        for tb, text in texts
        if 0 <= tb.y0 - box.y1 <= CAPTION_GAP and tb.x0 < box.x1 and tb.x1 > box.x0 and _CAPTION_RE.match(text)
    ]
    return min(below)[1] if below else None
