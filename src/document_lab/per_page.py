"""A PDF as one markdown file and one image per page: the layout for standards, manuals and data sheets.

A paper reads as sections (`document_lab.paper`); a standard is cited by clause on its page, its tables and figures
in place, so each page is kept whole. ``page-NNN.md`` is the page's text in markdown (`document_lab.markdown`, read
from the PDF's text layer -- a page of a capture needs `document_lab.ocr` first) under a link to ``page-NNN.png``,
the page as it is seen (`document_lab.pdf.read_raster`): a page one image covers is read at that image's own
resolution -- its own samples where the reader holds them as the page shows them -- and any other page is rendered
at :data:`RENDER_DPI`. Every PNG is written by `document_lab.png`.

Each figure, table and equation the page's text and drawn ink locate (`document_lab.regions`, told the picture's
drawings by `document_lab.ink`, less those half the pages repeat: a logo, a running frame) is cut out of that
picture as ``page-NNN-<kind>-<label>.png`` and linked in the markdown after the paragraph holding its caption or
number -- where OCR'd text cannot be trusted (a formula, a table's numbers), the picture beside it can. A region
reaching the page's foot is marked as continuing on the next page. ``INDEX.md`` lists the pages with the clause
headings each opens and the regions it holds. A page without text is written all the same, its markdown empty, and
named in :attr:`Paged.textless` -- never silently dropped.

The PDF itself goes with them, so the layout stands alone. ``out`` is a directory, or -- ending ``.zip`` -- one
archive holding the same files (written beside as ``<out>.part`` and moved into place when whole): a document is
one file to keep, send or index.
"""

from __future__ import annotations

import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab import layout, png
from document_lab.ink import Drawing, drawings
from document_lab.ir import BBox
from document_lab.markdown import page_markdowns
from document_lab.pdf import read_pdf, read_raster
from document_lab.regions import Region, regions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from document_lab.ir import Page

__all__ = ['RENDER_DPI', 'Paged', 'write_pages']

RENDER_DPI: Final = 150  # a page that is not one image is rendered at this resolution
_FOOT: Final = 1.0  # points from the page's foot within which a region runs on to the next page
_COVER: Final = 1.0  # points an image may fall short of each page edge and still cover the page
#: Ink within this many body sizes of other ink is one drawing (`document_lab.ink`); lines of text sit further apart.
DRAWING_REACH: Final = 0.5
#: A drawing at the same place on at least this share of the pages is the pages' furniture (a header's logo).
FURNITURE_SHARE: Final = 0.5


@dataclass(frozen=True)
class Paged:
    """What :func:`write_pages` wrote: ``pages`` pages to ``out``; ``textless`` the 1-based numbers without text."""

    out: Path
    pages: int
    textless: tuple[int, ...]


def _picture(pdf: Path, page: Page) -> tuple[np.ndarray, float]:
    """The page as seen, RGB rows, and its pixels per point: a page one image covers at that image's resolution."""
    whole = BBox(0, 0, page.width, page.height)
    covered = len(page.figures) == 1 and _covers(page.figures[0].provenance.bbox, whole)
    pixels = read_raster(pdf, page.number, whole, dpi=None if covered else RENDER_DPI).pixels
    return pixels, pixels.shape[1] / page.width


def _covers(box: BBox | None, page: BBox) -> bool:
    """Whether ``box`` holds the ``page`` within :data:`_COVER` of each edge."""
    return box is not None and (
        box.x0 <= page.x0 + _COVER
        and box.y0 <= page.y0 + _COVER
        and box.x1 >= page.x1 - _COVER
        and box.y1 >= page.y1 - _COVER
    )


def _drawings(pixels: np.ndarray, scale: float, page: Page, body: float) -> list[Drawing]:
    """The drawn ink of the page's picture (`document_lab.ink`), its text left out: what regions tell drawings by.

    Ink within :data:`DRAWING_REACH` ``body`` sizes of other ink is one drawing: a figure's parts sit that close.
    """
    text = [BBox(ln.x0, ln.y0, ln.x1, ln.y1) for ln in layout.lines(page, weighted=False)]
    return drawings(pixels, scale, text, DRAWING_REACH * body)


def _without_furniture(drawn: list[list[Drawing]], body: float) -> list[list[Drawing]]:
    """Each page's drawings less the pages' furniture: a header's logo, a running frame.

    Furniture is a drawing at the same place on :data:`FURNITURE_SHARE` of the pages or more (two at least). The same
    place is every edge within one ``body`` size; only the whole document can tell, so it is judged here.
    """

    def same(a: BBox, b: BBox) -> bool:
        return all(abs(u - v) <= body for u, v in zip(a, b, strict=True))

    def recurs(d: Drawing) -> bool:
        pages = sum(any(same(d.box, o.box) for o in page) for page in drawn)
        return pages >= max(2, FURNITURE_SHARE * len(drawn))

    return [[d for d in page if not recurs(d)] for page in drawn]


def _cut(pixels: np.ndarray, scale: float, region: Region) -> np.ndarray:
    """``region`` of the page's picture, its pixels as they are."""
    box = region.bbox
    x0, x1 = max(int(box.x0 * scale), 0), min(int(np.ceil(box.x1 * scale)), pixels.shape[1])
    y0, y1 = max(int(box.y0 * scale), 0), min(int(np.ceil(box.y1 * scale)), pixels.shape[0])
    return pixels[y0:y1, x0:x1]


def _file(name: str, region: Region) -> str:
    label = re.sub('(?i)(?:figure|table|equation)', '', region.label).strip('() ')  # the kind is named once
    return f'{name}-{region.kind}-{re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()}.png'


def _linked(text: str, links: list[tuple[str, str]]) -> str:
    """``text`` with each ``(label, link)`` after the first paragraph holding the label, else at its end."""
    blocks = text.rstrip('\n').split('\n\n') if text.strip() else []
    tail: list[str] = []
    for label, link in links:
        at = next((i for i, block in enumerate(blocks) if label in block), None)
        if at is None:
            tail.append(link)
        else:
            blocks.insert(at + 1, link)
    return '\n\n'.join(blocks + tail) + '\n' if blocks or tail else ''


@contextmanager
def _files(out: Path) -> Iterator[Callable[[str, bytes], None]]:
    """A ``put(name, data)`` into ``out``: a directory, or a ``.zip`` archive moved into place only when whole."""
    if out.suffix.lower() != '.zip':
        out.mkdir(parents=True, exist_ok=True)
        yield lambda name, data: (out / name).write_bytes(data)
        return
    out.parent.mkdir(parents=True, exist_ok=True)  # made where it was asked for, as a directory's own name is
    part = out.with_name(out.name + '.part')
    try:
        with zipfile.ZipFile(part, 'w') as archive:

            def put(name: str, data: bytes) -> None:
                stored = name.endswith('.png')  # a PNG is compressed already: deflating it again gains nothing
                archive.writestr(name, data, zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)

            yield put
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(out)


def _clauses(markdown: str) -> str:
    """The clause headings a page opens (``##`` and deeper), ``; ``-joined: what the index names it by."""
    return '; '.join(line.lstrip('#').strip() for line in markdown.splitlines() if line.startswith('##'))


def write_pages(pdf: Path, out: Path) -> Paged:
    """Write ``pdf`` page by page into ``out``: a directory (made if missing), or a ``.zip`` archive."""
    read = read_pdf(pdf)
    body = layout.body_size(read)
    texts = page_markdowns(read)
    width = max(3, len(str(len(texts))))
    index = [f'# {pdf.stem}', '']
    textless = []
    drawn = _without_furniture([_drawings(*_picture(pdf, page), page, body) for page in read.pages], body)
    with _files(out) as put:
        if out / pdf.name != pdf:  # the layout stands alone; a PDF already where it goes is not copied onto itself
            put(pdf.name, pdf.read_bytes())
        for number, (text, page) in enumerate(zip(texts, read.pages, strict=True), start=1):
            name = f'page-{number:0{width}d}'
            pixels, scale = _picture(pdf, page)
            put(f'{name}.png', png.encode(pixels))
            links = []
            for region in (found := regions(page, body, drawn[number - 1])):
                file = _file(name, region)
                put(file, png.encode(_cut(pixels, scale, region)))
                more = ' (continues on the next page)' if region.bbox.y1 >= page.height - _FOOT else ''
                links.append((region.label, f'![{region.label}]({file}){more}'))
            put(f'{name}.md', f'[{pdf.name}, page {number}]({name}.png)\n\n{_linked(text, links)}'.encode())
            if not text.strip():
                textless.append(number)
            notes = [part for part in (_clauses(text), ', '.join(r.label for r in found)) if part]
            index.append(f'- [page {number}]({name}.md)' + (f' -- {" | ".join(notes)}' if notes else ''))
        put('INDEX.md', ('\n'.join(index) + '\n').encode())
    return Paged(out, len(texts), tuple(textless))
