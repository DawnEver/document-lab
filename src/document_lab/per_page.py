"""A PDF as one markdown file and one image per page: the layout for standards, manuals and data sheets.

A paper reads as sections (`document_lab.paper`); a standard is cited by clause on its page, its tables and figures
in place, so each page is kept whole. ``page-NNN.md`` is the page's text in markdown (`document_lab.markdown`, read
from the PDF's text layer -- a page of a capture needs `document_lab.ocr` first) under a link to ``page-NNN.png``,
the page as it is seen: a page that is one image gives that image's own pixels, any other page is rendered at
:data:`RENDER_DPI`.

Each figure, table and equation the page's text locates (`document_lab.regions`) is cut out of that picture as
``page-NNN-<kind>-<label>.png`` and linked in the markdown after the paragraph holding its caption or number --
where OCR'd text cannot be trusted (a formula, a table's numbers), the picture beside it can. A region reaching the
page's foot is marked as continuing on the next page. ``INDEX.md`` lists the pages with the clause headings each
opens and the regions it holds. A page without text is written all the same, its markdown empty, and named in
:attr:`Paged.textless` -- never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pymupdf

from document_lab import layout
from document_lab.markdown import page_markdowns
from document_lab.pdf import read_pdf
from document_lab.regions import Region, regions

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['RENDER_DPI', 'Paged', 'write_pages']

RENDER_DPI: Final = 150  # a page that is not one image is rendered at this resolution
_FOOT: Final = 1.0  # points from the page's foot within which a region runs on to the next page


@dataclass(frozen=True)
class Paged:
    """What :func:`write_pages` wrote: ``pages`` pages to ``out``; ``textless`` the 1-based numbers without text."""

    out: Path
    pages: int
    textless: tuple[int, ...]


def _picture(page: pymupdf.Page) -> tuple[pymupdf.Pixmap, float]:
    """The page as seen, and its pixels per point: its one image's own pixels, else rendered."""
    images = page.get_images()
    if len(images) == 1:
        pix = pymupdf.Pixmap(page.parent, images[0][0])
        pix = pymupdf.Pixmap(pix, 0) if pix.alpha else pix
    else:
        pix = page.get_pixmap(dpi=RENDER_DPI)
    return pix, pix.width / page.rect.width


def _cut(pix: pymupdf.Pixmap, scale: float, region: Region) -> pymupdf.Pixmap:
    """``region`` of the page's picture, its pixels as they are."""
    rows = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    box = region.bbox
    x0, x1 = max(int(box.x0 * scale), 0), min(int(np.ceil(box.x1 * scale)), pix.width)
    y0, y1 = max(int(box.y0 * scale), 0), min(int(np.ceil(box.y1 * scale)), pix.height)
    part = np.ascontiguousarray(rows[y0:y1, x0:x1])
    return pymupdf.Pixmap(pix.colorspace, x1 - x0, y1 - y0, part.tobytes(), 0)


def _file(name: str, region: Region) -> str:
    label = region.label.removeprefix('Figure').removeprefix('Table').strip('() ')
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


def _clauses(markdown: str) -> str:
    """The clause headings a page opens (``##`` and deeper), ``; ``-joined: what the index names it by."""
    return '; '.join(line.lstrip('#').strip() for line in markdown.splitlines() if line.startswith('##'))


def write_pages(pdf: Path, out: Path) -> Paged:
    """Write ``pdf`` page by page into the directory ``out`` (made if missing)."""
    out.mkdir(parents=True, exist_ok=True)
    read = read_pdf(pdf, tables=False)
    body = layout.body_size(read)
    texts = page_markdowns(read)
    width = max(3, len(str(len(texts))))
    index = [f'# {pdf.stem}', '']
    textless = []
    with pymupdf.open(pdf) as doc:
        for number, (page, text, ir_page) in enumerate(zip(doc, texts, read.pages, strict=True), start=1):
            name = f'page-{number:0{width}d}'
            pix, scale = _picture(page)
            pix.save(out / f'{name}.png')
            links = []
            for region in (found := regions(ir_page, body)):
                file = _file(name, region)
                _cut(pix, scale, region).save(out / file)
                more = ' (continues on the next page)' if region.bbox.y1 >= page.rect.height - _FOOT else ''
                links.append((region.label, f'![{region.label}]({file}){more}'))
            (out / f'{name}.md').write_text(
                f'[{pdf.name}, page {number}]({name}.png)\n\n{_linked(text, links)}', encoding='utf-8'
            )
            if not text.strip():
                textless.append(number)
            notes = [part for part in (_clauses(text), ', '.join(r.label for r in found)) if part]
            index.append(f'- [page {number}]({name}.md)' + (f' -- {" | ".join(notes)}' if notes else ''))
    (out / 'INDEX.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return Paged(out, len(texts), tuple(textless))
