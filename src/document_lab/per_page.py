"""A PDF as one markdown file and one image per page: the layout for standards, manuals and data sheets.

A paper reads as sections (`document_lab.paper`); a standard is cited by clause on its page, its tables and figures
in place, so each page is kept whole. ``page-NNN.md`` is the page's text in markdown (`document_lab.markdown`, read
from the PDF's text layer -- a page of a capture needs `document_lab.ocr` first) under a link to ``page-NNN.png``,
the page as it is seen: a page that is one image gives that image's own pixels, any other page is rendered at
:data:`RENDER_DPI`. ``INDEX.md`` lists the pages with their first heading. A page without text is written all the
same, its markdown empty, and named in :attr:`Paged.textless` -- never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pymupdf

from document_lab.markdown import page_markdowns
from document_lab.pdf import read_pdf

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['RENDER_DPI', 'Paged', 'write_pages']

RENDER_DPI: Final = 150  # a page that is not one image is rendered at this resolution


@dataclass(frozen=True)
class Paged:
    """What :func:`write_pages` wrote: ``pages`` pages to ``out``; ``textless`` the 1-based numbers without text."""

    out: Path
    pages: int
    textless: tuple[int, ...]


def _picture(page: pymupdf.Page) -> pymupdf.Pixmap:
    images = page.get_images()
    if len(images) == 1:
        pix = pymupdf.Pixmap(page.parent, images[0][0])
        return pymupdf.Pixmap(pix, 0) if pix.alpha else pix
    return page.get_pixmap(dpi=RENDER_DPI)


def _heading(markdown: str) -> str:
    return next((line.lstrip('#').strip() for line in markdown.splitlines() if line.startswith('#')), '')


def write_pages(pdf: Path, out: Path) -> Paged:
    """Write ``pdf`` page by page into the directory ``out`` (made if missing)."""
    out.mkdir(parents=True, exist_ok=True)
    texts = page_markdowns(read_pdf(pdf, tables=False))
    width = max(3, len(str(len(texts))))
    index = [f'# {pdf.stem}', '']
    textless = []
    with pymupdf.open(pdf) as doc:
        for number, (page, text) in enumerate(zip(doc, texts, strict=True), start=1):
            name = f'page-{number:0{width}d}'
            _picture(page).save(out / f'{name}.png')
            (out / f'{name}.md').write_text(f'[{pdf.name}, page {number}]({name}.png)\n\n{text}', encoding='utf-8')
            if not text.strip():
                textless.append(number)
            heading = _heading(text)
            index.append(f'- [page {number}]({name}.md)' + (f' -- {heading}' if heading else ''))
    (out / 'INDEX.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return Paged(out, len(texts), tuple(textless))
