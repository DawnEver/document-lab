"""A text layer over a PDF of page images, by Tesseract: its words readable, its pixels exactly as they were.

Each page's image is recognised at the resolution it carries (`pymupdf.Pixmap.pdfocr_tobytes`) and only the words
Tesseract lays out -- invisible, over where they are seen -- are placed on the page; the page and its image are not
touched. A page that already has text, or holds no image, is left as it is, so a layer is never laid twice.
:func:`check` refuses a missing Tesseract or language before any work is spent.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

__all__ = ['check', 'text_layer']


def check(languages: str) -> None:
    """Refuse, naming what is missing, unless Tesseract and every ``+``-joined language in ``languages`` exist."""
    try:
        tessdata = Path(pymupdf.get_tessdata())
    except RuntimeError as exc:
        msg = f'OCR needs Tesseract (and TESSDATA_PREFIX if it is not on PATH): {exc}'
        raise OSError(msg) from exc
    missing = [lang for lang in languages.split('+') if not (tessdata / f'{lang}.traineddata').is_file()]
    if missing:
        msg = f'Tesseract in {tessdata} has no language data for {missing}'
        raise OSError(msg)


def _image(page: pymupdf.Page) -> pymupdf.Pixmap | None:
    """The page's one image at the resolution the page shows it, or None: a page with text, or not one image."""
    images = page.get_images()
    if len(images) != 1 or page.get_text().strip():
        return None
    pix = pymupdf.Pixmap(page.parent, images[0][0])
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    dpi = round(pix.width * 72 / page.rect.width)
    pix.set_dpi(dpi, dpi)  # Tesseract's page is sized from it: the words land where they are seen
    return pix


def _lay(page: pymupdf.Page, pix: pymupdf.Pixmap, languages: str) -> None:
    with pymupdf.open('pdf', pix.pdfocr_tobytes(language=languages)) as words:
        recognised = words[0]
        for xref, *_ in recognised.get_images():
            recognised.delete_image(xref)  # the words alone: the page keeps its own image
        page.show_pdf_page(page.rect, words, 0)


def text_layer(pdf: Path, out: Path, languages: str = 'eng') -> int:
    """``pdf`` with a text layer in ``languages`` over every page that is one image and has no text, to ``out``.

    ``out`` may be ``pdf`` itself: it is written beside as ``<out>.part`` and moved into place when whole. How many
    pages were given a layer (0: ``out`` is a copy).
    """
    laid = 0
    with pymupdf.open(pdf) as doc:
        for page in doc:
            if (pix := _image(page)) is None:
                continue
            if not laid:
                check(languages)  # only once there is work: a document already read needs no Tesseract
            _lay(page, pix, languages)
            laid += 1
        part = out.with_name(out.name + '.part')
        doc.save(part, garbage=3, deflate=True)
    part.replace(out)
    return laid
