"""A captured image to a PDF: one image per page, cut at a page height, pixels kept losslessly.

Sizes are in pixels; ``dpi`` turns them into PDF points (72 per inch). ``ocr`` names the Tesseract languages
(``'eng'``, ``'chi_sim+eng'``) of an invisible text layer laid over each page, so `document_lab.pdf.read_pdf` can read
the words back; :func:`check_ocr` refuses a missing Tesseract or language before any work is spent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pymupdf

__all__ = ['MAX_PAGE_PT', 'a4_page_height', 'check_ocr', 'check_page', 'save_png', 'to_pdf']

MAX_PAGE_PT: Final = 14400  # the largest page side a PDF reader must accept (ISO 32000, 200 inches)
SCREEN_DPI: Final = 96


def a4_page_height(width: int) -> int:
    """The page height in pixels that gives an A4-shaped page ``width`` pixels wide."""
    return round(width * 297 / 210)


def _pixmap(image: np.ndarray) -> pymupdf.Pixmap:
    rgb = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, rgb.tobytes(), 0)


def save_png(image: np.ndarray, path: Path) -> Path:
    """Write ``image`` (``(h, w, 3)`` uint8) to ``path`` as a PNG and return it."""
    _pixmap(image).save(path)
    return path


def check_ocr(languages: str) -> None:
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


def check_page(width: int, page_height: int | None, dpi: float) -> None:
    """Refuse a page size no PDF reader must accept; ``page_height`` 0 (one page) is checked by :func:`to_pdf`."""
    step = a4_page_height(width) if page_height is None else page_height
    if max(step, width) * 72 / dpi > MAX_PAGE_PT:
        msg = f'a page of {width}x{step} px at {dpi} dpi exceeds the {MAX_PAGE_PT} pt PDF limit: lower page_height'
        raise ValueError(msg)


def to_pdf(
    image: np.ndarray, path: Path, *, page_height: int | None = None, dpi: float = SCREEN_DPI, ocr: str | None = None
) -> Path:
    """Write ``image`` (``(h, w, 3)`` uint8) to ``path`` and return it.

    ``page_height`` in pixels: ``None`` cuts A4-shaped pages, ``0`` puts the whole image on one page.
    """
    height, width = image.shape[:2]
    step = a4_page_height(width) if page_height is None else (page_height or height)
    check_page(width, step, dpi)
    if ocr:
        check_ocr(ocr)
    with pymupdf.open() as doc:
        for start in range(0, height, step):
            pix = _pixmap(image[start : start + step])
            if ocr:
                pix.set_dpi(round(dpi), round(dpi))  # the OCR page is sized from the pixmap's resolution
                with pymupdf.open('pdf', pix.pdfocr_tobytes(language=ocr)) as page_pdf:
                    doc.insert_pdf(page_pdf)
            else:
                rect = pymupdf.Rect(0, 0, pix.width * 72 / dpi, pix.height * 72 / dpi)
                doc.new_page(width=rect.width, height=rect.height).insert_image(rect, pixmap=pix)
        doc.save(path, deflate=True)
    return path
