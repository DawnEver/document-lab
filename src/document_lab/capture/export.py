"""A captured image to a PDF: one image per page, cut at a page height, pixels kept losslessly.

Sizes are in pixels; ``dpi`` turns them into PDF points (72 per inch). ``ocr`` adds an invisible text layer through
Tesseract (it must be installed; PyMuPDF raises naming it otherwise) so `document_lab.pdf.read_pdf` can read the
words back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pymupdf

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['MAX_PAGE_PT', 'a4_page_height', 'save_png', 'to_pdf']

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


def to_pdf(
    image: np.ndarray, path: Path, *, page_height: int | None = None, dpi: float = SCREEN_DPI, ocr: bool = False
) -> Path:
    """Write ``image`` (``(h, w, 3)`` uint8) to ``path`` and return it.

    ``page_height`` in pixels: ``None`` cuts A4-shaped pages, ``0`` puts the whole image on one page.
    """
    height, width = image.shape[:2]
    step = a4_page_height(width) if page_height is None else (page_height or height)
    if max(step, width) * 72 / dpi > MAX_PAGE_PT:
        msg = f'a page of {width}x{step} px at {dpi} dpi exceeds the {MAX_PAGE_PT} pt PDF limit: lower page_height'
        raise ValueError(msg)
    with pymupdf.open() as doc:
        for start in range(0, height, step):
            pix = _pixmap(image[start : start + step])
            if ocr:
                with pymupdf.open('pdf', pix.pdfocr_tobytes()) as page_pdf:
                    doc.insert_pdf(page_pdf)
                page = doc[-1]
                page.set_mediabox(page.rect)  # OCR pages come out at the pixmap's own resolution
            else:
                rect = pymupdf.Rect(0, 0, pix.width * 72 / dpi, pix.height * 72 / dpi)
                doc.new_page(width=rect.width, height=rect.height).insert_image(rect, pixmap=pix)
        doc.save(path, deflate=True)
    return path
