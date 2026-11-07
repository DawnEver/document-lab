"""A captured image to a PDF or a PNG, written as its rows arrive: a capture of any length costs a page of memory.

:class:`PdfWriter` cuts pages at a page height, one image per page, pixels kept losslessly; it saves every
:data:`PAGES_PER_SAVE` pages incrementally, so the document is never held whole. :class:`PngWriter` compresses rows as
they come and fills the height in at the end. Both write ``<name>.part`` beside the target and rename it into place
when closed -- also when the capture stopped early -- so a reader never sees a half-written file under the real name.

Sizes are in pixels; ``dpi`` turns them into PDF points (72 per inch). ``ocr`` names the Tesseract languages
(``'eng'``, ``'chi_sim+eng'``) of an invisible text layer laid over each page, so `document_lab.pdf.read_pdf` can read
the words back; :func:`check_ocr` refuses a missing Tesseract or language before any work is spent.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, Self

import numpy as np
import pymupdf

if TYPE_CHECKING:
    from types import TracebackType

__all__ = [
    'MAX_PAGE_PT',
    'PAGES_PER_SAVE',
    'PdfWriter',
    'PngWriter',
    'a4_page_height',
    'check_ocr',
    'check_page',
    'save_png',
    'to_pdf',
]

MAX_PAGE_PT: Final = 14400  # the largest page side a PDF reader must accept (ISO 32000, 200 inches)
SCREEN_DPI: Final = 96
PAGES_PER_SAVE: Final = 16  # pages added between incremental saves: what a PDF holds in memory at most


def a4_page_height(width: int) -> int:
    """The page height in pixels that gives an A4-shaped page ``width`` pixels wide."""
    return round(width * 297 / 210)


def _pixmap(image: np.ndarray) -> pymupdf.Pixmap:
    rgb = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, rgb.tobytes(), 0)


def _part(path: Path) -> Path:
    return path.with_name(path.name + '.part')


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
    """Refuse a page size no PDF reader must accept; ``page_height`` 0 (one page) is checked when it is written."""
    step = a4_page_height(width) if page_height is None else page_height
    if max(step, width) * 72 / dpi > MAX_PAGE_PT:
        msg = f'a page of {width}x{step} px at {dpi} dpi exceeds the {MAX_PAGE_PT} pt PDF limit: lower page_height'
        raise ValueError(msg)


class PdfWriter:
    """A PDF at ``path`` of the rows written to it, ``width`` pixels wide.

    ``page_height`` in pixels: ``None`` cuts A4-shaped pages, ``0`` puts everything on one page (held until closed,
    so not for an unbounded capture). Closing writes the last, shorter page and moves the file into place.
    """

    def __init__(
        self, path: Path, width: int, *, page_height: int | None = None, dpi: float = SCREEN_DPI, ocr: str | None = None
    ) -> None:
        """Check the page size and OCR before anything is written."""
        self.path, self.width, self.dpi, self.ocr = path, width, dpi, ocr
        self.step = a4_page_height(width) if page_height is None else page_height
        if self.step:
            check_page(width, self.step, dpi)
        if ocr:
            check_ocr(ocr)
        self.pages = 0
        self._rows: list[np.ndarray] = []
        self._held = 0
        self._doc: pymupdf.Document | None = None
        self._saved = False

    def __enter__(self) -> Self:
        """The writer, closed on leaving."""
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        """Close: what was written is kept even when the capture failed."""
        self.close()

    def write(self, rows: np.ndarray) -> None:
        """Add ``rows`` (``(n, width, 3)`` uint8) below what was written; every full page is laid out."""
        if not len(rows):
            return
        self._rows.append(rows)
        self._held += len(rows)
        while self.step and self._held >= self.step:
            self._page(self._take(self.step))

    def _take(self, count: int) -> np.ndarray:
        held = np.concatenate(self._rows)
        self._rows, self._held = [held[count:]], len(held) - count
        return held[:count]

    def _page(self, image: np.ndarray) -> None:
        if self._doc is None:
            self._doc = pymupdf.open(_part(self.path)) if self._saved else pymupdf.open()
        pix = _pixmap(image)
        if self.ocr:
            pix.set_dpi(round(self.dpi), round(self.dpi))  # the OCR page is sized from the pixmap's resolution
            with pymupdf.open('pdf', pix.pdfocr_tobytes(language=self.ocr)) as page_pdf:
                self._doc.insert_pdf(page_pdf)
        else:
            rect = pymupdf.Rect(0, 0, pix.width * 72 / self.dpi, pix.height * 72 / self.dpi)
            self._doc.new_page(width=rect.width, height=rect.height).insert_image(rect, pixmap=pix)
        self.pages += 1
        if self.pages % PAGES_PER_SAVE == 0:
            self._save()

    def _save(self) -> None:
        if self._doc is None:
            return
        if self._saved:
            self._doc.save(_part(self.path), incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
        else:
            self._doc.save(_part(self.path), deflate=True)
        self._doc.close()
        self._doc, self._saved = None, True

    def close(self) -> Path | None:
        """Lay out the last page, save, and move the PDF into place; the path, or None if nothing was written."""
        if self._held:
            last = self._take(self._held)
            if not self.step:
                check_page(self.width, len(last), self.dpi)
            self._page(last)
        self._save()
        if not self._saved:
            return None
        _part(self.path).replace(self.path)
        self._saved = False
        return self.path


_PNG: Final = b'\x89PNG\r\n\x1a\n'


def _chunk(out: BinaryIO, kind: bytes, data: bytes) -> None:
    out.write(struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data)))


class PngWriter:
    """A PNG at ``path`` of the rows written to it, ``width`` pixels wide, compressed as they come."""

    def __init__(self, path: Path, width: int) -> None:
        """Open ``<path>.part`` and write the header; its height is filled in on closing."""
        self.path, self.width, self.height = path, width, 0
        self._out = _part(path).open('wb')
        self._out.write(_PNG)
        self._header = self._out.tell()
        _chunk(self._out, b'IHDR', struct.pack('>IIBBBBB', width, 0, 8, 2, 0, 0, 0))
        self._zip = zlib.compressobj(6)

    def __enter__(self) -> Self:
        """The writer, closed on leaving."""
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        """Close: what was written is kept even when the capture failed."""
        self.close()

    def write(self, rows: np.ndarray) -> None:
        """Add ``rows`` (``(n, width, 3)`` uint8) below what was written."""
        if not len(rows):
            return
        filtered = np.zeros((len(rows), 1 + self.width * 3), dtype=np.uint8)  # filter byte 0: the row as it is
        filtered[:, 1:] = np.ascontiguousarray(rows[..., :3], dtype=np.uint8).reshape(len(rows), -1)
        self.height += len(rows)
        if data := self._zip.compress(filtered.tobytes()):
            _chunk(self._out, b'IDAT', data)

    def close(self) -> Path | None:
        """Finish the PNG, fill in its height and move it into place; the path, or None if nothing was written."""
        if self._out.closed:
            return self.path if self.path.exists() else None
        _chunk(self._out, b'IDAT', self._zip.flush())
        _chunk(self._out, b'IEND', b'')
        self._out.seek(self._header)
        _chunk(self._out, b'IHDR', struct.pack('>IIBBBBB', self.width, self.height, 8, 2, 0, 0, 0))
        self._out.close()
        if not self.height:
            _part(self.path).unlink()
            return None
        _part(self.path).replace(self.path)
        return self.path


def to_pdf(
    image: np.ndarray, path: Path, *, page_height: int | None = None, dpi: float = SCREEN_DPI, ocr: str | None = None
) -> Path:
    """Write ``image`` (``(h, w, 3)`` uint8) to ``path`` through a :class:`PdfWriter` and return it."""
    with PdfWriter(path, image.shape[1], page_height=page_height, dpi=dpi, ocr=ocr) as pdf:
        pdf.write(image)
    return path


def save_png(image: np.ndarray, path: Path) -> Path:
    """Write ``image`` (``(h, w, 3)`` uint8) to ``path`` through a :class:`PngWriter` and return it."""
    with PngWriter(path, image.shape[1]) as png:
        png.write(image)
    return path
