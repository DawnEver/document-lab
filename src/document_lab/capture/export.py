"""A captured image to a PDF or a PNG, written as its rows arrive: a capture of any length costs a page of memory.

:class:`PdfWriter` cuts pages at a page height -- or, ``'auto'``, on the reader's own page separators
(`document_lab.capture.pages`) -- one image per page, pixels kept losslessly; it saves every
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
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, Self

import numpy as np
import pymupdf

from document_lab.capture.pages import Rhythm, page_rhythm

if TYPE_CHECKING:
    from types import TracebackType

__all__ = [
    'MAX_PAGE_PT',
    'PAGES_PER_SAVE',
    'PdfWriter',
    'PngWriter',
    'a4_page_height',
    'check_name',
    'check_ocr',
    'check_page',
    'save_png',
    'to_pdf',
]

MAX_PAGE_PT: Final = 14400  # the largest page side a PDF reader must accept (ISO 32000, 200 inches)
SCREEN_DPI: Final = 96
PAGES_PER_SAVE: Final = 16  # pages added between incremental saves: what a PDF holds in memory at most
AUTO_LOOK: Final = 4  # A4 pages of rows 'auto' holds before it looks for the separators, and each time again
AUTO_GIVE_UP: Final = 12  # A4 pages of rows after which 'auto' stops looking and cuts A4-shaped pages


def a4_page_height(width: int) -> int:
    """The page height in pixels that gives an A4-shaped page ``width`` pixels wide."""
    return round(width * 297 / 210)


def _pixmap(image: np.ndarray) -> pymupdf.Pixmap:
    rgb = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    height, width = rgb.shape[:2]
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, rgb.tobytes(), 0)


def _part(path: Path) -> Path:
    return path.with_name(path.name + '.part')


_UNNAMEABLE: Final = '<>:"/\\|?*'  # characters Windows refuses in a file name; ':' opens an NTFS data stream
_RESERVED: Final = frozenset(
    ['CON', 'PRN', 'AUX', 'NUL', *(f'COM{n}' for n in range(1, 10)), *(f'LPT{n}' for n in range(1, 10))]
)


def check_name(path: Path) -> None:
    """Refuse a file name Windows cannot hold -- before a capture is spent on it -- naming one it can.

    ``a:b.pdf`` is the worst case: Windows writes it as the hidden stream ``b.pdf`` of a file ``a``, and the move into
    place then fails. Refused on every system, so a name is portable wherever the capture is read.
    """
    name = path.name
    fixed = ''.join('_' if c in _UNNAMEABLE or ord(c) < 32 else c for c in name).rstrip(' .') or '_'  # noqa: PLR2004
    if fixed.split('.')[0].upper() in _RESERVED:
        fixed = f'_{fixed}'
    if fixed != name:
        msg = f'Windows cannot name a file {name!r}: try {fixed}'
        raise ValueError(msg)


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
    so not for an unbounded capture), ``'auto'`` cuts on the reader's page separators: it holds rows until it finds
    them (:func:`~document_lab.capture.pages.page_rhythm`) -- ``rhythm`` then says where -- and cuts A4-shaped pages
    if there are none in the first :data:`AUTO_GIVE_UP` pages' worth (``rhythm`` stays None). Closing writes the last,
    shorter page and moves the file into place.
    """

    def __init__(
        self,
        path: Path,
        width: int,
        *,
        page_height: int | Literal['auto'] | None = None,
        dpi: float = SCREEN_DPI,
        ocr: str | None = None,
    ) -> None:
        """Check the page size and OCR before anything is written."""
        self.path, self.width, self.dpi, self.ocr = path, width, dpi, ocr
        self.auto = page_height == 'auto'
        self.step = a4_page_height(width) if page_height is None or self.auto else int(page_height)
        if self.step:
            check_page(width, self.step, dpi)
        if ocr:
            check_ocr(ocr)
        self.rhythm: Rhythm | None = None
        self.pages = 0
        self._rows: list[np.ndarray] = []
        self._held = 0
        self._written = 0  # rows already laid out on pages
        self._look = AUTO_LOOK * self.step  # held rows at which 'auto' next looks for the separators
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
        if self.auto and self._held >= self._look:
            self._find_rhythm(final=False)
        self._lay_out()

    def _find_rhythm(self, *, final: bool) -> None:
        held = np.concatenate(self._rows)
        self._rows = [held]
        self.rhythm = page_rhythm(held)
        if self.rhythm is not None:
            check_page(self.width, round(self.rhythm.pitch), self.dpi)
        if self.rhythm is not None or final or self._held >= AUTO_GIVE_UP * self.step:
            self.auto = False  # found, or A4-shaped from here
        else:
            self._look += AUTO_LOOK * self.step

    def _next_cut(self) -> int | None:
        if self.auto or not self.step:
            return None
        if self.rhythm is None:
            return self._written + self.step
        return self.rhythm.cuts(self._written, self._written + 2 * round(self.rhythm.pitch))[0]

    def _lay_out(self) -> None:
        while (cut := self._next_cut()) is not None and self._written + self._held >= cut:
            self._page(self._take(cut - self._written))

    def _take(self, count: int) -> np.ndarray:
        held = np.concatenate(self._rows)
        self._rows, self._held = [held[count:]], len(held) - count
        self._written += count
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
        if self.auto and self._held:
            self._find_rhythm(final=True)
            self._lay_out()
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
    image: np.ndarray,
    path: Path,
    *,
    page_height: int | Literal['auto'] | None = None,
    dpi: float = SCREEN_DPI,
    ocr: str | None = None,
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
