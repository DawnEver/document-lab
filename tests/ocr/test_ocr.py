"""The ``ocr`` part: a captured image of known words, given a Tesseract text layer, reads back as those words, its
pixels and page size unchanged; so does a fax scan of them, handed over in a TIFF, upright on a page turned or not.
Its own part, never tests/base: it needs Tesseract with English data, and FAILS naming what is missing where there
is none; it is never skipped."""

import os

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import to_pdf
from document_lab.ocr import TesseractPlaces, check, text_layer
from document_lab.pdf import read_pdf
from tests.synthetic import fax_g4, image_pdf

pytestmark = pytest.mark.ocr

PLACES = TesseractPlaces.of(os.environ)  # where this machine's Tesseract may be, beyond PATH
WORDS = ('stator', 'winding', 'capture')


def rendered(words: tuple[str, ...]) -> np.ndarray:
    """``words``, one per line in large type, as the RGB pixels a screen capture would give."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=60 * len(words) + 40)
        for i, word in enumerate(words):
            page.insert_text((30, 60 + 60 * i), word, fontsize=36)
        pix = page.get_pixmap(dpi=96)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()


def test_the_text_layer_reads_back_over_the_same_pixels(tmp_path):
    check('eng', PLACES)
    image = rendered(WORDS)
    capture = to_pdf(image, tmp_path / 'capture.pdf', page_height=0, dpi=96)
    assert len(text_layer(capture, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    with pymupdf.open(capture) as before, pymupdf.open(tmp_path / 'ocr.pdf') as after:
        assert after.page_count == 1
        assert after[0].rect == before[0].rect
        ((xref, *_),) = after[0].get_images()
        pix = pymupdf.Pixmap(after, xref)
        assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(image.shape), image)
        assert all(word in after[0].get_text().casefold() for word in WORDS)  # an independent reader reads them
    drawn_where_they_are(tmp_path / 'ocr.pdf')
    laid_again = text_layer(tmp_path / 'ocr.pdf', tmp_path / 'again.pdf', places=PLACES)
    assert laid_again.readings == ()  # a page with text is never laid twice


def drawn_where_they_are(pdf) -> None:
    """Each of :data:`WORDS` read back from ``pdf`` over where :func:`rendered` draws it."""
    spans = {span.text.casefold(): span.provenance.bbox for span in read_pdf(pdf).pages[0].spans}
    for i, word in enumerate(WORDS):  # from x 30, on the baseline at y 60 + 60 i
        x0, y0, x1, y1 = spans[word]
        assert 25 < x0 < 35 < x1
        assert y0 < 60 + 60 * i - 10 < y1


@pytest.mark.parametrize('rotate', [0, 90])
def test_a_fax_scan_reads_back_upright_on_a_page_turned_or_not(tmp_path, rotate):
    check('eng', PLACES)
    white = rendered(WORDS)[..., 0] >= 128  # the scanner's threshold
    stored = np.rot90(white, rotate // 90)  # turned back as far as the page turns it: shown upright
    rows, columns = stored.shape
    image = f'/Width {columns} /Height {rows} /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode'
    params = f'/DecodeParms << /K -1 /Columns {columns} >>'  # spelling-ignore-line: the PDF name
    size = (columns * 72 / 96, rows * 72 / 96)
    pdf = image_pdf(tmp_path / 'fax.pdf', fax_g4(stored), f'{image} {params}', size=size, deflate=False, rotate=rotate)
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    with pymupdf.open(tmp_path / 'ocr.pdf') as after:
        assert all(word in after[0].get_text().casefold() for word in WORDS)  # an independent reader reads them
    drawn_where_they_are(tmp_path / 'ocr.pdf')
