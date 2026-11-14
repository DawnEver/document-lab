"""The ``ocr`` part: a captured image of known words, given a Tesseract text layer, reads back as those words, its
pixels and page size unchanged. Its own part, never tests/base: it needs Tesseract with English data, and FAILS
naming what is missing where there is none; it is never skipped."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import to_pdf
from document_lab.ocr import check, text_layer
from document_lab.pdf import read_pdf

pytestmark = pytest.mark.ocr

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
    check('eng')
    image = rendered(WORDS)
    capture = to_pdf(image, tmp_path / 'capture.pdf', page_height=0, dpi=96)
    assert text_layer(capture, tmp_path / 'ocr.pdf') == 1
    with pymupdf.open(capture) as before, pymupdf.open(tmp_path / 'ocr.pdf') as after:
        assert after.page_count == 1
        assert after[0].rect == before[0].rect
        ((xref, *_),) = after[0].get_images()
        pix = pymupdf.Pixmap(after, xref)
        assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(image.shape), image)
    text = ' '.join(span.text for page in read_pdf(tmp_path / 'ocr.pdf', tables=False).pages for span in page.spans)
    for word in WORDS:
        assert word in text.casefold()
    assert text_layer(tmp_path / 'ocr.pdf', tmp_path / 'again.pdf') == 0  # a page with text is never laid twice
