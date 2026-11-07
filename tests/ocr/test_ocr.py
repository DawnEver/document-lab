"""The ``ocr`` part: a captured image of known words, exported with a Tesseract text layer, reads back as those
words on pages of the asked size. Its own part, never tests/base: it needs Tesseract with English data, and FAILS
naming what is missing where there is none; it is never skipped."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import check_ocr, to_pdf
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


def test_the_text_layer_reads_back(tmp_path):
    check_ocr('eng')
    image = rendered(WORDS)
    out = to_pdf(image, tmp_path / 'ocr.pdf', page_height=0, dpi=96, ocr='eng')
    with pymupdf.open(out) as doc:
        assert doc.page_count == 1
        assert doc[0].rect.width == pytest.approx(image.shape[1] * 72 / 96, abs=1)
        assert doc[0].rect.height == pytest.approx(image.shape[0] * 72 / 96, abs=1)
    text = ' '.join(span.text for page in read_pdf(out, tables=False).pages for span in page.spans).casefold()
    for word in WORDS:
        assert word in text
