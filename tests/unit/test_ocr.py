"""The text layer's guards, without Tesseract: what it refuses before any work, and the pages it leaves alone.

That the words read back -- Tesseract itself -- is the ``ocr`` part's (tests/ocr)."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import to_pdf
from document_lab.ocr import check, text_layer


def no_tesseract():
    msg = 'No tessdata specified and Tesseract is not installed'
    raise RuntimeError(msg)


def test_no_tesseract_is_refused_naming_it(monkeypatch):
    monkeypatch.setattr(pymupdf, 'get_tessdata', no_tesseract)
    with pytest.raises(OSError, match='Tesseract'):
        check('eng')


def test_a_missing_language_is_refused_naming_it(tmp_path, monkeypatch):
    (tmp_path / 'eng.traineddata').write_bytes(b'')
    monkeypatch.setattr(pymupdf, 'get_tessdata', lambda: str(tmp_path))
    check('eng')
    with pytest.raises(OSError, match='chi_sim'):
        check('chi_sim+eng')


def test_a_pdf_that_has_text_is_left_as_it_is_and_needs_no_tesseract(tmp_path, monkeypatch):
    monkeypatch.setattr(pymupdf, 'get_tessdata', no_tesseract)
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'already searchable')
        doc.save(tmp_path / 'text.pdf')
    assert text_layer(tmp_path / 'text.pdf', tmp_path / 'out.pdf') == 0
    with pymupdf.open(tmp_path / 'out.pdf') as doc:
        assert 'already searchable' in doc[0].get_text()


def test_a_capture_without_tesseract_is_refused_and_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(pymupdf, 'get_tessdata', no_tesseract)
    image = np.random.default_rng(0).integers(0, 256, (120, 80, 3), dtype=np.uint8)
    pdf = to_pdf(image, tmp_path / 'capture.pdf')
    before = pdf.read_bytes()
    with pytest.raises(OSError, match='Tesseract'):
        text_layer(pdf, pdf)
    assert pdf.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['capture.pdf']
