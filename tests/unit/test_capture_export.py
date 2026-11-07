"""A stitched image to PDF: pages cut at the asked height, the pixels kept losslessly, the page size from the DPI."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import MAX_PAGE_PT, a4_page_height, check_ocr, save_png, to_pdf


def image(height: int, width: int = 80) -> np.ndarray:
    return np.random.default_rng(height).integers(0, 256, (height, width, 3), dtype=np.uint8)


def pixels(page: pymupdf.Page) -> np.ndarray:
    ((xref, *_),) = page.get_images()
    pix = pymupdf.Pixmap(page.parent, xref)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def test_pages_cut_at_the_page_height_and_rejoin_losslessly(tmp_path):
    tall = image(250)
    out = to_pdf(tall, tmp_path / 'a.pdf', page_height=100)
    with pymupdf.open(out) as doc:
        assert doc.page_count == 3
        assert np.array_equal(np.concatenate([pixels(p) for p in doc]), tall)
        assert doc[2].rect.height == pytest.approx(50 * 72 / 96)


def test_page_size_follows_the_dpi(tmp_path):
    with pymupdf.open(to_pdf(image(100), tmp_path / 'b.pdf', page_height=100, dpi=144)) as doc:
        assert doc[0].rect.width == pytest.approx(80 * 72 / 144)
        assert doc[0].rect.height == pytest.approx(100 * 72 / 144)


def test_default_page_is_a4_shaped():
    assert a4_page_height(1000) == round(1000 * 297 / 210)


def test_one_page_for_the_whole_image(tmp_path):
    with pymupdf.open(to_pdf(image(300), tmp_path / 'c.pdf', page_height=0)) as doc:
        assert doc.page_count == 1


def test_a_page_taller_than_pdf_allows_refuses(tmp_path):
    with pytest.raises(ValueError, match='14400'):
        to_pdf(image(10), tmp_path / 'd.pdf', page_height=MAX_PAGE_PT * 96 // 72 + 1)


def test_png_keeps_the_pixels(tmp_path):
    tall = image(120)
    pix = pymupdf.Pixmap(str(save_png(tall, tmp_path / 'e.png')))
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(120, 80, 3), tall)


def no_tesseract():
    msg = 'No tessdata specified and Tesseract is not installed'
    raise RuntimeError(msg)


def test_ocr_without_tesseract_refuses_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(pymupdf, 'get_tessdata', no_tesseract)
    out = tmp_path / 'f.pdf'
    with pytest.raises(OSError, match='Tesseract'):
        to_pdf(image(50), out, ocr='eng')
    assert not out.exists()


def test_ocr_without_the_language_refuses(tmp_path, monkeypatch):
    (tmp_path / 'eng.traineddata').write_bytes(b'')
    monkeypatch.setattr(pymupdf, 'get_tessdata', lambda: str(tmp_path))
    check_ocr('eng')
    with pytest.raises(OSError, match='chi_sim'):
        check_ocr('chi_sim+eng')
