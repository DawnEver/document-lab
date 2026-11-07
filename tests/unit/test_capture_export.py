"""A stitched image to PDF: pages cut at the asked height, the pixels kept losslessly, the page size from the DPI."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import (
    MAX_PAGE_PT,
    PAGES_PER_SAVE,
    PdfWriter,
    PngWriter,
    a4_page_height,
    check_ocr,
    save_png,
    to_pdf,
)


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


def test_rows_written_in_pieces_make_the_same_pdf_across_incremental_saves(tmp_path):
    tall = image(100 * (PAGES_PER_SAVE * 2 + 3) + 37)
    with PdfWriter(tmp_path / 'g.pdf', 80, page_height=100) as pdf:
        for start in range(0, len(tall), 73):
            pdf.write(tall[start : start + 73])
    with pymupdf.open(tmp_path / 'g.pdf') as doc:
        assert doc.page_count == PAGES_PER_SAVE * 2 + 4
        assert np.array_equal(np.concatenate([pixels(p) for p in doc]), tall)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['g.pdf']  # no .part left


def test_a_png_written_in_pieces_keeps_the_pixels(tmp_path):
    tall = image(531)
    with PngWriter(tmp_path / 'h.png', 80) as png:
        for start in range(0, len(tall), 50):
            png.write(tall[start : start + 50])
    pix = pymupdf.Pixmap(str(tmp_path / 'h.png'))
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(531, 80, 3), tall)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['h.png']


def fail_midway(path, tall):
    with PdfWriter(path, 80, page_height=100) as pdf:
        pdf.write(tall)
        msg = 'lost the window'
        raise RuntimeError(msg)


def test_what_was_written_before_a_failure_is_kept_under_the_real_name(tmp_path):
    tall = image(250)
    with pytest.raises(RuntimeError, match='lost the window'):
        fail_midway(tmp_path / 'i.pdf', tall)
    with pymupdf.open(tmp_path / 'i.pdf') as doc:
        assert np.array_equal(np.concatenate([pixels(p) for p in doc]), tall)


def test_nothing_written_leaves_no_file(tmp_path):
    PdfWriter(tmp_path / 'j.pdf', 80).close()
    PngWriter(tmp_path / 'j.png', 80).close()
    assert not list(tmp_path.iterdir())
