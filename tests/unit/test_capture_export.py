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
    check_name,
    save_png,
    to_pdf,
)
from document_lab.capture.pages import page_rhythm


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


# --page-height auto: a reader's long image -- pages of text lines between grey separator bands, as Acrobat draws
# them in continuous scroll -- cut on its separators. The three cases measured on a real capture: a 1775-row pitch,
# the same starting 37 rows into a page, and 1284 rows at another zoom (the page narrower than the view).

WIDE = 400


def scroll_border(out: np.ndarray, last_step: int) -> None:
    """A viewer border line the stitcher let through: once a step, the steps growing from one notch to ``last_step``."""
    at, step = 0, 7
    while at < len(out):
        if not (out[at] == 128).all():  # a separator row stays one
            out[at] = 60
        at += step
        step = min(2 * step, last_step)


def reader(pitch: int, pages: int, *, offset: int = 0, margin: int = 0, border_step: int = 0) -> np.ndarray:
    """A long reader image: ``pages`` pages ``pitch`` rows apart, each starting with a 6-row grey separator band
    (the first ``offset`` rows down), with a running header rule, text lines, ``margin`` grey columns each side, and a
    viewer border at ``border_step`` (0: none)."""
    rng = np.random.default_rng(pitch + offset)
    height = offset + pitch * pages
    out = np.full((height, WIDE, 3), 255, dtype=np.uint8)
    out[:, :margin] = 128
    out[:, WIDE - margin :] = 128
    text = slice(margin + 30, WIDE - margin - 30)
    for top in range(offset - pitch, height, pitch):
        if top >= 0:
            out[top : top + 6] = 128  # the separator: the whole width
        if 0 <= top + 40 < height:
            out[top + 40, text] = 0  # the running header rule: the text width
        for line in range(max(top + 60, 0), min(top + pitch - 30, height - 8), 14):
            out[line : line + 8, text] = rng.integers(0, 2, text.stop - text.start)[:, None] * 255  # never alike
    if border_step:
        scroll_border(out, border_step)
    return out


def page_tops(path) -> list[int]:
    with pymupdf.open(path) as doc:
        heights = [pixels(p).shape[0] for p in doc]
    return [int(t) for t in np.cumsum([0, *heights])[:-1]]


@pytest.mark.parametrize(
    ('pitch', 'offset', 'margin'),
    [(1775, 0, 0), (1775, 37, 0), (1284, 0, 60)],
    ids=['1775', '1775-offset-37', '1284-narrow'],
)
def test_auto_cuts_every_page_on_its_separator(tmp_path, pitch, offset, margin):
    long = reader(pitch, 8, offset=offset, margin=margin, border_step=500)
    out = tmp_path / 'auto.pdf'
    with PdfWriter(out, WIDE, page_height='auto') as pdf:
        for start in range(0, len(long), 700):  # streamed, as a capture writes it
            pdf.write(long[start : start + 700])
    assert pdf.rhythm is not None
    assert pdf.rhythm.pitch == pytest.approx(pitch, abs=0.01)
    assert page_tops(out) == sorted({0, *range(offset, len(long), pitch)})  # every page starts on its separator
    with pymupdf.open(out) as doc:
        assert np.array_equal(np.concatenate([pixels(p) for p in doc]), long)


def test_auto_without_separators_cuts_a4_pages_and_says_so(tmp_path):
    out = tmp_path / 'noise.pdf'
    with PdfWriter(out, WIDE, page_height='auto') as pdf:
        pdf.write(image(6000, WIDE))
    assert pdf.rhythm is None
    assert page_tops(out) == list(range(0, 6000, a4_page_height(WIDE)))


def test_a_border_line_at_the_scroll_step_is_not_taken_for_the_page():
    plain = np.full((5000, WIDE, 3), 255, dtype=np.uint8)
    scroll_border(plain, 400)
    assert page_rhythm(plain) is None  # the control: a scroll-step border alone is no page rhythm
    plain[1::1000] = 128
    rhythm = page_rhythm(plain)
    assert rhythm is not None
    assert (rhythm.pitch, rhythm.offset) == (pytest.approx(1000, abs=0.01), 1)


@pytest.mark.parametrize(
    'name', ['BS-EN-60034-1:2010.pdf', 'a<b.pdf', 'a|b.pdf', 'a?.pdf', 'a*.pdf', 'a".pdf', 'CON.pdf', 'x.pdf.']
)
def test_a_name_windows_cannot_hold_is_refused_with_one_it_can(tmp_path, name):
    with pytest.raises(ValueError, match='cannot name a file') as refused:
        check_name(tmp_path / name)
    suggested = str(refused.value).rsplit(': try ', 1)[1]
    check_name(tmp_path / suggested)


def test_a_drive_and_folders_are_not_the_name(tmp_path):
    check_name(tmp_path / 'sub' / 'BS-EN-60034-1_2010.pdf')
