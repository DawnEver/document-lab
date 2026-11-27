"""A PDF written page by page: a markdown file and an image per page, an index, and the pages without text named."""

import zipfile

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import to_pdf
from document_lab.per_page import RENDER_DPI, write_pages


def test_each_page_gets_its_markdown_and_image_and_the_index_names_them(tmp_path):
    with pymupdf.open() as doc:
        first = doc.new_page(width=400, height=300)
        first.insert_text((40, 60), '4 Duty', fontsize=14, fontname='hebo')
        first.insert_text((40, 100), 'The duty shall be declared by the purchaser.', fontsize=10)
        doc.new_page(width=400, height=300)  # blank
        third = doc.new_page(width=400, height=300)
        third.insert_text((40, 60), 'Annex A continues here.', fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    done = write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    assert done.pages == 3
    assert done.textless == (2,)
    names = sorted(p.name for p in (tmp_path / 'out').iterdir())
    assert names == ['INDEX.md', *(f'page-00{n}.{ext}' for n in (1, 2, 3) for ext in ('md', 'png')), 'std.pdf']
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert page1.startswith('[std.pdf, page 1](page-001.png)')
    assert 'declared by the purchaser' in page1
    assert 'Annex A' not in page1  # a page stands alone
    index = (tmp_path / 'out' / 'INDEX.md').read_text(encoding='utf-8')
    assert '- [page 3](page-003.md)\n' in index
    assert '- [page 1](page-001.md) -- 4 Duty' in index


def test_a_page_that_is_one_image_keeps_its_own_pixels(tmp_path, monkeypatch):
    image = np.random.default_rng(1).integers(0, 256, (90, 70, 3), dtype=np.uint8)
    capture = to_pdf(image, tmp_path / 'capture.pdf')
    monkeypatch.setattr(pymupdf.Page, 'get_pixmap', None)  # taken from the reader's samples, never rendered
    done = write_pages(capture, tmp_path / 'out')
    assert done.textless == (1,)
    pix = pymupdf.Pixmap(str(tmp_path / 'out' / 'page-001.png'))
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(90, 70, 3), image)


def test_a_page_of_text_placing_one_small_image_is_rendered_whole_not_taken_for_the_image(tmp_path):
    """A standard's cover: its text, and one image -- the publisher's logo -- that is not the page."""
    logo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 30, 20), False)
    logo.set_rect(logo.irect, (0, 90, 160))
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.insert_image(pymupdf.Rect(20, 20, 80, 60), pixmap=logo)
        page.insert_text((40, 120), 'Rotating electrical machines', fontsize=14)
        doc.save(tmp_path / 'cover.pdf')
    write_pages(tmp_path / 'cover.pdf', tmp_path / 'out')
    pix = pymupdf.Pixmap(str(tmp_path / 'out' / 'page-001.png'))
    assert abs(pix.width - 400 * RENDER_DPI / 72) <= 1  # the renderer rounds the page's edges outward
    assert abs(pix.height - 300 * RENDER_DPI / 72) <= 1


def test_figures_tables_and_equations_are_cut_out_and_linked_where_their_text_is(tmp_path):
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=840)
        page.insert_text((50, 100), prose, fontsize=10)
        page.draw_rect(pymupdf.Rect(150, 140, 450, 320), color=(0, 0, 0))
        page.insert_text((200, 360), 'Figure 2 - Short-time duty', fontsize=10)
        page.insert_text((50, 420), prose, fontsize=10)
        page.insert_text((230, 470), 'U = R I', fontsize=20)
        page.insert_text((520, 470), '(3)', fontsize=10)
        page.insert_text((50, 520), prose, fontsize=10)
        page.insert_text((200, 600), 'Table 1 - Preferred voltage ratings', fontsize=10)
        page.insert_text((120, 630), 'Rated voltage', fontsize=9)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    out = tmp_path / 'out'
    for name in ('page-001-figure-2.png', 'page-001-equation-3.png', 'page-001-table-1.png'):
        assert (out / name).is_file(), name
    md = (out / 'page-001.md').read_text(encoding='utf-8')
    figure = md.index('![Figure 2](page-001-figure-2.png)')
    assert md.index('Figure 2 - Short-time duty') < figure < md.index(prose, md.index('Figure 2'))
    assert '![(3)](page-001-equation-3.png)' in md
    assert '![Table 1](page-001-table-1.png) (continues on the next page)' in md
    assert 'Figure 2, (3), Table 1' in (out / 'INDEX.md').read_text(encoding='utf-8')


def test_a_zip_out_holds_the_pdf_and_everything_a_directory_would(tmp_path):
    with pymupdf.open() as doc:
        doc.new_page(width=400, height=300).insert_text((40, 60), '4 Duty', fontsize=12)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'dir')
    done = write_pages(tmp_path / 'std.pdf', tmp_path / 'std.zip')
    assert done.out == tmp_path / 'std.zip'
    with zipfile.ZipFile(tmp_path / 'std.zip') as archive:
        names = sorted(archive.namelist())
        assert names == sorted(p.name for p in (tmp_path / 'dir').iterdir())
        assert 'std.pdf' in names
        assert archive.read('page-001.md') == (tmp_path / 'dir' / 'page-001.md').read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['dir', 'std.pdf', 'std.zip']  # nothing left beside it


def test_a_ruled_table_whose_words_no_reader_has_is_cut_out_by_its_rules(tmp_path):
    """A two-column table ('Draft | Report on voting'): the page's picture has its rules, regions their box."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=840)
        prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
        page.insert_text((50, 100), prose, fontsize=10)
        for y in (150, 180, 210):
            page.draw_line((180, y), (420, y), width=1)
        for x in (180, 300, 420):
            page.draw_line((x, 150), (x, 210), width=1)
        page.insert_text((50, 300), prose, fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'out.zip')
    with zipfile.ZipFile(tmp_path / 'out.zip') as archive:
        assert 'page-001-table-untitled-1.png' in archive.namelist()
        assert '![Untitled table 1](page-001-table-untitled-1.png)' in archive.read('page-001.md').decode()


def test_a_drawing_every_page_repeats_is_furniture_and_one_page_s_table_is_still_cut(tmp_path):
    """A JMAG whitepaper's header logo: its letters' strokes read as a ruled table on every one of its pages."""

    def grid(page, x0, y0):
        for y in (y0, y0 + 30, y0 + 60):
            page.draw_line((x0, y), (x0 + 240, y), width=1)
        for x in (x0, x0 + 120, x0 + 240):
            page.draw_line((x, y0), (x, y0 + 60), width=1)

    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        for number in (1, 2):
            page = doc.new_page(width=600, height=840)
            grid(page, 180, 120)  # the same place on both pages: furniture
            page.insert_text((50, 250), prose, fontsize=10)
            if number == 1:
                grid(page, 180, 300)  # this page's own table
            page.insert_text((50, 450), prose, fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'out.zip')
    with zipfile.ZipFile(tmp_path / 'out.zip') as archive:
        cut = sorted(n for n in archive.namelist() if '-table-' in n)
    assert cut == ['page-001-table-untitled-1.png']


def _one_image_page(path, colorspace, samples, *, extra=None):
    """A 200 x 150 pt page that is one 8 x 6 image in ``colorspace`` (a PDF object), every sample ``samples``."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=150)
        if extra is not None:  # an object the color space refers to (an ICC profile's stream)
            ref = doc.get_new_xref()
            doc.update_object(ref, extra[0])
            doc.update_stream(ref, extra[1])
            colorspace = colorspace.replace('REF', f'{ref} 0 R')
        image = doc.get_new_xref()
        doc.update_object(
            image,
            f'<< /Type /XObject /Subtype /Image /Width 8 /Height 6 /ColorSpace {colorspace} /BitsPerComponent 8 >>',
        )
        doc.update_stream(image, bytes(samples) * 48)
        doc.xref_set_key(page.xref, 'Resources', f'<< /XObject << /Im0 {image} 0 R >> >>')
        content = doc.get_new_xref()
        doc.update_object(content, '<< >>')
        doc.update_stream(content, b'q 200 0 0 150 0 0 cm /Im0 Do Q')
        doc.xref_set_key(page.xref, 'Contents', f'{content} 0 R')
        doc.save(path)
    return path


@pytest.mark.parametrize(
    ('colorspace', 'samples', 'extra', 'seen'),
    [
        ('/DeviceCMYK', (0, 255, 255, 0), None, 'red'),  # magenta and yellow ink
        ('[/Indexed /DeviceRGB 1 <FF000000C000>]', (1,), None, 'green'),  # palette entry 1
        ('[/Lab << /WhitePoint [0.9505 1 1.089] >>]', (128, 128, 128), None, 'gray'),  # L* 50, neutral
        # a CMYK ICC profile no reader can use: its /Alternate, DeviceCMYK, stands for it
        ('[/ICCBased REF]', (0, 255, 255, 0), ('<< /N 4 /Alternate /DeviceCMYK >>', b'not a profile'), 'red'),
    ],
)
def test_a_page_that_is_one_image_in_any_color_space_is_written_in_its_colors(
    tmp_path, colorspace, samples, extra, seen
):
    """JMAG manuals and data sheets: a page that is one CMYK picture once crashed the PNG writer."""
    pdf = _one_image_page(tmp_path / 'scan.pdf', colorspace, samples, extra=extra)
    write_pages(pdf, tmp_path / 'out')
    pix = pymupdf.Pixmap(str(tmp_path / 'out' / 'page-001.png'))
    assert pix.n == 3
    r, g, b = (
        int(v) for v in np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)[pix.h // 2, pix.w // 2]
    )
    if seen == 'red':
        assert r > 180 and g < 90 and b < 90, (r, g, b)
    elif seen == 'green':
        assert g > 150 and r < 60 and b < 60, (r, g, b)
    else:
        assert 90 < r < 150 and abs(r - g) < 12 and abs(r - b) < 12, (r, g, b)


def test_a_caption_across_the_page_foot_cuts_no_empty_region(tmp_path):
    """A caption the page's foot cuts, prose right above it: its figure has no room on the page, and is no region."""
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=300)
        page.insert_text((50, 296), prose, fontsize=10)
        page.insert_text((50, 307), 'Figure 3 - At the foot', fontsize=10)
        doc.save(tmp_path / 'foot.pdf')
    write_pages(tmp_path / 'foot.pdf', tmp_path / 'out')
    assert not list((tmp_path / 'out').glob('page-001-figure-*.png'))
