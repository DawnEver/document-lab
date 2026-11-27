"""A PDF written page by page: a markdown file and an image per page, an index, and the pages without text named."""

import zipfile

import numpy as np
import pymupdf
import pytest

from document_lab import layout
from document_lab.__main__ import main
from document_lab.capture.export import to_pdf
from document_lab.markdown import table as markdown_table
from document_lab.pdf import read_pdf
from document_lab.per_page import RENDER_DPI, UNREAD, write_pages
from document_lab.regions import regions
from document_lab.tables import tables
from tests.synthetic import (
    LANDSCAPE_CAPTION,
    LANDSCAPE_HEAD,
    LANDSCAPE_NOTE,
    TABLE_ROWS,
    draw_booktabs,
    draw_ruled_table,
    landscape_pdf,
    table_cell_rect,
)

#: A line of prose: it bounds a region.
PROSE = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'


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
    assert done.unread == (1,)  # a picture without a text layer, said so in its markdown
    assert UNREAD in (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
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


def test_the_index_names_each_page_by_the_clauses_its_outline_states(tmp_path):
    with pymupdf.open() as doc:
        for clause in ('Scope', 'Terms and definitions'):
            page = doc.new_page(width=400, height=300)
            page.insert_text((40, 60), clause, fontsize=12)  # plain words: the rules cannot tell a clause by them
            page.insert_text((40, 100), 'The text of the clause, in a sentence of its own.', fontsize=10)
        doc.set_toc([[1, 'Scope', 1], [1, 'Terms and definitions', 2]])
        doc.save(tmp_path / 'std.pdf')
    done = write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    assert done.structure.source == 'outline'
    index = (tmp_path / 'out' / 'INDEX.md').read_text(encoding='utf-8')
    assert '- [page 1](page-001.md) -- Scope\n' in index
    assert '- [page 2](page-002.md) -- Terms and definitions\n' in index
    inferred = write_pages(tmp_path / 'std.pdf', tmp_path / 'inferred', outline=False)
    assert inferred.structure.source == 'inferred'
    assert '-- Scope' not in (tmp_path / 'inferred' / 'INDEX.md').read_text(encoding='utf-8')


def test_a_ruled_table_is_written_as_a_pipe_table_in_place_of_its_words(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 300), 'The limits are given below.', fontsize=10)
        draw_ruled_table(page)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert '| Parameter | Value | Unit |\n| --- | --- | --- |\n| alpha | 1.25 | mm |' in page1
    assert page1.count('alpha') == 1  # its words are the table's, not the running text's
    assert page1.index('The limits are given below.') < page1.index('| Parameter')


def test_a_table_ruled_by_its_rows_alone_is_written_as_a_pipe_table(tmp_path):
    """Booktabs: no rule down, its columns parted by white space; once its rules were no table and two-cell bands
    no rows, so its words ran on as text."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 300), 'The limits are given below.', fontsize=10)
        draw_booktabs(page)
        doc.save(tmp_path / 'paper.pdf')
    write_pages(tmp_path / 'paper.pdf', tmp_path / 'out')
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert '| Parameter | Value | Unit |\n| --- | --- | --- |\n| alpha | 1.25 | mm |' in page1
    assert page1.count('alpha') == 1  # its words are the table's, not the running text's


def test_a_scanned_table_reads_from_its_rules_in_pixels_and_its_text_layer(tmp_path):
    from document_lab import _pdf  # noqa: PLC0415
    from tests.synthetic import scan_image, scan_pdf  # noqa: PLC0415

    with pymupdf.open() as doc:
        drawn = doc.new_page(width=612, height=792)
        draw_ruled_table(drawn)
        words = [(w[4], tuple(w[:4]), 0.0, False) for w in drawn.get_text('words')]  # where the scan shows them
    scan = scan_pdf(tmp_path / 'scan.pdf', scan_image(draw_ruled_table, dpi=150))
    reader = _pdf.open(scan.read_bytes(), b'')
    writer = _pdf.create(tmp_path / 'ocr.pdf')
    writer.copy_page_with_text(reader, 1, words)
    writer.close()
    done = write_pages(tmp_path / 'ocr.pdf', tmp_path / 'out')
    assert done.unread == ()
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert '| Parameter | Value | Unit |\n| --- | --- | --- |\n| alpha | 1.25 | mm |' in page1
    assert '| gamma | 4.5e2 | s |' in page1


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


def test_running_heads_and_feet_leave_the_body_and_are_listed_in_the_page_s_front_matter(tmp_path):
    """ACT1210: its running head and its page numbers were read into the body of every page."""
    with pymupdf.open() as doc:
        for number in range(1, 4):
            page = doc.new_page(width=595, height=842)
            page.insert_text((42, 60), 'Data and Signal Line Chokes', fontsize=11)
            page.insert_text((499, 60), 'ACT1210', fontsize=11)
            page.insert_text((37, 120), f'Clause {number} states the rated current of the part.', fontsize=11)
            page.insert_text((291, 822), str(number), fontsize=11)
        doc.save(tmp_path / 'sheet.pdf')
    write_pages(tmp_path / 'sheet.pdf', tmp_path / 'out')
    md = (tmp_path / 'out' / 'page-002.md').read_text(encoding='utf-8')
    assert md.startswith(
        '---\nheader: ["Data and Signal Line Chokes", "ACT1210"]\nfooter: ["2"]\n---\n\n[sheet.pdf, page 2]'
    )
    body = md.split('(page-002.png)', 1)[1]
    assert body.strip() == 'Clause 2 states the rated current of the part.'


_COLUMNS = (37, 86, 132, 179, 225, 271, 317, 425)
_CODES = (
    ('11', '0.05', '300', '0.4', '300', '550', 'B82786C0113H002', 'ACT1210-110-2P-TL00'),
    ('22', '0.06', '250', '0.5', '500', '1100', 'B82786C0223H002', 'ACT1210-220-2P-TL00'),
    ('51', '0.09', '200', '0.7', '1000', '2600', 'B82786C0513H002', 'ACT1210-510-2P-TL00'),
    ('100', '0.13', '150', '1.5', '2200', '5100', 'B82786C0104H002', 'ACT1210-101-2P-TL00'),
)


def _data_sheet_tables(path):
    """ACT1210 page 4: names and values, rows ruled apart, one rule down between the columns and no frame; then the
    ordering codes: two header rows (symbols set with subscripts, units) over a rule, columns ruled, open at the sides.
    """

    def text(x, y, s, size=11, font='helv'):
        page.insert_text((x, y), s, fontsize=size, fontname=font)

    def rule(x0, y0, x1, y1):
        page.draw_line((x0, y0), (x1, y1), width=0.5)

    with pymupdf.open() as doc:
        page = doc.new_page(width=595, height=842)
        text(42, 60, 'Data and Signal Line Chokes')
        text(499, 60, 'ACT1210')
        text(37, 108, 'Technical data and measuring conditions', font='hebo')
        values = (
            ('Rated voltage', ('80 V DC',)),
            (
                'Rated inductance',
                (
                    'Measured with Keysight E4990A (or equivalent) at 100 kHz, 100 mV,',
                    '+20 C, inductance is specified in common-mode',
                ),
            ),
            ('Inductance tolerance', ('-30/+50% at +20 C',)),
            ('DC resistance', ('Measured at +20 C, specified per winding',)),
            ('Climatic category', ('55/150/56 (to IEC 60068-1)',)),
        )
        y = top = 117
        for name, lines in values:
            rule(37, y, 550, y)
            text(37, y + 13, name)
            for k, line in enumerate(lines):
                text(207, y + 13 + 13 * k, line)
            y += 6 + 13 * len(lines)
        rule(37, y, 550, y)
        rule(204, top, 204, y)
        text(37, y + 30, 'Characteristics and ordering codes', font='hebo')
        t = y + 40
        bottom = _ordering_codes(text, rule, t)
        text(37, 806, 'Please read Cautions and warnings and', size=9)
        text(291, 822, '4')
        doc.save(path)
    return path, t, bottom


def _ordering_codes(text, rule, t):
    """The ordering codes table from ``t`` down, drawn by ``text`` and ``rule``; its bottom rule's height."""
    symbols = (('L', 'R'), ('L', 'stray'), ('I', 'R'), ('R', 'max'), ('Z', 'min'), ('Z', 'typ'))
    for x, (symbol, sub) in zip(_COLUMNS, symbols, strict=False):
        text(x, t + 12, symbol)
        text(x + pymupdf.get_text_length(symbol, fontsize=11), t + 14, sub, size=7)
    text(317, t + 12, 'Internal code')
    text(425, t + 12, 'Ordering code')
    for x, unit in zip(_COLUMNS, ('uH', 'uH', 'mA'), strict=False):
        text(x, t + 29, unit)
    for x in _COLUMNS[3:6]:
        text(x, t + 29, 'W', font='symb')  # an ohm sign alone
    rule(37, t + 36, 550, t + 36)
    for k, row in enumerate(_CODES):
        for x, cell in zip(_COLUMNS, row, strict=True):
            text(x, t + 49 + 17 * k, cell)
    bottom = t + 36 + 4 * 17 + 4
    rule(37, t, 550, t)
    rule(37, bottom, 550, bottom)
    for x in _COLUMNS[1:]:
        rule(x - 3, t, x - 3, bottom)
    return bottom


def test_a_ruled_two_column_table_and_an_ordering_table_are_cut_whole_and_written_row_by_row(tmp_path):
    """ACT1210 page 4: the names-and-values table was no region and its markdown paired names with values one row out
    of step; the ordering table's crop left out its two header rows and its code columns landed after the footer."""
    pdf, top, bottom = _data_sheet_tables(tmp_path / 'sheet.pdf')
    write_pages(pdf, tmp_path / 'out')
    out = tmp_path / 'out'
    assert (out / 'page-001-table-untitled-1.png').is_file()
    codes = pymupdf.Pixmap(str(out / 'page-001-table-untitled-2.png'))
    assert codes.height >= (bottom - top - 6) * RENDER_DPI / 72  # its header rows inside it
    md = (out / 'page-001.md').read_text(encoding='utf-8')
    assert '| Rated voltage | 80 V DC |' in md
    measured = 'Measured with Keysight E4990A (or equivalent) at 100 kHz, 100 mV, +20 C, inductance is specified in'
    assert f'| Rated inductance | {measured} common-mode |' in md
    assert '| Inductance tolerance | -30/+50% at +20 C |' in md
    rows = [md.index('| ' + ' | '.join(row) + ' |') for row in _CODES]
    assert rows == sorted(rows)
    assert md.index('Internal code | Ordering code |') < rows[0] < rows[-1] < md.index('Please read Cautions')
    assert md.index('| Climatic category') < md.index('Characteristics and ordering codes') < rows[0]


def test_text_drawn_inside_an_untitled_figure_goes_with_it_not_into_the_body(tmp_path):
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=840)
        page.insert_text((50, 100), prose, fontsize=10)
        page.draw_rect(pymupdf.Rect(150, 140, 450, 320), color=(0, 0, 0))
        page.draw_line((150, 320), (450, 140), width=2)  # a characteristic, drawn
        page.draw_circle((300, 200), 40, color=(0, 0, 0), width=2)
        page.insert_text((290, 280), 'N', fontsize=10)  # a pole's letter in the drawing
        page.insert_text((50, 420), prose, fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    md = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    blocks = md.split('\n\n')
    assert 'N' not in blocks
    figure = blocks.index('![Untitled figure 1](page-001-figure-untitled-1.png)')
    assert blocks[figure + 1] == '*Text in the figure:* N'
    assert blocks[figure - 1].endswith(prose) and blocks[figure + 2].startswith(prose)


def test_the_unnumbered_equations_option_reaches_the_pages_written(tmp_path):
    """``document-lab pages --unnumbered-equations``: a display without a number is cut out at strict (the default)
    and left in the text at off, numbered ones either way."""
    pdf = tmp_path / 'sheet.pdf'
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    for y in (100, 112, 124):
        page.insert_text((60, y), prose, fontsize=10)
    page.insert_text((230, 150), 'E = m c2 + k', fontsize=10, fontname='tiit')
    page.insert_text((230, 180), 'P = T w', fontsize=10, fontname='tiit')
    page.insert_text((520, 180), '(3)', fontsize=10)
    for y in (210, 222):
        page.insert_text((60, y), prose, fontsize=10)
    doc.save(pdf)
    for level, files in (('strict', ['equation-3', 'equation-untitled-1']), ('off', ['equation-3'])):
        out = tmp_path / level
        assert main(['pages', str(pdf), str(out), '--unnumbered-equations', level]) == 0
        assert sorted(p.stem.removeprefix('page-001-') for p in out.glob('page-001-equation-*.png')) == files


@pytest.mark.parametrize('kind', ['hidden', 'in_background_color'])
def test_text_the_page_does_not_show_is_written_only_when_its_own_flag_asks(tmp_path, kind):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 60), 'Words the page shows.', fontsize=10)
        _unshown(doc, page, (40, 100), 'Words the page does not show.', kind, fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    own, other = ('--include-hidden', '--include-in-background-color')[:: 1 if kind == 'hidden' else -1]
    for flags, read in (([], False), ([other], False), ([own], True)):  # the other flag: the planted control
        out = tmp_path / '-'.join(['out', *flags])
        assert main(['pages', str(tmp_path / 'std.pdf'), str(out), *flags]) == 0
        written = (out / 'page-001.md').read_text(encoding='utf-8')
        assert 'Words the page shows.' in written
        assert ('does not show' in written) is read, flags


#: The text a page draws but its default view does not show, planted two ways, and the option that reads it.
UNSHOWN = {'hidden': 'include_hidden', 'in_background_color': 'include_in_background_color'}


def _unshown(doc, page, point, text, kind, **style):
    """``text`` set at ``point`` on ``page`` as the default view does not show it: on a layer turned off, or white."""
    if kind == 'hidden':
        page.insert_text(point, text, oc=doc.add_ocg('Off', on=False), **style)
    else:
        page.insert_text(point, text, color=(1, 1, 1), **style)


@pytest.mark.parametrize('kind', UNSHOWN)
def test_a_ruled_table_s_cells_read_what_the_page_shows(tmp_path, kind):
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 300), 'The limits are given below.', fontsize=10)
        draw_ruled_table(page)
        cell = table_cell_rect(1, 2)
        _unshown(doc, page, (cell.x0 + 60, cell.y1 - 6), 'unseen', kind, fontname='helv', fontsize=9)
        doc.save(tmp_path / 'std.pdf')
    for asked in (False, True):
        write_pages(tmp_path / 'std.pdf', tmp_path / f'out-{asked}', **{UNSHOWN[kind]: asked})
        page1 = (tmp_path / f'out-{asked}' / 'page-001.md').read_text(encoding='utf-8')
        (row,) = [line for line in page1.splitlines() if line.startswith('| alpha')]
        assert ('unseen' in row) is asked, row
        assert page1.count('unseen') == asked  # in its cell alone, never the running text


@pytest.mark.parametrize('kind', UNSHOWN)
def test_running_heads_are_what_the_page_shows(tmp_path, kind):
    with pymupdf.open() as doc:
        for number in range(1, 4):
            page = doc.new_page(width=595, height=842)
            page.insert_text((42, 60), 'Data and Signal Line Chokes', fontsize=11)
            _unshown(doc, page, (499, 60), 'ACT1210', kind, fontsize=11)
            page.insert_text((37, 120), f'Clause {number} states the rated current of the part.', fontsize=11)
        doc.save(tmp_path / 'sheet.pdf')
    for asked in (False, True):
        write_pages(tmp_path / 'sheet.pdf', tmp_path / f'out-{asked}', **{UNSHOWN[kind]: asked})
        md = (tmp_path / f'out-{asked}' / 'page-002.md').read_text(encoding='utf-8')
        heads = '["Data and Signal Line Chokes", "ACT1210"]' if asked else '["Data and Signal Line Chokes"]'
        assert md.startswith(f'---\nheader: {heads}\n---\n\n'), md


@pytest.mark.parametrize('kind', UNSHOWN)
def test_an_untitled_figure_takes_the_text_the_page_shows_in_it(tmp_path, kind):
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=840)
        page.insert_text((50, 100), prose, fontsize=10)
        # Its frame and its axis drawn as lines of their own: a mark under the white letter, by its box, would show it
        # (the reader judges a stroke by its path's box).
        for a, b in (
            ((150, 140), (450, 140)),
            ((450, 140), (450, 320)),
            ((450, 320), (150, 320)),
            ((150, 320), (150, 140)),
        ):
            page.draw_line(a, b, width=1)
        page.draw_line((150, 230), (450, 230), width=2)
        page.draw_circle((300, 200), 40, color=(0, 0, 0), width=2)
        page.insert_text((290, 280), 'N', fontsize=10)  # a pole's letter, shown
        _unshown(doc, page, (380, 300), 'S', kind, fontsize=10)  # another, not
        page.insert_text((50, 420), prose, fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    for asked in (False, True):
        write_pages(tmp_path / 'std.pdf', tmp_path / f'out-{asked}', **{UNSHOWN[kind]: asked})
        blocks = (tmp_path / f'out-{asked}' / 'page-001.md').read_text(encoding='utf-8').split('\n\n')
        figure = blocks.index('![Untitled figure 1](page-001-figure-untitled-1.png)')
        assert blocks[figure + 1] == ('*Text in the figure:* N S' if asked else '*Text in the figure:* N')


def test_a_landscape_table_s_caption_heads_it_and_its_region_is_cut_and_linked(tmp_path):
    """A landscape table printed sideways under an upright running head: the page reads sideways, so its caption --
    above the table in their frame -- comes first, then the region's link and the table, then the note after it; the
    head, turned against the page's text, is neither its body nor its front matter."""
    pdf = landscape_pdf(tmp_path / 'landscape.pdf')
    assert main(['pages', str(pdf), str(tmp_path / 'out')]) == 0
    md = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    blocks = md.rstrip('\n').split('\n\n')
    order = [
        blocks.index(LANDSCAPE_CAPTION[0]),
        blocks.index('![Table 10](page-001-table-10.png)'),
        blocks.index(markdown_table(TABLE_ROWS)),
        blocks.index(LANDSCAPE_NOTE[0]),
    ]
    assert order == sorted(order)
    assert (tmp_path / 'out' / 'page-001-table-10.png').exists()
    assert not any(head in md for head, _ in LANDSCAPE_HEAD)
    # The planted wrong truth: read as the page stands, the table -- its top above the caption's -- would come first.
    page = read_pdf(pdf).page(1)
    (table,) = tables(page)
    caption = next(s for s in page.spans if s.text == LANDSCAPE_CAPTION[0]).provenance.bbox
    assert table.provenance.bbox.y0 < caption.y0


def test_a_region_s_text_reading_more_rows_but_fewer_columns_than_its_rules_gives_way_to_them(tmp_path):
    """A captioned ruled table whose first column's words run up to its rule (the next column's 3 points on) and
    whose first cell wraps: its text reads four rows in two columns, its rules three rows in three. Text that runs
    columns its rules part into one reads less, however many rows it makes: the ruled table is written."""
    rows = (('Grade', 'Flux mT', 'Hc kA/m'), ('AlNiCo 9/5', '550', '44'), ('AlNiCo 12/6', '630', '52'))
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 380), 'Table 1 - Magnetic properties of the cast grades', fontname='helv', fontsize=9)
        draw_ruled_table(page, [('', '', '')] * len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                rect, font = table_cell_rect(r, c), 'hebo' if c == 0 else 'helv'
                lines = [text, 'cast'] if (r, c) == (1, 0) else [text]
                for i, line in enumerate(lines):
                    x = (
                        rect.x1 - 2 - pymupdf.get_text_length(line, fontname=font, fontsize=9)
                        if c == 0
                        else rect.x0 + 1
                    )
                    y = rect.y1 - 6 if len(lines) == 1 else rect.y0 + 9 + 9 * i
                    page.insert_text((x, y), line, fontname=font, fontsize=9)
        page.insert_text((72, 520), PROSE, fontname='helv', fontsize=9)
        doc.save(tmp_path / 'std.pdf')
    read = read_pdf(tmp_path / 'std.pdf')
    (region,) = regions(read.page(1), layout.body_face(read))
    (ruled,) = tables(read.page(1))
    # The planted wrong truth: the region's text reads more rows than the rules, in fewer columns.
    assert len(region.table.rows) > len(ruled.rows)
    assert max(map(len, region.table.rows)) < len(ruled.rows[0])
    assert main(['pages', str(tmp_path / 'std.pdf'), str(tmp_path / 'out')]) == 0
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert markdown_table(ruled.rows) in page1
    assert sum(line.startswith('| ---') for line in page1.splitlines()) == 1  # one table written


def test_untitled_rows_inside_a_ruled_table_no_caption_holds_are_its_own_and_written_once(tmp_path):
    """A ruled table with no caption -- a continued table -- whose rows a full-width note parts: its rows above and
    below the note are each told as an untitled table by their cells, inside the ruled one. They are its own rows:
    the ruled table is written once, with all of them, and no region of them alone is cut."""
    note = 'Typical values of further properties are given for information only'
    rows = (*TABLE_ROWS[:2], (note, '', ''), *TABLE_ROWS[2:])
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 300), PROSE, fontname='helv', fontsize=9)
        draw_ruled_table(page, rows, merged=((2, 0, 2, 2),))
        doc.save(tmp_path / 'std.pdf')
    read = read_pdf(tmp_path / 'std.pdf')
    (ruled,) = tables(read.page(1))
    # The planted wrong truth: told by their rows, the bands above and below the note are tables of their own.
    told = [r for r in regions(read.page(1), layout.body_face(read)) if r.table is not None]
    assert [r.table.rows for r in told] == [rows[:2], rows[3:]]
    assert main(['pages', str(tmp_path / 'std.pdf'), str(tmp_path / 'out')]) == 0
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert sum(line.startswith('| ---') for line in page1.splitlines()) == 1  # one table written
    assert markdown_table(ruled.rows, ruled.cell_bboxes) in page1
    assert page1.count('alpha') == 1
