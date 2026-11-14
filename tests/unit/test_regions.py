"""A page's figure, table and equation regions found from its text alone, as on an OCR'd capture: a caption and
the prose around it bound a figure or table, an equation number the noise lines beside it."""

import pymupdf

from document_lab import layout
from document_lab.pdf import read_pdf
from document_lab.regions import regions

PROSE = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'


def _page(tmp_path, draw):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=840)
    draw(page)
    doc.save(tmp_path / 'r.pdf')
    read = read_pdf(tmp_path / 'r.pdf', tables=False)
    return read.pages[0], layout.body_size(read)


def _text(page, y, text, *, x=50, size=10):
    page.insert_text((x, y), text, fontsize=size)


def test_a_figure_runs_from_the_prose_above_down_to_its_caption(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 115, PROSE)
        for y, size in ((180, 22), (230, 6), (260, 31)):  # a drawing's labels, read as noise
            _text(page, y, 'i)|\\', x=250, size=size)
        _text(page, 300, 'Key', size=9)
        _text(page, 315, 'P load', size=9)
        _text(page, 360, 'Figure 2 - Short-time duty', x=200)
        _text(page, 420, PROSE)

    found = regions(*_page(tmp_path, draw))
    assert [(r.kind, r.label) for r in found] == [('figure', 'Figure 2')]
    box = found[0].bbox
    assert 115 < box.y0 < 165  # below the prose above
    assert 360 <= box.y1 < 410  # its caption, not the prose below
    assert box.x0 <= 50 and box.x1 >= 470  # the text width: the prose lines' extent


def test_a_table_runs_from_its_caption_down_to_the_prose_after_it(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 160, 'Table 1 - Preferred voltage ratings', x=200)
        for y in (190, 210, 230):
            _text(page, y, 'Rated voltage', x=120, size=9)
            _text(page, y, '150 to 1 000', x=330, size=9)
        _text(page, 300, PROSE)
        _text(page, 315, PROSE)

    found = regions(*_page(tmp_path, draw))
    assert [(r.kind, r.label) for r in found] == [('table', 'Table 1')]
    box = found[0].bbox
    assert 140 < box.y0 <= 152  # its caption's top
    assert 232 < box.y1 < 292  # past its last row, short of the prose


def test_an_equation_is_its_number_and_the_noise_beside_it_between_prose(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 160, 'U,=qW,+a, +220]', x=230, size=20)
        _text(page, 162, '(A.1)', x=520)
        _text(page, 185, '1/. +0U,', x=260, size=14)
        _text(page, 240, PROSE)

    found = regions(*_page(tmp_path, draw))
    assert [(r.kind, r.label) for r in found] == [('equation', '(A.1)')]
    box = found[0].bbox
    assert 100 < box.y0 < 148 and 185 <= box.y1 < 230


def test_a_mention_in_prose_or_a_contents_entry_is_no_caption(tmp_path):
    def draw(page):
        _text(
            page, 100, 'Temperature rise limits in accordance with Table 7, item 1d for resistance method shall apply.'
        )
        _text(page, 130, 'Table 1 - Preferred voltage ratings ..............ccccecce 28', x=60)
        _text(page, 160, PROSE)

    assert regions(*_page(tmp_path, draw)) == []


def _row(page, y, cells, size=10):
    for x, text in cells:
        _text(page, y, text, x=x, size=size)


def test_a_table_without_a_caption_is_found_by_its_columns(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _row(
            page,
            160,
            ((50, 'IEC 60034-18-41'), (160, '2014'), (200, 'Rotating electrical machines'), (420, 'EN 60034-18-41')),
        )
        _row(page, 175, ((200, 'Partial discharge free systems'),))  # a cell's second line
        _row(page, 200, ((50, 'IEC 60085'), (160, '-'), (200, 'Electrical insulation'), (420, 'EN 60085')))
        _row(page, 225, ((50, 'IEC 62539'), (160, '-'), (200, 'Guide for the analysis'), (420, '-')))
        _text(page, 300, PROSE)

    found = regions(*_page(tmp_path, draw))
    assert [(r.kind, r.label) for r in found] == [('table', 'Untitled table 1')]
    box = found[0].bbox
    assert 100 < box.y0 < 152 and 225 <= box.y1 < 292


def test_two_columns_of_prose_are_no_table(tmp_path):
    def draw(page):
        for y in range(100, 400, 15):
            _row(page, y, ((50, 'the rated output of the machine is'), (320, 'expressed in watts at the shaft end')))

    assert regions(*_page(tmp_path, draw)) == []


def test_a_captioned_table_is_not_found_twice(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 140, 'Table 2 - Duty types', x=200)
        for y in (170, 190, 210):
            _row(page, y, ((60, 'S1'), (200, 'continuous'), (400, '1 h')))
        _text(page, 280, PROSE)

    assert [(r.kind, r.label) for r in regions(*_page(tmp_path, draw))] == [('table', 'Table 2')]


def test_an_untitled_table_ends_at_the_formula_after_it(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        for y in (150, 170, 190):
            _row(page, y, ((60, 'Zero-sequence'), (220, '7.1'), (330, 'Single-phase voltage')))
        _text(page, 240, 'U,=qW,+a, +220]', x=230, size=18)  # a formula, OCR'd
        _text(page, 265, '1/. +0U,', x=260, size=14)

    found = regions(*_page(tmp_path, draw))
    assert [(r.kind, r.label) for r in found] == [('table', 'Untitled table 1')]
    assert found[0].bbox.y1 < 225


def test_two_columns_of_prose_split_into_three_pieces_are_no_table(tmp_path):
    def draw(page):  # a back-matter page: two columns of small prose, OCR'd into three pieces a line
        for y in range(100, 260, 14):
            _row(
                page,
                y,
                (
                    (50, 'All the content in BSI publications'),
                    (190, 'including British Standards is the'),
                    (320, 'documentation is current for further information'),
                ),
                size=8,
            )

    assert regions(*_page(tmp_path, draw)) == []


def test_rows_with_specks_read_as_cells_are_no_table(tmp_path):
    def draw(page):  # columns of prose with OCR specks beside them: a speck is no cell
        for y in (100, 120, 140):
            _row(page, y, ((50, 'All documentation'), (200, 'is current in BSI'), (380, 'oo'), (420, '.'), (460, 'i')))

    assert regions(*_page(tmp_path, draw)) == []


def test_a_long_line_in_a_narrow_column_is_prose_not_a_cell(tmp_path):
    def draw(page):  # a two-column back page: a long line of words beside short pieces
        for y in (100, 120, 140):
            _row(
                page,
                y,
                (
                    (50, 'you may not transfer share or disseminate any portion of the standard to'),
                    (330, 'Our British'),
                    (450, 'to benefit'),
                ),
                size=6,
            )

    assert regions(*_page(tmp_path, draw)) == []
