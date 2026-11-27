"""A page's figure, table and equation regions found from its text alone, as on an OCR'd capture: a caption and
the prose around it bound a figure or table, an equation number the noise lines beside it."""

import pymupdf
import pytest

from document_lab import layout
from document_lab.ink import Drawing
from document_lab.ir import BBox
from document_lab.pdf import read_pdf
from document_lab.regions import regions
from tests.synthetic import LANDSCAPE_CAPTION, TABLE_ROWS, hidden_layer, in_background_color, landscape_pdf

PROSE = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'


def _page(tmp_path, draw):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=840)
    draw(page)
    doc.save(tmp_path / 'r.pdf')
    read = read_pdf(tmp_path / 'r.pdf')
    return read.pages[0], layout.body_face(read)


def _text(page, y, text, *, x=50, size=10, ocr=False):
    page.insert_text((x, y), text, fontsize=size, render_mode=3 if ocr else 0)


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
        _text(page, 160, 'U,=qW,+a, +220]', x=230, size=20, ocr=True)
        _text(page, 162, '(A.1)', x=520, ocr=True)
        _text(page, 185, '1/. +0U,', x=260, size=14, ocr=True)
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


def _row(page, y, cells, size=10, *, ocr=False):
    for x, text in cells:
        _text(page, y, text, x=x, size=size, ocr=ocr)


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

    found = regions(*_page(tmp_path, draw))  # the formula is a display of its own, no number printed
    assert [(r.kind, r.label) for r in found] == [('table', 'Untitled table 1'), ('equation', 'Untitled equation 1')]
    assert found[0].bbox.y1 < 225 and found[0].bbox.y1 <= found[1].bbox.y0


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
    def draw(page):  # columns of prose with OCR specks beside them, an OCR layer: a speck is no cell
        for y in (100, 120, 140):
            cells = ((50, 'All documentation'), (200, 'is current in BSI'), (380, 'oo'), (420, '.'), (460, 'i'))
            _row(page, y, cells, ocr=True)

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


def _drawn(box, rules=(0, 0), loose=0.0, stroke=100.0):
    return Drawing(BBox(*box), rules, loose, stroke)


def test_a_ruled_table_of_two_columns_its_words_unread_is_a_table(tmp_path):
    """IEC voting tables ('Draft | Report on voting'): two columns, words too small for OCR; the rules tell it."""

    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 300, PROSE)

    page, face = _page(tmp_path, draw)
    found = regions(page, face, [_drawn((150, 150, 440, 230), rules=(3, 3), loose=6)])
    assert [(r.kind, r.label) for r in found] == [('table', 'Untitled table 1')]


def test_a_frame_round_a_lines_words_a_margin_note_and_a_short_line_are_no_regions(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 200, 'IEC 60034-18-41 applies', x=60)

    page, face = _page(tmp_path, draw)
    link = _drawn((55, 190, 150, 204), rules=(2, 2))  # a link's frame round words of the line
    margin = _drawn((5, 100, 20, 700))  # a sideways license note beside the text column
    rule = _drawn((50, 400, 500, 401))  # a rule under a header
    words = _drawn((60, 500, 120, 540), stroke=8)  # words OCR missed: glyphs, no drawn line
    # a link's frame just before the line, its words unread: under three of its line's heights (10 pt Helvetica: 9.25)
    beside = _drawn((42, 190, 58, 216))
    assert regions(page, face, [link, margin, rule, words, beside]) == []


def test_a_drawings_views_side_by_side_are_one_figure_and_prose_parts_two(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 420, PROSE)

    page, face = _page(tmp_path, draw)
    left, right = _drawn((80, 150, 220, 380)), _drawn((300, 160, 460, 390))
    below = _drawn((80, 450, 460, 650))
    found = regions(page, face, [left, right, below])
    assert [(r.kind, r.label) for r in found] == [('figure', 'Untitled figure 1'), ('figure', 'Untitled figure 2')]
    assert found[0].bbox.x0 < 80 and found[0].bbox.x1 > 460 and found[0].bbox.y1 < 420


def test_a_short_drawing_with_no_word_in_it_is_an_equation_without_a_number(tmp_path):
    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 160, 'he = h - d1 - d2', x=200)  # a formula, OCR'd, beside its fraction bars
        _text(page, 260, PROSE)

    page, face = _page(tmp_path, draw)
    found = regions(page, face, [_drawn((180, 145, 420, 175), stroke=60)])  # three lines high, under four
    assert [(r.kind, r.label) for r in found] == [('equation', 'Untitled equation 1')]


def _captioned_figure(page):
    _text(page, 100, PROSE)
    for y, size in ((180, 22), (230, 6), (260, 31)):
        _text(page, y, 'i)|\\', x=250, size=size)
    _text(page, 360, 'Figure 2 - Short-time duty', x=200)
    _text(page, 420, PROSE)


def test_what_a_hidden_layer_draws_is_no_region_unless_asked(tmp_path):
    page, face = _page(tmp_path, _captioned_figure)
    hidden = hidden_layer(page)
    assert regions(hidden, face) == []
    assert regions(hidden, face, include_hidden=True) == regions(page, face) != []


def test_text_filled_in_the_color_under_it_is_no_region_unless_asked(tmp_path):
    page, face = _page(tmp_path, _captioned_figure)
    unseen = in_background_color(page)
    assert regions(unseen, face) == []
    assert regions(unseen, face, include_in_background_color=True) == regions(page, face) != []


def test_a_region_the_page_edge_leaves_no_room_for_is_none(tmp_path):
    """A caption across the page's foot, prose right above it: clamped to the page, its figure's top falls below its
    bottom -- once a crop of negative height. A region is never a box without area."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=300)
    _text(page, 296, PROSE)
    _text(page, 307, 'Figure 3 - At the foot')
    doc.save(tmp_path / 'r.pdf')
    read = read_pdf(tmp_path / 'r.pdf')
    assert regions(read.pages[0], layout.body_face(read)) == []


def test_a_short_drawing_without_math_in_it_is_a_figure_not_an_equation(tmp_path):
    """A cover's logo: drawn, a few body sizes high, no text in it (or only its brand's): nothing says formula."""

    def draw(page):
        _text(page, 100, PROSE)
        _text(page, 300, PROSE)
        _text(page, 470, 'TDK', x=330, size=14)  # brand text set inside the second logo
        _text(page, 600, PROSE)

    page, face = _page(tmp_path, draw)
    logo, branded = _drawn((300, 150, 420, 176), stroke=30), _drawn((300, 450, 420, 480), stroke=30)
    found = regions(page, face, [logo, branded])
    assert [(r.kind, r.label) for r in found] == [('figure', 'Untitled figure 1'), ('figure', 'Untitled figure 2')]


def _data_sheet_drawings(page):
    """A data sheet's drawings page: each drawing under its own title, two of them in two parts."""
    _text(page, 60, 'Data and Signal Line Chokes', x=42, size=11)
    _text(page, 60, 'ACT1210', x=499, size=11)
    _text(page, 110, 'Dimensional drawing and pin configuration', x=37, size=11)
    _text(page, 110, 'Layout recommendation', x=313, size=11)
    _text(page, 278, 'Dimensions in mm', x=37, size=11)
    _text(page, 306, 'Circuit diagram', x=37, size=11)
    _text(page, 410, 'Taping and packing', x=37, size=11)
    _text(page, 436, 'Blister tape', x=37, size=11)
    _text(page, 436, 'Reel', x=356, size=11)
    _text(page, 624, 'Dimensions in mm', x=37, size=11)


def test_drawings_under_their_own_titles_and_apart_are_separate_figures(tmp_path):
    """ACT1210 page 3: a dimensional drawing (two views touching), a layout beside it under its own title, a circuit
    below, a blister tape (two parts close together) and a reel (two views) under theirs -- once one figure."""
    page, face = _page(tmp_path, _data_sheet_drawings)
    views = [_drawn((36.9, 119.0, 159.8, 263.4)), _drawn((139.6, 119.0, 229.8, 213.0))]
    layout_ = _drawn((313.3, 119.0, 419.9, 212.6))
    circuit = _drawn((42.7, 315.3, 122.4, 382.4), stroke=19.7)  # its longest line: a pin to its coil
    tape = [_drawn((124.3, 442.4, 293.7, 610.8)), _drawn((45.6, 490.4, 105.1, 571.5))]
    reel = [_drawn((434.3, 442.9, 514.9, 586.4)), _drawn((359.9, 447.2, 416.0, 573.4))]
    found = regions(page, face, [*views, layout_, circuit, *tape, *reel])
    assert [r.kind for r in found] == ['figure'] * 5

    def holding(*parts):
        return [
            r.label
            for r in found
            if all(r.bbox.x0 <= p.box.x0 and p.box.x1 <= r.bbox.x1 for p in parts)
            and all(r.bbox.y0 <= p.box.y0 and p.box.y1 <= r.bbox.y1 for p in parts)
        ]

    assert len({*holding(*views), *holding(layout_), *holding(circuit), *holding(*tape), *holding(*reel)}) == 5


def test_a_unit_row_of_lone_signs_the_document_sets_is_a_row_of_its_table(tmp_path):
    """ACT1210's ordering codes: a units row (uH, mA, an ohm sign alone) under the header; set, not OCR'd."""

    def draw(page):
        _text(page, 100, PROSE)
        _row(page, 150, ((50, 'L'), (150, 'I'), (250, 'R'), (350, 'Ordering code')))
        _row(page, 165, ((50, 'uH'), (150, 'mA'), (250, 'W')))
        for y, (value, code) in zip((185, 200, 215), (('11', 'B110'), ('22', 'B220'), ('51', 'B510')), strict=True):
            _row(page, y, ((50, value), (150, '300'), (250, '0.4'), (350, code)))
        _text(page, 300, PROSE)

    found = regions(*_page(tmp_path, draw))
    assert [(r.kind, r.label) for r in found] == [('table', 'Untitled table 1')]
    assert found[0].bbox.y0 < 140  # its header and units rows inside it
    assert found[0].table.rows == (
        ('L', 'I', 'R', 'Ordering code'),
        ('uH', 'mA', 'W', ''),
        ('11', '300', '0.4', 'B110'),
        ('22', '300', '0.4', 'B220'),
        ('51', '300', '0.4', 'B510'),
    )


def test_rows_ruled_apart_with_text_are_a_table_a_striped_icon_without_text_is_not(tmp_path):
    """ACT1210's names and values: long value lines cross the rules; a manual's striped fan icon has stripes alone."""

    def draw(page):
        _text(page, 100, PROSE)
        for y in (160, 180, 200):
            _text(page, y, 'Rated voltage', x=50)
            _text(page, y, 'Measured with a meter of the class stated at the rated frequency, 20 C.', x=160)
        _text(page, 400, PROSE)

    page, face = _page(tmp_path, draw)
    ruled = _drawn((48, 148, 470, 206), rules=(4, 1), loose=1)
    icon = _drawn((200, 260, 260, 320), rules=(5, 0), loose=8)
    found = regions(page, face, [ruled, icon])
    assert [(r.kind, r.label) for r in found] == [('table', 'Untitled table 1'), ('figure', 'Untitled figure 1')]


def test_a_caption_up_the_side_of_a_landscape_table_cuts_its_region_in_its_own_frame(tmp_path):
    """A landscape table printed sideways, its caption above it in its frame (left of it on the page, read bottom to
    top): the caption is matched in its frame and the table runs across the page from it to the note after it; its
    cells are read in that frame, each box turned back to where the page shows it."""
    read = read_pdf(landscape_pdf(tmp_path / 'landscape.pdf'))
    page = read.page(1)
    (region,) = regions(page, layout.body_face(read))
    assert (region.kind, region.label) == ('table', 'Table 10')
    cells = {cell for row in TABLE_ROWS for cell in row}

    def wrongly(box):
        """The spans ``box`` holds and should not, or should hold and does not: the caption and the cells it holds."""
        return [s.text for s in page.spans if box.contains(*s.center) != (s.text in cells | {LANDSCAPE_CAPTION[0]})]

    assert wrongly(region.bbox) == []
    caption = next(s for s in page.spans if s.text == LANDSCAPE_CAPTION[0]).provenance.bbox
    # The planted wrong truth: the caption read as the page stands, its table running down the page from it.
    assert wrongly(BBox(0, caption.y0, page.width, page.height))
    assert region.table.rows == TABLE_ROWS
    boxes = {s.text: tuple(s.provenance.bbox) for s in page.spans}
    seen = [v for r, row in enumerate(TABLE_ROWS) for c in range(len(row)) for v in region.table.cell_bboxes[r][c]]
    assert seen == pytest.approx([v for row in TABLE_ROWS for cell in row for v in boxes[cell]])
    # The planted wrong truth: a cell's box left in the table's frame stands nowhere near its text on the page.
    framed = layout.framed(region.table.cell_bboxes[0][0], (0.0, -1.0))
    assert not framed.contains(*next(s for s in page.spans if s.text == TABLE_ROWS[0][0]).center)
