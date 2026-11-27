"""Ruled tables found from a page's rules and text alone, on synthetic PDFs whose every cell is known."""

import dataclasses

import pymupdf
import pytest

from document_lab.extract import Field, Schema, from_columns, from_rows
from document_lab.ink import Rulings, rulings
from document_lab.ir import BBox, Page
from document_lab.pdf import read_pdf
from document_lab.tables import NotATable, grids, tables
from tests.synthetic import (
    GROUPED_HEADER,
    GROUPED_ROWS,
    LIST_ROWS,
    MERGED,
    MERGED_ROWS,
    RED,
    SCAN_PAGE,
    TABLE_ROWS,
    WRAPPED_ROWS,
    Axis,
    PlotSpec,
    broken_rules,
    double_border,
    draw_booktabs,
    draw_grouped_rows,
    draw_plot,
    draw_ruled_table,
    draw_sideways_booktabs,
    draw_sideways_table,
    draw_table,
    draw_wrapped_rows,
    mixed_pdf,
    scan_image,
    table_cell_rect,
    table_differences,
    turned,
)


def _page(tmp_path, *draws):
    """Page 1 of a PDF each of ``draws`` drew on, as read."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        for draw in draws:
            draw(page)
        doc.save(tmp_path / 'drawn.pdf')
    return read_pdf(tmp_path / 'drawn.pdf').page(1)


def _box(rect: pymupdf.Rect) -> tuple[float, ...]:
    return pytest.approx(tuple(rect), abs=0.5)


def test_a_table_of_cell_rectangles_reads_as_its_grid_with_each_cells_box(tmp_path):
    page = read_pdf(mixed_pdf(tmp_path / 'mixed.pdf')).page(1)
    (table,) = tables(page)
    assert table.rows == TABLE_ROWS
    for r, row in enumerate(TABLE_ROWS):
        for c in range(len(row)):
            assert tuple(table.cell_bboxes[r][c]) == _box(table_cell_rect(r, c))
    assert tuple(table.provenance.bbox) == _box(table_cell_rect(0, 0) | table_cell_rect(3, 2))
    assert (table.provenance.page, table.provenance.locator) == (1, 'table[0]')
    assert table.cell(2, 1).provenance.locator == 'table[0]/row[2]/col[1]'


@pytest.mark.parametrize('filled', [False, True], ids=['stroked lines', 'thin filled rectangles'])
def test_rules_read_the_same_drawn_as_lines_or_as_thin_filled_rectangles(tmp_path, filled):
    page = _page(tmp_path, lambda p: draw_ruled_table(p, filled=filled))
    assert all((path.fill is not None) == filled for path in page.paths)  # the rules are drawn as asked
    (table,) = tables(page)
    assert table.rows == TABLE_ROWS
    assert tuple(table.cell_bboxes[3][2]) == _box(table_cell_rect(3, 2))


def test_a_rule_painted_with_a_pattern_is_a_rule(tmp_path):
    """A pattern fill has no color components (``()``): its color is unknown, so it is ink, never taken for white."""
    page = _page(tmp_path, lambda p: draw_ruled_table(p, filled=True))
    patterned = dataclasses.replace(page, paths=tuple(dataclasses.replace(p, fill=()) for p in page.paths))
    (table,) = tables(patterned)
    assert table.rows == TABLE_ROWS


def test_a_dense_drawing_elsewhere_on_the_page_moves_no_line_of_a_table(tmp_path):
    """Upright strokes every 0.8 pt across the page's width, below the table, chain no two of its lines into one."""

    def hatch(page):
        for i in range(500):
            page.draw_line((60 + 0.8 * i, 700), (60 + 0.8 * i, 720), color=(0, 0, 0), width=0.2)

    (table,) = tables(_page(tmp_path, hatch, draw_table))
    assert table.rows == TABLE_ROWS
    assert tuple(table.cell_bboxes[1][1]) == _box(table_cell_rect(1, 1))


def test_a_double_rule_between_a_header_and_its_body_is_one_line(tmp_path):
    """The header's boxes end 2.5 pt above the body's: two rules, one line between the two parts of one table."""

    def split(page):
        for r, row in enumerate(TABLE_ROWS):
            for c, text in enumerate(row):
                rect = table_cell_rect(r, c)
                if r == 0:
                    rect.y1 -= 2.5
                page.draw_rect(rect, color=(0, 0, 0), width=0.5)
                page.insert_text((rect.x0 + 4, rect.y1 - 6), text, fontsize=9)

    (table,) = tables(_page(tmp_path, split))
    assert table.rows == TABLE_ROWS
    assert table.cell_bboxes[0][0].y1 == pytest.approx(table_cell_rect(0, 0).y1 - 1.25, abs=0.1)  # the mean line


def test_a_merged_cell_repeats_its_text_and_its_own_box_in_every_slot_it_covers(tmp_path):
    page = _page(tmp_path, lambda p: draw_ruled_table(p, MERGED_ROWS, merged=MERGED))
    (table,) = tables(page)
    assert table.rows == MERGED_ROWS
    parameter = table_cell_rect(0, 0) | table_cell_rect(1, 0)
    limits = table_cell_rect(0, 1) | table_cell_rect(0, 2)
    assert tuple(table.cell_bboxes[0][0]) == tuple(table.cell_bboxes[1][0]) == _box(parameter)
    assert tuple(table.cell_bboxes[0][1]) == tuple(table.cell_bboxes[0][2]) == _box(limits)
    assert tuple(table.cell_bboxes[1][1]) == _box(table_cell_rect(1, 1))  # a slot of its own keeps its own box


def test_a_cells_lines_read_top_to_bottom_and_words_apart(tmp_path):
    def draw(page):
        draw_table(page, [('Rated voltage', 'V'), ('x', 'y')])
        page.insert_text((table_cell_rect(0, 0).x0 + 4, table_cell_rect(0, 0).y0 + 8), 'Top', fontsize=6)

    (table,) = tables(_page(tmp_path, draw))
    assert table.rows == (('Top Rated voltage', 'V'), ('x', 'y'))


def test_two_tables_are_numbered_top_to_bottom(tmp_path):
    def lower(page):
        page.draw_rect(pymupdf.Rect(72, 600, 192, 620), color=(0, 0, 0), width=0.5)
        page.draw_rect(pymupdf.Rect(192, 600, 312, 620), color=(0, 0, 0), width=0.5)
        page.draw_rect(pymupdf.Rect(72, 620, 192, 640), color=(0, 0, 0), width=0.5)
        page.draw_rect(pymupdf.Rect(192, 620, 312, 640), color=(0, 0, 0), width=0.5)
        for (x, y), text in zip([(76, 614), (196, 614), (76, 634), (196, 634)], 'abcd', strict=True):
            page.insert_text((x, y), text, fontsize=9)

    found = tables(_page(tmp_path, lower, draw_table))
    assert [t.rows for t in found] == [TABLE_ROWS, (('a', 'b'), ('c', 'd'))]
    assert [t.provenance.locator for t in found] == ['table[0]', 'table[1]']


def _single_box(page):
    page.draw_rect(pymupdf.Rect(72, 100, 300, 140), color=(0, 0, 0), width=0.5)
    page.insert_text((80, 125), 'NOTE A framed remark.', fontsize=9)


def _one_column(page):
    draw_table(page, [('Notes',), ('one',), ('two',)])


def _plot_with_grid(page):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, [0.0, 50.0, 100.0], RED)], grid=True)
    draw_plot(page, spec)


def _plot_with_tick_marks(page):
    draw_plot(page, PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, [0.0, 50.0, 100.0], RED)]))


def _empty_grid(page):
    draw_ruled_table(page, [('', '', ''), ('', '', '')])


def _one_label(page):
    """A plot's grid with one label inside it: a single text in a grid relates nothing."""
    draw_ruled_table(page, [('', '', ''), ('', 'V_s', ''), ('', '', '')])


def _text_across_a_rule(page):
    draw_ruled_table(page)
    page.insert_text((table_cell_rect(1, 0).x1 - 20, table_cell_rect(1, 0).y0 + 10), 'overflowing', fontsize=9)


def _rule_stopping_inside_a_cell(page):
    draw_ruled_table(page)
    cell = table_cell_rect(1, 1)
    page.draw_line((cell.x0 + 40, cell.y0), (cell.x0 + 40, cell.y0 + 10), color=(0, 0, 0), width=0.5)


def _open_sides(page, *, ragged: bool = False):
    """Rules across, and between the columns, but none down either side; ``ragged``: one rule across stops short."""
    whole = table_cell_rect(0, 0) | table_cell_rect(3, 2)
    for r in range(len(TABLE_ROWS)):
        y = table_cell_rect(r, 0).y0
        start = whole.x0 + (10 if ragged and r == 2 else 0)
        page.draw_line((start, y), (whole.x1, y), color=(0, 0, 0), width=0.5)
    page.draw_line(whole.bl, whole.br, color=(0, 0, 0), width=0.5)
    for c in (1, 2):
        x = table_cell_rect(0, c).x0
        page.draw_line((x, whole.y0), (x, whole.y1), color=(0, 0, 0), width=0.5)
    for r, row in enumerate(TABLE_ROWS):
        for c, text in enumerate(row):
            page.insert_text((table_cell_rect(r, c).x0 + 4, table_cell_rect(r, c).y1 - 6), text, fontsize=9)


def test_a_side_no_rule_closes_is_closed_where_every_rule_across_it_ends(tmp_path):
    (table,) = tables(_page(tmp_path, _open_sides))
    assert table.rows == TABLE_ROWS
    assert tuple(table.cell_bboxes[2][0]) == _box(table_cell_rect(2, 0))
    assert tuple(table.cell_bboxes[3][2]) == _box(table_cell_rect(3, 2))
    assert tuple(table.provenance.bbox) == _box(table_cell_rect(0, 0) | table_cell_rect(3, 2))


def _l_shaped_cell(page):
    """A 2 x 2 grid whose inner rules are only an L round the bottom-right cell: the rest is one L-shaped cell."""
    a, d = table_cell_rect(0, 0), table_cell_rect(1, 1)
    page.draw_rect(a | d, color=(0, 0, 0), width=0.5)
    page.draw_line((d.x0, d.y0), (d.x1, d.y0), color=(0, 0, 0), width=0.5)
    page.draw_line((d.x0, d.y0), (d.x0, d.y1), color=(0, 0, 0), width=0.5)
    page.insert_text((a.x0 + 4, a.y1 - 6), 'L', fontsize=9)
    page.insert_text((d.x0 + 4, d.y1 - 6), 'square', fontsize=9)


@pytest.mark.parametrize(
    ('draw', 'reason'),
    [
        (_single_box, 'a single box'),
        (_one_column, 'one column'),
        (_plot_with_grid, 'a drawing crosses its rules'),
        (_plot_with_tick_marks, 'a rule stops inside a cell'),
        (_empty_grid, 'fewer than two cells hold text'),
        (_one_label, 'fewer than two cells hold text'),
        (_text_across_a_rule, 'text crosses a rule'),
        (_rule_stopping_inside_a_cell, 'a rule stops inside a cell'),
        (lambda page: _open_sides(page, ragged=True), 'a rule runs past'),
        (_l_shaped_cell, 'not a rectangle'),
    ],
)
def test_a_grid_that_is_no_table_is_refused_by_name(tmp_path, draw, reason):
    page = _page(tmp_path, draw)
    assert tables(page) == ()
    (grid,) = grids(page)
    assert isinstance(grid, NotATable)
    assert reason in grid.reason


def test_a_refused_grid_leaves_the_tables_beside_it_found(tmp_path):
    page = _page(tmp_path, _plot_with_grid, draw_table)
    found = grids(page)
    assert [type(g).__name__ for g in found] == ['NotATable', 'Table']  # top to bottom: the plot, then the table
    assert tables(page)[0].rows == TABLE_ROWS


def test_a_shaded_background_is_no_rule_and_breaks_no_table(tmp_path):
    def shade(page):
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1))  # a page-sized white ground
        page.draw_rect(table_cell_rect(0, 0) | table_cell_rect(0, 2), color=None, fill=(0.8, 0.8, 0.8))  # header row

    (table,) = tables(_page(tmp_path, shade, draw_table))
    assert table.rows == TABLE_ROWS


def test_rules_drawn_in_white_are_no_rules(tmp_path):
    def white(page):
        for r in range(len(TABLE_ROWS)):
            for c in range(3):
                page.draw_rect(table_cell_rect(r, c), color=(1, 1, 1), width=0.5)
                page.insert_text(table_cell_rect(r, c).bl + pymupdf.Point(4, -6), TABLE_ROWS[r][c], fontsize=9)

    assert grids(_page(tmp_path, white)) == ()


def test_a_detected_table_feeds_extract_with_each_values_cell(tmp_path):
    (table,) = tables(read_pdf(mixed_pdf(tmp_path / 'mixed.pdf')).page(1))
    keyed = from_rows(table, Schema((Field('alpha', 'number'), Field('beta', 'number'), Field('gamma', 'number'))))
    assert {k: v.value for k, v in keyed.items()} == {'alpha': 1.25, 'beta': -3.0, 'gamma': 450.0}
    assert tuple(keyed['beta'].provenance.bbox) == _box(table_cell_rect(2, 1))
    assert keyed['beta'].provenance.locator == 'table[0]/row[2]/col[1]'
    schema = Schema((Field('name', 'text', labels=('Parameter',)), Field('value', 'number'), Field('unit', 'text')))
    records = from_columns(table, schema)
    assert [(r['name'].value, r['value'].value, r['unit'].value) for r in records] == [
        ('alpha', 1.25, 'mm'),
        ('beta', -3.0, 'kg'),
        ('gamma', 450.0, 's'),
    ]
    assert tuple(records[2]['unit'].provenance.bbox) == _box(table_cell_rect(3, 2))


def test_a_page_without_rules_has_no_grid(tmp_path):
    page = _page(tmp_path, lambda p: p.insert_text((72, 72), 'Only prose.', fontsize=9))
    assert grids(page) == ()


def test_a_cell_reads_a_stated_space_as_a_space_and_a_zero_width_space_over_a_rule_breaks_no_grid(tmp_path):
    page = _page(tmp_path, draw_table)
    stated = tuple(dataclasses.replace(s, text='al\u00a0\u200bpha') if s.text == 'alpha' else s for s in page.spans)
    rule = table_cell_rect(1, 0).x1
    across = stated[0].provenance.at(bbox=BBox(rule - 3, 425, rule + 3, 434))
    gap = dataclasses.replace(stated[0], text='\u200b', provenance=across)
    (table,) = tables(dataclasses.replace(page, spans=(*stated, gap)))
    assert table.rows[1] == ('al pha', '1.25', 'mm')


# A scanned page: its rules come from its picture (`ink.rulings`), its words from OCR. Here the words are the drawn
# text's own, placed where the scan turns them (each box's center turned, its size kept), so every expected value is
# known and Tesseract is not needed; the ``ocr`` part runs the same tables through it.


def _scanned(tmp_path, *draws, angle=0.0, noise=0.0, dpi=300):
    """The page of ``draws`` as scanned: its words where the scan shows them, and its picture's rulings."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=SCAN_PAGE.width, height=SCAN_PAGE.height)
        for draw in draws:
            draw(page)
        doc.save(tmp_path / 'drawn.pdf')
    drawn = read_pdf(tmp_path / 'drawn.pdf').page(1)

    def seen(span):
        box = span.provenance.bbox
        x, y = turned((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, angle)
        moved = BBox(x - box.width / 2, y - box.height / 2, x + box.width / 2, y + box.height / 2)
        return dataclasses.replace(span, provenance=span.provenance.at(bbox=moved))

    words = Page(1, drawn.width, drawn.height, spans=tuple(seen(s) for s in drawn.spans))
    pixels = scan_image(*draws, dpi=dpi, angle=angle, noise=noise)
    return words, rulings(pixels, pixels.shape[1] / SCAN_PAGE.width)


@pytest.mark.parametrize(('angle', 'noise'), [(0.0, 0.0), (1.0, 0.0), (-0.7, 1.0), (1.5, 1.0)])
def test_a_scanned_ruled_table_reads_from_its_pictures_rules_and_its_words(tmp_path, angle, noise):
    page, found = _scanned(tmp_path, draw_ruled_table, angle=angle, noise=noise)
    assert found.skew == pytest.approx(angle, abs=0.05)
    (table,) = tables(page, rulings=found)
    assert table_differences(table, TABLE_ROWS, angle=angle) == ([], [])
    assert (table.provenance.page, table.provenance.locator) == (1, 'table[0]')


def test_a_scanned_merged_cell_repeats_its_text_and_box_where_the_scan_turns_it(tmp_path):
    page, found = _scanned(tmp_path, lambda p: draw_ruled_table(p, MERGED_ROWS, merged=MERGED), angle=-1.3, noise=1.0)
    (table,) = tables(page, rulings=found)
    assert table_differences(table, MERGED_ROWS, merged=MERGED, angle=-1.3) == ([], [])


@pytest.mark.parametrize(
    ('draws', 'within'),
    [
        ((draw_ruled_table, broken_rules), 0.5),
        ((lambda p: draw_ruled_table(p, width=2.0),), 0.5),
        ((draw_ruled_table, double_border), 1.3),  # a double rule is one line, at its rules' mean
    ],
    ids=['broken rules', 'thick rules', 'a double outline'],
)
def test_a_scans_broken_thick_or_double_rules_still_rule_its_table(tmp_path, draws, within):
    page, found = _scanned(tmp_path, *draws, angle=0.6, noise=0.5, dpi=200)
    (table,) = tables(page, rulings=found)
    assert table_differences(table, TABLE_ROWS, angle=0.6, within=within) == ([], [])


@pytest.mark.parametrize(
    ('wrong', 'kind'),
    [
        ({'rows': MERGED_ROWS}, 'text'),
        ({'rows': TABLE_ROWS[:3]}, 'structure'),
        ({'rows': TABLE_ROWS, 'merged': ((0, 1, 0, 2),)}, 'structure'),
        ({'rows': TABLE_ROWS, 'angle': -1.0}, 'structure'),
    ],
    ids=['other words', 'a row fewer', 'a merged cell that is not there', 'the skew the other way'],
)
def test_the_comparison_reports_every_planted_wrong_truth(tmp_path, wrong, kind):
    page, found = _scanned(tmp_path, draw_ruled_table, angle=1.0)
    (table,) = tables(page, rulings=found)
    structure, text = table_differences(table, **{'angle': 1.0, **wrong})
    assert structure if kind == 'structure' else text


def test_a_scan_without_rules_has_no_skew_and_no_table(tmp_path):
    page, found = _scanned(tmp_path, lambda p: p.insert_text((72, 100), 'Only prose, no rule.', fontsize=11))
    assert found == Rulings(None, ())
    assert tables(page, rulings=found) == ()


def _text_only(tmp_path, rows=TABLE_ROWS):
    """Page 1 of a PDF of :func:`draw_ruled_table`'s text without its rules, as read."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                rect = table_cell_rect(r, c)
                page.insert_text((rect.x0 + 4, rect.y1 - 6), text, fontname='helv', fontsize=9)
        doc.save(tmp_path / 'text.pdf')
    return read_pdf(tmp_path / 'text.pdf').page(1)


def _picture_grid(rows: int, extra: tuple[BBox, ...] = ()) -> Rulings:
    """:func:`draw_ruled_table`'s grid of ``rows`` rows by three as picture rules (thin boxes), and ``extra``."""
    whole = table_cell_rect(0, 0) | table_cell_rect(rows - 1, 2)
    level = tuple(BBox(whole.x0, y - 0.25, whole.x1, y + 0.25) for y in range(400, 401 + 20 * rows, 20))
    upright = tuple(BBox(x - 0.25, whole.y0, x + 0.25, whole.y1) for x in (72, 192, 312, 432))
    return Rulings(0.0, level + upright + extra)


def test_a_picture_rule_inside_one_words_box_is_that_words_ink(tmp_path):
    """The tops of ``mm`` side by side read as a short level rule: within the word's box, it parts no cell."""
    page = _text_only(tmp_path)
    word = next(s for s in page.spans if s.text == 'mm').provenance.bbox
    tops = BBox(word.x0 + 0.5, word.y0 + 1, word.x1 - 0.5, word.y0 + 1.3)
    (table,) = tables(page, rulings=_picture_grid(4, (tops,)))
    assert table.rows == TABLE_ROWS
    (broken,) = grids(page, rulings=Rulings(0.0, (*_picture_grid(4).rules, tops._replace(x0=word.x0 - 5))))
    assert isinstance(broken, NotATable)  # reaching out of the word, it is a rule: a planted control


def test_a_stroke_a_picture_rule_runs_through_is_that_rule_read_as_text_but_a_dash_in_a_cell_is_kept(tmp_path):
    page = _text_only(tmp_path, [('a', '-', 'b'), ('c', 'd', 'e')])
    on_rule = page.spans[0].provenance.at(bbox=BBox(191.0, 402.0, 193.0, 412.0))  # across the rule at x 192
    read_rule = dataclasses.replace(page.spans[0], text='|', provenance=on_rule)
    (table,) = tables(dataclasses.replace(page, spans=(*page.spans, read_rule)), rulings=_picture_grid(2))
    assert table.rows == (('a', '-', 'b'), ('c', 'd', 'e'))
    (kept,) = tables(
        dataclasses.replace(page, spans=(*page.spans, dataclasses.replace(read_rule, text='x'))),
        rulings=_picture_grid(2),
    )
    assert kept.rows[0][:2] != ('a', '-')  # a letter on a rule is no rule read as text: a planted control


def _sideways_differences(table, turn):
    """How ``table`` differs from :func:`draw_sideways_table`'s drawn ``turn`` degrees: rows, then each cell's box."""
    out = [] if table.rows == TABLE_ROWS else [f'rows {table.rows}']
    for r, row in enumerate(TABLE_ROWS):
        for c in range(len(row)):
            drawn, box = table_cell_rect(r, c), table.cell_bboxes[r][c]
            x, y = turned((drawn.x0 + drawn.x1) / 2, (drawn.y0 + drawn.y1) / 2, turn)
            seen = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, box.width, box.height)
            if seen != pytest.approx((x, y, drawn.height, drawn.width), abs=0.5):
                out.append(f'cell ({r}, {c}) at {seen}')
    return out


@pytest.mark.parametrize('turn', [-90, 90], ids=['read bottom to top', 'read top to bottom'])
def test_a_table_printed_sideways_reads_in_the_frame_its_text_stands_upright_in(tmp_path, turn):
    def head(page):
        page.insert_text((72, 60), 'An upright running head', fontsize=10)

    (table,) = tables(_page(tmp_path, head, lambda page: draw_sideways_table(page, turn=turn)))
    assert _sideways_differences(table, turn) == []
    assert _sideways_differences(table, -turn)  # a planted wrong truth: turned the other way
    mirrored = dataclasses.replace(table, rows=tuple(row[::-1] for row in table.rows))
    assert _sideways_differences(mirrored, turn)  # ... and its cells read in another order


def _booktabs(page, rows=TABLE_ROWS, **options):
    return draw_booktabs(page, rows, **options)


@pytest.mark.parametrize('filled', [True, False], ids=['thin filled bars', 'stroked lines'])
def test_a_table_ruled_by_its_rows_alone_reads_its_columns_from_its_text(tmp_path, filled):
    """Booktabs: a top rule, one under the header, a bottom rule, nothing down -- white space parts its columns."""
    page = _page(tmp_path, lambda p: _booktabs(p, filled=filled))
    (table,) = tables(page)
    assert table.rows == TABLE_ROWS
    assert tuple(table.provenance.bbox) == _box(table_cell_rect(0, 0) | table_cell_rect(3, 2))
    assert table.provenance.locator == 'table[0]'
    value = next(s for s in page.spans if s.text == '1.25').provenance.bbox
    assert table.cell_bboxes[1][1] == value  # a cell's box is its text's: no rule bounds it
    assert table_differences(table, TABLE_ROWS)[1] == []


def test_a_scan_s_table_ruled_by_its_rows_alone_reads_from_its_picture_rules(tmp_path):
    whole = table_cell_rect(0, 0) | table_cell_rect(3, 2)
    level = tuple(BBox(whole.x0, y - 0.4, whole.x1, y + 0.4) for y in (whole.y0, whole.y0 + 20, whole.y1))
    (table,) = tables(_text_only(tmp_path), rulings=Rulings(0.0, level))
    assert table.rows == TABLE_ROWS
    assert tables(_text_only(tmp_path), rulings=Rulings(0.0, level[::2])) == ()  # planted: two rules make no stack


def test_a_partial_rule_under_a_header_s_columns_breaks_no_table(tmp_path):
    """A ``cmidrule`` under two of the header's columns: a rule of its own extent, no row of the stack."""

    def cmidrule(page):
        page.draw_rect(pymupdf.Rect(196, 416, 428, 416.4), color=None, fill=(0, 0, 0))

    (table,) = tables(_page(tmp_path, _booktabs, cmidrule))
    assert table.rows == TABLE_ROWS


def test_a_ground_under_a_table_ruled_by_its_rows_is_not_drawn_in_it(tmp_path):
    """A page's tint, or a panel behind the table: painted under the whole stack, no figure's ink."""

    def ground(page):
        page.draw_rect(page.rect, color=None, fill=(0.9, 0.9, 0.85))

    (table,) = tables(_page(tmp_path, ground, _booktabs))
    assert table.rows == TABLE_ROWS
    shaded_row = pymupdf.Rect(72, 420, 432, 440)  # planted: shading one row is drawn in it

    (refused,) = grids(_page(tmp_path, ground, _booktabs, lambda p: p.draw_rect(shaded_row, color=None, fill=RED)))
    assert isinstance(refused, NotATable)


def test_a_header_over_two_columns_parts_no_columns_of_the_body(tmp_path):
    """A header set across the columns below it: the body's white space parts the columns, the header's cell in the
    column it starts in. Columns read from all the text, it once ran the two columns of values together."""
    rows = (('Machine', 'Rated speed at base and top', ''), ('A', '2100', '10000'), ('B', '1500', '8000'))
    (table,) = tables(_page(tmp_path, lambda p: _booktabs(p, rows)))
    assert table.rows == rows
    assert table.rows != (('Machine', 'Rated speed at base and top'), ('A', '2100 10000'), ('B', '1500 8000'))


def _fraction_bars(width):
    """Three fractions stacked, their bars ``width`` points long and one under the other, a sign beside each."""

    def draw(page):
        for y, (top, bottom) in zip((420, 444, 468), (('U', '2'), ('V', '3'), ('W', '4')), strict=True):
            page.draw_rect(pymupdf.Rect(300, y - 0.2, 300 + width, y + 0.2), color=None, fill=(0, 0, 0))
            page.insert_text((302, y - 3), top, fontname='helv', fontsize=9)
            page.insert_text((302 + width - 8, y + 10), bottom, fontname='helv', fontsize=9)

    return draw


def test_fraction_bars_stacked_alike_are_no_table(tmp_path):
    """A formula's fractions, one under the other, their bars one length: a stack of rules round short cells."""
    page = _page(tmp_path, _fraction_bars(30))
    assert tables(page) == ()
    (grid,) = grids(page)
    assert 'short' in grid.reason
    assert tables(_page(tmp_path, _fraction_bars(90)))  # planted: rules a table's length make one


def test_two_tables_ruled_by_their_rows_one_under_the_other_are_two(tmp_path):
    def two(page):
        _booktabs(page)
        page.insert_text((72, 506), 'Table 2. Second set of values', fontname='helv', fontsize=9)
        _booktabs(page, at=(72.0, 520.0))

    found = tables(_page(tmp_path, two))
    assert [t.rows for t in found] == [TABLE_ROWS, TABLE_ROWS]
    assert [t.provenance.locator for t in found] == ['table[0]', 'table[1]']


def _figure_between_rules(page):
    """A plot set between rules, as some journals frame a figure: a legend's two labels, tick values, a curve."""
    _booktabs(page, (('Model A', 'Model B', ''), ('0.5', '', ''), ('1.0', '', ''), ('1.5', '', '')))
    page.draw_polyline([(110, 470), (200, 440), (300, 455), (420, 425)], color=RED, width=1)


def _prose_between_rules(page):
    """Two columns of prose under a running head's rule, a section's rule and the foot's: no table."""
    words = ('the', 'rated', 'output', 'is', 'the', 'mechanical', 'power', 'at', 'the', 'shaft', 'expressed', 'in')
    for i, y in enumerate((415, 426, 437, 456, 467, 478)):
        for x in (72, 260):
            line = ' '.join(words[(i + x) % 5 :][:5])
            page.insert_text((x, y), line, fontname='helv', fontsize=9)
    for y in (400, 446, 485):
        page.draw_rect(pymupdf.Rect(72, y - 0.4, 432, y + 0.4), color=None, fill=(0, 0, 0))


@pytest.mark.parametrize(
    ('draw', 'reason'),
    [
        (lambda p: _booktabs(p, LIST_ROWS), 'a list'),
        (_figure_between_rules, 'drawn in it'),
        (_prose_between_rules, 'prose'),
    ],
    ids=['a list', 'a figure', 'prose'],
)
def test_rules_round_a_list_a_figure_or_prose_are_no_table(tmp_path, draw, reason):
    page = _page(tmp_path, draw)
    assert tables(page) == ()
    (grid,) = grids(page)
    assert isinstance(grid, NotATable)
    assert reason in grid.reason


def _cell_differences(table, rows):
    """How ``table``'s cells differ from ``rows``: its shape, else each cell whose text is not the one set."""
    if [len(row) for row in table.rows] != [len(row) for row in rows]:
        return [f'shape {[len(row) for row in table.rows]}, set {[len(row) for row in rows]}']
    return [
        f'cell ({r}, {c}) reads {got!r}, set {want!r}'
        for r, (read, row) in enumerate(zip(table.rows, rows, strict=True))
        for c, (got, want) in enumerate(zip(read, row, strict=True))
        if got != want
    ]


def _joined(rows):
    """Each cell of ``rows`` (its lines) as one text."""
    return tuple(tuple(' '.join(cell) for cell in row) for row in rows)


@pytest.mark.parametrize('first_ruled', [True, False], ids=['ruled across', 'its first column ruled apart'])
def test_a_table_ruled_under_every_row_reads_each_row_whole_its_wrapped_cells_centered(tmp_path, first_ruled):
    """IEEE sets a rule under every row and centers each cell down its row: a short cell stands between the lines of
    a wrapped one. Row 2 alone -- two cells of two lines beside short ones -- looks like two rows with labels; row 1,
    whose cells of one, two and three lines stand on no common lines, proves each row ruled. Ruled apart, the rules
    under the rows start at the second column, and the first column's own short rules make no table."""
    (table,) = tables(_page(tmp_path, lambda p: draw_wrapped_rows(p, first_ruled=first_ruled)))
    assert _cell_differences(table, _joined(WRAPPED_ROWS)) == []
    as_rows = (*_joined(WRAPPED_ROWS[:2]), ('[2]', 'Rotor', 'Minimize torque', 'Only half a pole'))
    as_rows += (('[2]', 'Rotor', 'ripple', 'was modeled'), *_joined(WRAPPED_ROWS[3:]))
    assert _cell_differences(table, as_rows)  # a planted wrong truth: row 2 read as two rows, its labels in each
    one_a_line = (*_joined(WRAPPED_ROWS[:1]), ('', '', 'Minimize the noise', 'Mesh dependence'))
    assert _cell_differences(table, one_a_line)  # ... and the table read one line a row


def test_a_table_ruled_between_groups_reads_the_rows_of_each_group_its_label_in_each(tmp_path):
    """Rules between groups of rows, none between rows; a group's label set beside its rows, centered down them, as a
    merged cell is (repeated in each row). The first row of a group holds no first cell: it carries on no row across
    the rule above it."""
    (table,) = tables(_page(tmp_path, draw_grouped_rows))
    groups = tuple((label, *row) for label, rows in GROUPED_ROWS for row in rows)
    assert _cell_differences(table, (GROUPED_HEADER, *groups)) == []
    once = tuple(
        (label if r == len(rows) // 2 else '', *row) for label, rows in GROUPED_ROWS for r, row in enumerate(rows)
    )
    assert _cell_differences(table, (GROUPED_HEADER, *once))  # a planted wrong truth: the label in one row
    whole = tuple((label, *(' '.join(column) for column in zip(*rows, strict=True))) for label, rows in GROUPED_ROWS)
    assert _cell_differences(table, (GROUPED_HEADER, *whole))  # ... and each group read as one row


def test_a_header_cell_wrapped_beside_centered_ones_is_one_cell(tmp_path):
    """A header cell set on two lines, its neighbors centered beside them: its lines stand on no common levels with
    theirs, so the header is one row -- and nothing of the body carries into it across the rule under it."""

    def draw(page):
        for y in (400, 424, 484):
            page.draw_rect(pymupdf.Rect(72, y - 0.4, 432, y + 0.4), color=None, fill=(0, 0, 0))
        for x, y, text in ((76, 415, 'Parameter'), (196, 410, 'Value'), (196, 420, '(SI)'), (316, 415, 'Unit')):
            page.insert_text((x, y), text, fontname='helv', fontsize=9)
        for r, row in enumerate(TABLE_ROWS[1:]):
            for c, text in enumerate(row):
                page.insert_text((76 + 120 * c, 438 + 20 * r), text, fontname='helv', fontsize=9)

    (table,) = tables(_page(tmp_path, draw))
    header = ('Parameter', 'Value (SI)', 'Unit')
    assert _cell_differences(table, (header, *TABLE_ROWS[1:])) == []
    two = (('', 'Value', ''), ('Parameter', '(SI)', 'Unit'), *TABLE_ROWS[1:])
    assert _cell_differences(table, two)  # a planted wrong truth: the header read as two rows


def _sideways_stack_differences(table, page, turn):
    """How ``table`` differs from :func:`draw_sideways_booktabs`' drawn ``turn`` degrees: its rows, its box (the
    rules' outline turned), and each cell's box (its text's, as the page shows it)."""
    out = [] if table.rows == TABLE_ROWS else [f'rows {table.rows}']
    drawn = table_cell_rect(0, 0) | table_cell_rect(3, 2)
    corners = [turned(x, y, turn) for x in (drawn.x0, drawn.x1) for y in (drawn.y0, drawn.y1)]
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    if tuple(table.provenance.bbox) != pytest.approx((min(xs), min(ys), max(xs), max(ys)), abs=0.5):
        out.append(f'box {table.provenance.bbox}')
    spans = {s.text: s.provenance.bbox for s in page.spans}
    for r, row in enumerate(TABLE_ROWS):
        for c, text in enumerate(row):
            box = table.cell_bboxes[r][c]
            if tuple(box) != pytest.approx(tuple(spans[text]), abs=0.01):
                out.append(f'cell ({r}, {c}) at {box}')
    return out


@pytest.mark.parametrize('turn', [-90, 90], ids=['read bottom to top', 'read top to bottom'])
def test_a_table_ruled_by_its_rows_printed_sideways_reads_in_the_frame_its_text_stands_upright_in(tmp_path, turn):
    """Its rules stand upright on the page: read a quarter turn round, as a grid printed sideways is."""

    def head(page):
        page.insert_text((72, 60), 'An upright running head', fontsize=10)

    page = _page(tmp_path, head, lambda p: draw_sideways_booktabs(p, turn=turn))
    (table,) = tables(page)
    assert _sideways_stack_differences(table, page, turn) == []
    assert table.provenance.locator == 'table[0]'
    assert _sideways_stack_differences(table, page, -turn)  # a planted wrong truth: turned the other way
    mirrored = dataclasses.replace(table, rows=tuple(row[::-1] for row in table.rows))
    assert _sideways_stack_differences(mirrored, page, turn)  # ... and its cells read in another order


#: Columns set close: each value of the second column flush right, the value of the third a set gap after it.
CLOSE_ROWS = (
    ('Parameter', 'Min', 'Max'),
    ('Rated speed', '1.25', '2.50'),
    ('beta', '-3', '4'),
    ('gamma', '450', '600'),
)


def _close_columns(rows=CLOSE_ROWS, *, apart, font='helv'):
    """``rows`` between booktabs' rules, the third column ``apart`` text sizes after the second, row by row: closer
    than the reader's word break (a text size), each row's two values are one span, its words apart."""

    def draw(page):
        for y in (400, 420, 500):
            page.draw_rect(pymupdf.Rect(72, y - 0.4, 432, y + 0.4), color=None, fill=(0, 0, 0))
        for r, (name, low, high) in enumerate(rows):
            y = 414 + (r and 20 * r + 6)
            page.insert_text((76, y), name, fontname=font, fontsize=9)
            page.insert_text((240 - pymupdf.get_text_length(low, font, 9), y), low, fontname=font, fontsize=9)
            page.insert_text((240 + apart * 9, y), high, fontname=font, fontsize=9)

    return draw


@pytest.mark.parametrize(
    ('apart', 'font'),
    [(0.7, 'helv'), (0.42, 'tiro')],
    ids=['closer than the reader s word break', 'closer than COLUMN_GAP, wider than the table s word spaces'],
)
def test_columns_set_close_part_where_the_words_of_a_span_stand_apart(tmp_path, apart, font):
    """Two values under a text size apart, drawn one after the other, are one span (a gap under a text size reads as a
    space): its words, as the reader places them, stand in two columns, so the span parts between them. Two columns
    0.42 text sizes apart stand closer than :data:`COLUMN_GAP`, but wider than one and a half of the table's own word
    space (Times' 0.25 em, between ``Rated`` and ``speed``, ``Parameter`` across it): its columns all the same."""
    page = _page(tmp_path, _close_columns(apart=apart, font=font))
    assert '1.25 2.50' in {s.text for s in page.spans}  # the reader joined them: one span, two words
    (table,) = tables(page)
    assert _cell_differences(table, CLOSE_ROWS) == []
    joined = tuple((row[0], f'{row[1]} {row[2]}') for row in CLOSE_ROWS)
    assert _cell_differences(table, joined)  # a planted wrong truth: each pair read as one cell
    words = next(s for s in page.spans if s.text == '1.25 2.50').words
    assert (table.cell_bboxes[1][1], table.cell_bboxes[1][2]) == words  # each cell's box its word's, as placed


def test_a_table_stating_no_word_space_parts_its_columns_at_column_gap(tmp_path):
    """No cell of two words, no word space of its own: columns 0.42 text sizes apart stand closer than
    :data:`COLUMN_GAP`, and nothing the table states says they are columns (a value it does not state is no guess)."""
    rows = tuple((('alpha' if row[0] == 'Rated speed' else row[0]), *row[1:]) for row in CLOSE_ROWS)
    (table,) = tables(_page(tmp_path, _close_columns(rows, apart=0.42, font='tiro')))
    assert _cell_differences(table, tuple((row[0], f'{row[1]} {row[2]}') for row in rows)) == []
    assert _cell_differences(table, rows)  # a planted wrong truth: the columns parted without a measure
