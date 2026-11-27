"""Ruled tables found from a page's rules and text alone, on synthetic PDFs whose every cell is known."""

import dataclasses

import pymupdf
import pytest

from document_lab.extract import Field, Schema, from_columns, from_rows
from document_lab.pdf import read_pdf
from document_lab.tables import NotATable, grids, tables
from tests.synthetic import (
    MERGED,
    MERGED_ROWS,
    RED,
    TABLE_ROWS,
    Axis,
    PlotSpec,
    draw_plot,
    draw_ruled_table,
    draw_table,
    mixed_pdf,
    table_cell_rect,
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
