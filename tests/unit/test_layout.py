"""Layout of vertical writing: a column is a line read top to bottom, columns side by side read right to left."""

from document_lab import layout
from document_lab.ir import BBox, Page, Provenance, TextSpan

SRC = Provenance('0' * 64, 'file:///synthetic.pdf', page=1)
SIZE = 10.0


def _vertical(text: str, x: float, top: float) -> TextSpan:
    """``text`` down a column centered on ``x`` from ``top``, an em a character."""
    box = BBox(x - SIZE / 2, top, x + SIZE / 2, top + SIZE * len(text))
    return TextSpan(text, SIZE, 'Song', SRC.at(bbox=box), vertical=True)


def _flat(text: str, x0: float, y0: float) -> TextSpan:
    return TextSpan(text, SIZE, 'Helvetica', SRC.at(bbox=BBox(x0, y0, x0 + 5 * len(text), y0 + SIZE)))


def _page(*spans: TextSpan) -> Page:
    return Page(number=1, width=600, height=800, spans=spans)


def test_spans_down_one_column_are_one_line_read_top_to_bottom():
    # Drawn bottom first: the line still reads top to bottom; a gap over WORD_GAP ems is a space.
    page = _page(_vertical('CD', 500, 120), _vertical('AB', 500, 100), _vertical('EF', 500, 147))
    (line,) = layout.lines(page, weighted=True)
    assert (line.vertical, line.text, line.column) == (True, 'ABCD EF', 'R')
    assert (line.x0, line.y0, line.x1, line.y1) == (495, 100, 505, 167)


def test_a_vertical_span_never_joins_a_flat_line():
    page = _page(_flat('side', 480, 100), _vertical('AB', 506, 100))
    assert sorted((ln.text, ln.vertical) for ln in layout.lines(page, weighted=False)) == [
        ('AB', True),
        ('side', False),
    ]


def test_columns_side_by_side_read_right_to_left_as_one_block():
    # Three columns 15 pt apart (half an em between them), drawn left to right; a fourth far off is its own block.
    page = _page(
        _vertical('GHI', 470, 100),
        _vertical('DEF', 485, 100),
        _vertical('ABC', 500, 100),
        _vertical('XY', 400, 100),
    )
    assert [ln.text for ln in layout.ordered(layout.lines(page, weighted=True))] == ['ABC', 'DEF', 'GHI', 'XY']
    assert [b.text for b in layout.blocks(page)] == ['ABC\nDEF\nGHI', 'XY']


def test_a_vertical_block_across_the_middle_is_full_width_and_reads_right_to_left():
    # A page set in vertical columns from x 320 to x 280: the block crosses the middle (300), so no column split.
    page = _page(*(_vertical(text, x, 100) for text, x in (('A', 320), ('B', 305), ('C', 290), ('D', 275))))
    found = layout.lines(page, weighted=True)
    assert {ln.column for ln in found} == {'F'}
    assert [ln.text for ln in layout.ordered(found)] == ['A', 'B', 'C', 'D']


def test_vertical_columns_stacked_read_top_band_first():
    page = _page(_vertical('LOW', 500, 300), _vertical('TOP2', 485, 100), _vertical('TOP1', 500, 100))
    assert [ln.text for ln in layout.ordered(layout.lines(page, weighted=True))] == ['TOP1', 'TOP2', 'LOW']
