import hashlib

import pymupdf
import pytest

from document_lab.pdf import read_pdf
from tests.synthetic import RED, TABLE_ROWS, mixed_pdf, table_cell_rect


@pytest.fixture
def doc(tmp_path):
    return read_pdf(mixed_pdf(tmp_path / 'mixed.pdf'))


def test_the_document_is_named_by_its_content_hash(tmp_path):
    path = mixed_pdf(tmp_path / 'mixed.pdf')
    doc = read_pdf(path)
    assert doc.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.uri == path.resolve().as_uri()
    assert [p.number for p in doc.pages] == [1, 2]
    assert (doc.page(1).width, doc.page(1).height) == (612, 792)


def test_a_span_carries_its_page_and_box(doc):
    (span,) = [s for s in doc.spans() if s.text == 'Hello document']
    assert span.provenance.page == 1
    assert span.provenance.sha256 == doc.sha256
    box = span.provenance.bbox
    assert box.x0 == pytest.approx(72, abs=0.5)
    assert box.y0 < 72 < box.y1  # the baseline sits inside the box
    assert span.size == pytest.approx(14, abs=1e-6)


def test_a_drawn_polyline_is_one_vector_path_with_its_points_and_colour(doc):
    (path,) = [p for p in doc.page(2).paths if p.stroke == RED]
    assert [c for p in path.points for c in p] == pytest.approx([100, 100, 200, 150, 300, 120], abs=1e-3)
    assert path.width == pytest.approx(1.5, abs=1e-6)
    assert not path.closed
    assert path.provenance.page == 2


def test_a_drawn_rectangle_is_a_closed_path(doc):
    rects = [p for p in doc.page(1).paths if p.closed]
    first = table_cell_rect(0, 0)
    assert any(p.points[0] == pytest.approx((first.x0, first.y0), abs=1e-3) and len(p.points) == 4 for p in rects)


def test_a_ruled_table_reads_as_its_grid_with_per_cell_boxes(doc):
    (table,) = doc.page(1).tables
    assert table.rows == TABLE_ROWS
    cell = table.cell(2, 1)
    assert cell.text == '-3'
    expected = table_cell_rect(2, 1)
    assert tuple(cell.provenance.bbox) == pytest.approx(tuple(expected), abs=0.5)


def test_an_image_is_a_figure_with_its_box_and_caption(doc):
    (figure,) = doc.page(1).figures
    assert tuple(figure.provenance.bbox) == pytest.approx((72, 600, 172, 700), abs=0.5)
    assert figure.caption == 'Figure 1. A blue square.'


def test_tables_can_be_skipped_for_speed(tmp_path):
    doc = read_pdf(mixed_pdf(tmp_path / 'm.pdf'), tables=False)
    assert doc.page(1).tables == ()


def test_a_bold_span_says_so(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'Strong', fontname='hebo', fontsize=10)
    page.insert_text((72, 100), 'plain', fontname='helv', fontsize=10)
    doc.save(tmp_path / 'b.pdf')
    spans = {s.text: s.bold for s in read_pdf(tmp_path / 'b.pdf').spans()}
    assert spans == {'Strong': True, 'plain': False}
