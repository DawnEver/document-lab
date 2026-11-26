import hashlib

import pymupdf
import pytest

from document_lab.ir import BBox
from document_lab.pdf import read_pdf, read_raster
from tests.synthetic import RED, mixed_pdf, table_cell_rect


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
    assert doc.repairs == ()
    assert doc.page(1).warnings == ()


def test_a_span_carries_its_page_and_box(doc):
    (span,) = [s for s in doc.spans() if s.text == 'Hello document']
    assert span.provenance.page == 1
    assert span.provenance.sha256 == doc.sha256
    box = span.provenance.bbox
    assert box.x0 == pytest.approx(72, abs=0.5)
    assert box.y0 < 72 < box.y1  # the baseline sits inside the box
    assert span.size == pytest.approx(14, abs=1e-6)
    assert span.font == 'Helvetica'


def test_a_drawn_polyline_is_one_vector_path_with_its_points_and_color(doc):
    (path,) = [p for p in doc.page(2).paths if p.stroke == RED]
    assert [c for p in path.points for c in p] == pytest.approx([100, 100, 200, 150, 300, 120], abs=1e-3)
    assert path.width == pytest.approx(1.5, abs=1e-6)
    assert not path.closed
    assert path.fill is None
    assert path.provenance.page == 2


def test_a_drawn_rectangle_is_a_closed_path_of_its_four_corners(doc):
    first = table_cell_rect(0, 0)
    corners = {(first.x0, first.y0), (first.x1, first.y0), (first.x1, first.y1), (first.x0, first.y1)}
    rects = [p for p in doc.page(1).paths if p.closed and len(p.points) == 4]
    assert any({(round(x, 3), round(y, 3)) for x, y in p.points} == corners for p in rects)


def test_an_image_is_a_figure_with_its_box_and_native_size(doc):
    (figure,) = doc.page(1).figures
    assert tuple(figure.provenance.bbox) == pytest.approx((72, 600, 172, 700), abs=0.5)
    assert (figure.width, figure.height) == (8, 8)
    assert figure.image is not None


def test_a_filled_shape_says_its_fill_and_rule(tmp_path):
    doc = pymupdf.open()
    shape = doc.new_page().new_shape()
    shape.draw_rect(pymupdf.Rect(10, 10, 50, 30))
    shape.finish(color=None, fill=(0, 0, 1), even_odd=True)
    shape.commit()
    doc.save(tmp_path / 'f.pdf')
    (path,) = read_pdf(tmp_path / 'f.pdf').page(1).paths
    assert (path.stroke, path.fill, path.even_odd, path.closed) == (None, (0.0, 0.0, 1.0), True, True)


def test_a_curve_is_flattened_to_the_samples_asked(tmp_path):
    doc = pymupdf.open()
    doc.new_page().draw_bezier((10, 10), (20, 0), (30, 0), (40, 10), color=(0, 0, 0))
    doc.save(tmp_path / 'c.pdf')
    (coarse,) = read_pdf(tmp_path / 'c.pdf', curve_samples=2).page(1).paths
    (fine,) = read_pdf(tmp_path / 'c.pdf', curve_samples=16).page(1).paths
    assert (len(coarse.points), len(fine.points)) == (3, 17)
    assert fine.points[-1] == pytest.approx((40, 10), abs=1e-6)


def test_a_bold_span_says_so(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'Strong', fontname='hebo', fontsize=10)
    page.insert_text((72, 100), 'plain', fontname='helv', fontsize=10)
    doc.save(tmp_path / 'b.pdf')
    spans = {s.text: s.bold for s in read_pdf(tmp_path / 'b.pdf').spans()}
    assert spans == {'Strong': True, 'plain': False}


def test_an_encrypted_file_opens_with_its_password_and_refuses_without(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), 'secret', fontname='helv', fontsize=10)
    doc.save(tmp_path / 'e.pdf', encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw='u', owner_pw='o')
    opened = read_pdf(tmp_path / 'e.pdf', password='u')  # noqa: S106 -- the test file's own password
    assert [s.text for s in opened.spans()] == ['secret']
    with pytest.raises(ValueError, match='password'):
        read_pdf(tmp_path / 'e.pdf')


def test_a_raster_renders_at_the_native_resolution_of_the_image_under_it(doc, tmp_path):
    path = mixed_pdf(tmp_path / 'mixed.pdf')
    raster = read_raster(path, 1, BBox(72, 600, 172, 700))
    rows, columns = raster.pixels.shape[:2]  # 8 samples across 100 points: 5.76 dpi, a pixel per sample
    assert abs(rows - 8) <= 1 and abs(columns - 8) <= 1  # the renderer rounds the clip's edges outward


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / 'absent.pdf')
