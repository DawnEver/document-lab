import hashlib

import numpy as np
import pymupdf
import pytest

from document_lab.ir import BBox
from document_lab.pdf import CURVE_SAMPLES, Locked, Malformed, NotPdf, Refused, Unsupported, read_pdf, read_raster
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
    assert list(doc.repairs) == []
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
    np.testing.assert_allclose(path.points, [(100, 100), (200, 150), (300, 120)], atol=1e-3)
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
    np.testing.assert_allclose(fine.points[-1], (40, 10), atol=1e-6)


def test_a_curve_is_flattened_within_the_flatness_asked(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_bezier((100, 100), (100, 400), (400, 400), (400, 100), color=(0, 0, 0))  # a large arch
    page.draw_bezier((10, 10), (10.1, 9.9), (10.2, 9.9), (10.3, 10), color=(0, 0, 0))  # a hook bowing 0.075 pt
    doc.save(tmp_path / 'c.pdf')
    t = np.linspace(0, 1, 2001)[:, None]
    arch = (1 - t) ** 3 * [100, 100] + 3 * (1 - t) ** 2 * t * [100, 400] + 3 * (1 - t) * t**2 * [400, 400]
    arch += t**3 * [400, 100]
    for flatness in (0.5, 0.1, 0.02):
        large, small = read_pdf(tmp_path / 'c.pdf', flatness=flatness).page(1).paths
        start, step = large.points[:-1], np.diff(large.points, axis=0)
        reach = np.clip(((arch[:, None] - start) * step).sum(-1) / (step * step).sum(-1), 0, 1)
        nearest = np.linalg.norm(arch[:, None] - (start + reach[..., None] * step), axis=-1).min(axis=1)
        assert nearest.max() <= flatness
        assert len(small.points) == (2 if flatness >= 0.1 else 3)  # one segment unless it bows past the flatness
    default = read_pdf(tmp_path / 'c.pdf').page(1).paths[0]
    assert len(default.points) == CURVE_SAMPLES + 1


@pytest.mark.parametrize(
    'choice', [{'curve_samples': 8, 'flatness': 0.1}, {'curve_samples': 0}, {'flatness': 0.0}, {'flatness': -1.0}]
)
def test_an_ambiguous_or_empty_flattening_is_refused_at_once(tmp_path, choice):
    with pytest.raises(ValueError, match='flatten'):
        read_pdf(mixed_pdf(tmp_path / 'mixed.pdf'), **choice)


def test_points_are_a_read_only_array_of_rows(doc):
    paths = doc.page(1).paths
    for path in paths:
        assert (path.points.dtype, path.points.ndim, path.points.shape[1]) == (np.float64, 2, 2)
        assert not path.points.flags.writeable
    box = paths[0].provenance.bbox
    assert tuple(box) == (*paths[0].points.min(axis=0), *paths[0].points.max(axis=0))


def test_a_path_is_itself_and_no_other(doc):
    a, b = doc.page(1).paths[:2]
    assert a == a  # noqa: PLR0124 -- identity is the comparison
    assert a != b
    assert len({a, b}) == 2


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
    with pytest.raises(Locked):
        read_pdf(tmp_path / 'e.pdf')
    with pytest.raises(Locked):
        read_pdf(tmp_path / 'e.pdf', password='wrong')  # noqa: S106 -- a wrong password, planted


def test_each_kind_of_refusal_is_its_own_exception(tmp_path):
    (tmp_path / 'text.pdf').write_bytes(b'hello, no PDF')
    with pytest.raises(NotPdf):
        read_pdf(tmp_path / 'text.pdf')
    handler = b'%PDF-1.7\n1 0 obj << /Filter /Adobe.PubSec /R 4 >> endobj\ntrailer << /Encrypt 1 0 R /Root 2 0 R >>\n'
    (tmp_path / 'pubsec.pdf').write_bytes(handler)
    with pytest.raises(Unsupported):
        read_pdf(tmp_path / 'pubsec.pdf')
    (tmp_path / 'empty.pdf').write_bytes(b'%PDF-1.7\n')
    with pytest.raises(Malformed):
        read_pdf(tmp_path / 'empty.pdf')
    assert all(issubclass(kind, Refused) for kind in (NotPdf, Locked, Unsupported, Malformed))
    assert issubclass(Refused, ValueError)


def test_a_raster_renders_at_the_native_resolution_of_the_image_under_it(doc, tmp_path):
    path = mixed_pdf(tmp_path / 'mixed.pdf')
    raster = read_raster(path, 1, BBox(72, 600, 172, 700))
    rows, columns = raster.pixels.shape[:2]  # 8 samples across 100 points: 5.76 dpi, a pixel per sample
    assert abs(rows - 8) <= 1 and abs(columns - 8) <= 1  # the renderer rounds the clip's edges outward


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / 'absent.pdf')


def test_a_page_is_read_when_first_asked_for_and_kept(tmp_path, monkeypatch):
    import document_lab.pdf as pdf_module  # noqa: PLC0415 -- the reading counted where it happens

    read = []
    page = pdf_module._page

    def counted(reader, number, *rest):
        read.append(number)
        return page(reader, number, *rest)

    monkeypatch.setattr(pdf_module, '_page', counted)
    doc = read_pdf(mixed_pdf(tmp_path / 'mixed.pdf'))
    assert (len(doc.pages), read) == (2, [])
    assert doc.pages[-1].number == 2
    assert doc.page(2) is doc.pages[1]
    assert read == [2]
    assert [p.number for p in doc.pages[:1]] == [1]
    assert [p.number for p in doc.pages] == [1, 2]
    assert read == [2, 1]


def test_pages_read_after_the_file_is_gone_are_of_the_bytes_hashed(tmp_path):
    path = mixed_pdf(tmp_path / 'mixed.pdf')
    doc = read_pdf(path)
    path.unlink()
    assert [s.text for s in doc.page(1).spans if s.text == 'Hello document'] == ['Hello document']


def test_a_page_past_the_last_is_refused(doc):
    with pytest.raises(IndexError):
        doc.pages[2]


def test_a_repair_made_reading_a_page_is_said_once_the_page_is_read(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), 'Moved', fontsize=12)
    contents = doc[0].get_contents()[0]
    data = bytearray(doc.tobytes(garbage=0, deflate=False, use_objstms=0))
    first = data.index(b'\n', data.index(b'\nxref\n') + len(b'\nxref\n')) + 1  # past the subsection's `0 n` line
    entry = first + 20 * contents  # a table entry is 20 bytes
    offset = int(data[entry : entry + 10])
    data[entry : entry + 10] = b'%010d' % (offset + 1)  # the table now points one byte into the object
    (tmp_path / 'm.pdf').write_bytes(bytes(data))
    read = read_pdf(tmp_path / 'm.pdf')
    assert list(read.repairs) == []
    assert [s.text for s in read.page(1).spans] == ['Moved']
    assert len(read.repairs) == 1
