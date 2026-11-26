import hashlib

import numpy as np
import pymupdf
import pytest

from document_lab.ir import BBox
from document_lab.pdf import Locked, Malformed, NotPdf, Refused, Unsupported, read_pdf, read_raster
from tests.synthetic import RED, image_pdf, mixed_pdf, table_cell_rect


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


SAMPLES = np.random.default_rng(7).integers(0, 256, (40, 60, 3), dtype=np.uint8)  # rows x columns x RGB
RGB = '/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8'
GRAY = '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8'


def _shown(path, clip=None, zoom=1.0):
    """What the page shows, rendered by PyMuPDF at ``zoom`` pixels a point: the truth a raster is held to."""
    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()


def _refuse_rendering(monkeypatch):
    def rendered(*_args, **_kwargs):
        msg = 'rendered: the samples were not taken'
        raise AssertionError(msg)

    monkeypatch.setattr(pymupdf.Page, 'get_pixmap', rendered)


def _whole_page(path, **kwargs):
    page = read_pdf(path).page(1)
    return read_raster(path, 1, BBox(0, 0, page.width, page.height), **kwargs).pixels


# The cm placing a 60 x 40 image, a sample a point (PDF space, y up), and the page it fills: upright, flipped,
# mirrored, half turned, and the four quarter-turned placements on a page standing on its side.
PLACED = {
    'upright': ((60, 0, 0, 40, 0, 0), (60, 40)),
    'flipped': ((60, 0, 0, -40, 0, 40), (60, 40)),
    'mirrored': ((-60, 0, 0, 40, 60, 0), (60, 40)),
    'half-turned': ((-60, 0, 0, -40, 60, 40), (60, 40)),
    'quarter-turned': ((0, 60, -40, 0, 40, 0), (40, 60)),
    'quarter-turned-back': ((0, -60, 40, 0, 0, 60), (40, 60)),
    'transposed': ((0, 60, 40, 0, 0, 0), (40, 60)),
    'transposed-back': ((0, -60, -40, 0, 40, 60), (40, 60)),
}


@pytest.mark.parametrize('placed', PLACED)
def test_a_region_that_is_one_image_whole_is_its_own_samples_as_the_page_shows_them(placed, tmp_path, monkeypatch):
    matrix, size = PLACED[placed]
    path = image_pdf(tmp_path / 'one.pdf', SAMPLES.tobytes(), RGB, size=size, matrix=matrix)
    shown = _shown(path)
    _refuse_rendering(monkeypatch)
    assert np.array_equal(_whole_page(path), shown)
    if placed == 'upright':
        assert np.array_equal(shown, SAMPLES)  # the planted truth the renderer is held to first


def test_a_turned_page_turns_its_image_with_it(tmp_path, monkeypatch):
    path = image_pdf(tmp_path / 'turned.pdf', SAMPLES.tobytes(), RGB, size=(60, 40), rotate=90)
    shown = _shown(path)
    _refuse_rendering(monkeypatch)
    assert np.array_equal(_whole_page(path), shown)


@pytest.mark.parametrize('bits', [1, 2, 4, 8])
def test_a_gray_image_of_any_depth_is_its_levels_spread_over_0_to_255(bits, tmp_path, monkeypatch):
    levels = np.random.default_rng(bits).integers(0, 2**bits, (40, 60), dtype=np.uint8)
    rows = np.unpackbits(levels[..., None], axis=2)[..., 8 - bits :].reshape(40, -1)
    stored = np.packbits(rows, axis=1).tobytes()  # each row padded to a whole byte
    path = image_pdf(tmp_path / 'gray.pdf', stored, GRAY.replace('Component 8', f'Component {bits}'), size=(60, 40))
    shown = _shown(path)
    _refuse_rendering(monkeypatch)
    assert np.array_equal(_whole_page(path), shown)
    assert np.array_equal(shown[..., 0], levels * (255 // (2**bits - 1)))


def test_a_decode_stating_the_default_mapping_keeps_the_samples(tmp_path, monkeypatch):
    path = image_pdf(tmp_path / 'stated.pdf', SAMPLES.tobytes(), f'{RGB} /Decode [0 1 0 1 0 1]', size=(60, 40))
    assert np.array_equal(_shown(path), SAMPLES)
    _refuse_rendering(monkeypatch)
    assert np.array_equal(_whole_page(path), SAMPLES)


def test_an_ocr_layer_over_the_image_does_not_hide_its_samples(tmp_path, monkeypatch):
    def ocr_layer(page):
        page.insert_text((2, 20), 'invisible words', fontsize=8, render_mode=3)

    path = image_pdf(tmp_path / 'ocr.pdf', SAMPLES.tobytes(), RGB, size=(60, 40), over=ocr_layer)
    _refuse_rendering(monkeypatch)
    assert np.array_equal(_whole_page(path), SAMPLES)


def _jpeg() -> bytes:
    return pymupdf.Pixmap(pymupdf.csRGB, 60, 40, SAMPLES.tobytes(), 0).tobytes('jpeg')


PALETTE = np.random.default_rng(3).integers(0, 256, (4, 3), dtype=np.uint8)
INDICES = np.random.default_rng(4).integers(0, 4, (40, 60), dtype=np.uint8)
CMYK = np.random.default_rng(5).integers(0, 256, (40, 60, 4), dtype=np.uint8)
GRAY16 = np.random.default_rng(6).integers(0, 2**16, (40, 60), dtype=np.uint16).astype('>u2')  # PDF is big-endian
# What the samples alone do not show -- each case one the page shows unlike its samples read as 8-bit RGB -- with
# the truth it is held to where it can be planted (None: the renderer's word alone).
UNSEEN = {
    'decode-inverted': (
        {'stored': SAMPLES[..., 0].tobytes(), 'image': f'{GRAY} /Decode [1 0]'},
        np.repeat(255 - SAMPLES[..., :1], 3, axis=2),
    ),
    'soft-masked-away': (
        {'stored': SAMPLES.tobytes(), 'image': RGB, 'soft_mask': bytes(60 * 40)},
        np.full((40, 60, 3), 255, dtype=np.uint8),
    ),
    'color-key-masked': ({'stored': SAMPLES.tobytes(), 'image': f'{RGB} /Mask [0 127 0 255 0 255]'}, None),
    'indexed': (
        {
            'stored': INDICES.tobytes(),
            'image': GRAY.replace('/DeviceGray', f'[/Indexed /DeviceRGB 3 <{PALETTE.tobytes().hex()}>]'),
        },
        PALETTE[INDICES],
    ),
    'cmyk': ({'stored': CMYK.tobytes(), 'image': GRAY.replace('Gray', 'CMYK')}, None),
    '16-bit': ({'stored': GRAY16.tobytes(), 'image': GRAY.replace('Component 8', 'Component 16')}, None),
    'jpeg': ({'stored': _jpeg(), 'image': f'{RGB} /Filter /DCTDecode', 'deflate': False}, None),
    'text-over': (
        {'stored': SAMPLES.tobytes(), 'image': RGB, 'over': lambda p: p.insert_text((2, 20), 'Ink', fontsize=14)},
        None,
    ),
    'path-over': (
        {'stored': SAMPLES.tobytes(), 'image': RGB, 'over': lambda p: p.draw_line((0, 0), (60, 40), width=3)},
        None,
    ),
}


@pytest.mark.parametrize('case', UNSEEN)
def test_what_the_samples_alone_do_not_show_is_rendered(case, tmp_path):
    made, truth = UNSEEN[case]
    path = image_pdf(tmp_path / 'unseen.pdf', size=(60, 40), **made)
    shown = _shown(path)
    assert not np.array_equal(shown, SAMPLES)  # the planted premise: the samples are not what shows
    if truth is not None:
        assert np.array_equal(shown, truth)
    assert np.array_equal(_whole_page(path), shown)


def test_part_of_an_image_or_another_resolution_is_rendered(tmp_path):
    path = image_pdf(tmp_path / 'one.pdf', SAMPLES.tobytes(), RGB, size=(60, 40))
    half = BBox(0, 0, 30, 40)
    assert np.array_equal(read_raster(path, 1, half).pixels, _shown(path, pymupdf.Rect(*half)))
    assert np.array_equal(_whole_page(path, dpi=144), _shown(path, zoom=2))


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / 'absent.pdf')


def _raw_page(tmp_path, resources: str, content: bytes, objects: dict[str, str]):
    """A one-page PDF (200 x 100) drawing ``content`` with ``resources``, its ``objects`` named in them by key."""
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    for key, source in objects.items():
        xref = doc.get_new_xref()
        doc.update_object(xref, source)
        resources = resources.replace(key, f'{xref} 0 R')
    doc.xref_set_key(page.xref, 'Resources', resources)
    stream = doc.get_new_xref()
    doc.update_object(stream, '<< >>')
    doc.update_stream(stream, content)
    doc.xref_set_key(page.xref, 'Contents', f'{stream} 0 R')
    doc.save(tmp_path / 'raw.pdf')
    return read_pdf(tmp_path / 'raw.pdf').page(1)


_AXIAL = (
    '<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 1 0] '
    '/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>'
)


def test_a_shading_is_the_closed_rectangle_of_its_clip_filled_with_a_shading(tmp_path):
    page = _raw_page(tmp_path, '<< /Shading << /S SHADING >> >>', b'q 10 10 50 50 re W n /S sh Q', {'SHADING': _AXIAL})
    (path,) = page.paths
    assert path.points == ((10, 40), (60, 40), (60, 90), (10, 90))
    assert (path.closed, path.stroke, path.fill, path.pattern) == (True, None, (), 'shading')
    assert tuple(path.provenance.bbox) == (10, 40, 60, 90)


def test_a_fill_with_a_pattern_names_it(tmp_path):
    page = _raw_page(
        tmp_path,
        '<< /Pattern << /G PATTERN >> >>',
        b'0 0 1 rg 0 0 5 5 re f /Pattern cs /G scn 10 10 30 30 re f',
        {'PATTERN': f'<< /PatternType 2 /Shading {_AXIAL} >>'},
    )
    assert [(p.fill, p.pattern) for p in page.paths] == [((0, 0, 1), None), ((), 'shading')]


def test_text_of_a_layer_turned_off_is_read_flagged_hidden(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    off = doc.add_ocg('Off', on=False)
    page.insert_text((72, 72), 'shown', fontname='helv', fontsize=10)
    page.insert_text((72, 100), 'layered', fontname='helv', fontsize=10, oc=off)
    doc.save(tmp_path / 'oc.pdf')
    spans = {s.text: s.hidden for s in read_pdf(tmp_path / 'oc.pdf').spans()}
    assert spans == {'shown': False, 'layered': True}


def test_annotations_are_read_only_when_asked_and_cite_their_subtype(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'body', fontname='helv', fontsize=10)
    page.add_freetext_annot(pymupdf.Rect(72, 200, 272, 230), 'note', fontsize=10)
    doc.save(tmp_path / 'a.pdf')
    assert [s.text for s in read_pdf(tmp_path / 'a.pdf').spans()] == ['body']
    spans = {s.text: s.annotation for s in read_pdf(tmp_path / 'a.pdf', annotations=True).spans()}
    assert spans == {'body': None, 'note': 'FreeText'}
    (note,) = [s for s in read_pdf(tmp_path / 'a.pdf', annotations=True).spans() if s.text == 'note']
    assert 72 <= note.provenance.bbox.x0 < note.provenance.bbox.x1 <= 272  # inside the annotation's rectangle
