import hashlib
import itertools

import numpy as np
import pymupdf
import pytest

from document_lab import _pdf
from document_lab.ir import BBox
from document_lab.pdf import CURVE_SAMPLES, Locked, Malformed, NotPdf, Refused, Unsupported, read_pdf, read_raster
from tests.synthetic import (
    PAGE,
    RED,
    cid_font_pdf,
    content_pdf,
    icc_profile,
    image_pdf,
    mixed_pdf,
    rgb_in_srgb,
    table_cell_rect,
)


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


def test_a_span_carries_its_fill_and_text_not_filled_has_none(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'red', fontname='helv', fontsize=10, color=RED)
    page.insert_text((72, 100), 'plain', fontname='helv', fontsize=10)
    page.insert_text((72, 128), 'hidden', fontname='helv', fontsize=10, render_mode=3)  # an OCR layer's mode
    doc.save(tmp_path / 'c.pdf')
    fills = {s.text: s.fill for s in read_pdf(tmp_path / 'c.pdf').spans()}
    assert fills == {'red': (1.0, 0.0, 0.0), 'plain': (0.0, 0.0, 0.0), 'hidden': None}


def test_a_span_says_which_way_its_baseline_runs(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'across', fontname='helv', fontsize=10)
    page.insert_text((30, 400), 'up the margin', fontname='helv', fontsize=10, rotate=90)
    doc.save(tmp_path / 'd.pdf')
    directions = {s.text: tuple(round(c, 6) for c in s.direction) for s in read_pdf(tmp_path / 'd.pdf').spans()}
    assert directions == {'across': (1, 0), 'up the margin': (0, -1)}  # page space: y grows down


def test_a_span_says_its_render_mode(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'hidden', fontname='helv', fontsize=10, render_mode=3)  # invisible, an OCR layer
    page.insert_text((72, 100), 'shown', fontname='helv', fontsize=10)
    doc.save(tmp_path / 'r.pdf')
    modes = {s.text: s.render_mode for s in read_pdf(tmp_path / 'r.pdf').spans()}
    assert modes == {'hidden': 3, 'shown': 0}


def background_pdf(path):
    """White on the white page, white on a black box (the visible control), black on a black box, black on white."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(60, 85, 300, 140), color=None, fill=(0, 0, 0))
    page.insert_text((72, 72), 'white on white', fontname='helv', fontsize=10, color=(1, 1, 1))
    page.insert_text((72, 100), 'white on black', fontname='helv', fontsize=10, color=(1, 1, 1))
    page.insert_text((72, 128), 'black on black', fontname='helv', fontsize=10, color=(0, 0, 0))
    page.insert_text((72, 156), 'black on white', fontname='helv', fontsize=10, color=(0, 0, 0))
    doc.save(path)
    return read_pdf(path).page(1)


def test_text_filled_in_the_color_under_it_shows_nothing_and_the_shown_page_leaves_it_out(tmp_path):
    page = background_pdf(tmp_path / 'b.pdf')
    assert {s.text: s.in_background_color for s in page.spans} == {
        'white on white': True,
        'white on black': False,
        'black on black': True,
        'black on white': False,
    }
    assert [s.text for s in page.shown().spans] == ['white on black', 'black on white']
    assert len(page.shown(include_in_background_color=True).spans) == 4


#: White text, its baseline at (150, 150) in user space, on a :data:`tests.synthetic.PAGE`-sized page.
WHITE = 'BT 1 g /F1 10 Tf 150 150 Td (white) Tj ET'


def _framed(margin: float, x0: float, b: float, x1: float, t: float) -> str:
    """The rectangle ``margin`` round the text's box (user space: left, bottom, right, top), as ``re`` draws it."""
    return f'{x0 - margin} {b - margin} {x1 - x0 + 2 * margin} {t - b + 2 * margin} re'


def _across(x0: float, b: float, x1: float, t: float) -> str:
    """A line through the middle of the text's box, past both its ends."""
    return f'{x0 - 5} {(b + t) / 2} m {x1 + 5} {(b + t) / 2} l'


def _short(x0: float, b: float, x1: float, t: float) -> str:
    """A line along the text's middle ending 2 left of it."""
    return f'{x0 - 40} {(b + t) / 2} m {x0 - 2} {(b + t) / 2} l'


def _corner(x0: float, b: float, x1: float, t: float) -> str:
    """A sharp corner 3 below the text pointing up at it: 4 wide, mitered, its tip 9.4 from it."""
    mid = (x0 + x1) / 2
    return f'{mid - 10} {b - 49} m {mid} {b - 3} l {mid + 10} {b - 49} l'


#: Strokes drawn in black (unless they say otherwise) under white text, each with whether the text then shows
#: nothing; each case is followed by its planted wrong truth, the same strokes but painting into the text.
STROKES_UNDER_WHITE = {
    'a table cell frame round it': (lambda *box: f'1 w {_framed(3, *box)} S', True),
    'the frame, its band reaching in': (lambda *box: f'8 w {_framed(3, *box)} S', False),
    'a dashed frame round it': (lambda *box: f'1 w [3 2] 0 d {_framed(3, *box)} S', True),
    'the dashed frame, its dashes reaching in': (lambda *box: f'8 w [3 2] 0 d {_framed(3, *box)} S', False),
    'a white cell framed black': (lambda *box: f'1 g 1 w {_framed(3, *box)} B', True),
    'a black cell framed black': (lambda *box: f'0 g 1 w {_framed(3, *box)} B', False),
    'a thick white stroke across it': (lambda *box: f'1 G 30 w {_across(*box)} S', True),
    'a thick black stroke across it': (lambda *box: f'30 w {_across(*box)} S', False),
    'a butt cap short of it': (lambda *box: f'6 w 0 J {_short(*box)} S', True),
    'a square cap reaching in': (lambda *box: f'6 w 2 J {_short(*box)} S', False),
    'a round cap reaching in': (lambda *box: f'6 w 1 J {_short(*box)} S', False),
    'a beveled corner below it': (lambda *box: f'4 w 2 j {_corner(*box)} S', True),
    'a mitered corner reaching in': (lambda *box: f'4 w 0 j 10 M {_corner(*box)} S', False),
}


@pytest.mark.parametrize('case', STROKES_UNDER_WHITE)
def test_white_text_is_judged_by_what_strokes_paint_in_its_box_not_by_their_paths_box(tmp_path, case):
    strokes, unseen = STROKES_UNDER_WHITE[case]
    (alone,) = read_pdf(content_pdf(tmp_path / 'alone.pdf', [WHITE])).spans()
    x0, top, x1, bottom = alone.provenance.bbox
    height = PAGE[1]
    path = content_pdf(tmp_path / 'stroked.pdf', [f'0 G {strokes(x0, height - bottom, x1, height - top)} {WHITE}'])
    (span,) = read_pdf(path).spans()
    assert span.in_background_color is unseen
    # The truth, from another renderer: whether anything but white shows in the text's box (4 pixels a point, a
    # pixel in from its edges, where anti-aliasing shades them).
    with pymupdf.open(path) as doc:
        pixmap = doc[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), colorspace=pymupdf.csGRAY, alpha=False)
    grey = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width)
    inside = grey[round(top * 4) + 1 : round(bottom * 4) - 1, round(x0 * 4) + 1 : round(x1 * 4) - 1]
    assert bool(inside.min() == 255) is unseen


#: Black text, its baseline at (150, 150) in user space, on a :data:`tests.synthetic.PAGE`-sized page.
BLACK = 'BT 0 g /F1 10 Tf 150 150 Td (black) Tj ET'

#: Strokes drawn in black (unless they say otherwise) under black text, each with whether the text then shows
#: nothing; the first a band that holds the text, then that band's planted wrong truths, each leaving page white in it.
STROKES_UNDER_BLACK = {
    'a thick black stroke across it': (lambda *box: f'30 w {_across(*box)} S', True),
    'clipped round it': (lambda *box: f'q {_framed(3, *box)} W n 30 w {_across(*box)} S Q', True),
    'dashed across it': (lambda *box: f'30 w [3 2] 0 d {_across(*box)} S', False),
    'thinner than it is tall': (lambda *box: f'6 w {_across(*box)} S', False),
    'ending inside it': (
        lambda x0, b, x1, t: f'30 w {x0 - 5} {(b + t) / 2} m {(x0 + x1) / 2} {(b + t) / 2} l S',
        False,
    ),
    'clipped to half of it': (
        lambda x0, b, x1, t: f'q {x0 - 3} {b - 3} {(x1 - x0) / 2} {t - b + 6} re W n 30 w {_across(x0, b, x1, t)} S Q',
        False,
    ),
}


@pytest.mark.parametrize('case', STROKES_UNDER_BLACK)
def test_text_inside_a_solid_band_of_its_color_shows_nothing(tmp_path, case):
    strokes, unseen = STROKES_UNDER_BLACK[case]
    (alone,) = read_pdf(content_pdf(tmp_path / 'alone.pdf', [BLACK])).spans()
    x0, top, x1, bottom = alone.provenance.bbox
    height = PAGE[1]
    path = content_pdf(tmp_path / 'stroked.pdf', [f'0 G {strokes(x0, height - bottom, x1, height - top)} {BLACK}'])
    (span,) = read_pdf(path).spans()
    assert span.in_background_color is unseen
    # The truth, from another renderer: whether anything but black shows in the text's box (as for white text).
    with pymupdf.open(path) as doc:
        pixmap = doc[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), colorspace=pymupdf.csGRAY, alpha=False)
    grey = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width)
    inside = grey[round(top * 4) + 1 : round(bottom * 4) - 1, round(x0 * 4) + 1 : round(x1 * 4) - 1]
    assert bool(inside.max() == 0) is unseen


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
    assert raster.provenance.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


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


# Adobe RGB (1998): its primaries and white (CIE xy) and its gamma (its specification, 4.3.4).
ADOBE = {'primaries': ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), 'white': (0.3127, 0.3290), 'gamma': 563 / 256}


def test_an_icc_based_image_is_its_samples_through_its_profile(tmp_path, monkeypatch):
    path = image_pdf(
        tmp_path / 'adobe.pdf',
        SAMPLES.tobytes(),
        RGB.replace(' /ColorSpace /DeviceRGB', ''),
        size=(60, 40),
        profile=(icc_profile(**ADOBE), 3),
    )
    truth = rgb_in_srgb(SAMPLES, **ADOBE)  # by the formulas, apart from any color management
    assert np.abs(truth.astype(int) - SAMPLES).max() > 20  # the planted premise: the profile changes what shows
    assert np.abs(_shown(path).astype(int) - truth).max() <= 2  # PyMuPDF, color-managed, corroborates the truth
    _refuse_rendering(monkeypatch)
    assert np.abs(_whole_page(path).astype(int) - truth).max() <= 1


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


def test_a_missing_file_raises_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=r'absent\.pdf'):
        read_pdf(tmp_path / 'absent.pdf')


def test_a_file_opened_by_its_path_reads_as_its_bytes_in_memory(tmp_path):
    path = mixed_pdf(tmp_path / 'mixed.pdf')
    by_path, in_memory = _pdf.open(path, b''), _pdf.open(path.read_bytes(), b'')
    assert by_path.sha256 == in_memory.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert _pdf.open(str(path), b'').sha256 == by_path.sha256
    assert by_path.pages == in_memory.pages
    assert repr(by_path.page(1, annotations=False, curve_samples=8)) == repr(
        in_memory.page(1, annotations=False, curve_samples=8)
    )


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
    assert path.points.tolist() == [[10, 40], [60, 40], [60, 90], [10, 90]]
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


_NONE = '[/Separation /None /DeviceRGB << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [1 0 0] /N 1 >>]'


def test_what_the_colorant_none_paints_is_read_flagged_inkless(tmp_path):
    page = _raw_page(
        tmp_path,
        '<< /ColorSpace << /N NONE >> >>',
        b'0 0 1 rg 0 0 5 5 re f /N cs 1 sc 10 10 5 5 re f 0 1 0 RG 20 20 5 5 re B',
        {'NONE': _NONE},
    )
    found = [(p.fill, p.stroke, p.inkless, p.hidden) for p in page.paths]
    assert found == [
        ((0, 0, 1), None, False, False),
        ((1,), None, True, False),  # no RGB: it marks nothing; its tint is kept
        (None, (0, 1, 0), False, False),  # `B`: the stroke marks, the fill does not -- two paintings
        ((1,), None, True, False),
    ]
    assert [p.group for p in page.paths] == [0, 1, 2, 3]
    assert [p.fill for p in page.shown().paths] == [(0, 0, 1), None]
    assert len(page.shown(include_hidden=True).paths) == 2  # a hidden layer's content is another matter
    assert page.shown(include_inkless=True) == page


def test_an_uncolored_tiling_pattern_fills_in_its_tint(tmp_path):
    tiling = '<< /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 /Resources << >> >>'
    page = _raw_page(
        tmp_path,
        '<< /Pattern << /T TILING >> >>',
        b'[/Pattern /DeviceGray] cs 0.25 /T scn 10 10 30 30 re f',
        {'TILING': tiling},
    )
    assert [(p.fill, p.pattern) for p in page.paths] == [((0.25, 0.25, 0.25), 'tiling')]


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


def test_vertical_writing_is_a_vertical_span_read_top_to_bottom_in_its_column(tmp_path):
    # PyMuPDF sets the text with a predefined Unicode CMap; its -V twin (vertical, ISO 32000-2 9.7.5.2) is swapped in.
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((100, 40), '\u4e00\u4e8c\u4e09', fontname='china-s', fontsize=10)
    page.insert_text((20, 150), 'side', fontsize=10)
    (cjk,) = [xref for xref, *_, encoding in page.get_fonts() if encoding == 'UniGB-UTF16-H']
    doc.xref_set_key(cjk, 'Encoding', '/UniGB-UTF16-V')
    doc.save(tmp_path / 'v.pdf')
    spans = {s.text: s for s in read_pdf(tmp_path / 'v.pdf').page(1).spans}
    column = spans['\u4e00\u4e8c\u4e09']
    assert column.vertical
    assert not spans['side'].vertical
    # From the vertical origin (100, 40): an em across, centered on it; three ems (the default advance) down.
    assert tuple(column.provenance.bbox) == pytest.approx((95, 40, 105, 70), abs=1e-6)


#: CIDs of Adobe-Japan1 and the text Adobe-Japan1-UCS2 names for each: a code unit, a character past U+FFFF, and a
#: variation sequence (a slashed zero).
JAPAN1_CIDS = (34, 843, 3284, 3722, 13839, 230)
JAPAN1_TEXT = 'A\u3042\u65e5\u672c\U0002000b0\ufe00'
#: The text Adobe-GB1-UCS2 names for the same CIDs: what reading them in the wrong collection gives.
GB1_TEXT = 'A>\u816e\u553e\u687d\u2460'


def misread(path, truth):
    """Where the text ``path`` reads differs from ``truth``, character by character; empty when it reads as truth."""
    read = ''.join(s.text for s in read_pdf(path).spans())
    return [(i, r, t) for i, (r, t) in enumerate(itertools.zip_longest(read, truth)) if r != t]


def test_a_font_not_embedded_on_an_adobe_collection_reads_the_text_adobe_names_for_its_cids(tmp_path):
    japan1 = cid_font_pdf(tmp_path / 'japan1.pdf', JAPAN1_CIDS)
    gb1 = cid_font_pdf(tmp_path / 'gb1.pdf', JAPAN1_CIDS, ordering='GB1')
    assert misread(japan1, JAPAN1_TEXT) == []
    assert misread(gb1, GB1_TEXT) == []
    # Planted: each collection's text is the other's wrong truth.
    assert misread(japan1, GB1_TEXT)
    assert misread(gb1, JAPAN1_TEXT)


def test_a_cid_no_table_maps_reads_as_unknown_never_a_guess(tmp_path):
    # `.notdef` and the first CID past Adobe-Japan1-7's last (23059) around one it maps; then a registry not Adobe's.
    beyond = cid_font_pdf(tmp_path / 'beyond.pdf', (0, 34, 23060))
    other = cid_font_pdf(tmp_path / 'other.pdf', (843,), registry='Acme')
    assert misread(beyond, '\ufffdA\ufffd') == []
    assert misread(other, '\ufffd') == []
    # Planted: a guess -- the code as its code point, the collection's text under another registry.
    assert misread(beyond, '\0A\u5a14')
    assert misread(other, '\u3042')


def test_the_to_unicode_a_font_states_wins_over_its_collection(tmp_path):
    stated = cid_font_pdf(tmp_path / 'stated.pdf', (843, 3284), to_unicode={843: 'X'})
    assert misread(stated, 'X\u65e5') == []  # the code ToUnicode does not map falls to the collection
    assert misread(stated, '\u3042\u65e5')  # planted: the collection's text of the code ToUnicode maps
