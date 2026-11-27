"""The own renderer accepted against PyMuPDF's renders on synthetic pages (``tests/rendering.py`` says how): paths,
dashes, caps and joins, clips, compositing, and images in every space and mask it draws. Each case is paired with a
planted wrong truth -- the same page drawn a little wrong -- that the comparison must report.

Two conversions are the reader's own and differ from PyMuPDF's default by design: ``DeviceCMYK`` is converted by
the standard's formula, as PyMuPDF does with color management off (it is compared so), not through a press profile;
``Lab`` colors outside sRGB are clipped channel by channel, and a ``Lab`` image's samples span its ``/Range`` (Table
90's default ``/Decode``) where PyMuPDF spans -128..127 whatever the range (the case's range is that). A line of
width 0 is one pixel wide as ISO 32000-2 8.4.3.2 says (``crates/render`` tests it); PyMuPDF draws it fainter, and
round caps a little smaller, so no case holds one or round caps that touch.
"""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest

from document_lab.ir import BBox
from document_lab.pdf import Unsupported, read_pdf, read_raster
from tests.rendering import agrees, disagreement, own, page_pdf, reference
from tests.synthetic import icc_profile

DPI = 150

_y, _x = np.mgrid[0:40, 0:60]
RGB = np.stack([_x * 4, _y * 6, (_x + _y) * 2], -1).astype(np.uint8)
GRAY = (_x * 4).astype(np.uint8)


def _bits(bits: int, samples: np.ndarray) -> bytes:
    """``samples`` (rows x values) packed ``bits`` a value, each row on a whole byte."""
    out = b''
    for row in samples.reshape(samples.shape[0], -1):
        text = ''.join(format(int(v), f'0{bits}b') for v in row)
        text += '0' * (-len(text) % 8)
        out += int(text, 2).to_bytes(len(text) // 8, 'big')
    return out


def _image(entries: str, data: bytes) -> tuple[str, bytes]:
    return f'<< /Type /XObject /Subtype /Image {entries} >>', data


def _jpeg(samples: np.ndarray, space: pymupdf.Colorspace) -> bytes:
    height, width = samples.shape[:2]
    return pymupdf.Pixmap(space, width, height, samples.tobytes(), False).tobytes('jpeg', jpg_quality=95)


PLACE = 'q 120 0 0 80 40 35 cm /Im1 Do Q'
SHIFTED = 'q 120 0 0 80 43 35 cm /Im1 Do Q'
PROFILE = icc_profile(primaries=[(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], white=(0.3127, 0.3290), gamma=1.8)
STAR = '100 75 m 130 10 l 160 75 l 90 35 l 170 35 l h'
DISC = '100 75 m 100 116 78 140 50 140 c 22 140 0 116 0 75 c 0 34 22 10 50 10 c 78 10 100 34 100 75 c'
ALPHA = '/ExtGState << /Half << /ca 0.5 >> /Faint << /CA 0.3 >> /Mul << /BM /Multiply >> >>'
SEPARATION = '[/Separation /Spot /DeviceRGB << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 0.5 1] /N 1 >>]'
LAB = '[/Lab << /WhitePoint [0.9505 1 1.089] /Range [-128 127 -128 127] >>]'
JOINS = (
    '10 w 0 j 30 20 m 60 70 l 90 20 l S {round} j 110 20 m 140 70 l 170 20 l S 2 j 30 90 m 60 140 l 90 90 l S '
    '0 j {limit} M 125 90 m 140 140 l 155 90 l S'
)
COMPOSITED = (
    '{half} 1 0 0 rg 20 20 100 80 re f 0 0 1 rg 70 50 100 80 re f /Faint gs 0 1 0 RG 8 w 10 140 m 190 10 l S '
    '/Mul gs 1 1 0 rg 0 0 60 60 re f'
)
INDEXED = '/Width 60 /Height 40 /ColorSpace [/Indexed /DeviceRGB 3 <ff000000ff000000ffffff00>] /BitsPerComponent 4'
SHADING = (
    '<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 200 0] '
    '/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>'
)
INLINE_RGB = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 128, 128, 128, 0, 255, 255])

#: name: (content, objects, resources, the planted wrong content, its objects, whether PyMuPDF manages color).
CASES = {
    'fills': (
        '1 0 0 rg 20 20 60 40 re f 0.5 g 100 20 m 180 20 l 140 120 l h f 0 0.6 0.3 rg 20 80 40 40 re f',
        None,
        '',
        '1 0 0 rg 20 20 60 40 re f 0.5 g 100 20 m 180 20 l 140 120 l h f 0 0.5 0.3 rg 20 80 40 40 re f',
        None,
        True,
    ),
    'even-odd': (f'0 0 1 rg {STAR} f*', None, '', f'0 0 1 rg {STAR} f', None, True),
    'caps': (
        '6 w 0 J 20 20 m 80 20 l S 1 J 20 50 m 80 50 l S 2 J 20 80 m 80 80 l S',
        None,
        '',
        '6 w 0 J 20 20 m 80 20 l S 1 J 20 50 m 80 50 l S 0 J 20 80 m 80 80 l S',
        None,
        True,
    ),
    'joins': (
        JOINS.format(round=1, limit=2),
        None,
        '',
        JOINS.format(round=2, limit=10),
        None,
        True,
    ),
    'dashes': (
        '2 w [6 3] 0 d 10 20 m 190 20 l S [5] 2 d 10 50 m 190 50 l S 1 J [8 4 1 4] 0 d 10 80 m 190 80 l S',
        None,
        '',
        '2 w [6 3] 0 d 10 20 m 190 20 l S [5] 7 d 10 50 m 190 50 l S 1 J [8 4 1 4] 0 d 10 80 m 190 80 l S',
        None,
        True,
    ),
    'curves': (
        f'3 w 0 0 1 RG 20 20 m 60 140 100 -20 180 120 c S 1 0 0 rg {DISC} f',
        None,
        '',
        f'3 w 0 0 1 RG 20 22 m 60 142 100 -18 180 122 c S 1 0 0 rg {DISC} f',
        None,
        True,
    ),
    'uneven-matrix': (
        'q 4 0 0 1 0 0 cm 2 w 5 20 m 40 120 l S Q',
        None,
        '',
        'q 1 0 0 4 0 0 cm 2 w 20 5 m 160 30 l S Q',
        None,
        True,
    ),
    'clips': (
        f'q 30 30 140 90 re W n q {DISC} W n 1 0 0 rg 0 0 200 150 re f Q 0 0 1 rg 130 0 70 150 re f Q',
        None,
        '',
        (
            f'q 30 30 140 90 re W n q 1 0 0 1 15 0 cm {DISC} W n 1 0 0 rg -15 0 200 150 re f Q '
            '0 0 1 rg 130 0 70 150 re f Q'
        ),
        None,
        True,
    ),
    'even-odd-clip': (
        f'q {STAR} W* n 0 0.5 0 rg 0 0 200 150 re f Q',
        None,
        '',
        f'q {STAR} W n 0 0.5 0 rg 0 0 200 150 re f Q',
        None,
        True,
    ),
    'compositing': (
        COMPOSITED.format(half='/Half gs'),
        None,
        ALPHA,
        COMPOSITED.format(half=''),
        None,
        True,
    ),
    'separation-fill': (
        '/S1 cs 0.7 scn 20 20 160 110 re f',
        None,
        f'/ColorSpace << /S1 {SEPARATION} >>',
        '/S1 cs 0.4 scn 20 20 160 110 re f',
        None,
        True,
    ),
    'cmyk': (
        '0.8 0.1 0 0.1 k 20 20 160 110 re f',
        None,
        '',
        '0.8 0.1 0 0.3 k 20 20 160 110 re f',
        None,
        False,
    ),
    'rgb-image': (
        PLACE,
        {'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8', RGB.tobytes())},
        '',
        SHIFTED,
        None,
        True,
    ),
    'shrunk-image': (
        'q 30 0 0 20 40 35 cm /Im1 Do Q',
        {'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8', RGB.tobytes())},
        '',
        'q 30 0 0 20 40 37 cm /Im1 Do Q',
        None,
        True,
    ),
    'turned-image': (
        'q 60 50 -40 60 110 20 cm /Im1 Do Q',
        {'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8', RGB.tobytes())},
        '',
        'q 60 -50 40 60 110 20 cm /Im1 Do Q',
        None,
        True,
    ),
    'gray-1-bit': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 1', _bits(1, (_x // 3 + _y // 3) % 2)
            )
        },
        '',
        SHIFTED,
        None,
        True,
    ),
    'gray-4-bit': (
        PLACE,
        {'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 4', _bits(4, _x // 4))},
        '',
        SHIFTED,
        None,
        True,
    ),
    'gray-16-bit': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 16', (_x * 1000).astype('>u2').tobytes()
            )
        },
        '',
        SHIFTED,
        None,
        True,
    ),
    'decode': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0]', GRAY.tobytes()
            )
        },
        '',
        PLACE,
        {'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8', GRAY.tobytes())},
        True,
    ),
    'indexed-image': (
        PLACE,
        {
            'Im1': _image(
                INDEXED,
                _bits(4, (_x // 15) % 4),
            )
        },
        '',
        SHIFTED,
        None,
        True,
    ),
    'cmyk-image': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceCMYK /BitsPerComponent 8',
                np.stack([_x * 4, _y * 6, _x * 0, _y * 2], -1).astype(np.uint8).tobytes(),
            )
        },
        '',
        SHIFTED,
        None,
        False,
    ),
    'icc-image': (
        PLACE,
        {
            'Im1': _image('/Width 60 /Height 40 /ColorSpace [/ICCBased {P}] /BitsPerComponent 8', RGB.tobytes()),
            'P': ('<< /N 3 >>', PROFILE),
        },
        '',
        PLACE,
        {'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8', RGB.tobytes())},
        True,
    ),
    'separation-image': (
        PLACE,
        {'Im1': _image(f'/Width 60 /Height 40 /ColorSpace {SEPARATION} /BitsPerComponent 8', GRAY.tobytes())},
        '',
        SHIFTED,
        None,
        True,
    ),
    'lab-image': (
        PLACE,
        {
            'Im1': _image(
                f'/Width 60 /Height 40 /ColorSpace {LAB} /BitsPerComponent 8',
                np.stack([128 + _x, 100 + _y, 150 - _y], -1).astype(np.uint8).tobytes(),
            )
        },
        '',
        SHIFTED,
        None,
        True,
    ),
    'stencil-mask': (
        '0 0 1 rg ' + PLACE,
        {'Im1': _image('/Width 60 /Height 40 /ImageMask true', _bits(1, (_x // 6 + _y // 6) % 2))},
        '',
        '0 0 1 rg ' + PLACE,
        {'Im1': _image('/Width 60 /Height 40 /ImageMask true /Decode [1 0]', _bits(1, (_x // 6 + _y // 6) % 2))},
        True,
    ),
    'soft-mask': (
        PLACE,
        {
            'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask {S}', RGB.tobytes()),
            'S': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8', (_y * 6).astype(np.uint8).tobytes()
            ),
        },
        '',
        PLACE,
        {
            'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask {S}', RGB.tobytes()),
            'S': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0]',
                (_y * 6).astype(np.uint8).tobytes(),
            ),
        },
        True,
    ),
    'color-key-mask': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8 /Mask [0 100]', GRAY.tobytes()
            )
        },
        '',
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8 /Mask [100 255]', GRAY.tobytes()
            )
        },
        True,
    ),
    'stencil-mask-image': (
        PLACE,
        {
            'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask {M}', RGB.tobytes()),
            'M': _image('/Width 30 /Height 20 /ImageMask true', _bits(1, (_x[:20, :30] // 5) % 2)),
        },
        '',
        PLACE,
        {
            'Im1': _image('/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Mask {M}', RGB.tobytes()),
            'M': _image('/Width 30 /Height 20 /ImageMask true /Decode [1 0]', _bits(1, (_x[:20, :30] // 5) % 2)),
        },
        True,
    ),
    'jpeg': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode',
                _jpeg(RGB, pymupdf.csRGB),
            )
        },
        '',
        SHIFTED,
        None,
        True,
    ),
    'gray-jpeg': (
        PLACE,
        {
            'Im1': _image(
                '/Width 60 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode',
                _jpeg(GRAY, pymupdf.csGRAY),
            )
        },
        '',
        SHIFTED,
        None,
        True,
    ),
    'inline-image': (
        'q 120 0 0 80 40 35 cm BI /W 4 /H 2 /CS /RGB /BPC 8 ID ' + INLINE_RGB.decode('latin-1') + ' EI Q',
        None,
        '',
        'q 120 0 0 80 40 35 cm BI /W 4 /H 2 /CS /RGB /BPC 8 ID ' + INLINE_RGB[::-1].decode('latin-1') + ' EI Q',
        None,
        True,
    ),
}


def _case(name: str, *, wrong: bool) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    content, objects, resources, wrong_content, wrong_objects, managed = CASES[name]
    truth = page_pdf(content, objects, resources=resources)
    drawn = page_pdf(wrong_content, wrong_objects or objects, resources=resources) if wrong else truth
    pixels, undrawn = own(drawn, DPI)
    return pixels, reference(truth, DPI, managed=managed), undrawn


@pytest.mark.parametrize('name', CASES)
def test_the_own_renderer_draws_as_the_reference_does(name):
    pixels, truth, undrawn = _case(name, wrong=False)
    assert undrawn == {}
    mean, share = disagreement(pixels, truth)
    assert agrees(mean, share), f'{name}: mean {mean:.3f}, {share:.4%} of the inked pixels off'


@pytest.mark.parametrize('name', CASES)
def test_a_planted_wrong_render_is_reported(name):
    pixels, truth, _ = _case(name, wrong=True)
    mean, share = disagreement(pixels, truth)
    assert not agrees(mean, share), name


def test_a_region_drawn_by_the_own_renderer_is_the_reader_s_render_of_it(tmp_path):
    path = tmp_path / 'p.pdf'
    path.write_bytes(page_pdf(CASES['clips'][0]))
    raster = read_raster(path, 1, BBox(20, 20, 120, 100), dpi=DPI, renderer='own')
    whole, _ = own(path.read_bytes(), DPI)
    # The region's pixels: columns 41.67.. rounded out to 41..250, rows 41..209.
    assert raster.pixels.shape == (209 - 41, 250 - 41, 3)
    assert agrees(*disagreement(raster.pixels, whole[41:209, 41:250]))


def test_the_own_renderer_refuses_a_region_holding_text_by_name(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=150)
    page.insert_text((20, 40), 'Hello', fontsize=12)
    path = tmp_path / 't.pdf'
    doc.save(path)
    with pytest.raises(Unsupported, match='no text until its stage 2'):
        read_raster(path, 1, BBox(0, 0, 200, 150), dpi=DPI, renderer='own')
    assert read_raster(path, 1, BBox(100, 100, 200, 150), dpi=DPI, renderer='own').pixels.min() == 255


def test_the_own_renderer_draws_a_region_whose_only_text_shows_nothing(tmp_path):
    whole = BBox(0, 0, 200, 150)
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=150)
    page.draw_rect(pymupdf.Rect(10, 20, 190, 60), color=None, fill=(0, 0, 0))
    page.insert_text((20, 45), 'black on black', fontsize=12, color=(0, 0, 0))
    page.insert_text((20, 100), 'white on white', fontsize=12, color=(1, 1, 1))
    doc.save(tmp_path / 'unseen.pdf')
    assert all(s.in_background_color for s in read_pdf(tmp_path / 'unseen.pdf').spans())  # the premise
    drawn = read_raster(tmp_path / 'unseen.pdf', 1, whole, dpi=DPI, renderer='own').pixels
    assert agrees(*disagreement(drawn, read_raster(tmp_path / 'unseen.pdf', 1, whole, dpi=DPI).pixels))
    # The planted control: one span the page shows beside them, and the region is refused, counting it alone.
    page.insert_text((20, 130), 'shown', fontsize=12)
    doc.save(tmp_path / 'shown.pdf')
    with pytest.raises(Unsupported, match=r'holds 1 span\(s\)'):
        read_raster(tmp_path / 'shown.pdf', 1, whole, dpi=DPI, renderer='own')


def test_the_own_renderer_refuses_a_region_holding_what_it_does_not_draw_by_name(tmp_path):
    path = tmp_path / 's.pdf'
    path.write_bytes(page_pdf('q 0 0 100 150 re W n /Sh1 sh Q', resources=f'/Shading << /Sh1 {SHADING} >>'))
    with pytest.raises(Unsupported, match=r'shading \(stage 3\) x1'):
        read_raster(path, 1, BBox(0, 0, 200, 150), dpi=DPI, renderer='own')


def test_an_unknown_renderer_is_refused(tmp_path):
    path = tmp_path / 'p.pdf'
    path.write_bytes(page_pdf(''))
    with pytest.raises(ValueError, match='one of pymupdf, own'):
        read_raster(path, 1, BBox(0, 0, 10, 10), renderer='cairo')  # type: ignore[arg-type]
