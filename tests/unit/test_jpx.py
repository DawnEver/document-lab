"""JPEG 2000 images drawn by the own renderer (``read_raster``), against known pixels.

The truth is the pixels themselves: each image is known samples coded losslessly by OpenJPEG at test time
(``tests.synthetic.jpx``, Pillow's plugin) -- an encoder independent of the renderer's decoder, hayro-jpeg2000 -- drawn
one sample a pixel, so what is drawn must be those samples as the dictionary maps them: through ``/Decode``, the
space the dictionary or else the data states (an embedded ICC profile: by the formulas, ``rgb_in_srgb``), and the
opacity the data carries as ``/SMaskInData`` says. Each case is paired with a planted wrong decode -- what a decoder
reading the case otherwise would draw -- that the comparison must report.

Where ISO 32000 and ``MuPDF`` read apart (measured 2026-11-27), the standard's reading is the default and
``MuPDF``'s a selectable option, each held to ``PyMuPDF`` drawing the same image: an opacity channel under
``/SMaskInData 0`` (its default) is ignored -- "If present, encoded soft-mask image information shall be ignored"
(Table 87) -- where ``MuPDF`` applies it (``jpx_opacity='mupdf'``); ``/Decode`` on an image that is no stencil mask is
ignored -- "If the image uses the JPXDecode filter and ImageMask is false, Decode shall be ignored" (Table 87, 7.4.9)
-- where ``MuPDF`` applies it (``jpx_decode='mupdf'``), but to an image in an ``Indexed`` space, whose ``/Decode`` it
ignores at every depth: an ``Indexed`` image's indices are its samples under either reading (2-bit indices
included). 16-bit samples are rounded to 8 bits, where ``MuPDF`` truncates them (a level apart at most).
"""

from __future__ import annotations

import numpy as np
import pytest

from document_lab import _pdf
from document_lab.ir import BBox
from document_lab.pdf import Unsupported, read_raster
from tests.rendering import page_pdf, reference
from tests.synthetic import icc_profile, jpx, rgb_in_srgb

_y, _x = np.mgrid[0:40, 0:60]
RGB = np.stack([_x * 4, _y * 6, (_x + _y) * 2], -1).astype(np.uint8)
GRAY = (_x * 4).astype(np.uint8)
#: Four bands of opacity across: 0, 85, 170, 255.
ALPHA = ((_x // 15) * 85).astype(np.uint8)
GRAY16 = (_x * 1111 + _y * 7).astype(np.uint16)
INDICES = ((_x // 15 + _y // 20) % 4).astype(np.uint8)
PALETTE = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
INDEXED = f'/ColorSpace [/Indexed /DeviceRGB 3 <{PALETTE.tobytes().hex()}>]'
PRIMARIES = [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)]
D65 = (0.3127, 0.3290)
PROFILE = icc_profile(primaries=PRIMARIES, white=D65, gamma=1.8)
#: Levels a drawn sample may be off its truth: color conversion's rounding.
LEVELS = 1


def _gray(samples: np.ndarray) -> np.ndarray:
    return np.repeat(samples[..., None], 3, -1)


def _over_white(color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``color`` (RGB) at ``alpha`` over white paper."""
    a = alpha[..., None] / 255.0
    return np.round(color * a + 255 * (1 - a)).astype(np.uint8)


def _premultiplied(color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.round(color * (alpha[..., None] / 255.0)).astype(np.uint8)


_PRE = _premultiplied(RGB, ALPHA)

#: name: (the image's dictionary entries, its JPEG 2000 data, the pixels it draws, the pixels a wrong decode draws).
CASES = {
    'gray': ('/ColorSpace /DeviceGray', jpx(GRAY), _gray(GRAY), _gray(GRAY[:, ::-1])),  # each row read backwards
    'rgb-codestream': ('/ColorSpace /DeviceRGB', jpx(RGB, codestream=True), RGB, RGB[..., ::-1]),  # BGR
    'space-from-data': ('', jpx(RGB), RGB, _gray(RGB[..., 0])),  # its first component as gray
    'gray-16-bit': (
        '/ColorSpace /DeviceGray',
        jpx(GRAY16),
        _gray(np.round(GRAY16 / 257).astype(np.uint8)),
        _gray((GRAY16 & 0xFF).astype(np.uint8)),  # the low byte
    ),
    'decode-ignored': ('/ColorSpace /DeviceRGB /Decode [1 0 1 0 1 0]', jpx(RGB), RGB, 255 - RGB),  # applied
    'opacity-as-soft-mask': (
        '/ColorSpace /DeviceRGB /SMaskInData 1',
        jpx(np.concatenate([RGB, ALPHA[..., None]], -1)),
        _over_white(RGB, ALPHA),
        RGB,  # its opacity ignored
    ),
    'opacity-ignored': (
        '/ColorSpace /DeviceRGB',
        jpx(np.concatenate([RGB, ALPHA[..., None]], -1)),
        RGB,
        _over_white(RGB, ALPHA),  # applied though /SMaskInData is 0
    ),
    'opacity-preblended': (
        '/ColorSpace /DeviceRGB /SMaskInData 2',
        jpx(np.concatenate([_PRE, ALPHA[..., None]], -1)),
        np.round(_PRE + 255.0 * (1 - ALPHA[..., None] / 255.0)).clip(0, 255).astype(np.uint8),
        _over_white(_PRE, ALPHA),  # premultiplied twice
    ),
    'gray-opacity': (
        '/ColorSpace /DeviceGray /SMaskInData 1',
        jpx(np.stack([GRAY, ALPHA], -1)),
        _over_white(_gray(GRAY), ALPHA),
        _gray(GRAY),
    ),
    'embedded-profile': (
        '',
        jpx(RGB, profile=PROFILE),
        rgb_in_srgb(RGB, primaries=PRIMARIES, white=D65, gamma=1.8),
        RGB,  # the profile ignored
    ),
    'indices': (
        INDEXED,
        jpx(INDICES),
        PALETTE[INDICES],
        _gray(INDICES),  # the indices read as gray levels
    ),
    'two-bit-indices-decode-ignored': (
        f'{INDEXED} /Decode [3 0]',
        jpx(INDICES, depth=2),
        PALETTE[INDICES],
        PALETTE[3 - INDICES],  # /Decode applied over 0..3
    ),
}

#: name: as :data:`CASES`, drawn with MuPDF's readings (``jpx_decode`` and ``jpx_opacity`` ``'mupdf'``): what PyMuPDF
#: draws.
MUPDF_CASES = {
    'decode-applied': ('/ColorSpace /DeviceRGB /Decode [1 0 1 0 1 0]', jpx(RGB), 255 - RGB, RGB),  # ignored
    'opacity-applied': (
        '/ColorSpace /DeviceRGB',
        jpx(np.concatenate([RGB, ALPHA[..., None]], -1)),
        _over_white(RGB, ALPHA),
        RGB,  # ignored, as /SMaskInData 0 says
    ),
    'two-bit-indices-decode-ignored': (
        f'{INDEXED} /Decode [3 0]',
        jpx(INDICES, depth=2),
        PALETTE[INDICES],
        PALETTE[np.full_like(INDICES, 3)],  # /Decode applied over 0..255, the samples taken as 8 bits (O48)
    ),
    'indices-decode-ignored': (
        f'{INDEXED} /Decode [3 0]',
        jpx(INDICES),
        PALETTE[INDICES],
        PALETTE[3 - INDICES],  # /Decode applied as stated
    ),
}


def _page(entries: str, data: bytes) -> bytes:
    """A page placing the image (60 x 40 samples, ``entries``, ``data``) at its top left, one sample a point."""
    image = f'<< /Type /XObject /Subtype /Image /Width 60 /Height 40 {entries} /Filter /JPXDecode >>'
    return page_pdf('q 60 0 0 40 0 110 cm /Im1 Do Q', {'Im1': (image, data)})


def _drawn(tmp_path, entries: str, data: bytes, *, mupdf: bool = False) -> np.ndarray:
    """The image of :func:`_page` read at 72 dpi by the own renderer -- with MuPDF's readings when ``mupdf``."""
    path = tmp_path / 'jpx.pdf'
    path.write_bytes(_page(entries, data))
    reading = 'mupdf' if mupdf else 'standard'
    return read_raster(path, 1, BBox(0, 0, 60, 40), dpi=72, jpx_decode=reading, jpx_opacity=reading).pixels


def _off(drawn: np.ndarray, truth: np.ndarray) -> int:
    """The most levels a drawn sample is off ``truth``."""
    return int(np.abs(drawn.astype(int) - truth.astype(int)).max())


@pytest.mark.parametrize('name', CASES)
def test_a_jpx_image_draws_its_known_pixels(tmp_path, name):
    entries, data, truth, _ = CASES[name]
    assert _off(_drawn(tmp_path, entries, data), truth) <= LEVELS


@pytest.mark.parametrize('name', CASES)
def test_a_planted_wrong_jpx_decode_is_reported(tmp_path, name):
    entries, data, _, wrong = CASES[name]
    assert _off(_drawn(tmp_path, entries, data), wrong) > 8 * LEVELS, name


@pytest.mark.parametrize('name', MUPDF_CASES)
def test_mupdf_s_jpx_readings_draw_what_pymupdf_draws(tmp_path, name):
    entries, data, truth, wrong = MUPDF_CASES[name]
    drawn = _drawn(tmp_path, entries, data, mupdf=True)
    assert _off(drawn, truth) <= LEVELS
    # The reading is MuPDF's: PyMuPDF draws the same truth (color management off: the samples as they stand).
    assert _off(reference(_page(entries, data), 72, managed=False)[:40, :60], truth) <= LEVELS
    # Planted wrong reading.
    assert _off(drawn, wrong) > 8 * LEVELS, name


def test_a_jpx_reading_other_than_the_standard_s_or_mupdf_s_is_refused(tmp_path):
    path = tmp_path / 'jpx.pdf'
    path.write_bytes(_page('/ColorSpace /DeviceGray', jpx(GRAY)))
    with pytest.raises(ValueError, match=r"jpx_decode 'acrobat': one of standard, mupdf"):
        read_raster(path, 1, BBox(0, 0, 60, 40), dpi=72, jpx_decode='acrobat')  # type: ignore[arg-type]


def test_a_jpx_soft_mask_gives_its_samples_their_alpha(tmp_path):
    # The InDesign corpus file's shape: an image whose /SMask is a JPEG 2000 gray image.
    entries = '/ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask {S}'
    image = f'<< /Type /XObject /Subtype /Image /Width 60 /Height 40 {entries} >>'
    mask = '<< /Type /XObject /Subtype /Image /Width 60 /Height 40 /ColorSpace /DeviceGray /Filter /JPXDecode >>'
    path = tmp_path / 'masked.pdf'
    path.write_bytes(
        page_pdf('q 60 0 0 40 0 110 cm /Im1 Do Q', {'Im1': (image, RGB.tobytes()), 'S': (mask, jpx(ALPHA))})
    )
    drawn = read_raster(path, 1, BBox(0, 0, 60, 40), dpi=72).pixels
    assert _off(drawn, _over_white(RGB, ALPHA)) <= LEVELS
    # Planted wrong decode: the mask ignored.
    assert _off(drawn, RGB) > 8 * LEVELS


def test_a_jpx_image_whose_size_is_not_its_dictionary_s_is_refused_by_name(tmp_path):
    with pytest.raises(Unsupported, match=r'JPXDecode image of 30x40 samples where its dictionary states 60x40 x1'):
        _drawn(tmp_path, '/ColorSpace /DeviceGray', jpx(GRAY[:, :30]))


@pytest.mark.parametrize(
    ('data', 'family', 'components'),
    [(jpx(RGB), 'DeviceRGB', 3), (jpx(GRAY), 'DeviceGray', 1), (jpx(RGB, profile=PROFILE), 'ICCBased', 3)],
)
def test_the_reader_states_the_space_a_jpx_image_s_data_states_when_its_dictionary_states_none(
    tmp_path, data, family, components
):
    image = '<< /Type /XObject /Subtype /Image /Width 60 /Height 40 /Filter /JPXDecode >>'
    pdf = page_pdf('q 60 0 0 40 0 110 cm /Im1 Do Q', {'Im1': (image, data)})
    reader = _pdf.open(pdf, b'')
    (row,) = reader.page(1, annotations=False, curve_samples=1)[2]
    _width, _height, space, count, _bits, stored, codec, *_ = reader.image(row[3])
    # Its space read off the header; its samples handed over still encoded (D3).
    assert (space, count, codec, stored) == (family, components, 'JPXDecode', data)
