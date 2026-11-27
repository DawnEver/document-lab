"""A JBIG2 image whose generic region is MMR-coded, drawn by the own renderer (``read_raster``).

The truth is known pixels, coded at test time by an encoder independent of the renderer's decoder (hayro-jbig2):
an MMR generic region's data is T.6 (T.88 6.2.6), here MuPDF's Group 4 coding (``tests.synthetic.jbig2_mmr``), so
the image must draw those pixels. The arithmetic-coded regions are checked in ``crates/render/tests/codecs.rs``
against jbig2enc-rust's coding; that encoder writes no MMR generic region.
"""

from __future__ import annotations

import numpy as np

from document_lab.ir import BBox
from document_lab.pdf import read_raster
from tests.rendering import page_pdf
from tests.synthetic import jbig2_mmr

_y, _x = np.mgrid[0:48, 0:80]
#: Runs of every length across the rows, and solid rows: 1 white, 0 black.
WHITE = (((_x * (_y % 7 + 1)) // 9 + _y // 12) % 2).astype(np.uint8)
WHITE[20:23] = 0


def _drawn(tmp_path, entries: str, data: bytes) -> np.ndarray:
    """The image (80 x 48 samples, ``entries``) placed at the page's top left one sample a point, read at 72 dpi by
    the own renderer, as gray levels."""
    image = f'<< /Type /XObject /Subtype /Image /Width 80 /Height 48 {entries} /Filter /JBIG2Decode >>'
    path = tmp_path / 'jbig2.pdf'
    path.write_bytes(page_pdf('q 80 0 0 48 0 102 cm /Im1 Do Q', {'Im1': (image, data)}))
    pixels = read_raster(path, 1, BBox(0, 0, 80, 48), dpi=72).pixels
    assert (pixels == pixels[..., :1]).all()  # gray
    return pixels[..., 0]


def test_an_mmr_generic_region_draws_its_known_pixels(tmp_path):
    drawn = _drawn(tmp_path, '/ColorSpace /DeviceGray /BitsPerComponent 1', jbig2_mmr(WHITE))
    assert (drawn == WHITE * 255).all()
    # Planted wrong decodes: the samples inverted, and drawn a row low.
    assert (drawn != (1 - WHITE) * 255).any()
    assert (drawn != np.roll(WHITE, 1, axis=0) * 255).any()


def test_an_mmr_generic_region_maps_through_its_decode_array(tmp_path):
    drawn = _drawn(tmp_path, '/ColorSpace /DeviceGray /BitsPerComponent 1 /Decode [1 0]', jbig2_mmr(WHITE))
    assert (drawn == (1 - WHITE) * 255).all()
    # Planted wrong decode: /Decode ignored.
    assert (drawn != WHITE * 255).any()
