"""PNG files written from pixels: what another reader reads back is the pixels written."""

import numpy as np
import pymupdf
import pytest

from document_lab.png import encode


@pytest.mark.parametrize('shape', [(37, 53, 3), (37, 53)])
def test_an_image_reads_back_as_written(shape):
    image = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
    pix = pymupdf.Pixmap(encode(image))
    assert (pix.width, pix.height, pix.n) == (53, 37, 1 if len(shape) == 2 else 3)
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(shape), image)


def test_an_image_neither_gray_nor_rgb_is_refused():
    with pytest.raises(ValueError, match='gray or RGB'):
        encode(np.zeros((2, 2, 4), dtype=np.uint8))


@pytest.mark.parametrize('shape', [(0, 5, 3), (5, 0), (0, 0)])
def test_an_image_without_a_pixel_is_refused(shape):
    with pytest.raises(ValueError, match='at least one pixel'):
        encode(np.zeros(shape, dtype=np.uint8))
