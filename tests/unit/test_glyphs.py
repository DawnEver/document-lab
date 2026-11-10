"""Digits read from glyph shapes, and the shapes the reader must refuse rather than guess (the planted controls)."""

import numpy as np
import pymupdf
import pytest

from document_lab.glyphs import Glyph, UnreadableGlyph, read_glyph, read_number


def _ink(text: str, *, size: float = 24, font: str = 'helv', dpi: int = 72) -> np.ndarray:
    with pymupdf.open() as doc:
        page = doc.new_page(width=40 * len(text) + 20, height=60)
        page.insert_text((10, 40), text, fontname=font, fontsize=size)
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return 1 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w) / 255  # darkness


@pytest.mark.parametrize('digit', list('0123456789'))
def test_each_digit_reads_at_label_size_in_pixels(digit):
    assert read_glyph(_ink(digit, size=12, dpi=130)) == digit  # about 16 px high


@pytest.mark.parametrize('letter', list('ASFHmBeg'))
def test_a_letter_is_refused_not_read_as_its_nearest_digit(letter):  # an O IS a 0 to a shape reader
    with pytest.raises(UnreadableGlyph):
        read_glyph(_ink(letter))


def test_a_glyph_too_few_pixels_high_is_refused():
    with pytest.raises(UnreadableGlyph, match='px high'):
        read_glyph(_ink('5', size=12, dpi=85))  # about 11 px high: a 6 and an 8 come too close to tell


def test_blank_ink_is_refused():
    with pytest.raises(UnreadableGlyph):
        read_glyph(np.zeros((20, 10), dtype=bool))


def _box(x0, y0, x1, y1):
    return Glyph(np.ones((max(1, round(y1 - y0)), max(1, round(x1 - x0))), dtype=bool), (x0, y0, x1, y1))


def test_a_short_wide_mark_at_mid_height_is_a_leading_minus_and_a_low_one_a_point():
    two, five = _ink('2'), _ink('5')
    rows = two.shape[0]
    digits = [Glyph(two, (10, 0, 20, rows)), Glyph(five, (30, 0, 40, rows))]
    minus = _box(2, rows / 2 - 1, 8, rows / 2 + 1)
    point = _box(23, rows - 3, 26, rows)
    assert read_number([minus, *digits]) == -25
    assert read_number([digits[0], point, digits[1]]) == 2.5
    with pytest.raises(UnreadableGlyph, match='not a number'):
        read_number([digits[0], _box(21, rows / 2 - 1, 27, rows / 2 + 1), digits[1]])


@pytest.mark.parametrize('font', ['tiro', 'cour'])
def test_digits_of_another_font_are_refused_or_read_right_never_misread(font):
    for digit in '0123456789':
        try:
            assert read_glyph(_ink(digit, size=12, dpi=200, font=font)) == digit
        except UnreadableGlyph:
            continue
