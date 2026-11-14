"""Digits read from glyph shapes, and the shapes the reader must refuse rather than guess (the planted controls)."""

import numpy as np
import pymupdf
import pytest

from document_lab.glyphs import STROKE, Font, Glyph, UnreadableGlyph, read_glyph, read_number, read_word, stroke
from document_lab.strokefont import TEMPLATES


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


def _lines(char: str) -> list[list[tuple[float, float]]]:
    return [
        [tuple(float(v) / 100 for v in p.split(',')) for p in line.split()] for line in TEMPLATES[char][2].split(';')
    ]


def _stroked(char: str, pieces: int = 3, width: float | None = None) -> np.ndarray:
    """A stroke-font character's ink, its polylines broken into ``pieces`` hairlines, as the SIFERRIT sheets draw it."""
    broken = []
    for line in _lines(char):
        step = max(1, -(-(len(line) - 1) // pieces))
        broken += [line[k : k + step + 1] for k in range(0, max(1, len(line) - 1), step)]
    return stroke(broken, TEMPLATES[char][1] if width is None else width)


@pytest.mark.parametrize('char', sorted(TEMPLATES))
def test_each_stroke_font_character_reads_as_itself(char):
    """TDK SIFERRIT N27/N30/N41/N72 draw every label in this font: each character, broken into hairlines, reads back."""
    ink = _stroked(char)
    assert read_word([Glyph(ink, (0, 0, 4, 7))], STROKE) == char
    if char.isdigit():
        assert read_glyph(ink, STROKE) == char


class _Variant(Font):
    """The stroke font with its letters changed by ``letters``: a font a test plants a defect in."""

    def __init__(self, letters):
        self.name = 'variant'
        self._letters = letters

    def digits(self, rows):
        return STROKE.digits(rows)

    def letters(self, rows):
        return self._letters(STROKE.letters(rows))

    def ascent(self):
        return STROKE.ascent()

    def x_height(self):
        return STROKE.x_height()


def test_a_stroked_letter_nearer_than_any_digit_is_no_digit():
    """SIFERRIT N22 p3: the ``C`` of ``°C`` on the label row read as ``0``, the nearest digit and clear of the next.

    A heavier ``C`` is such a glyph: among digits alone it reads as ``0`` (the planted precondition), but the
    letter ``C`` is nearer than any digit, so it is no digit.
    """
    bold_c = _stroked('C', width=0.085)  # of its height: the template's hairline is 0.057
    digits_only = _Variant(lambda refs: {c: r for c, r in refs.items() if c.isdigit()})
    assert read_glyph(bold_c, digits_only) == '0'
    with pytest.raises(UnreadableGlyph, match="the letter 'C'"):
        read_glyph(bold_c, STROKE)


def test_a_stroked_glyph_of_no_character_is_refused():
    """The control: a spiral is no character of the stroke font, so it is refused rather than read as its nearest."""
    spiral = [(0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1.5), (2, -1.5), (2, 2.5), (-2, 2.5), (-2, -3.5), (2, -3.5)]
    with pytest.raises(UnreadableGlyph):
        read_word([Glyph(stroke([spiral], 0.4), (0, 0, 4, 7))], STROKE)


def test_a_character_the_font_draws_as_another_is_refused_as_a_twin():
    """A clear match says nothing when the font draws another character within ``TWIN`` of it: an ``O`` 0.9 off."""

    def with_o(refs):
        variants, place = refs['0']
        return {**refs, 'O': ([(f, a + 0.45) for f, a in variants], place)}  # aspect weight 2: 0.9 apart

    zero = [Glyph(_stroked('0'), (0, 0, 4, 7))]
    with pytest.raises(UnreadableGlyph, match='within'):
        read_word(zero, _Variant(with_o))
    assert read_word(zero, STROKE) == '0'  # the control: without the twin, a 0
