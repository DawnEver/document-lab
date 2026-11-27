"""Drawn ink found in a page's pixels apart from its text: its boxes, the rules it is drawn with, what lies off them."""

import numpy as np
import pytest

from document_lab.ink import Rulings, drawings, rulings, skew
from document_lab.ir import BBox
from tests.synthetic import draw_ruled_table, scan_image


def _page(h=200, w=300):
    return np.full((h, w), 255, dtype=np.uint8)


def test_a_ruled_grid_reads_as_its_rules_with_its_words_off_them_small():
    px = _page()
    for y in (20, 50, 80):
        px[y, 40:241] = 0
    for x in (40, 140, 240):
        px[20:81, x] = 0
    px[23:30, 43:50] = 0  # a word OCR did not read, in a cell, within reach of its rules
    (grid,) = drawings(px, 1.0, [], reach=3.0)
    assert grid.rules == (3, 3)
    assert grid.loose <= 8
    assert grid.box == BBox(40, 20, 241, 81)


def test_a_curve_is_no_rule_and_lies_off_them_large():
    px = _page()
    for x in range(40, 240):
        y = int(150 - 100 * ((x - 40) / 200) ** 2)
        px[y : y + 2, x] = 0
    (curve,) = drawings(px, 1.0, [], reach=3.0)
    assert curve.loose > 50
    assert curve.stroke < 50  # a slant has no long straight run


def test_the_text_boxes_ink_is_no_drawing_and_far_ink_is_two():
    px = _page()
    px[30:40, 30:90] = 0  # words
    px[100, 30:200] = 0  # a rule
    px[160, 30:200] = 0  # another, far below
    found = drawings(px, 1.0, [BBox(28, 28, 92, 42)], reach=3.0)
    assert [d.box.y0 for d in found] == [100, 160]
    assert all(d.stroke >= 170 for d in found)


# Rulings: a picture's rules as boxes (points) in the frame its skew turns back. Pictures here are drawn pixel by
# pixel at 4 px/pt, so every rule's place, length and thickness is known.

_SCALE = 4.0


def _sheet(points=(300, 400)):
    """A white picture ``points`` (width, height) at :data:`_SCALE` px/pt."""
    return np.full((points[1] * 4, points[0] * 4), 255, dtype=np.uint8)


def _line(px, start, end, *, width=2, grey=0):
    """A straight line from ``start`` to ``end`` (points), ``width`` pixels thick, drawn by sampling it."""
    (x0, y0), (x1, y1) = start, end
    steps = int(max(abs(x1 - x0), abs(y1 - y0)) * _SCALE * 2) + 1
    for t in np.linspace(0, 1, steps):
        x, y = (x0 + t * (x1 - x0)) * _SCALE, (y0 + t * (y1 - y0)) * _SCALE
        if abs(x1 - x0) >= abs(y1 - y0):
            px[int(y) : int(y) + width, int(x)] = grey
        else:
            px[int(y), int(x) : int(x) + width] = grey


def _turn(x, y, degrees, center=(150, 200)):
    a = np.radians(degrees)
    cx, cy = center
    return cx + (x - cx) * np.cos(a) - (y - cy) * np.sin(a), cy + (x - cx) * np.sin(a) + (y - cy) * np.cos(a)


def _grid(px, degrees=0.0, **kw):
    """A grid of three level and three upright rules over (50..250, 100..200) points, turned ``degrees``."""
    for y in (100, 150, 200):
        _line(px, _turn(50, y, degrees), _turn(250, y, degrees), **kw)
    for x in (50, 150, 250):
        _line(px, _turn(x, 100, degrees), _turn(x, 200, degrees), **kw)


def _level(found):
    return sorted((round((b.y0 + b.y1) / 2, 1), round(b.x0), round(b.x1)) for b in found.rules if b.width >= b.height)


def _upright(found):
    return sorted((round((b.x0 + b.x1) / 2, 1), round(b.y0), round(b.y1)) for b in found.rules if b.width < b.height)


def test_a_grid_of_rules_reads_as_its_level_and_upright_boxes_with_no_skew():
    px = _sheet()
    _grid(px)
    found = rulings(px, _SCALE)
    assert found.skew == pytest.approx(0.0, abs=0.01)
    assert _level(found) == [(100.2, 50, 250), (150.2, 50, 250), (200.2, 50, 250)]
    assert _upright(found) == [(50.2, 100, 200), (150.2, 100, 200), (250.2, 100, 200)]
    assert all(0.25 <= min(b.width, b.height) <= 0.75 for b in found.rules)  # 2 px thick: half a point


@pytest.mark.parametrize('degrees', [0.8, -1.5, 2.5])
def test_a_skewed_grid_reads_level_and_upright_in_the_frame_its_measured_skew_turns_back(degrees):
    px = _sheet()
    _grid(px, degrees)
    found = rulings(px, _SCALE)
    assert found.skew == pytest.approx(degrees, abs=0.05)
    for got, want in zip(_level(found), [100, 150, 200], strict=True):
        assert got[0] == pytest.approx(want + 0.2, abs=0.6)
        assert (got[1], got[2]) == pytest.approx((50, 250), abs=1.5)
    assert [u[0] for u in _upright(found)] == pytest.approx([50.2, 150.2, 250.2], abs=0.6)


def test_a_skew_past_the_limit_is_no_rule_at_all_and_the_limit_is_an_option():
    px = _sheet()
    _grid(px, 4.0)
    assert rulings(px, _SCALE) == Rulings(None, ())  # a slant of 4 degrees is no rule's
    assert rulings(px, _SCALE, skew_limit=5.0).skew == pytest.approx(4.0, abs=0.05)


def test_a_rule_broken_within_the_gap_is_one_rule_and_a_wider_break_parts_it():
    px = _sheet()
    _line(px, (50, 100), (250, 100))
    px[:, int(120 * _SCALE) : int(122 * _SCALE)] = 255  # a 2 pt break
    assert _level(rulings(px, _SCALE)) == [(100.2, 50, 250)]
    assert _level(rulings(px, _SCALE, gap=1.0)) == [(100.2, 50, 120), (100.2, 122, 250)]


def test_specks_dropped_from_a_thin_rule_and_a_rule_grey_across_two_rows_still_rule():
    px = _sheet()
    _line(px, (50, 100), (250, 100), width=1)
    rng = np.random.default_rng(0)
    px[400, rng.choice(np.arange(200, 1000), 40, replace=False)] = 255  # one-pixel dropouts
    _line(px, (50, 150), (250, 150), width=2, grey=150)  # drawn between two rows: grey in both, dark in neither
    assert [r[0] for r in _level(rulings(px, _SCALE))] == pytest.approx([100.1, 150.2], abs=0.2)


def test_glyphs_are_no_rules_nor_are_specks_or_a_filled_block():
    """Glyphs' strokes are shorter than a rule. A longer stem, or a word's strokes specks run together, is one here,
    which `document_lab.tables` finds within its word's box."""
    glyphs = _sheet()
    for row in range(10):  # lines of 'glyphs': stems 4 pt high, bars 2 pt wide, 1 pt apart
        for col in range(60):
            y, x = int((40 + 12 * row) * _SCALE), int((20 + 3 * col) * _SCALE)
            glyphs[y : y + 16, x : x + 2] = 0
            glyphs[y + 8 : y + 10, x : x + 8] = 0
    assert rulings(glyphs, _SCALE) == Rulings(None, ())
    specks = _sheet()
    specks[np.random.default_rng(1).random(specks.shape) < 0.003] = 0
    specks[1300:1500, 400:900] = 0  # a filled block, far thicker than a rule
    assert rulings(specks, _SCALE) == Rulings(None, ())


def test_a_diagonal_is_no_rule_and_a_dashed_line_is_none_but_as_its_cover_option_says():
    px = _sheet()
    _grid(px)
    _line(px, (60, 110), (240, 190))  # a slant across the grid
    found = rulings(px, _SCALE)
    assert len(found.rules) == 6
    for x in range(50, 250, 6):  # beside it, a dashed line: 4 pt on, 2 pt off
        _line(px, (x, 300), (x + 4, 300))
    assert [r[0] for r in _level(rulings(px, _SCALE))] == pytest.approx([100.2, 150.2, 200.2], abs=0.2)
    assert (300.2, 50, 252) in _level(rulings(px, _SCALE, cover=0.6))


def test_thickness_bounds_a_rule_and_is_an_option():
    px = _sheet()
    _line(px, (50, 100), (250, 100), width=20)  # 5 pt thick
    assert rulings(px, _SCALE) == Rulings(None, ())
    (bar,) = rulings(px, _SCALE, thickness=6.0).rules
    assert bar.height == pytest.approx(5.0, abs=0.3)


def test_rgb_and_grey_pictures_read_the_same():
    px = _sheet()
    _grid(px, 1.0)
    assert rulings(np.repeat(px[..., None], 3, axis=2), _SCALE) == rulings(px, _SCALE)


# Skew: the angle a picture's lines -- of text and rules alike -- stand turned, from synthetic scans of known angle.


def _prose(page):
    for i in range(14):
        page.insert_text(
            (72, 90 + 18 * i), 'The rated output of a machine is the value included in its rating', fontsize=10
        )


def _sideways(page):
    for i in range(10):
        page.insert_text((300 + 16 * i, 700), 'The rated output of a machine is its rating', fontsize=10, rotate=90)


@pytest.mark.parametrize('degrees', [0.0, 1.0, 3.0, -5.0, 8.0])
@pytest.mark.parametrize(
    'draws', [(_prose,), (_prose, draw_ruled_table), (_sideways,)], ids=['prose', 'ruled', 'sideways']
)
def test_a_pictures_skew_is_told_by_its_lines_of_text_and_rules_level_or_sideways(draws, degrees):
    found = skew(scan_image(*draws, dpi=100, angle=degrees, noise=0.5, seed=4))
    assert found == pytest.approx(degrees, abs=0.15)  # a pixel across a short line's length at 100 dpi
    assert degrees == 0 or abs(found + degrees) > 1  # a planted wrong truth: the skew the other way


def test_no_lines_tell_no_skew_and_the_search_is_an_option():
    assert skew(np.full((300, 400), 255, dtype=np.uint8)) is None  # blank
    assert skew(np.random.default_rng(0).integers(0, 256, (300, 400), dtype=np.uint8)) is None  # noise
    steep = scan_image(_prose, dpi=100, angle=12.0)
    assert skew(steep) != pytest.approx(12.0, abs=0.5)  # beyond the search
    assert skew(steep, search=15.0) == pytest.approx(12.0, abs=0.05)
