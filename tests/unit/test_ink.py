"""Drawn ink found in a page's pixels apart from its text: its boxes, the rules it is drawn with, what lies off them."""

import numpy as np

from document_lab.ink import drawings
from document_lab.ir import BBox


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
