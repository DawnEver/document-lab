"""Equation crops (`document_lab.regions.equations`): one display equation and its number, nothing else -- read off
synthetic pages whose boxes are known.

Each layout is one the real papers set (and the old crop got wrong): an equation across a one-column page, one across
both columns of a two-column page, a paragraph's short last line just above a formula, one number for two rows, two
equations stacked close, a formula beside its number in a column, a page number at the page's head. Every check is
paired with a planted wrong box it must report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymupdf

from document_lab import layout
from document_lab.ir import BBox
from document_lab.pdf import read_pdf
from document_lab.regions import equations, regions
from tests.synthetic import Run, text_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from document_lab.ir import Document, Page

_TEXT = (
    'the rated output is the mechanical power available at the shaft and shall be expressed in watts while the '
    'machine runs at its rated speed under the stated conditions of supply cooling and duty for the whole test'
)


def prose(x: float, y: float, width: float, start: int = 0) -> Run:
    """A line of prose (Helvetica 10) from ``x`` as wide as ``width`` allows: words, nothing a formula holds."""
    words: list[str] = []
    every = _TEXT.split()
    for word in every[start:] + every[:start]:
        if pymupdf.get_text_length(' '.join([*words, word]), 'helv', 10) > width:
            break
        words.append(word)
    return (x, y, ' '.join(words), 'helv', 10)


def column(x: float, width: float, ys: range | list[float]) -> list[Run]:
    return [prose(x, y, width, start=i) for i, y in enumerate(ys)]


def math(x: float, y: float, text: str, size: float = 10) -> Run:
    return (x, y, text, 'tiit', size)


def number(x: float, y: float, n: int) -> Run:
    return (x, y, f'({n})', 'helv', 10)


def read(tmp_path: Path, *pages: list[Run], bars: tuple[list[pymupdf.Rect], ...] = ()) -> Document:
    return read_pdf(text_pdf(tmp_path / 'page.pdf', pages, bars))


def numbered(page: Page, n: int | str, face: tuple[str, float]) -> BBox | None:
    """The box of the equation numbered ``(n)`` on ``page``, as `document_lab.regions.equations` finds it."""
    found = [r.bbox for r in equations(page, face) if r.label == f'({n})']
    return found[0] if found else None


def crop(doc: Document, n: int | str, page: int = 0) -> BBox | None:
    return numbered(doc.pages[page], n, layout.body_face(doc))


def faults(page: Page, box: BBox | None, formula: list[Run]) -> list[str]:
    """How ``box`` departs from holding exactly ``formula`` (its runs, the number among them): a run cut, one let in."""
    if box is None:
        return ['no crop']
    found = []
    for span in page.spans:
        b = span.provenance.bbox
        inside = box.x0 <= b.x0 and b.x1 <= box.x1 and box.y0 <= b.y0 and b.y1 <= box.y1
        meets = b.x0 < box.x1 and box.x0 < b.x1 and b.y0 < box.y1 and box.y0 < b.y1
        held = any(span.text == text and abs(b.x0 - x) < 1 and b.y0 < y < b.y1 for x, y, text, _, _ in formula)
        if held and not inside:
            found.append(f'{span.text!r} cut')
        elif not held and meets:
            found.append(f'{span.text!r} let in')
    return found


def without(box: BBox, left: float = 0.0, top: float = 0.0, right: float = 0.0, bottom: float = 0.0) -> BBox:
    """``box`` moved in from its edges by so many points (negative: grown): a planted wrong crop."""
    return BBox(box.x0 + left, box.y0 + top, box.x1 - right, box.y1 - bottom)


def test_an_equation_across_a_one_column_page_is_cropped_whole(tmp_path):
    """Its left part stands left of the page middle, far from its number: it is the equation's all the same."""
    formula = [math(150, 220, 'P = R i'), math(330, 220, 'a + b + c'), number(525, 220, 1)]
    doc = read(tmp_path, [*column(72, 468, range(100, 200, 12)), *formula, *column(72, 468, range(250, 330, 12))])
    box = crop(doc, 1)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=100), formula)  # planted: its left part cut off


def test_an_equation_across_both_columns_of_a_two_column_page_is_cropped_whole(tmp_path):
    formula = [math(110, 420, 'W = sum k a b'), math(330, 420, 'c d + e f g'), number(545, 420, 2)]
    columns = [*column(50, 250, range(100, 400, 12)), *column(312, 250, range(100, 400, 12))]
    below = [*column(50, 250, range(450, 600, 12)), *column(312, 250, range(450, 600, 12))]
    doc = read(tmp_path, [*columns, *formula, *below])
    box = crop(doc, 2)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=230), formula)  # planted: the left column's half cut off


def test_an_equation_in_a_column_takes_nothing_of_the_other(tmp_path):
    """A formula beside its number in the right column; prose runs on in the left column beside it."""
    lead = (312, 288, 'sentence, as in:', 'helv', 10)
    formula = [math(417, 300, 'a + b = g'), number(551, 300, 1)]
    left = column(50, 250, range(100, 400, 12))
    right = [*column(312, 250, range(100, 280, 12)), lead, *column(312, 250, range(320, 400, 12))]
    doc = read(tmp_path, [*left, *right, *formula])
    box = crop(doc, 1)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=-150), formula)  # planted: grown into the left column


def test_a_paragraphs_short_last_line_above_a_formula_is_left_out(tmp_path):
    tail = (50, 196, 'values [13].', 'helv', 10)
    formula = [math(130, 210, 'w = x + y'), number(288, 210, 4)]
    left = [*column(50, 250, range(100, 190, 12)), tail, *formula, *column(50, 250, range(230, 330, 12))]
    doc = read(tmp_path, left)
    box = crop(doc, 4)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=-80, top=-10), formula)  # planted: grown over its tail


def test_one_number_for_two_rows_takes_both(tmp_path):
    """Two rows of fractions, the number centered between them, the rows close: one display."""
    rows = [
        math(150, 200, '1', 7),
        math(127, 207, 'w1 ='),
        math(150, 214, 'T avg', 7),
        math(150, 223, '1', 7),
        math(127, 230, 'w2 ='),
        math(150, 237, 'L cu', 7),
    ]
    formula = [*rows, number(288, 219, 4)]
    bars = [pymupdf.Rect(148, 208, 170, 208.6), pymupdf.Rect(148, 231, 170, 231.6)]
    left = [*column(50, 250, range(100, 190, 12)), *formula, *column(50, 250, range(255, 340, 12))]
    doc = read(tmp_path, left, bars=(bars,))
    box = crop(doc, 4)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, bottom=15), formula)  # planted: the second row cut off


def test_two_equations_stacked_close_are_cropped_apart(tmp_path):
    first = [math(120, 210, 'A = k V cos a', 9), math(140, 222, 'w (1 - k)', 9), number(274, 216, 7)]
    second = [math(120, 233, 'B = k V sin b', 9), math(140, 245, 'w (1 + k)', 9), number(274, 239, 8)]
    left = [*column(50, 250, range(100, 200, 12)), *first, *second, *column(50, 250, range(262, 340, 12))]
    doc = read(tmp_path, left)
    assert faults(doc.pages[0], crop(doc, 7), first) == []
    assert faults(doc.pages[0], crop(doc, 8), second) == []
    assert faults(doc.pages[0], without(crop(doc, 7), bottom=-12), first)  # planted: grown into equation (8)


def test_a_running_page_number_is_no_part_of_an_equation_at_the_head_of_a_page(tmp_path):
    """Each page's number at its head's right; an equation set right under it, on the page without its running lines."""
    pages = []
    for n, y in ((1, 140), (2, 62), (3, 200)):  # numbers at their equations' places, not lined up as a running line
        formula = [math(200, y, f'u = R i + L d{n}'), number(525, y, n)]
        pages.append([(530, 48, str(70 + n), 'helv', 10), *formula, *column(72, 468, range(90 + y, 300 + y, 12))])
    doc = read(tmp_path, *pages)
    body = layout.without_running(doc)
    box = numbered(body.pages[1], 2, layout.body_face(doc))
    formula = [math(200, 62, 'u = R i + L d2'), number(525, 62, 2)]
    assert faults(doc.pages[1], box, formula) == []
    assert faults(doc.pages[1], crop(doc, 2, page=1), formula)  # planted: the page with its running lines


def test_a_reference_in_prose_is_no_equation_number_and_two_numbers_refuse(tmp_path):
    reference = (72, 112, 'the power is given in (3)', 'helv', 10)
    formula = [math(200, 220, 'P = T w'), number(525, 220, 3)]
    doc = read(tmp_path, [*column(72, 468, range(100, 112, 12)), reference, *formula])
    assert faults(doc.pages[0], crop(doc, 3), formula) == []
    twice = read(tmp_path, [*formula, math(200, 400, 'Q = T v'), number(525, 400, 3)])
    assert crop(twice, 3) is None  # planted: a number printed twice, each beside a formula, refuses, never a guess


def test_a_formula_row_set_flush_left_is_no_prose(tmp_path):
    """A display broken over lines opens flush left with a variable's letters (``cos hweTs``); its number ends it."""
    formula = [math(50, 210, 'cos hweTs'), math(64, 224, '= 1 - x2 + x4 - x6'), number(288, 238, 25)]
    left = [*column(50, 250, range(100, 200, 12)), *formula, *column(50, 250, range(256, 340, 12))]
    doc = read(tmp_path, left)
    box = crop(doc, 25)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, top=12), formula)  # planted: its first row cut off


def test_a_heading_right_below_an_equation_is_left_out(tmp_path):
    """An italic subsection heading, its words hyphenated, set just under the equation."""
    heading = (58, 222, '3) Context-Based Re-Exploration', 'tiit', 10)
    formula = [math(50, 208, 'a = b, k = 1, N'), number(288, 208, 25)]
    left = [*column(50, 250, range(100, 196, 12)), *formula, heading, *column(50, 250, range(236, 340, 12))]
    doc = read(tmp_path, left)
    box = crop(doc, 25)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, bottom=-12), formula)  # planted: grown over the heading


def test_a_number_beyond_the_prose_takes_nothing_of_the_next_equation(tmp_path):
    """Equation (12), a line of prose, then a tall matrix whose number (13) stands below it: its top row is nearer
    (12) than (13), but the prose between parts them."""
    first = [math(100, 100, 'p = M q'), number(275, 100, 12)]
    lead = (50, 120, 'The matrix is', 'helv', 10)
    rows = [math(100, y, f'r{i} 0 0 L{i}') for i, y in enumerate((136, 151, 166, 181))]
    second = [math(60, 158, 'L ='), *rows, number(275, 196, 13)]
    left = [*column(50, 250, range(60, 90, 12)), *first, lead, *second, *column(50, 250, range(214, 300, 12))]
    doc = read(tmp_path, left)
    box = crop(doc, 13)
    assert faults(doc.pages[0], box, second) == []
    assert faults(doc.pages[0], without(box, top=16), second)  # planted: the matrix's top row cut off


def test_a_wide_formula_whose_letters_run_together_is_no_prose(tmp_path):
    """Letters run into each other across a formula (``Ie5jwet``) read as no words, however wide it is set."""
    formula = [math(60, 210, 'T = (k1 - k3) Ie5jwet + (k2 - k4) Ie7jwet + c'), number(288, 210, 19)]
    left = [*column(50, 250, range(100, 196, 12)), *formula, *column(50, 250, range(226, 340, 12))]
    doc = read(tmp_path, left)
    box = crop(doc, 19)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=200), formula)  # planted: the number alone


def test_a_number_after_a_formula_whose_letters_spell_a_word_is_its_number(tmp_path):
    """Math italic letters that spell a word (``iron``) before the number on its line: a formula, not prose."""
    formula = [math(120, 210, '(u r, iron - u r, air) (7)')]
    left = [*column(50, 250, range(100, 196, 12)), *formula, *column(50, 250, range(226, 340, 12))]
    doc = read(tmp_path, left)
    box = crop(doc, 7)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, top=-20), formula)  # planted: grown into the prose above


def test_a_number_alone_on_its_line_is_chosen_over_a_short_line_ending_in_it(tmp_path):
    """A short line ends in ``as (11)``; the equation's own number stands alone at the right of its formula."""
    stray = (50, 112, 'as (11)', 'helv', 10)
    formula = [math(120, 210, 'P = T w'), number(288, 210, 11)]
    left = [*column(50, 250, range(100, 112, 12)), stray, *column(50, 250, range(124, 196, 12)), *formula]
    doc = read(tmp_path, [*left, *column(50, 250, range(226, 340, 12))])
    box = crop(doc, 11)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=60), formula)  # planted: the formula cut off


def found(doc: Document, page: int = 0) -> list[tuple[str, BBox]]:
    """The equations `document_lab.regions.regions` finds on the page -- as `document-lab pages` and a paper's picture
    page cut them out -- each its label and box."""
    return [(r.label, r.bbox) for r in regions(doc.pages[page], layout.body_face(doc)) if r.kind == 'equation']


def noise(x: float, y: float, text: str, size: float = 10) -> Run:
    """A formula as OCR reads it: noise, set in the prose's own font (an OCR layer has one font)."""
    return (x, y, text, 'helv', size)


def test_a_left_column_number_is_found_and_its_equation_takes_its_column_alone(tmp_path):
    """A two-column page: the left column's number stands short of the page's middle, the right column's prose runs
    level with the formula. Once no number left of 0.6 of the page counted, and every region spanned the text."""
    formula = [noise(110, 300, 'U,=qW +a,'), noise(150, 312, '1/. +0U'), number(282, 306, 1)]
    left = [*column(50, 250, range(100, 290, 12)), *formula, *column(50, 250, range(330, 500, 12))]
    doc = read(tmp_path, [*left, *column(312, 250, range(100, 500, 12))])
    ((label, box),) = found(doc)
    assert label == '(1)'
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, right=-280), formula)  # planted: the text's width, the right column in


def test_a_right_column_equation_takes_nothing_of_the_left_column(tmp_path):
    formula = [noise(400, 300, 'a +b= g'), number(545, 300, 2)]
    right = [*column(312, 250, range(100, 290, 12)), *formula, *column(312, 250, range(324, 500, 12))]
    doc = read(tmp_path, [*column(50, 250, range(100, 500, 12)), *right])
    ((label, box),) = found(doc)
    assert label == '(2)'
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=-300), formula)  # planted: grown over the left column


def test_an_equation_across_both_columns_is_found_whole(tmp_path):
    formula = [math(110, 420, 'W = sum k a b'), math(330, 420, 'c d + e f g'), number(545, 420, 3)]
    columns = [*column(50, 250, range(100, 400, 12)), *column(312, 250, range(100, 400, 12))]
    below = [*column(50, 250, range(450, 600, 12)), *column(312, 250, range(450, 600, 12))]
    doc = read(tmp_path, [*columns, *formula, *below])
    ((label, box),) = found(doc)
    assert label == '(3)'
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=230), formula)  # planted: the left column's half cut off


def test_a_number_at_the_left_of_its_column_is_no_equation(tmp_path):
    """A list's ``(1)`` set apart before its item: a number all the same, but at its column's left, no equation's."""
    item = [(50, 300, '(1)', 'helv', 10), prose(90, 300, 210)]
    doc = read(tmp_path, [*column(50, 250, range(100, 290, 12)), *item, *column(312, 250, range(100, 500, 12))])
    assert found(doc) == []


def test_a_value_in_parentheses_is_no_equation_number_a_chapter_s_number_is(tmp_path):
    """A table's spread ``(0.167)`` set alone at a line's right reads like a number; none starts with 0."""
    value = [noise(150, 300, 'x = 0.5'), (270, 300, '(0.167)', 'helv', 10)]
    doc = read(tmp_path, [*column(50, 250, range(100, 290, 12)), *value, *column(50, 250, range(320, 400, 12))])
    assert [label for label, _ in found(doc)] == ['Untitled equation 1']  # its formula, no number
    chapter = [noise(150, 300, 'x = 0.5'), (270, 300, '(2.8)', 'helv', 10)]
    doc = read(tmp_path, [*column(50, 250, range(100, 290, 12)), *chapter, *column(50, 250, range(320, 400, 12))])
    assert [label for label, _ in found(doc)] == ['(2.8)']  # planted: a number that is one


def displays(doc: Document, page: int = 0) -> list[tuple[str, BBox]]:
    """The display equations `document_lab.regions.equations` finds on the page -- as `document-lab paper` cuts them
    out -- each its label and box."""
    return [(r.label, r.bbox) for r in equations(doc.pages[page], layout.body_face(doc))]


def test_a_display_without_a_number_is_found_by_its_formula(tmp_path):
    formula = [math(250, 220, 'E = m c2 + k')]
    doc = read(tmp_path, [*column(72, 468, range(100, 200, 12)), *formula, *column(72, 468, range(250, 330, 12))])
    ((label, box),) = displays(doc)
    assert label == 'Untitled equation 1'
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, right=30), formula)  # planted: its right part cut off


def test_a_display_without_a_number_in_a_column_takes_nothing_of_the_other(tmp_path):
    formula = [math(400, 300, 'a + b = g')]
    right = [*column(312, 250, range(100, 290, 12)), *formula, *column(312, 250, range(320, 500, 12))]
    doc = read(tmp_path, [*column(50, 250, range(100, 500, 12)), *right])
    ((label, box),) = displays(doc)
    assert label == 'Untitled equation 1'
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, left=-150), formula)  # planted: grown over the left column


def test_inline_math_in_prose_is_no_display(tmp_path):
    """A line of prose with a formula in it -- flush with its column, set among its paragraph's lines -- stays prose;
    the same formula set apart on its own line is a display."""
    inline = (72, 208, 'where the power P = T w is the product of the torque and the speed at the shaft', 'helv', 10)
    above, below = column(72, 468, range(100, 200, 12)), column(72, 468, range(220, 300, 12))
    assert displays(read(tmp_path, [*above, inline, *below])) == []
    apart = [
        *above,
        (72, 208, 'where the power is', 'helv', 10),
        math(250, 226, 'P = T w'),
        *column(72, 468, range(250, 300, 12)),
    ]
    assert [label for label, _ in displays(read(tmp_path, apart))] == ['Untitled equation 1']  # planted: a display


def test_a_cases_block_whose_rows_read_as_words_is_taken_whole(tmp_path):
    """A definition by cases in the right column, its rows set in from the column's edge: ``f(x) > 0 Solid magnetic
    material`` reads as words and spans half the column, and is a formula's row all the same (it states ``>``)."""
    rows = [
        math(376, 300, 'f(x) < 0 air'),
        math(336, 312, 'f(x) = { f(x) = 0 boundary edge'),
        math(376, 324, 'f(x) > 0 Solid magnetic material'),
    ]
    formula = [*rows, number(548, 312, 2)]
    right = [*column(312, 250, range(100, 290, 12)), *formula, *column(312, 250, range(345, 500, 12))]
    doc = read(tmp_path, [*column(50, 250, range(100, 500, 12)), *right])
    box = crop(doc, 2)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, bottom=12), formula)  # planted: its last row cut off


def test_a_sub_numbered_pair_is_two_equations(tmp_path):
    first = [math(150, 216, 'B n = 0'), math(300, 216, 'on G1,'), (505, 216, '(16a)', 'helv', 10)]
    second = [math(146, 231, 'n x H = 0'), math(300, 231, 'on G2,'), (505, 231, '(16b)', 'helv', 10)]
    doc = read(
        tmp_path, [*column(72, 468, range(100, 200, 12)), *first, *second, *column(72, 468, range(250, 330, 12))]
    )
    assert faults(doc.pages[0], crop(doc, '16a'), first) == []
    assert faults(doc.pages[0], crop(doc, '16b'), second) == []
    assert faults(doc.pages[0], without(crop(doc, '16a'), bottom=-15), first)  # planted: grown into (16b)


def test_a_number_printed_twice_is_the_one_beside_its_formula(tmp_path):
    """A figure's legend in the right column repeats ``(1)`` alone; the equation's ``(1)`` has its formula level with
    it in the left column. Once the page refused both."""
    formula = [math(110, 300, 'L = a + b'), number(282, 300, 1)]
    left = [*column(50, 250, range(100, 290, 12)), *formula, *column(50, 250, range(320, 500, 12))]
    legend = [(330, 250, 'Tab', 'helv', 10), (540, 262, '(1)', 'helv', 10)]  # set as large as a number
    right = [*column(312, 250, range(100, 230, 12)), *legend, *column(312, 250, range(300, 500, 12))]
    doc = read(tmp_path, [*left, *right])
    box = crop(doc, 1)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], BBox(536, 252, 560, 266), formula)  # planted: the legend's number taken


def test_a_paragraph_tail_that_is_no_word_is_left_out(tmp_path):
    """A paragraph ends ``... the constitutive relation reads [55]:`` -- its last line ``[55]:``, flush with the
    column, in the body's font, no word and no math -- right above the formula."""
    tail = (72, 192, '[55]:', 'helv', 10)
    formula = [math(250, 210, 'B = u H + Br'), number(525, 210, 3)]
    doc = read(tmp_path, [*column(72, 468, range(100, 190, 12)), tail, *formula, *column(72, 468, range(240, 300, 12))])
    box = crop(doc, 3)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, top=-20), formula)  # planted: grown over the tail


def test_a_mark_inside_a_matrix_is_no_number(tmp_path):
    """A tall matrix whose entries carry ``(1)`` marks set inside it: they stand short of the column's right, so no
    piece of the matrix is taken for another equation's."""
    rows = [math(250, y, f'| a{i} b{i} |') for i, y in enumerate(range(200, 270, 10))]
    marks = [(300, 214, '(1)', 'helv', 6), (300, 244, '(1)', 'helv', 6)]
    formula = [*rows, *marks, number(525, 230, 19)]
    doc = read(tmp_path, [*column(72, 468, range(100, 190, 12)), *formula, *column(72, 468, range(290, 360, 12))])
    box = crop(doc, 19)
    assert faults(doc.pages[0], box, formula) == []
    assert faults(doc.pages[0], without(box, top=20), formula)  # planted: the matrix's top rows cut off


def test_a_formula_set_in_a_paragraph_or_a_listing_is_no_display(tmp_path):
    """A relation the reader split off a line of prose (a subscript ``<0``), a paragraph's flush tail stating one
    (``e = 99 %.``), and a code listing's ``x = 1;`` are no displays; the same tail set apart below its paragraph is."""
    above, below = column(72, 468, range(100, 200, 12)), column(72, 468, range(240, 300, 12))
    line = prose(72, 208, 300)
    subscript = [line, (73 + pymupdf.get_text_length(line[2], 'helv', 10), 210, 'a<0', 'helv', 6)]
    tail = [prose(72, 208, 468), (72, 220, 'e = 99 %.', 'helv', 10)]
    listing = [(100, 214, 'x = 1;', 'cour', 10), (100, 226, 'y = x + 2;', 'cour', 10)]
    for inline in (subscript, tail, listing):
        assert displays(read(tmp_path, [*above, *inline, *below])) == []
    apart = [prose(72, 208, 468), math(250, 230, 'e = 99 %'), *column(72, 468, range(256, 300, 12))]
    assert [label for label, _ in displays(read(tmp_path, [*above, *apart]))] == ['Untitled equation 1']  # planted


def test_a_number_in_a_plots_legend_box_is_no_equations(tmp_path):
    """A plot's legend lists ``(1)`` .. ``(3)`` in a tall, narrow box at the right of its column: no equation's
    numbers, though each stands alone at the column's right; the same numbers set without the box are taken."""
    legend = [(540, y, f'({n})', 'helv', 10) for n, y in ((1, 250), (2, 270), (3, 290))]
    page = [*column(50, 250, range(100, 500, 12)), *column(312, 200, range(100, 230, 12)), *legend]
    box = [pymupdf.Rect(532, 236, 562, 300)]
    assert displays(read(tmp_path, page, bars=(box,))) == []
    assert [label for label, _ in displays(read(tmp_path, page))] == ['(1)', '(2)', '(3)']  # planted: no box
