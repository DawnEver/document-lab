"""Equation crops: one display equation and its number, nothing else -- read off synthetic pages whose boxes are known.

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
from document_lab.paper.crop import crop_equation
from document_lab.pdf import read_pdf
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


def crop(doc: Document, n: int, page: int = 0) -> BBox | None:
    return crop_equation(doc.pages[page], str(n), layout.body_face(doc))


def faults(page: Page, box: BBox | None, formula: list[Run]) -> list[str]:
    """How ``box`` departs from holding exactly ``formula`` (its runs, the number among them): a run cut, one let in."""
    if box is None:
        return ['no crop']
    held = {text for _, _, text, _, _ in formula}
    found = []
    for span in page.spans:
        b = span.provenance.bbox
        inside = box.x0 <= b.x0 and b.x1 <= box.x1 and box.y0 <= b.y0 and b.y1 <= box.y1
        meets = b.x0 < box.x1 and box.x0 < b.x1 and b.y0 < box.y1 and box.y0 < b.y1
        if span.text in held and not inside:
            found.append(f'{span.text!r} cut')
        elif span.text not in held and meets:
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
    box = crop_equation(body.pages[1], '2', layout.body_face(doc))
    formula = [math(200, 62, 'u = R i + L d2'), number(525, 62, 2)]
    assert faults(doc.pages[1], box, formula) == []
    assert faults(doc.pages[1], crop(doc, 2, page=1), formula)  # planted: the page with its running lines


def test_a_reference_in_prose_is_no_equation_number_and_two_numbers_refuse(tmp_path):
    reference = (72, 112, 'the power is given in (3)', 'helv', 10)
    formula = [math(200, 220, 'P = T w'), number(525, 220, 3)]
    doc = read(tmp_path, [*column(72, 468, range(100, 112, 12)), reference, *formula])
    assert faults(doc.pages[0], crop(doc, 3), formula) == []
    twice = read(tmp_path, [*formula, number(525, 400, 3)])
    assert crop(twice, 3) is None  # planted: an ambiguous number refuses, never a guess


def test_a_formula_row_set_flush_left_is_no_prose(tmp_path):
    """A multline display opens flush left with a variable's letters (``cos hweTs``); its number closes it below."""
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
