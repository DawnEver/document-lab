"""Layout: a line spaces its spans by their source's word gaps; vertical writing reads a column top to bottom, columns
side by side right to left.
"""

import dataclasses
import math

import pymupdf
import pytest

from document_lab import layout
from document_lab.ir import BBox, Page, Provenance, TextSpan
from document_lab.layout import lines
from document_lab.pdf import read_pdf
from tests.synthetic import ACTUAL_TEXT_LINES, actual_text_pdf, layer_pdf


def _line_texts(path):
    return [line.text for line in lines(read_pdf(path).page(1), weighted=True)]


def test_a_line_spaces_its_spans_apart_but_where_one_continues_a_word(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    bold = pymupdf.get_text_length('Head', fontname='hebo', fontsize=10)
    page.insert_text((72, 72), 'Head', fontname='hebo', fontsize=10)
    page.insert_text((72 + bold, 72), 'line one', fontname='helv', fontsize=10)
    page.insert_text((72, 100), 'Bold', fontname='hebo', fontsize=10)
    page.insert_text((72 + pymupdf.get_text_length('Bold ', fontname='hebo', fontsize=10), 100), 'plain', fontsize=10)
    doc.save(tmp_path / 'w.pdf')
    assert sorted(_line_texts(tmp_path / 'w.pdf')) == ['Bold plain', 'Headline one']


SRC = Provenance('0' * 64, 'file:///synthetic.pdf', page=1)
SIZE = 10.0


def _vertical(text: str, x: float, top: float, unspaced: int | None = None) -> TextSpan:
    """``text`` down a column centered on ``x`` from ``top``, an em a character."""
    box = BBox(x - SIZE / 2, top, x + SIZE / 2, top + SIZE * len(text))
    return TextSpan(text, SIZE, 'Song', SRC.at(bbox=box), unspaced=unspaced, vertical=True)


def _flat(text: str, x0: float, y0: float) -> TextSpan:
    return TextSpan(text, SIZE, 'Helvetica', SRC.at(bbox=BBox(x0, y0, x0 + 5 * len(text), y0 + SIZE)))


def _page(*spans: TextSpan) -> Page:
    return Page(number=1, width=600, height=800, spans=spans)


def test_spans_down_one_column_are_one_line_read_top_to_bottom():
    # Drawn bottom first: the line still reads top to bottom, spaced but where spans continue one word.
    page = _page(
        _vertical('CD', 500, 120, unspaced=1), _vertical('AB', 500, 100, unspaced=1), _vertical('EF', 500, 147)
    )
    (line,) = layout.lines(page, weighted=True)
    assert (line.vertical, line.text, line.column) == (True, 'ABCD EF', 'R')
    assert (line.x0, line.y0, line.x1, line.y1) == (495, 100, 505, 167)


def test_a_vertical_span_never_joins_a_flat_line():
    page = _page(_flat('side', 480, 100), _vertical('AB', 506, 100))
    assert sorted((ln.text, ln.vertical) for ln in layout.lines(page, weighted=False)) == [
        ('AB', True),
        ('side', False),
    ]


def test_columns_side_by_side_read_right_to_left_as_one_block():
    # Three columns 15 pt apart (half an em between them), drawn left to right; a fourth far off is its own block.
    page = _page(
        _vertical('GHI', 470, 100),
        _vertical('DEF', 485, 100),
        _vertical('ABC', 500, 100),
        _vertical('XY', 400, 100),
    )
    assert [ln.text for ln in layout.ordered(layout.lines(page, weighted=True))] == ['ABC', 'DEF', 'GHI', 'XY']
    assert [b.text for b in layout.blocks(page)] == ['ABC\nDEF\nGHI', 'XY']


def test_a_vertical_block_across_the_middle_is_full_width_and_reads_right_to_left():
    # A page set in vertical columns from x 320 to x 280: the block crosses the middle (300), so no column split.
    page = _page(*(_vertical(text, x, 100) for text, x in (('A', 320), ('B', 305), ('C', 290), ('D', 275))))
    found = layout.lines(page, weighted=True)
    assert {ln.column for ln in found} == {'F'}
    assert [ln.text for ln in layout.ordered(found)] == ['A', 'B', 'C', 'D']


def test_vertical_columns_stacked_read_top_band_first():
    page = _page(_vertical('LOW', 500, 300), _vertical('TOP2', 485, 100), _vertical('TOP1', 500, 100))
    assert [ln.text for ln in layout.ordered(layout.lines(page, weighted=True))] == ['TOP1', 'TOP2', 'LOW']


def test_a_stated_space_reads_as_a_space_a_word_joiner_as_none_and_a_ligature_as_its_letters(tmp_path):
    page = read_pdf(actual_text_pdf(tmp_path / 'actual.pdf')).page(1)
    # The IR states each replacement as the document does; reading it as lines is layout's.
    top_down = sorted(page.spans, key=lambda s: s.provenance.bbox.y0)
    assert [s.text for s in top_down] == [before + actual + after for before, _, actual, after in ACTUAL_TEXT_LINES]
    assert [ln.text for ln in layout.ordered(lines(page, weighted=True))] == ['80 V', '80 V', '80V', 'Exfinal']


def test_a_span_of_a_zero_width_space_holds_no_text():
    bold = TextSpan('Head', SIZE, 'Helvetica-Bold', SRC.at(bbox=BBox(100, 100, 120, 110)), bold=True)
    gap = TextSpan('\u200b', SIZE, 'Helvetica', SRC.at(bbox=BBox(120, 100, 123, 110)))
    (line,) = layout.lines(_page(bold, gap), weighted=True)
    assert (line.text, line.bold) == ('Head', True)
    assert layout.lines(_page(gap), weighted=True) == []


def _running_pdf(path, pages=4):
    """A data sheet's pages: a running head (title left, part number right) and a foot (page number, issue)."""
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((42, 60), 'Data and Signal Line Chokes', fontsize=11)
        page.insert_text((499, 60), 'ACT1210', fontsize=11)
        page.insert_text((37, 120), f'Section {number} opens the page with its own words', fontsize=11)
        page.insert_text((37, 400), 'Notes', fontsize=11)  # the same words, the same place, in the body
        page.insert_text((37, 600), 'The body closes the page' + ' here' * number, fontsize=11)
        page.insert_text((291, 822), str(number), fontsize=11)
        page.insert_text((319, 822), '6/25', fontsize=11)
    doc.save(path)
    return path


def test_running_heads_and_feet_repeat_at_the_page_edges_page_numbers_varying(tmp_path):
    found = layout.running(read_pdf(_running_pdf(tmp_path / 'sheet.pdf')))
    assert [[ln.text for ln in page.head] for page in found] == [['Data and Signal Line Chokes', 'ACT1210']] * 4
    assert [[ln.text for ln in page.foot] for page in found] == [[str(n), '6/25'] for n in range(1, 5)]


def test_an_equation_number_closing_pages_at_one_place_is_no_running_foot(tmp_path):
    """A paper's text block ends at one baseline on every page: pages ending in a numbered equation set its number at
    the same place, ``(7)``, ``(12)``, ``(19)`` -- each its own equation, never a page number repeating."""
    doc = pymupdf.open()
    for number, equation in enumerate((7, 12, 19, 23), start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((37, 120), f'Section {number} opens the page with its own words', fontsize=11)
        page.insert_text((200, 780), f'x{number} = a + b', fontsize=11)
        page.insert_text((520, 780), f'({equation})', fontsize=11)
        page.insert_text((291, 822), str(number), fontsize=11)
    doc.save(tmp_path / 'paper.pdf')
    found = layout.running(read_pdf(tmp_path / 'paper.pdf'))
    assert [[ln.text for ln in page.foot] for page in found] == [[str(n)] for n in range(1, 5)]  # the page numbers


def test_a_page_alone_and_a_line_on_too_few_pages_run_nowhere(tmp_path):
    assert [(p.head, p.foot) for p in layout.running(read_pdf(_running_pdf(tmp_path / 'one.pdf', pages=1)))] == [
        ((), ())
    ]
    doc = pymupdf.open()
    for number in range(4):  # a head on two pages of four: what most pages lack is no running head
        page = doc.new_page(width=595, height=842)
        if number < 2:
            page.insert_text((42, 60), 'Draft for comment', fontsize=11)
        page.insert_text((37, 120), f'Clause {number} text', fontsize=11)
    doc.save(tmp_path / 'few.pdf')
    assert all(p.head == () for p in layout.running(read_pdf(tmp_path / 'few.pdf')))


def test_vertical_writing_beside_the_text_is_no_furniture_but_a_sideways_stamp_is():
    """A column of vertical writing runs down the page as a turned stamp does, but it is the page's text: only text
    turned sideways, outside the box of the page's text (vertical columns included), is furniture.
    """
    column = TextSpan('ABCDE', SIZE, 'Song', SRC.at(bbox=BBox(495, 100, 505, 150)), direction=(0.0, 1.0), vertical=True)
    stamp = TextSpan('arXiv:0000.00000v1', SIZE, 'Times', SRC.at(bbox=BBox(20, 200, 30, 380)), direction=(0.0, -1.0))
    page = _page(_flat('body text of the page', 100, 100), _flat('more body text here', 100, 120), column, stamp)
    assert layout.furniture(page) == [stamp]
    assert 'ABCDE' in [line.text for line in lines(page, weighted=False)]


def test_a_sideways_stamp_on_every_page_is_furniture_never_also_a_running_line(tmp_path):
    doc = pymupdf.open()
    for number in range(1, 5):
        page = doc.new_page(width=595, height=842)
        page.insert_text((20, 600), 'arXiv:2301.01621v3 [cs.CL] 9 Jan 2023', fontsize=11, rotate=90)
        page.insert_text((42, 60), 'Data and Signal Line Chokes', fontsize=11)
        page.insert_text((60, 120), f'Section {number} opens the page with its own words', fontsize=11)
        page.insert_text((291, 822), str(number), fontsize=11)
    doc.save(tmp_path / 'stamped.pdf')
    read = read_pdf(tmp_path / 'stamped.pdf')
    stamps = [{id(s) for s in layout.furniture(page)} for page in read.pages]
    assert all(stamps)
    for run, stamped in zip(layout.running(read), stamps, strict=True):
        assert [ln.text for ln in run.head] == ['Data and Signal Line Chokes']
        assert not stamped & {id(s) for ln in (*run.head, *run.foot) for s in ln.spans}


#: A page whose caption and notes are turned sideways beside a landscape table (left out), between upright lines:
#: each line ``(text, baseline origin, degrees turned clockwise)``, set word by word as an OCR lays its words.
SIDEWAYS = (
    ('The table below is printed sideways, as a standard prints a landscape table.', (72, 72), 0),
    ('Table 3 - Magnetic properties and densities', (100, 700), -90),  # read bottom to top, the next line right
    ('of the sintered magnets', (114, 700), -90),
    ('Beside the caption', (150, 560), 0),  # upright, its top between the caption lines' tops
    ('Notes read top to bottom', (390, 150), 90),  # read top to bottom, the next line left
    ('and their next line', (376, 150), 90),
    ('Both turned runs read along their own lines, each placed as a block.', (72, 760), 0),
)


def _sideways_words(size: float = 10) -> list[tuple[str, tuple[float, float], tuple[float, float, float, float], int]]:
    """Each word of :data:`SIDEWAYS`: its text, its baseline origin, its box and its turn."""
    out = []
    for line, origin, turn in SIDEWAYS:
        x, y = origin
        dx, dy = round(math.cos(math.radians(turn))), round(math.sin(math.radians(turn)))
        up = (dy, -dx)  # toward the glyphs' tops
        for word in line.split():
            length = pymupdf.get_text_length(word, fontsize=size)
            xs = [x + a * dx + b * size * up[0] for a in (0, length) for b in (-0.2, 0.8)]
            ys = [y + a * dy + b * size * up[1] for a in (0, length) for b in (-0.2, 0.8)]
            out.append((word, (x, y), (min(xs), min(ys), max(xs), max(ys)), turn))
            step = pymupdf.get_text_length(word + ' ', fontsize=size)
            x, y = x + step * dx, y + step * dy
    return out


def _sideways_pdf(path, kind: str):
    if kind == 'layer':  # a scan's text layer: invisible words over their boxes
        return layer_pdf(path, [(word, box, turn, False) for word, _, box, turn in _sideways_words()])
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        for word, origin, _, turn in _sideways_words():
            page.insert_text(origin, word, fontname='helv', fontsize=10, rotate=-turn % 360)
        doc.save(path)
    return path


@pytest.mark.parametrize('kind', ['vector', 'layer'])
def test_text_turned_sideways_reads_along_its_own_lines_each_run_placed_as_a_block(tmp_path, kind):
    page = read_pdf(_sideways_pdf(tmp_path / 'sideways.pdf', kind)).page(1)
    assert {s.direction for s in page.spans} == {(1.0, 0.0), (0.0, -1.0), (0.0, 1.0)}
    read = [ln.text for ln in layout.ordered(layout.lines(page, weighted=False))]
    assert read == [text for text, _, _ in SIDEWAYS]
    # The planted wrong reading: the turned words read as if upright, top to bottom -- what a reader of every span
    # as left to right makes of them -- is told from the true one.
    planted = [' '.join(reversed(text.split())) if turn == -90 else text for text, _, turn in SIDEWAYS]
    assert read != planted
    caption, notes = (next(b for b in layout.blocks(page) if b.text.startswith(start)) for start in ('Table', 'Notes'))
    assert (caption.text, notes.text) == (
        'Table 3 - Magnetic properties and densities\nof the sintered magnets',
        'Notes read top to bottom\nand their next line',
    )


#: Four columns of vertical writing, right to left (Chinese: the stator winding of a hydro generator heats and
#: vibrates in service, so its insulation is inspected).
COLUMNS = (
    '\u6c34\u7535\u673a\u7ec4\u7684\u5b9a\u5b50\u7ed5',
    '\u7ec4\u5728\u8fd0\u884c\u65f6\u4ea7\u751f\u70ed',
    '\u91cf\u548c\u632f\u52a8\u56e0\u6b64\u9700\u8981',
    '\u5b9a\u671f\u68c0\u67e5\u7edd\u7f18\u72b6\u6001',
)
#: Each column's words as Tesseract boxes them in a 24 pt column (the chi_sim_vert round trip, tests/ocr): its
#: characters; how far its box stops short of its cells at the top and at the foot (its ink); its half width across;
#: its center's offset across. The second starts 22 pt below the first (more than its box is wide); the third's box
#: is 32 pt across a 24 pt column, off its center.
OCR_WORDS = ((2, 6.8, 8.8, 10.1, 0.6), (1, 13.2, 1.2, 11.0, 1.1), (3, 2.0, 2.0, 16.0, -3.6), (2, 3.0, 3.0, 10.8, 0.0))
EM, PITCH, MARGIN = 24.0, 38.4, 40.0  # the font size, column middle to middle (1.6 em), the page's margin


def _ocr_columns_pdf(path):
    width, height = len(COLUMNS) * PITCH + 2 * MARGIN, max(map(len, COLUMNS)) * EM + 2 * MARGIN
    words = []
    for i, column in enumerate(COLUMNS):
        middle, start = width - MARGIN - EM / 2 - PITCH * i, 0
        for chars, top, foot, half, offset in OCR_WORDS:
            y0, y1 = MARGIN + EM * start + top, MARGIN + EM * (start + chars) - foot
            box = (middle + offset - half, y0, middle + offset + half, y1)
            words.append((column[start : start + chars], box, 0, True))
            start += chars
    return layer_pdf(path, words, width, height)


def test_ocr_d_vertical_columns_read_right_to_left_however_wide_their_boxes_run(tmp_path):
    page = read_pdf(_ocr_columns_pdf(tmp_path / 'columns.pdf')).page(1)
    assert all(s.vertical for s in page.spans)
    found = layout.lines(page, weighted=False)
    assert len(found) == len(COLUMNS)  # each column one line, however far apart its words' boxes stand
    read = [ln.text.replace(' ', '') for ln in layout.ordered(found)]
    assert read == list(COLUMNS)
    assert [b.text.replace(' ', '') for b in layout.blocks(page)] == ['\n'.join(COLUMNS)]
    # The planted wrong reading: the page's halves read left one first (columns 3, 4, 1, 2).
    assert read != [*COLUMNS[2:], *COLUMNS[:2]]


def test_a_caption_turned_sideways_is_one_paragraph_of_the_page_markdown(tmp_path):
    from document_lab.markdown import page_markdowns  # noqa: PLC0415

    (text,) = page_markdowns(read_pdf(_sideways_pdf(tmp_path / 'sideways.pdf', 'layer')))
    paragraphs = text.split('\n\n')
    caption = ' '.join(text for text, _, turn in SIDEWAYS if turn == -90)
    assert caption in paragraphs
    assert ' '.join(text for text, _, turn in SIDEWAYS if turn == 90) in paragraphs
    # The planted wrong truth: the caption run on into the upright line beside it.
    assert f'{caption} Beside the caption' not in paragraphs


#: Two lines set 1.1 sizes apart (the control: close, yet two lines), each glyph drawn on its own.
GREY_LINES = ((100.0, 'Glyphs each in a grey of their own'), (111.0, 'read as the run their source set.'))


def _grey_pdf(path, greys: tuple[float, ...]):
    """:data:`GREY_LINES` glyph by glyph, cycling through ``greys`` and jittering each baseline by up to 0.04 sizes."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        for y, text in GREY_LINES:
            x = 72.0
            for i, char in enumerate(text):
                page.insert_text(
                    (x, y + 0.4 * (i % 3 - 1)), char, fontname='helv', fontsize=10, color=(greys[i % 3],) * 3
                )
                x += pymupdf.get_text_length(char, fontname='helv', fontsize=10)
        doc.save(path)
    return read_pdf(path).page(1)


def test_a_run_split_by_its_fill_reads_as_the_run_its_source_set(tmp_path):
    """A reader splits a run where its fill changes: glyphs each in a grey of their own are as many spans. Their lines
    -- text, frame, so every paragraph gap measured from them -- are those of the same glyphs in one grey."""
    split = _grey_pdf(tmp_path / 'greys.pdf', (0.2, 0.3, 0.45))
    whole = _grey_pdf(tmp_path / 'grey.pdf', (0.3, 0.3, 0.3))
    assert (len(split.spans), len(whole.spans)) == (sum(len(t) - t.count(' ') for _, t in GREY_LINES), 2)

    def read(page):
        return [(ln.text, ln.frame) for ln in layout.ordered(layout.lines(page, weighted=False))]

    assert read(split) == read(whole)
    assert [text for text, _ in read(split)] == [text for _, text in GREY_LINES]  # the control: two lines stay two


def test_a_span_that_continues_a_run_is_one_run_with_the_span_before_it():
    """The reader says which piece split off a run goes on with it (``continues_run``), and layout reads the pieces as
    that one run, judging nothing again: two pieces half a size apart across are one line. The control: the same
    pieces, the second starting a run, are two lines."""
    piece = TextSpan('two', SIZE, 'Helvetica', SRC.at(bbox=BBox(87, 105, 102, 115)), fill=(1.0, 0.0, 0.0))
    for continues, texts in ((True, ['one two']), (False, ['one', 'two'])):
        page = _page(_flat('one', 72, 100), dataclasses.replace(piece, continues_run=continues))
        assert [ln.text for ln in layout.ordered(layout.lines(page, weighted=True))] == texts
