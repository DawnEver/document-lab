"""The IR markdown writer on a synthetic two-column page whose reading order and headings are known exactly."""

import pymupdf
import pytest

from document_lab.ir import BBox
from document_lab.markdown import page_markdowns, table, to_markdown
from document_lab.pdf import read_pdf

LEFT, RIGHT, WIDTH = 50, 320, 240


def _write(page, x, y, text, *, size=10, font='tiro'):
    page.insert_text((x, y), text, fontname=font, fontsize=size)


@pytest.fixture
def md(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, 150, 60, 'A Synthetic Paper Title', size=22)
    _write(page, LEFT, 120, 'I. INTRODUCTION')
    _write(page, LEFT, 140, 'The left column starts here and its')
    _write(page, LEFT, 152, 'sentence runs on to a second line.')
    _write(page, LEFT, 180, 'II. A HEADING THAT WRAPS ONTO')
    _write(page, LEFT, 192, 'A SECOND LINE')
    _write(page, LEFT, 212, 'Body of the second section ends with an inter-')
    _write(page, RIGHT, 120, 'rupted word that the right column completes.')
    _write(page, RIGHT, 150, '#1 legend text')
    _write(page, RIGHT, 180, 'REFERENCES')
    _write(page, RIGHT, 200, '[1] A. Author, A paper.')
    doc.save(tmp_path / 'p.pdf')
    return to_markdown(read_pdf(tmp_path / 'p.pdf'))


def test_the_title_and_sections_are_headings_in_reading_order(md):
    headings = [line for line in md.splitlines() if line.startswith('#')]
    assert headings == [
        '# A Synthetic Paper Title',
        '## I. INTRODUCTION',
        '## II. A HEADING THAT WRAPS ONTO A SECOND LINE',
        '## REFERENCES',
    ]


def test_lines_of_a_paragraph_join(md):
    assert 'The left column starts here and its sentence runs on to a second line.' in md


def test_a_word_broken_across_columns_rejoins(md):
    assert 'ends with an interrupted word that the right column completes.' in md


def test_body_text_starting_with_a_hash_is_not_a_heading(md):
    assert '\\#1 legend text' in md


def test_a_table_of_contents_page_opens_no_sections(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, 150, 60, 'A Standard', size=22)
    contents = doc.new_page(width=612, height=792)
    _write(contents, LEFT, 100, 'Contents', size=22)
    entries = ['1 Scope', '2 Normative references', 'II. TERMS AND DEFINITIONS', 'REFERENCES']
    for i, entry in enumerate(entries):
        _write(contents, LEFT, 140 + 20 * i, f'{entry} ' + '.' * 30 + f' {i + 5}', font='tibo')
    body = doc.new_page(width=612, height=792)
    _write(body, LEFT, 100, '1 Scope', font='tibo')
    _write(body, LEFT, 120, 'This standard applies to rotating machines.')
    _write(body, LEFT, 760, '2', font='tibo')  # a page number in bold: no letters, no heading
    doc.save(tmp_path / 'toc.pdf')
    headings = [ln for ln in to_markdown(read_pdf(tmp_path / 'toc.pdf')).splitlines() if ln[:1] == '#']
    assert headings == ['# A Standard', '## 1 Scope']


def test_one_contents_line_is_not_a_heading_but_a_page_of_them_is_a_contents_page(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, LEFT, 100, 'II. METHOD ........ 4')
    _write(page, LEFT, 120, 'III. RESULTS')  # the control: a real heading on a page with one entry stays one
    doc.save(tmp_path / 'one.pdf')
    headings = [ln for ln in to_markdown(read_pdf(tmp_path / 'one.pdf')).splitlines() if ln[:1] == '#']
    assert headings == ['## III. RESULTS']


def test_each_page_stands_alone_and_a_page_without_text_is_empty(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), 'the paragraph runs on to the next page and', fontsize=10)
    doc.new_page()
    doc.new_page().insert_text((72, 72), 'ends here.', fontsize=10)
    doc.save(tmp_path / 'p.pdf')
    pages = page_markdowns(read_pdf(tmp_path / 'p.pdf'))
    assert pages[1] == ''
    assert 'ends here' not in pages[0]
    assert pages[2].startswith('ends here.')
    assert 'next page and ends here.' in to_markdown(read_pdf(tmp_path / 'p.pdf'))


def test_a_clause_number_set_apart_from_its_title_heads_the_clause_even_not_bold(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for y, number, title in ((80, '4', 'Effects on performance'), (120, '4.1', 'General')):
        _write(page, 57, y, number, size=11)
        _write(page, 100, y, title, size=11)  # four font sizes apart, as a standard sets a clause title
    _write(page, 57, 150, 'The effects of unbalanced voltages are as described in 4.2 to 4.6.', size=10)
    _write(page, 57, 165, '-', size=10)
    _write(page, 75, 165, 'IEC Electropedia: available online', size=10)
    _write(page, 57, 180, '1', size=10)
    _write(page, 100, 180, 'The motor shall be rated for the duty declared by the purchaser.', size=10)
    doc.save(tmp_path / 's.pdf')
    md = to_markdown(read_pdf(tmp_path / 's.pdf'))
    assert '## 4 Effects on performance' in md
    assert '### 4.1 General' in md
    assert '1 The motor shall be rated' in md  # a numbered sentence stays text
    assert '# 1 The motor' not in md
    assert '- IEC Electropedia: available online' in md  # a bullet keeps its text


def test_ocr_noise_heads_nothing_in_an_unweighted_document(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 57, 80, '5.8 Machines with more than one rating ..............ccccececeece', size=11)
    _write(page, 57, 110, '1 H', size=11)
    _write(page, 57, 140, '1 OO OO', size=11)
    _write(page, 300, 200, 'NN I ERE Ee', size=30)  # a figure's big lettering, read as text
    _write(page, 57, 240, '4.2 Duty types', size=11)
    _write(page, 57, 270, 'The duty shall be declared by the purchaser of the machine.', size=10)
    doc.save(tmp_path / 's.pdf')
    md = to_markdown(read_pdf(tmp_path / 's.pdf'))
    assert '### 4.2 Duty types' in md
    heads = [line for line in md.splitlines() if line.startswith('#')]
    assert heads == ['### 4.2 Duty types']


def test_a_table_row_led_by_a_year_heads_nothing(tmp_path):
    """BS EN 60034-18-1 Annex ZA: '2011 Evaluation and qualification of electrical' (a year, then a title) was a
    clause heading; a clause number has one or two digits before its first point, a year four.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 57, 80, '4.2 Duty types', size=11)
    _write(page, 57, 110, '2011 Evaluation and qualification of electrical', size=11)
    _write(page, 57, 140, '2014 Electrical insulation systems - Thermal', size=11)
    _write(page, 57, 170, '12 Tests', size=11)
    _write(page, 57, 200, 'The duty shall be declared by the purchaser of the machine.', size=10)
    doc.save(tmp_path / 's.pdf')
    heads = [line for line in to_markdown(read_pdf(tmp_path / 's.pdf')).splitlines() if line[:1] == '#']
    assert heads == ['### 4.2 Duty types', '## 12 Tests']


def test_a_table_s_merged_cells_are_written_once_given_its_cells_boxes():
    rows = (('Parameter', 'Limits', 'Limits'), ('Parameter', 'min', 'max'), ('a|b', '1', '2'))
    merged, wide = BBox(0, 0, 10, 20), BBox(10, 0, 30, 10)
    boxes = (
        (merged, wide, wide),
        (merged, BBox(10, 10, 20, 20), BBox(20, 10, 30, 20)),
        (BBox(0, 20, 10, 30), BBox(10, 20, 20, 30), BBox(20, 20, 30, 30)),
    )
    assert table(rows, boxes) == '| Parameter | Limits |  |\n| --- | --- | --- |\n|  | min | max |\n| a\\|b | 1 | 2 |'
    assert table(rows).count('Limits') == 2  # without boxes no merge is known


def test_a_placed_block_stands_where_its_box_is_in_place_of_the_lines_inside_it(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, LEFT, 100, 'Technical data and measuring conditions.')
    _write(page, LEFT, 140, 'Rated voltage')
    _write(page, 220, 140, '80 V DC')
    _write(page, LEFT, 160, 'Weight')
    _write(page, 220, 160, 'Approx. 0.075 g')
    _write(page, LEFT, 220, 'Characteristics and ordering codes.')
    doc.save(tmp_path / 'p.pdf')
    block = table([('Rated voltage', '80 V DC'), ('Weight', 'Approx. 0.075 g')])
    (text,) = page_markdowns(read_pdf(tmp_path / 'p.pdf'), [[(BBox(40, 125, 400, 170), block)]])
    assert text.split('\n\n') == [
        'Technical data and measuring conditions.',
        '| Rated voltage | 80 V DC |\n| --- | --- |\n| Weight | Approx. 0.075 g |',
        'Characteristics and ordering codes.\n',
    ]


def test_a_title_wrapping_beside_a_placed_logo_stays_whole(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 54, 300, 'Data and Signal Line Chokes', size=18)
    _write(page, 54, 366, 'Common-mode Chokes, EIA 1210 and more', size=18)
    _write(page, 54, 500, 'Series and type of the part, the date of its issue and the rights reserved to it.')
    _write(page, 54, 512, 'Reproduction of this publication without prior express consent is prohibited.')
    doc.save(tmp_path / 'p.pdf')
    (text,) = page_markdowns(read_pdf(tmp_path / 'p.pdf'), [[(BBox(419, 69, 546, 101), '![logo](logo.png)')]])
    assert text.split('\n\n')[:2] == [
        '# Data and Signal Line Chokes Common-mode Chokes, EIA 1210 and more',
        '![logo](logo.png)',
    ]


def test_a_table_is_written_with_its_header_rule_a_pipe_escaped_and_short_rows_padded():
    assert table([('a', 'b|c'), ('1',)]) == '| a | b\\|c |\n| --- | --- |\n| 1 |  |'


def _heads(md: str) -> list[str]:
    return [line for line in md.splitlines() if line[:1] == '#']


def test_a_bold_letter_in_a_data_sheets_table_heads_nothing(tmp_path):
    """A ferrite handbook's tables head their columns with bold dimension letters (``A B``, ``C Dmin``) and a
    material sheet letter-spaces its bold column heads (``U N I T``), all in the body size: none is an appendix
    section, which is set larger than its body. A lettered clause (``A.1``) heads in the body size, as ``1.1`` does.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 57, 80, '1 Scope', font='tibo')
    _write(page, 57, 100, 'The cores of this sheet are made of a ferrite material for power use.')
    for x, label in ((57, 'A'), (90, 'B'), (123, 'C'), (156, 'Dmin'), (200, 'U'), (215, 'CORES')):
        _write(page, x, 130, label, font='tibo')  # a table's head row, its cells a few ems apart
    for i, letter in enumerate('UNIT'):
        _write(page, 300 + 9 * i, 130, letter, font='tibo')
    _write(page, 57, 150, '0.464 mm 71300 mm3 182 mm')
    _write(page, 57, 180, 'A.1 Dimensions', font='tibo')
    _write(page, 57, 200, 'The dimensions of the cores are given in the table above for each type.')
    sheet = doc.new_page(width=595, height=842)
    _write(sheet, 113, 190, 'C O N D I T I O N S', size=12, font='tibo')  # letter-spaced, larger: one span
    _write(sheet, 113, 210, '25 C; 10 kHz; 0.25 mT')
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == ['## 1 Scope', '### A.1 Dimensions']


def test_a_stamp_set_sideways_in_the_margin_is_neither_title_nor_text(tmp_path):
    """arXiv 2204.08094: the margin's sideways stamp, the largest text on page 1, was the title and joined the
    abstract's line beside it. Text turned sideways outside the page's text frame is furniture; text turned sideways
    inside it (a figure's axis label) stays.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 70, 70, 'A Title Set Larger Than Its Body', size=14)
    for i in range(10):
        _write(page, 55, 200 + 12 * i, f'the abstract runs on over line {i} of its column and', size=10)
    for i in range(3):
        _write(page, 55, 600 + 12 * i, f'the text after the figure runs on over line {i} and', size=10)
    page.insert_text((40, 520), 'arXiv:0000.00000v1 [cs.XX] 1 Jan 2000', fontname='tiro', fontsize=20, rotate=90)
    page.insert_text((150, 540), 'Amplitude', fontname='tiro', fontsize=8, rotate=90)  # inside the frame
    doc.save(tmp_path / 's.pdf')
    md = to_markdown(read_pdf(tmp_path / 's.pdf'))
    assert _heads(md) == ['# A Title Set Larger Than Its Body']
    assert 'arXiv' not in md
    assert 'line 4 of its column and the abstract runs on over line 5' in md
    assert 'Amplitude' in md


def test_only_the_largest_text_of_page_one_is_a_title_below_the_title_scale(tmp_path):
    doc = pymupdf.open()
    first = doc.new_page(width=595, height=842)
    _write(first, 70, 70, 'A Title Set Larger Than Its Body', size=14)
    _write(first, 70, 100, 'A Subtitle Set Smaller Than The Title', size=12)
    _write(first, 70, 140, 'The body text of the first page runs on here.', size=10)
    later = doc.new_page(width=595, height=842)
    _write(later, 70, 70, 'A Large Line On A Later Page', size=14)
    _write(later, 70, 100, 'The body text of the later page runs on here.', size=10)
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == ['# A Title Set Larger Than Its Body']


def test_a_bold_number_set_apart_from_its_title_heads_with_it(tmp_path):
    """arXiv 2608.20674 sets a section's number and its name an em apart, in bold larger than the body; an appendix
    section is lettered so. A back-matter word heads in title case too, and in title size it is a section, no title.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, 150, 60, 'A Preprint Title', size=17)
    _write(page, 72, 90, 'Hitoshi KOSHIBA', font='tibo')  # a bold author: no number, no heading
    _write(page, 72, 120, '1', size=12, font='tibo')
    _write(page, 92, 120, 'Introduction', size=12, font='tibo')  # 1.2 em after its number
    _write(page, 72, 140, 'The body of the introduction is set in the body size here.')
    _write(page, 72, 170, '1.1 Premises and Hypotheses', font='tibo')
    _write(page, 72, 190, 'The body of the subsection is set in the body size here.')
    _write(page, 72, 210, '1')  # not bold: a list's number set apart stays text
    _write(page, 92, 210, 'The motor shall be rated for its duty')
    _write(page, 72, 240, 'References', size=12, font='tibo')
    _write(page, 72, 260, '[1] A. Author. A paper on the subject. 2025.')
    appendix = doc.new_page(width=612, height=792)
    _write(appendix, 72, 80, 'Appendix', size=17, font='tibo')
    _write(appendix, 72, 110, 'A', size=12, font='tibo')
    _write(appendix, 93, 110, 'Sentence-Level Estimation', size=12, font='tibo')
    _write(appendix, 72, 130, 'The body of the appendix is set in the body size here.')
    _write(appendix, 72, 160, 'Fiscal Year', font='tibo')  # a table's bold head heads nothing
    _write(appendix, 200, 160, 'Total', font='tibo')
    doc.save(tmp_path / 's.pdf')
    md = to_markdown(read_pdf(tmp_path / 's.pdf'))
    assert _heads(md) == [
        '# A Preprint Title',
        '## 1 Introduction',
        '### 1.1 Premises and Hypotheses',
        '## References',
        '# Appendix',
        '## A Sentence-Level Estimation',
    ]
    assert 'The motor shall be rated for its duty' in md  # text, under no heading of its own


def test_an_unweighted_paper_heads_its_capitalized_sections_and_no_footnote(tmp_path):
    """arXiv 2204.08094 as this reader weighs it: no span bold, sections numbered in capitals, a footnote numbered
    in a smaller size. A heading is never set smaller than the body.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 70, 70, 'A TITLE SET LARGER THAN ITS BODY', size=14)
    _write(page, 120, 120, '1. INTRODUCTION')
    _write(page, 55, 140, 'The body of the introduction is set in the body size here.')
    _write(page, 55, 170, '2.1 Baseline Model')
    _write(page, 55, 190, 'The body of the subsection is set in the body size here.')
    _write(page, 55, 220, 'Acknowledgments')
    _write(page, 55, 240, 'The body of the acknowledgments is set in the body size here.')
    _write(page, 55, 700, '3 All code is available at https://example.org/code', size=8)
    doc.save(tmp_path / 's.pdf')
    md = to_markdown(read_pdf(tmp_path / 's.pdf'))
    assert _heads(md) == [
        '# A TITLE SET LARGER THAN ITS BODY',
        '## 1. INTRODUCTION',
        '### 2.1 Baseline Model',
        '## Acknowledgments',
    ]


def test_a_capital_section_heads_in_small_capitals_below_the_floor_and_a_small_label_does_not(tmp_path):
    """IEEE 10583296 sets 'IV. MATCH OF ...' at 0.815x the body, W4309593298 'REFERENCES' at 0.8x: small capitals.
    A bold title-case 'References' as small, a table's label (W3213764655), stays under the floor.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, 72, 80, 'IV. MATCH OF INTERPOLATION AND CALCULATION METHODS', size=8.15)
    _write(page, 72, 100, 'The body of the section is set in the body size here and runs on.')
    _write(page, 72, 130, 'References', size=8, font='tibo')
    _write(page, 72, 150, 'The body after the label is set in the body size here and runs on.')
    _write(page, 72, 180, 'REFERENCES', size=8)
    _write(page, 72, 200, '[1] A. Author. A paper on the subject. 2025.')
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == [
        '## IV. MATCH OF INTERPOLATION AND CALCULATION METHODS',
        '## REFERENCES',
    ]


def test_a_cover_sheet_set_large_throughout_has_no_title_and_a_url_is_never_one(tmp_path):
    """W3213764655's repository cover sheet sets its eight lines at 12 pt over 10 pt text: no line of it is set apart
    as a title is. The paper's own title follows on page 2.
    """
    doc = pymupdf.open()
    cover = doc.new_page(width=595, height=842)
    for i, text in enumerate(
        (
            'Deposited via The University of Sheffield.',
            'White Rose Research Online URL for this paper:',
            'https://eprints.example.org/id/eprint/180427/',
            'Version: Published Version',
            'Article:',
            'Zhu, Z., Wang, S. et al. (2021) Advances in dual-three-phase machines.',
            'Energies, 14 (22). 7508.',
            'https://doi.org/10.3390/en14227508',
        )
    ):
        _write(cover, 57, 120 + 14 * i, text, size=12)
    for i in range(6):
        _write(cover, 57, 600 + 12 * i, f'This article is distributed under the terms of the license, line {i} of it.')
    paper = doc.new_page(width=595, height=842)
    _write(paper, 57, 80, 'Advances in Dual-Three-Phase Machines', size=18)
    _write(paper, 57, 120, 'https://www.example.org/a-url-set-as-large-as-a-title', size=18)
    for i in range(20):
        _write(paper, 57, 160 + 12 * i, f'The body of the paper is set in the body size and runs on, line {i}.')
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == ['# Advances in Dual-Three-Phase Machines']


def test_back_matter_set_as_large_as_the_chapters_takes_their_level(tmp_path):
    """Thesis W7220810123 sets ACKNOWLEDGMENTS and REFERENCES at its chapters' 20.66 pt: they are chapters too, not
    sections of the chapter before.
    """
    doc = pymupdf.open()
    for heading in ('ACKNOWLEDGMENTS', 'CHAPTER 1', 'REFERENCES'):
        page = doc.new_page(width=612, height=792)
        _write(page, 72, 100, heading, size=20.66)
        for i in range(20):
            _write(page, 72, 140 + 14 * i, f'The body is set in the body size and runs on here, line {i}.', size=12)
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == ['# ACKNOWLEDGMENTS', '# CHAPTER 1', '# REFERENCES']


def test_a_numbered_line_in_its_own_face_heads_where_it_continues_the_numbering(tmp_path):
    """W7153671269 sets its sections in bold and its subsections in italic (``2.1. Basics of magnetostatics``), arXiv
    2204.08094 its ``3.1.1 GuitarSet`` in italic under a bold ``3.1``, a manual its subsections in a larger roman under
    a title-size ``1.``: a numbered line set wholly in a face other than the body's heads at its depth when its parent
    number already heads -- ``2.1`` after ``2``. A list item whose number is in the body's face stays text.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, 72, 80, '2. Fundamentals', font='tibo')
    _write(page, 72, 100, 'The body of the section is set in the body face, in roman, at the body size.')
    _write(page, 72, 125, '2.1. Basics of magnetostatics', font='tiit')
    _write(page, 72, 145, 'The body of the subsection is set in the body face, in roman, at the body size.')
    _write(page, 72, 170, '2.1.1 GuitarSet', font='tiit')
    _write(page, 72, 190, 'The body of the subsubsection is set in the body face, in roman, at the body size.')
    _write(page, 72, 215, '1.')
    _write(page, 86, 215, 'Fully immersed', font='tiit')  # a list item: its number is the body's face
    _write(page, 72, 240, '4.1 An orphan subsection', font='tiit')  # no section 4 heads
    manual = doc.new_page(width=612, height=792)
    _write(manual, 72, 80, '3. Convection', size=20)
    _write(manual, 72, 110, 'The body of the chapter is set in the body face, in roman, at the body size.')
    _write(manual, 72, 135, '3.2 Natural convection', size=14)
    _write(manual, 72, 160, 'The body of the subsection is set in the body face, in roman, at the body size.')
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == [
        '## 2. Fundamentals',
        '### 2.1. Basics of magnetostatics',
        '#### 2.1.1 GuitarSet',
        '# 3. Convection',
        '### 3.2 Natural convection',
    ]


def test_a_data_sheets_numbered_rows_in_another_face_head_nothing(tmp_path):
    """Data sheets set short numbered rows in faces other than the body's (``09 August 2018``, ``0 A/m``,
    ``13.56 MHz``, ``1. Chamfered holes and sides.``): none continues a numbering the document heads, so none heads --
    a top-level number never heads by its face, nor a ``2.5`` whose ``2`` heads nothing (or heads only later).
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _write(page, 57, 80, '1 Scope', font='tibo')
    _write(page, 57, 100, 'The cores of this sheet are made of a ferrite material for power use.')
    rows = ('09 August 2018', '13.56 MHz', '2.5 Height of the core', '1. Chamfered holes and sides.', '0 A/m')
    for i, row in enumerate(rows):
        _write(page, 57, 130 + 20 * i, row, size=8, font='helv')
    _write(page, 57, 250, '2 Dimensions', font='tibo')
    doc.save(tmp_path / 's.pdf')
    assert _heads(to_markdown(read_pdf(tmp_path / 's.pdf'))) == ['## 1 Scope', '## 2 Dimensions']
