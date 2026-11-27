"""The IR markdown writer on a synthetic two-column page whose reading order and headings are known exactly."""

import pymupdf
import pytest

from document_lab.markdown import headings, page_markdowns, to_markdown
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


def test_text_of_a_layer_turned_off_is_written_only_when_asked(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, LEFT, 120, 'Shown in every view.')
    _write(page, LEFT, 160, 'Drawn on a layer turned off.')
    page.insert_text((LEFT, 200), 'Off too.', fontname='tiro', fontsize=10, oc=doc.add_ocg('Off', on=False))
    doc.save(tmp_path / 'oc.pdf')
    read = read_pdf(tmp_path / 'oc.pdf')
    assert 'Off too.' not in to_markdown(read)
    assert 'Off too.' not in page_markdowns(read)[0]
    assert 'Off too.' in to_markdown(read, include_hidden=True)
    assert 'Off too.' in page_markdowns(read, include_hidden=True)[0]


def _outlined(tmp_path, toc, lines_per_page):
    """A PDF whose pages set ``lines_per_page`` (each ``(y, text, size)``) and whose outline is ``toc``."""
    doc = pymupdf.open()
    for lines in lines_per_page:
        page = doc.new_page(width=612, height=792)
        for y, text, size in lines:
            _write(page, LEFT, y, text, size=size)
    doc.set_toc(
        [
            [level, title, page, {'kind': pymupdf.LINK_GOTO, 'page': page - 1, 'to': pymupdf.Point(LEFT, y)}]
            for level, title, page, y in toc
        ]
    )
    doc.save(tmp_path / 'o.pdf')
    return read_pdf(tmp_path / 'o.pdf')


BODY = 'The body of a section runs on in plain words for a line.'


def test_a_stated_outline_names_the_sections_the_rules_miss(tmp_path):
    doc = _outlined(
        tmp_path,
        [(1, 'Overview', 1, 90), (2, 'Scope of work', 1, 150), (1, 'Results and their meaning', 2, 90)],
        [
            [(100, 'Overview', 12), (120, BODY, 10), (160, 'Scope of', 11), (174, 'work', 11), (190, BODY, 10)],
            [(100, 'RESULTS AND THEIR MEANING', 12), (120, BODY, 10)],
        ],
    )
    found = headings(doc)
    assert (found.source, found.stated, found.disagreements) == ('outline', 3, ())
    md = to_markdown(doc, sections=found)
    assert [line for line in md.splitlines() if line.startswith('#')] == [
        '## Overview',
        '### Scope of work',  # a title wrapped over two lines is one heading
        '## RESULTS AND THEIR MEANING',  # matched up to case
    ]
    inferred = headings(doc, outline=False)
    assert (inferred.source, inferred.stated) == ('inferred', 0)
    assert not any(line.startswith('#') for line in to_markdown(doc, sections=inferred).splitlines())


def test_an_outline_not_found_on_its_pages_leaves_the_rules_to_head_and_says_why(tmp_path):
    doc = _outlined(
        tmp_path,
        [(1, 'Introduction', 1, 90), (1, 'Page two', 2, 90)],
        [[(100, '1 Introduction', 12), (120, BODY, 10)], [(100, '2 Method', 12), (120, BODY, 10)]],
    )
    found = headings(doc)
    assert found.source == 'inferred'
    assert found.unlocated == ("'Page two' (page 2)",)
    assert found.disagreements == ("outline item 'Page two' (page 2) is not found on its page",)
    md = to_markdown(doc, sections=found)
    assert [line for line in md.splitlines() if line.startswith('#')] == ['## 1 Introduction', '## 2 Method']


def test_headings_the_rules_find_beside_a_stated_outline_are_reported(tmp_path):
    doc = _outlined(
        tmp_path,
        [(1, 'Introduction', 1, 90), (1, 'Method', 1, 690)],  # Method opens at the top of page 2, its anchor above
        [
            [(100, '1 Introduction', 12), (120, BODY, 10), (300, '1.1 Aside', 12), (320, BODY, 10)],
            [(100, '2 Method', 12), (120, BODY, 10)],
        ],
    )
    found = headings(doc)
    assert found.source == 'outline'
    assert found.unstated == ('1.1 Aside',)
    md = page_markdowns(doc, sections=found)
    assert md[1].startswith('## 2 Method')  # found on the page after its destination's
    assert '1.1 Aside' in md[0]
    assert not any(line.startswith('#') and 'Aside' in line for line in md[0].splitlines())  # the outline alone heads


def test_an_outline_naming_none_of_the_headings_the_rules_find_is_no_section_structure(tmp_path):
    """A repository's cover sheet: its outline names the title and the terms of use, found there, not the sections."""
    doc = _outlined(
        tmp_path,
        [(1, 'Terms of use', 1, 90)],
        [[(100, 'Terms of use', 10), (120, BODY, 10)], [(100, '1 Introduction', 12), (120, BODY, 10)]],
    )
    found = headings(doc)
    assert (found.source, found.shared, found.unstated) == ('inferred', 0, ('1 Introduction',))
    assert '## 1 Introduction' in to_markdown(doc, sections=found)


def test_roman_numbers_and_untitled_items_do_not_keep_an_outline_from_its_pages(tmp_path):
    doc = _outlined(
        tmp_path,
        [(1, 'I Introduction', 1, 90), (1, '', 1, 90), (1, 'II Method', 2, 90)],
        [[(100, 'I. INTRODUCTION', 12), (120, BODY, 10)], [(100, 'II. METHOD', 12), (120, BODY, 10)]],
    )
    found = headings(doc)
    assert (found.source, found.shared, found.untitled) == ('outline', 2, ('page 1',))
    assert found.disagreements == ('an outline item (page 1) states no title: it names no heading',)
