"""The IR markdown writer on a synthetic two-column page whose reading order and headings are known exactly."""

import pymupdf
import pytest

from document_lab.paper.markdown import to_markdown
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
    return to_markdown(read_pdf(tmp_path / 'p.pdf', tables=False))


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
    headings = [ln for ln in to_markdown(read_pdf(tmp_path / 'toc.pdf', tables=False)).splitlines() if ln[:1] == '#']
    assert headings == ['# A Standard', '## 1 Scope']


def test_one_contents_line_is_not_a_heading_but_a_page_of_them_is_a_contents_page(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _write(page, LEFT, 100, 'II. METHOD ........ 4')
    _write(page, LEFT, 120, 'III. RESULTS')  # the control: a real heading on a page with one entry stays one
    doc.save(tmp_path / 'one.pdf')
    headings = [ln for ln in to_markdown(read_pdf(tmp_path / 'one.pdf', tables=False)).splitlines() if ln[:1] == '#']
    assert headings == ['## III. RESULTS']
