"""The ``ocr`` part's orientation and skew: synthetic scans Tesseract misreads as placed are read right.

A landscape table printed sideways on a portrait page under an upright running head (as standards set them) is read
turned as OSD says beside the page as placed: the head and the table both read, and the table, its rules found in
the picture, reads right side up. A page fed askew by 3, 5 and 8 degrees, of which Tesseract reads almost nothing as
placed, is read turned back, its words laid back where the page shows them. Each check is paired with the reading it
replaces, which must fail it. Its own part: it needs Tesseract with English and OSD data, and FAILS naming what is
missing where there is none.
"""

import os
from collections import Counter

import numpy as np
import pymupdf
import pytest

from document_lab.ink import rulings
from document_lab.ir import BBox
from document_lab.ocr import TesseractPlaces, check, text_layer
from document_lab.pdf import read_pdf, read_raster
from document_lab.tables import tables
from tests.synthetic import (
    SCAN_PAGE,
    TABLE_ROWS,
    draw_ruled_table,
    draw_sideways_table,
    scan_image,
    scan_pdf,
    turned,
)

pytestmark = pytest.mark.ocr

PLACES = TesseractPlaces.of(os.environ)
#: Cells of twelve whose OCR'd text may differ from the drawn one's (Tesseract's misreads; as tests/ocr's tables).
MISREAD = 3
HEAD = 'Magnetic materials specifications for individual materials'
NOTES = (
    'Typical values of the additional magnetic properties are given for information only',
    'Temperature coefficient of remanent flux density for twenty to one hundred degrees',
)


def _sideways_page(page):
    """An upright running head over a table printed sideways, its caption and notes sideways with it."""
    page.insert_text((72, 50), HEAD, fontsize=11)
    draw_sideways_table(page)
    for i, line in enumerate(('Table 7 Magnetic properties of the materials', *NOTES)):
        x, y = turned(72, 520 + 16 * i, -90)
        page.insert_text((x, y), line, fontsize=10, rotate=90)


def _read(tmp_path, grey, **options):
    check('eng+osd', PLACES)
    scan = scan_pdf(tmp_path / 'scan.pdf', grey)
    out = tmp_path / f'ocr-{len(list(tmp_path.iterdir()))}.pdf'
    (reading,) = text_layer(scan, out, places=PLACES, **options).readings
    return reading, out


def _table(pdf):
    page = read_pdf(pdf).page(1)
    pixels = read_raster(pdf, 1, BBox(0, 0, page.width, page.height), dpi=None).pixels
    return page, tables(page, rulings=rulings(pixels, pixels.shape[1] / SCAN_PAGE.width))


def _misread(table):
    return [
        (table.rows[r][c], cell)
        for r, row in enumerate(TABLE_ROWS)
        for c, cell in enumerate(row)
        if r < len(table.rows) and c < len(table.rows[r]) and table.rows[r][c] != cell
    ]


def test_a_table_printed_sideways_reads_right_side_up_under_its_upright_head(tmp_path):
    grey = scan_image(_sideways_page, dpi=200, noise=0.5, seed=5)
    reading, pdf = _read(tmp_path, grey)
    assert (reading.turn, reading.unsure) == (90, False), reading
    page, (table,) = _table(pdf)
    assert len(table.rows) == len(TABLE_ROWS)
    assert len(_misread(table)) <= MISREAD, table.rows
    text = ' '.join(s.text for s in page.spans)
    assert all(word in text for word in HEAD.split()), text  # the head, read as placed, is kept beside
    assert all(word in text for word in NOTES[0].split()), text
    # The reading it replaces: as placed, Tesseract reads the sideways text reversed.
    _, placed = _read(tmp_path, grey, orientation_confidence=None)
    _, found = _table(placed)
    assert not found or len(_misread(found[0])) > MISREAD
    assert sum(word in ' '.join(s.text for s in read_pdf(placed).page(1).spans) for word in NOTES[0].split()) < 5


PROSE = (
    'The rated output of a machine is the value of the output included in the rating',
    'Each machine shall be marked with the rated voltage and the rated frequency',
    'Temperature rise of windings shall not exceed the limits given in the table below',
)


def _prose(page):
    for i, line in enumerate(PROSE * 4):
        page.insert_text((72, 90 + 18 * i), line, fontsize=10)


DRAWN = Counter(w for line in PROSE * 4 for w in line.split()) + Counter(
    w for row in TABLE_ROWS for cell in row for w in cell.split()
)


def _words(pdf):
    return sum((Counter(w for s in read_pdf(pdf).page(1).spans for w in s.text.split()) & DRAWN).values())


@pytest.mark.parametrize('degrees', [3.0, 5.0, 8.0])
def test_a_page_fed_askew_is_read_turned_back_its_words_where_the_page_shows_them(tmp_path, degrees):
    grey = scan_image(_prose, draw_ruled_table, dpi=300, angle=degrees, noise=0.5, seed=6)
    reading, pdf = _read(tmp_path, grey)
    assert reading.skew == pytest.approx(degrees, abs=0.1)
    assert reading.deskewed
    after = _words(pdf)
    _, placed = _read(tmp_path, grey, deskew_above=None)
    before = _words(placed)
    print(f'{degrees} degrees: {before} of {sum(DRAWN.values())} words read as placed, {after} deskewed')  # noqa: T201
    assert after >= 0.9 * sum(DRAWN.values())
    assert before < after / 2  # the reading it replaces
    spans = {s.text: s.provenance.bbox for s in read_pdf(pdf).page(1).spans}
    with pymupdf.open() as doc:  # where each cell word is drawn, before the page is fed askew
        page = doc.new_page(width=SCAN_PAGE.width, height=SCAN_PAGE.height)
        draw_ruled_table(page)
        drawn = {w[4]: w[:4] for w in page.get_text('words')}
    for word in ('Parameter', 'alpha', 'beta', 'gamma'):  # each alone in its cell, where the skewed page shows it
        x0, y0, x1, y1 = drawn[word]
        x, y = turned((x0 + x1) / 2, (y0 + y1) / 2, degrees)
        box = spans[word]
        off = np.hypot((box.x0 + box.x1) / 2 - x, (box.y0 + box.y1) / 2 - y)
        assert off < 3, (word, box, (x, y))  # a font's box (ascent to descent) beside Tesseract's ink box: 2.2 pt
