"""The ``ocr`` part's scanned tables: a ruled table drawn, scanned (grey pixels, turned, noisy, its rules broken or
doubled), given a text layer by `document_lab.ocr` and read back -- its rules from its picture (`ink.rulings`), its
cells' text from the OCR words -- is the table drawn: every cell where the scan puts it, every word OCR read in the
cell the drawing has it in, nearly every text as drawn (Tesseract misreads a glyph now and then, which no table can
mend; around broken rules it misreads many), and every planted wrong truth reported. Its own part: it needs
Tesseract with English data, and FAILS naming what is missing where there is none."""

import dataclasses
import os

import pytest

from document_lab.ink import rulings
from document_lab.ir import BBox
from document_lab.ocr import TesseractPlaces, check, text_layer
from document_lab.pdf import read_pdf, read_raster
from document_lab.tables import STROKES, tables
from tests.synthetic import (
    MERGED,
    MERGED_ROWS,
    SCAN_PAGE,
    TABLE_ROWS,
    broken_rules,
    double_border,
    draw_ruled_table,
    scan_image,
    scan_pdf,
    table_cell_rect,
    table_differences,
    turned,
)

pytestmark = pytest.mark.ocr

PLACES = TesseractPlaces.of(os.environ)
#: Cells whose OCR'd text may differ from the drawn one's, of twelve: Tesseract's misreads, measured at most 3 where
#: the rules are whole (around broken rules it read 7 wrong, so their text is not compared).
MISREAD: int = 3


def _merged(page):
    draw_ruled_table(page, MERGED_ROWS, merged=MERGED)


#: name: (draws, rows, merged, scan options, how near each cell's box is in points, whether its text is compared)
CASES = {
    'clean': ((draw_ruled_table,), TABLE_ROWS, (), {}, 0.5, True),
    'merged cells': ((_merged,), MERGED_ROWS, MERGED, {'angle': -1.3, 'noise': 1.0, 'seed': 7}, 0.5, True),
    'skewed': ((draw_ruled_table,), TABLE_ROWS, (), {'angle': 1.0}, 0.5, True),
    'skewed and noisy': ((draw_ruled_table,), TABLE_ROWS, (), {'angle': 1.5, 'noise': 1.0, 'seed': 2}, 0.5, True),
    '200 dpi': ((draw_ruled_table,), TABLE_ROWS, (), {'dpi': 200, 'angle': 0.5, 'noise': 0.5}, 0.5, True),
    'broken rules': ((draw_ruled_table, broken_rules), TABLE_ROWS, (), {'angle': 0.4, 'noise': 0.5}, 0.5, False),
    'double outline': ((draw_ruled_table, double_border), TABLE_ROWS, (), {'angle': 0.8, 'noise': 0.5}, 1.3, True),
}


def _read(tmp_path, draws, options):
    """The scanned page read back after OCR, its tables, and its measured skew."""
    check('eng', PLACES)
    scan = scan_pdf(tmp_path / 'scan.pdf', scan_image(*draws, **options))
    assert len(text_layer(scan, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    page = read_pdf(tmp_path / 'ocr.pdf').page(1)
    pixels = read_raster(tmp_path / 'ocr.pdf', 1, BBox(0, 0, page.width, page.height), dpi=None).pixels
    found = rulings(pixels, pixels.shape[1] / SCAN_PAGE.width)
    return page, tables(page, rulings=found), found.skew


def _misplaced(page, table, rows, angle):
    """The OCR words missing from the text of the cell the drawing has them in (a rule read as a stroke aside)."""
    out = []
    for span in page.spans:
        box = span.provenance.bbox
        x, y = turned((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, -angle)  # where it is drawn
        cell = next(
            ((r, c) for r in range(len(rows)) for c in range(len(rows[0])) if table_cell_rect(r, c).contains((x, y))),
            None,
        )
        if cell is not None and not set(span.text) <= STROKES and span.text not in table.rows[cell[0]][cell[1]]:
            out.append(f'{span.text!r} drawn in cell {cell}, not in {table.rows[cell[0]][cell[1]]!r}')
    return out


@pytest.mark.parametrize('case', CASES)
def test_a_scanned_table_ocrd_reads_back_as_the_table_drawn(tmp_path, case):
    draws, rows, merged, options, within, compared = CASES[case]
    page, found, skew = _read(tmp_path, draws, options)
    angle = options.get('angle', 0.0)
    assert skew == pytest.approx(angle, abs=0.05)
    (table,) = found
    structure, text = table_differences(table, rows, merged=merged, angle=angle, within=within)
    assert structure == []
    assert _misplaced(page, table, rows, angle) == []
    if compared:
        assert len(text) <= MISREAD, text


def test_every_planted_wrong_truth_is_reported(tmp_path):
    """The same scan against a truth that is not it: each comparison that passes above reports it."""
    page, (table,), _ = _read(tmp_path, (draw_ruled_table,), {'angle': 1.0})
    swapped = [(r[1], r[0], r[2]) for r in TABLE_ROWS]
    assert len(table_differences(table, swapped, angle=1.0)[1]) > MISREAD  # columns swapped
    assert table_differences(table, TABLE_ROWS, merged=((1, 1, 2, 1),), angle=1.0)[0]  # a merged cell not there
    assert table_differences(table, TABLE_ROWS, angle=-1.0)[0]  # the skew the other way
    assert table_differences(table, TABLE_ROWS[:3], angle=1.0)[0]  # a row fewer
    shifted = dataclasses.replace(table, rows=tuple((r[2], r[0], r[1]) for r in table.rows))
    assert _misplaced(page, shifted, TABLE_ROWS, 1.0)  # its words in other cells than drawn
