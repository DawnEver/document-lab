"""A page's figures, tables and equations as regions, found from its text alone.

Nothing here reads a drawing or an image: an OCR'd capture is one picture with words over it, so a region is told
by its words. A figure's caption (``Figure 2 - ...``) is below it: the figure runs up from the caption to the
nearest prose above. A table's caption is above it: the table runs down from the caption to the next prose, clause
heading or caption. An equation number standing alone at the right (``(3)``, ``(A.1)``) is an equation: its region
takes in every line beside it that is not prose -- a formula OCR'd reads as noise -- up to the prose around it.
Prose is a line that reads as words (`document_lab.layout.wordy`), set near the body size, across half the text
width or :data:`PROSE_WORDS` words long. A mention in prose (``Table 7, item 1d``) or a contents entry (a leader) is
no caption. Each region spans the text width; its label is the caption's or the number's, as read.

A table without a caption (a standard's normative-reference annex, a list set in columns) is told by its rows:
:data:`TABLE_ROWS` or more bands of :data:`ROW_CELLS` or more cells side by side, set near the body size, no prose
among them; a cell's further lines -- words, starting within one of the row's cells -- belong to it, and anything
else (a formula, a note, prose) ends it. It runs from its first row to its last and is
labelled ``Untitled table N`` in page order. Two columns of prose have two cells a band, so they are no table; a
band a captioned region already holds is not found again. A table whose cells OCR read no words from has nothing to
be told by here, and is not found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from document_lab import layout
from document_lab.ir import BBox

if TYPE_CHECKING:
    from document_lab.ir import Page

__all__ = ['Region', 'regions']

#: A caption: ``Figure``/``Table``, its number (``2``, ``A.1``), then a dash, as standards set it.
_CAPTION: Final = re.compile(
    r'^(Figure|Table)\s+([A-Z]?\.?\d+(?:\.\d+)?)\s*[-\N{HYPHEN}\N{EN DASH}\N{EM DASH}\N{MINUS SIGN}]'
)
#: An equation number on a line of its own: ``(3)``, ``(A.1)``.
_NUMBER: Final = re.compile(r'^\(([A-Z]?\.?\d+(?:\.\d+)?)\)$')
#: An equation number stands right of this share of the page width.
NUMBER_RIGHT: Final = 0.6
#: Prose is set within this factor of the body size, across at least this share of the text width.
PROSE_SIZE: Final = 1.4
PROSE_WIDTH: Final = 0.5
#: A line of at least this many words is prose at any width (a column of a two-column page); no table cell's is.
PROSE_WORDS: Final = 12
#: A region stops this many body sizes short of the prose that bounds it.
PAD: Final = 0.3
#: An equation takes a line not prose within this many body sizes of it, above or below.
EQUATION_REACH: Final = 2.0
#: An untitled table: at least this many rows of at least this many cells, each cell set within ROW_SIZE of the body
#: size, a row at most ROW_GAP body sizes below the table so far.
TABLE_ROWS: Final = 2
ROW_CELLS: Final = 3
ROW_SIZE: Final = 1.5
ROW_GAP: Final = 3.0
#: A table's row has a short cell -- at most this many words -- among its cells; columns of prose do not.
SHORT_CELL: Final = 3
#: A cell reads at least this many letters or digits (or is a dash); fewer is a speck OCR read.
CELL_CHARS: Final = 2
_DASHES: Final = frozenset('-\N{HYPHEN}\N{EN DASH}\N{EM DASH}\N{MINUS SIGN}')
_HEADING: Final = re.compile(r'^\d+(?:\.\d+)*\.?\s+[A-Z]')

Kind = Literal['figure', 'table', 'equation']


@dataclass(frozen=True)
class Region:
    """A ``figure``, ``table`` or ``equation`` region of a page: its ``label`` as read, its ``bbox`` in points."""

    kind: Kind
    label: str
    bbox: BBox


class _Page:
    """A page's lines with what regions are judged by: its body size and text extent, and which lines are prose."""

    def __init__(self, page: Page, body: float) -> None:
        self.page, self.body = page, body
        self.lines = sorted(layout.lines(page, weighted=False), key=lambda ln: ln.y0)
        # The text width: every line set near the body size -- words or not, a table's cells and a page's furniture
        # count -- so one wide column of words is not taken for the whole width.
        text = [ln for ln in self.lines if body / ROW_SIZE <= ln.size <= ROW_SIZE * body]
        self.left = min((ln.x0 for ln in text), default=0.0)
        self.right = max((ln.x1 for ln in text), default=page.width)

    def prose(self, line: layout.Line) -> bool:
        return (
            layout.wordy(line.text)
            and line.size <= PROSE_SIZE * self.body
            and '....' not in line.text
            and (line.x1 - line.x0 >= PROSE_WIDTH * (self.right - self.left) or len(line.text.split()) >= PROSE_WORDS)
        )

    def box(self, top: float, bottom: float) -> BBox:
        side = PAD * self.body  # as above and below: a glyph's edge is not cut
        return BBox(
            max(self.left - side, 0.0),
            max(top, 0.0),
            min(self.right + side, self.page.width),
            min(bottom, self.page.height),
        )


def _caption(line: layout.Line) -> tuple[str, str] | None:
    if '....' in line.text or (found := _CAPTION.match(line.text)) is None:
        return None
    return found.group(1).lower(), f'{found.group(1)} {found.group(2)}'


def _figure(page: _Page, caption: layout.Line, label: str) -> Region:
    above = [ln.y1 for ln in page.lines if ln.y1 <= caption.y0 and page.prose(ln)]
    top = max(above) + PAD * page.body if above else 0.0
    return Region('figure', label, page.box(top, caption.y1 + PAD * page.body))


def _table(page: _Page, caption: layout.Line, label: str) -> Region:
    def ends(line: layout.Line) -> bool:
        return page.prose(line) or _caption(line) is not None

    below = [ln.y0 for ln in page.lines if ln.y0 >= caption.y1 and ends(ln)]
    bottom = min(below) - PAD * page.body if below else page.page.height
    return Region('table', label, page.box(caption.y0 - PAD * page.body, bottom))


def _equation(page: _Page, number: layout.Line, label: str) -> Region:
    top, bottom = number.y0, number.y1
    grew = True
    while grew:  # take in every line not prose within reach, until none is left
        grew = False
        for line in page.lines:
            near = line.y1 >= top - EQUATION_REACH * page.body and line.y0 <= bottom + EQUATION_REACH * page.body
            if near and not page.prose(line) and (line.y0 < top or line.y1 > bottom):
                top, bottom, grew = min(top, line.y0), max(bottom, line.y1), True
    return Region('equation', label, page.box(top - PAD * page.body, bottom + PAD * page.body))


def _bands(page: _Page) -> list[list[layout.Line]]:
    """The page's lines in bands, top to bottom: lines whose middles lie within half a body size share one."""
    bands: list[list[layout.Line]] = []
    for line in page.lines:
        middle = (line.y0 + line.y1) / 2
        if bands and abs(middle - sum((ln.y0 + ln.y1) / 2 for ln in bands[-1]) / len(bands[-1])) <= page.body / 2:
            bands[-1].append(line)
        else:
            bands.append([line])
    return bands


def _noise(text: str) -> bool:
    """A speck OCR read as a cell: under :data:`CELL_CHARS` letters or digits, and not a dash (an empty entry)."""
    stripped = text.strip()
    return sum(c.isalnum() for c in stripped) < CELL_CHARS and stripped not in _DASHES


def _row(page: _Page, band: list[layout.Line]) -> bool:
    """Is ``band`` a table's row: cells side by side, none of them prose nor a contents entry?

    :data:`ROW_CELLS` of them are set near the body size and none larger (a lone ``-`` may read smaller); one is short,
    at most :data:`SHORT_CELL` words -- a code, a year, a dash -- which columns of prose, however OCR splits them,
    are not.
    """
    near = [ln for ln in band if page.body / ROW_SIZE <= ln.size <= ROW_SIZE * page.body]
    short = any(len(ln.text.split()) <= SHORT_CELL for ln in band)  # a code, a year, a '-': columns of prose have none
    return (
        len(near) >= ROW_CELLS
        and short
        and not any(_noise(ln.text) for ln in band)
        and all(ln.size <= ROW_SIZE * page.body and not page.prose(ln) and '....' not in ln.text for ln in band)
    )


def _continues(page: _Page, band: list[layout.Line], row: list[layout.Line]) -> bool:
    """Is ``band`` a further line of the cells of ``row``, the table's last row?

    Each of its lines reads as words -- a formula OCR'd does not -- and starts within one of the row's cells (as a
    wrapped title starts under its own); none is prose, a clause heading, a caption or a contents entry.
    """

    def in_cell(line: layout.Line) -> bool:
        return any(cell.x0 - page.body <= line.x0 <= cell.x1 for cell in row)

    return all(
        layout.wordy(ln.text)
        and in_cell(ln)
        and not page.prose(ln)
        and not _HEADING.match(ln.text)
        and _caption(ln) is None
        and '....' not in ln.text
        for ln in band
    )


def _untitled_tables(page: _Page, held: list[Region]) -> list[Region]:
    found: list[Region] = []
    bands = _bands(page)
    i = 0
    while i < len(bands):
        if not _row(page, bands[i]):
            i += 1
            continue
        first, last, rows, j = bands[i], bands[i], 1, i + 1
        bottom = max(ln.y1 for ln in first)
        while j < len(bands) and min(ln.y0 for ln in bands[j]) - bottom <= ROW_GAP * page.body:
            if _row(page, bands[j]):
                last, rows = bands[j], rows + 1
            elif not _continues(page, bands[j], last):
                break
            bottom = max(bottom, *(ln.y1 for ln in bands[j]))
            j += 1
        top = min(ln.y0 for ln in first)
        inside = any(r.bbox.y0 <= top <= r.bbox.y1 for r in held)
        if rows >= TABLE_ROWS and not inside:
            box = page.box(top - PAD * page.body, bottom + PAD * page.body)
            found.append(Region('table', f'Untitled table {len(found) + 1}', box))
        i = j if rows >= TABLE_ROWS else i + 1
    return found


def regions(page: Page, body: float) -> list[Region]:
    """The figures, tables and equations of ``page`` (``body``: the document's body size), top to bottom."""
    if not page.spans or body <= 0:
        return []
    judged = _Page(page, body)
    found: list[Region] = []
    for line in judged.lines:
        if caption := _caption(line):
            kind, label = caption
            found.append(_figure(judged, line, label) if kind == 'figure' else _table(judged, line, label))
        elif (number := _NUMBER.match(line.text)) and line.x0 >= NUMBER_RIGHT * page.width:
            found.append(_equation(judged, line, f'({number.group(1)})'))
    found += _untitled_tables(judged, found)
    return sorted(found, key=lambda r: r.bbox.y0)
