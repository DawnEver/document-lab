"""A page's figures, tables and equations as regions, found from its text.

Nothing here reads pixels: an OCR'd capture is one picture with words over it, so a region is told by its words.
A figure's caption (``Figure 2 - ...``) is below it: the figure runs up from the caption to the nearest prose above.
A table's caption is above it: the table runs down from the caption to the next prose, clause heading or caption.
Prose is a line that reads as words (`document_lab.layout.wordy`), set near the body size, across half the text
width or :data:`PROSE_WORDS` words long. A mention in prose (``Table 7, item 1d``) or a contents entry (a leader) is
no caption. A figure or table spans the text width; its label is the caption's, as read.

An equation number ending a line right of :data:`NUMBER_RIGHT` of its column -- alone, or after a formula (``(3)``,
``(A.1)``) -- is an equation, cut out by :func:`equation`: its formula's lines from the number to the prose around it,
in the number's column on a page set in two columns there (a left column's number stands short of the page's middle),
across both when the formula does, with the bars and small images set among its lines; a formula OCR'd reads as noise
and is taken all the same. One function crops every numbered equation -- a paper's (`document_lab.paper`) and a captured
standard's -- so each is cut alike.

A table without a caption (a standard's normative-reference annex, a list set in columns) is told by its rows:
:data:`TABLE_ROWS` or more bands of :data:`ROW_CELLS` or more cells side by side, set near the body size, no prose
among them; a cell's further lines -- words, starting within one of the row's cells -- belong to it, and anything
else (a formula, a note, prose) ends it. It runs from its first row to its last and is
labeled ``Untitled table N`` in page order. Two columns of prose have two cells a band, so they are no table; a
band a captioned region already holds is not found again. A table whose cells OCR read no words from has nothing to
be told by here, and is not found.

The caller that holds the page's picture may also hand over its drawings (`document_lab.ink`): a drawing no caption,
number or row of cells told is a region too -- a ruled table of two columns, a figure without a caption
(``Untitled figure N``), an equation without a number (``Untitled equation N``). Regions still read no pixels: a
drawing is its box and what it is drawn with. Drawings are one figure unless prose runs between them or, set apart by
white space, each stands under its own title (a data sheet's dimensional drawing, circuit and reel); a short drawing
is an equation only when its text reads as math (a sign, a Greek letter, a math font) -- a logo is a figure. A region
the page's edge leaves without area (its text set across the edge) shows nothing and is none.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from document_lab import layout
from document_lab.ir import BBox, union
from document_lab.tables import SHORT_CELL, aligned

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.ink import Drawing
    from document_lab.ir import Page, Table, TextSpan

__all__ = ['Region', 'equation', 'regions']

#: A caption: ``Figure``/``Table``, its number (``2``, ``A.1``), then a dash, as standards set it.
_CAPTION: Final = re.compile(
    r'^(Figure|Table)\s+([A-Z]?\.?\d+(?:\.\d+)?)\s*[-\N{HYPHEN}\N{EN DASH}\N{EM DASH}\N{MINUS SIGN}]'
)
#: An equation number ending a line: ``(3)``, ``(12b)``, an annex's ``(A.1)``, a chapter's ``(3.2)`` -- never a value
#: such as ``(0.167)``, as no number starts with 0.
_NUMBERED: Final = re.compile(r'\(\s*((?:[A-Z]\.?)?[1-9]\d*(?:\.\d+)*[a-z]?)\s*\)$')
#: A number found on a page ends right of this share of its column's width (a list's ``(1)`` stands at its left).
NUMBER_RIGHT: Final = 0.6
#: Prose is set within this factor of the body size; it spans at least PROSE_WIDTH of the width it is set across --
#: the text's (around a caption), its column's (around an equation).
PROSE_SIZE: Final = 1.4
PROSE_WIDTH: Final = 0.5
#: A line of at least this many words is prose at any width (a column of a two-column page); no table cell's is.
PROSE_WORDS: Final = 12
#: A region stops this many body sizes short of the prose that bounds it.
PAD: Final = 0.3
#: An equation takes a piece within this many body sizes of what it has taken, above or below; OCR_REACH where its
#: number is OCR'd (an invisible text layer), whose boxes hug the ink where a set font's carry its ascent and descent.
EQUATION_REACH: Final = 1.0
OCR_REACH: Final = 2.0
#: An equation's box stands this many body sizes clear of what it holds, so no glyph's edge is cut.
EQUATION_PAD: Final = 0.3
#: An image taller than this many body sizes is a figure, not an equation set as a picture.
EQUATION_IMAGE: Final = 10.0
#: An equation's prose may instead start within this many body sizes of its column's left edge (a paragraph's tail).
PROSE_INDENT: Final = 1.5
#: A line of at least this many words that reads as words is the page's text: its columns and its extent.
TEXT_WORDS: Final = 5
#: Lines of text within this many body sizes of an equation's number tell whether the page is set in two columns there.
COLUMN_WINDOW: Final = 20.0
_WORD: Final = re.compile(r'[^\W\d_]{4,}')  # a word of four letters
_LETTERS: Final = re.compile(r'[^\W\d_]+(?:-[^\W\d_]+)*')  # letters alone, hyphened or not: Context-Based
_PUNCTUATION: Final = '.,:;!?()[]{}"\''
#: An untitled table: at least this many rows of at least this many cells, each cell set within ROW_SIZE of the body
#: size, a row at most ROW_GAP body sizes below the table so far.
TABLE_ROWS: Final = 2
ROW_CELLS: Final = 3
ROW_SIZE: Final = 1.5
ROW_GAP: Final = 3.0
#: A cell reads at least this many letters or digits (or is a dash); fewer is a speck OCR read.
CELL_CHARS: Final = 2
#: The PDF text render mode of an OCR layer: its glyphs invisible over the picture they were read from.
INVISIBLE: Final = 3
_DASHES: Final = frozenset('-\N{HYPHEN}\N{EN DASH}\N{EM DASH}\N{MINUS SIGN}')
#: A drawing at least this many body sizes high and wide may be a region (`document_lab.ink`); a rule is not.
DRAWN_MIN: Final = 1.5
#: ...and draws a straight stroke at least this many body sizes long: words OCR missed are glyphs, none so long.
DRAWN_STROKE: Final = 1.5
#: A drawing of at least this many rules across it, only words off them and TABLE_ROWS lines of text in it, is a table
#: even where lines long enough to be prose cross it (a data sheet's names and values, its rows ruled apart; a
#: striped icon holds no text).
RULED_ROWS: Final = 3
#: A drawing in a text line's band at most this many times the line's height frames its words (a link, a header).
FRAME_LINES: Final = 3.0
FRAME_FILL: Final = 0.8
#: A drawing at most this many body sizes high whose text reads as no word, and as math, is an equation without a
#: number.
EQUATION_HEIGHT: Final = 4.0
#: Math: a relation or operator sign, a Greek letter, or a math font (TeX's, a Unicode math font).
_MATH_SIGNS: Final = frozenset(
    '=<>+'
    '\N{MINUS SIGN}\N{PLUS-MINUS SIGN}\N{MULTIPLICATION SIGN}\N{DIVISION SIGN}\N{N-ARY SUMMATION}'
    '\N{N-ARY PRODUCT}\N{INTEGRAL}\N{SQUARE ROOT}\N{PARTIAL DIFFERENTIAL}\N{NABLA}\N{INFINITY}'
    '\N{ALMOST EQUAL TO}\N{NOT EQUAL TO}\N{IDENTICAL TO}\N{LESS-THAN OR EQUAL TO}'
    '\N{GREATER-THAN OR EQUAL TO}\N{PROPORTIONAL TO}'
)
_MATH_FONT: Final = re.compile(r'(?i)math|cmmi|cmsy|cmex|stix')
#: Drawings further apart than this many body sizes, each under its own title, are separate figures.
JOIN_GAP: Final = 3.0
#: A drawing's title: the nearest line of words above it, across it, within this many body sizes.
TITLE_REACH: Final = 3.0
#: An untitled region's label starts so: no caption or number told it.
_UNTITLED: Final = 'Untitled '
_HEADING: Final = re.compile(r'^\d+(?:\.\d+)*\.?\s+[A-Z]')

Kind = Literal['figure', 'table', 'equation']


@dataclass(frozen=True)
class Region:
    """A ``figure``, ``table`` or ``equation`` region of a page: its ``label`` as read, its ``bbox`` in points.

    ``table``: a table's cells as its text sets them (:func:`_cells`), its box theirs; None for a figure or an
    equation, and for a table whose text does not make two rows of two columns (its words unread).
    """

    kind: Kind
    label: str
    bbox: BBox
    table: Table | None = None

    @property
    def captioned(self) -> bool:
        """Was it told by its caption or number (``Table 1``, ``(3)``), not as an ``Untitled ...`` one?"""
        return not self.label.startswith(_UNTITLED)


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


def _bands(page: _Page) -> list[list[layout.Line]]:
    """The page's lines in bands, top to bottom: lines whose middles lie within half a body size share one."""
    return _bands_of(page, page.lines)


def _speck(line: layout.Line) -> bool:
    """A speck OCR read as a cell.

    OCR'd (an invisible text layer), under :data:`CELL_CHARS` letters or digits, and not a dash (an empty entry).
    Text the document sets itself is what it sets: a lone sign is no speck.
    """
    stripped = line.text.strip()
    ocr = all(s.render_mode == INVISIBLE for s in line.spans)
    return ocr and sum(c.isalnum() for c in stripped) < CELL_CHARS and stripped not in _DASHES


def _row(page: _Page, band: list[layout.Line]) -> bool:
    """Is ``band`` a table's row: cells side by side, none of them prose nor a contents entry?

    :data:`ROW_CELLS` of them are set near the body size and none larger (a lone ``-`` may read smaller); one is short,
    at most :data:`SHORT_CELL` words -- a code, a year, a dash -- which columns of prose, however OCR splits them,
    are not. No cell is a speck OCR read (:func:`_speck`); a unit row's lone sign the document sets itself (an ohm
    sign) is a cell.
    """
    near = [ln for ln in band if page.body / ROW_SIZE <= ln.size <= ROW_SIZE * page.body]
    short = any(len(ln.text.split()) <= SHORT_CELL for ln in band)  # a code, a year, a '-': columns of prose have none
    return (
        len(near) >= ROW_CELLS
        and short
        and not any(_speck(ln) for ln in band)
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
            found.append(Region('table', _UNTITLED + 'table', box))
        i = j if rows >= TABLE_ROWS else i + 1
    return found


def _drawn(page: _Page, drawings: Sequence[Drawing], held: list[Region]) -> list[Region]:
    """The drawings no caption, number or row of cells told, labeled ``Untitled ...``.

    A drawing counts when it lies within the text's column (a margin's sideways license note, a cover's band to the
    page edge do not), is at least :data:`DRAWN_MIN` body sizes high and wide (a rule under a header is not), is no
    box round a line's words (a link's frame), draws a stroke :data:`DRAWN_STROKE` body sizes long (words OCR missed
    draw none), crosses no prose (a box round a note, links over lines do) unless its rows are ruled apart, and lies
    in no region found. A drawing ruled with a frame and a rule inside it, or with :data:`RULED_ROWS` rules across
    it, nothing off its rules larger than a body size (words, which may be too small for OCR to read), is a table;
    a bare rectangle round words (a link's frame) is no region. Other
    drawings are joined into figures (:func:`_joined`); one at most :data:`EQUATION_HEIGHT` body sizes high whose text
    reads as no word and as math (:func:`_math`: a formula, OCR'd or set, its fraction bars drawn) is an equation,
    else a figure -- a cover's logo, drawn without text or round its brand's, is no formula.
    """
    found: list[Region] = []
    pieces: list[BBox] = []
    for drawing in drawings:
        box = drawing.box
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        words_only = drawing.loose <= page.body  # off its rules, only words
        # rows ruled apart, text between the rules (an icon's stripes hold none): a table, its lines however long
        held_lines = sum(
            box.x0 <= ln.x0 and ln.x1 <= box.x1 and box.y0 <= ln.y0 and ln.y1 <= box.y1 for ln in page.lines
        )
        rowed = drawing.rules[0] >= RULED_ROWS and words_only and held_lines >= TABLE_ROWS
        if (
            box.x0 < page.left - page.body
            or box.x1 > page.right + page.body
            or min(box.width, box.height) < DRAWN_MIN * page.body
            or any(_round_words(box, ln) for ln in page.lines)
            or (
                not rowed
                and any(
                    page.prose(ln) and ln.x0 < box.x1 and box.x0 < ln.x1 and ln.y0 < box.y1 and box.y0 < ln.y1
                    for ln in page.lines
                )
            )
            or any(r.bbox.x0 <= cx <= r.bbox.x1 and r.bbox.y0 <= cy <= r.bbox.y1 for r in [*held, *found])
            or (drawing.rules == (2, 2) and drawing.loose <= page.body)  # a bare rectangle, words at most
            or drawing.stroke < DRAWN_STROKE * page.body  # words OCR missed: glyphs, no drawn line
        ):
            continue
        pad = PAD * page.body
        grown = BBox(max(box.x0 - pad, 0.0), max(box.y0 - pad, 0.0), box.x1 + pad, box.y1 + pad)
        ruled = min(drawing.rules) >= 2 and max(drawing.rules) >= 3  # noqa: PLR2004 -- a frame and a rule in it
        if (ruled and words_only) or rowed:
            found.append(Region('table', _UNTITLED + 'table', grown))
        else:
            pieces.append(grown)
    for box in _joined(page, pieces):
        words = [ln for ln in page.lines if box.x0 <= ln.x0 and ln.x1 <= box.x1 and box.y0 <= ln.y0 and ln.y1 <= box.y1]
        formula = not any(layout.wordy(ln.text) for ln in words) and _math(words)
        if box.height <= EQUATION_HEIGHT * page.body and formula:
            found.append(Region('equation', _UNTITLED + 'equation', box))
        else:
            found.append(Region('figure', _UNTITLED + 'figure', box))
    return found


def _math(lines: list[layout.Line]) -> bool:
    """Does the text of ``lines`` read as math: a relation or operator sign, a Greek letter, or set in a math font?"""
    text = ''.join(ln.text for ln in lines)
    signs = any(c in _MATH_SIGNS or unicodedata.name(c, '').startswith('GREEK') for c in text)
    return signs or any(_MATH_FONT.search(s.font) for ln in lines for s in ln.spans)


def _joined(page: _Page, boxes: list[BBox]) -> list[BBox]:
    """``boxes`` joined into figures, the closest two first, until no two may be: a drawing's parts, one figure.

    Two are apart when prose runs in the gap down the page between them, or when they stand more than
    :data:`JOIN_GAP` body sizes apart and each has its own title (:func:`_title`): a data sheet's drawings, each under
    its heading. A drawing's views side by side under one title, or none, are one figure; parts close together are
    one whatever their titles, and being joined first, they meet another drawing as one.
    """
    out = [(box, _title(page, box)) for box in boxes]

    def gap(a: BBox, b: BBox) -> float:
        return max(a.x0 - b.x1, b.x0 - a.x1, a.y0 - b.y1, b.y0 - a.y1, 0.0)

    def apart(a: tuple[BBox, frozenset[int]], b: tuple[BBox, frozenset[int]]) -> bool:
        top, bottom = (a[0], b[0]) if a[0].y0 <= b[0].y0 else (b[0], a[0])
        if any(page.prose(ln) and top.y1 <= ln.y0 and ln.y1 <= bottom.y0 for ln in page.lines):
            return True
        return gap(a[0], b[0]) > JOIN_GAP * page.body and bool(a[1]) and bool(b[1]) and a[1].isdisjoint(b[1])

    while True:
        pairs = sorted((gap(out[i][0], out[j][0]), i, j) for i in range(len(out)) for j in range(i + 1, len(out)))
        pair = next(((i, j) for _, i, j in pairs if not apart(out[i], out[j])), None)
        if pair is None:
            break
        (a, ta), (b, tb) = out[pair[0]], out[pair[1]]
        out[pair[0]] = (BBox(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)), ta | tb)
        del out[pair[1]]
    return sorted((box for box, _ in out), key=lambda b: (b.y0, b.x0))


def _title(page: _Page, box: BBox) -> frozenset[int]:
    """The title of the drawing in ``box`` (its line's index), or none.

    The nearest line of words above it and across it, within :data:`TITLE_REACH` body sizes.
    """
    above = [
        (ln.y1, i)
        for i, ln in enumerate(page.lines)
        if ln.y1 <= box.y0
        and box.y0 - ln.y1 <= TITLE_REACH * page.body
        and ln.x0 < box.x1
        and box.x0 < ln.x1
        and layout.wordy(ln.text)
    ]
    return frozenset({max(above)[1]}) if above else frozenset()


def _round_words(box: BBox, line: layout.Line) -> bool:
    """Is ``box`` a frame round ``line``'s words, as a link's or a header's is?

    Sharing half the line's band or more, across the line or beside it, at most :data:`FRAME_LINES` its height, the
    line reading as words or filling
    :data:`FRAME_FILL` of it (a header's code): a formula's bars frame no such line. Across the line's words or
    beside them: OCR may miss the framed words, leaving their ink inside the frame.
    """
    height = line.y1 - line.y0
    shared = min(box.y1, line.y1) - max(box.y0, line.y0)  # how much of the line's band the box shares
    touches = line.x0 - height <= box.x1 and box.x0 <= line.x1 + height  # across the line or just beside it
    framed = shared >= height / 2 and touches and box.height <= FRAME_LINES * height
    fills = line.x1 - line.x0 >= FRAME_FILL * box.width  # a header's code fills its frame; a formula's noise does not
    return framed and (layout.wordy(line.text) or fills)


def _bands_of(page: _Page, lines: list[layout.Line]) -> list[list[layout.Line]]:
    """``lines`` in bands, as :func:`_bands` makes them of the whole page."""
    bands: list[list[layout.Line]] = []
    for line in sorted(lines, key=lambda ln: ln.y0):
        middle = (line.y0 + line.y1) / 2
        if bands and abs(middle - sum((ln.y0 + ln.y1) / 2 for ln in bands[-1]) / len(bands[-1])) <= page.body / 2:
            bands[-1].append(line)
        else:
            bands.append([line])
    return bands


def _cells(page: _Page, region: Region) -> Table | None:
    """The cells of table ``region`` as its text sets them (`document_lab.tables.aligned`).

    Its text is the spans whose middle lies in it, below its caption.
    """
    caption = next((ln for ln in page.lines if (c := _caption(ln)) and c[1] == region.label), None)
    box, below = region.bbox, caption.y1 if caption is not None else region.bbox.y0
    return aligned([s for s in page.page.spans if box.x0 <= s.center[0] <= box.x1 and below < s.center[1] <= box.y1])


def equation(page: Page, number: str, face: tuple[str, float]) -> BBox | None:
    """The box of display equation ``(number)`` on ``page``: the formula and its number, nothing else.

    ``face``: the font and size the document's prose is set in (`document_lab.layout.body_face`); ``page`` is best
    handed without its running heads and feet (`document_lab.layout.without_running`), which no equation holds.

    - The number: the line ending ``(number)`` that is the number alone, or a formula then its number (a line that
      reads as words, ending in a reference, is neither); of several, the one line that is the number alone. None
      when no line is it, or several are and not one of them stands alone.
    - The column: where lines of words (:data:`TEXT_WORDS` words or more) within :data:`COLUMN_WINDOW` body sizes of
      the number are set mostly in two columns, a number in the left column takes the left column's lines, one in the
      right column the right column's -- and every column's when what it takes crosses the page middle, or the left
      column holds level with it a formula's pieces alone, no prose and no number (an equation set across both
      columns); else the page is set in one column there and the equation may span it all. Never past the page's
      lines of words by more than a body size (a line number in the margin).
    - The pieces: from the number, every piece -- a line that is not prose, an image no taller than
      :data:`EQUATION_IMAGE` body sizes -- within :data:`EQUATION_REACH` body sizes of what is taken (:data:`OCR_REACH`
      where the number is OCR'd), until none is left; never past the prose above or below the number, and never a
      piece nearer another equation's number between them. A piece's box is its spans' whole boxes (a sum's limits,
      a big bracket), not a line's median.
    - Prose: a line that reads as words -- a word of four letters set in ``face``'s font, or two in any font (an
      italic heading; :func:`_words`) -- and spans :data:`PROSE_WIDTH` of its column or starts within
      :data:`PROSE_INDENT` body sizes of the column's left edge, as a paragraph's last line does (``values [13].``
      above a formula). A formula's letters (``cos hweTs`` flush left, ``Lvsd =``, ``Ie5jwet``) and its short words
      set inside it (``with``, ``otherwise``) are no prose.
    - The drawings whose middle lies in what is taken (a fraction's bar, a root's) are taken too; the box stands
      :data:`EQUATION_PAD` body sizes clear of it all, inside the page -- but never into text it leaves out
      (:func:`_padded`: the next equation set a point below).
    """
    if face[1] <= 0:
        return None
    lines = layout.lines(page, weighted=True)
    found = _number(lines, number, face)
    return None if found is None else _equation_box(page, lines, found, face)


def _equation_box(page: Page, lines: list[layout.Line], found: layout.Line, face: tuple[str, float]) -> BBox:
    """The box of the equation numbered on line ``found`` of ``lines`` (``page``'s), as :func:`equation` says."""
    body = face[1]
    seed = _spans_box(found)
    middle = (seed.y0 + seed.y1) / 2
    columns = _Columns(page, lines, middle, body)
    ocr = all(s.render_mode == INVISIBLE for s in found.spans)

    def prose(line: layout.Line) -> bool:
        left, width = columns.edge(line)
        return _reads(line.spans, line.text, face) and (
            line.x1 - line.x0 >= PROSE_WIDTH * width or line.x0 - left <= PROSE_INDENT * body
        )

    def take(*, across: bool) -> BBox:
        here = [ln for ln in lines if ln is not found and columns.holds(_spans_box(ln), ln.column, seed, across=across)]
        bounds = [_spans_box(ln) for ln in here if prose(ln)]
        top = max((b.y1 for b in bounds if (b.y0 + b.y1) / 2 < middle), default=0.0)
        bottom = min((b.y0 for b in bounds if (b.y0 + b.y1) / 2 > middle), default=page.height)
        # the other equations' numbers between the prose that bounds it
        numbers = [m for ln in here if _numbered(ln, face) and top <= (m := _middle(ln)) <= bottom]
        pieces = [_spans_box(ln) for ln in here if not prose(ln)]
        pieces += [
            b
            for b in _images(page)
            if b.height <= EQUATION_IMAGE * body
            and columns.holds(b, layout.column(b.x0, b.x1, columns.mid), seed, across=across)
        ]
        pieces = [
            b
            for b in pieces
            if top <= (b.y0 + b.y1) / 2 <= bottom
            and all(abs((b.y0 + b.y1) / 2 - middle) <= abs((b.y0 + b.y1) / 2 - other) for other in numbers)
        ]
        box, reach = seed, (OCR_REACH if ocr else EQUATION_REACH) * body
        while near := [p for p in pieces if p.y0 <= box.y1 + reach and p.y1 >= box.y0 - reach]:
            pieces = [p for p in pieces if p not in near]
            box = union([box, *near])
        return box

    def open_left(box: BBox) -> bool:
        """Does the left column hold, level with ``box``, a formula's pieces alone -- no prose, no number of its own?"""
        level = [ln for ln in lines if ln.column == 'L' and _spans_box(ln).y0 < box.y1 and box.y0 < _spans_box(ln).y1]
        return bool(level) and not any(prose(ln) or _numbered(ln, face) for ln in level)

    box = take(across=False)
    if columns.two and not columns.left_of(seed) and (box.x0 < columns.mid - layout.GUTTER or open_left(box)):
        box = take(across=True)  # an equation set across both columns
    drawn = [
        d
        for d in _painted(page)
        if box.x0 <= (d.x0 + d.x1) / 2 <= box.x1
        and box.y0 <= (d.y0 + d.y1) / 2 <= box.y1
        and d.width <= box.width + body
    ]
    return _padded(page, union([box, *drawn]), EQUATION_PAD * body)


def _equations(page: Page, face: tuple[str, float]) -> list[Region]:
    """The page's numbered equations, each number once, labeled as read (``(3)``): see :func:`equation`.

    A number counts where it ends its line right of :data:`NUMBER_RIGHT` of the column it stands in, judged as
    :func:`equation` judges columns; a number the page prints twice refuses, as there.
    """
    lines = layout.lines(page, weighted=True)
    labels: list[str] = []
    for line in lines:
        label = _numbered(line, face)
        if label is None or label in labels:
            continue
        left, width = _Columns(page, lines, _middle(line), face[1]).edge(line)
        if line.x1 >= left + NUMBER_RIGHT * width:
            labels.append(label)
    found = [(label, _number(lines, label, face)) for label in labels]
    return [Region('equation', f'({label})', _equation_box(page, lines, ln, face)) for label, ln in found if ln]


def _spans_box(line: layout.Line) -> BBox:
    """The whole box of a line's spans: a tall glyph's top and bottom, which the line's median leaves out."""
    return union([s.provenance.bbox for s in line.spans])


def _middle(line: layout.Line) -> float:
    box = _spans_box(line)
    return (box.y0 + box.y1) / 2


def _words(text: str) -> int:
    """How many of ``text``'s words (spaced apart, punctuation aside, of letters alone) have four letters or more.

    Letters run into a formula's digits (``Ie5jwet``) make no word.
    """
    words = (token.strip(_PUNCTUATION) for token in text.split())
    return sum(bool(_LETTERS.fullmatch(w)) and sum(c.isalpha() for c in w) >= 4 for w in words)  # noqa: PLR2004


def _reads(spans: list[TextSpan], text: str, face: tuple[str, float]) -> bool:
    """Does ``text`` (of ``spans``) read as words: a word of four letters set in ``face``'s font, or two in any font?"""
    return _words(text) >= 2 or any(s.font == face[0] and _WORD.search(s.text) for s in spans)  # noqa: PLR2004


def _numbered(line: layout.Line, face: tuple[str, float]) -> str | None:
    """The equation number ``line`` ends in -- alone, or after a formula, not after words -- else None."""
    found = _NUMBERED.search(line.text)
    if found is None or _reads(line.spans, line.text[: found.start()], face):
        return None
    return found.group(1)


def _number(lines: list[layout.Line], number: str, face: tuple[str, float]) -> layout.Line | None:
    """The line that ends in equation number ``(number)`` (:func:`_numbered`), of several the one that is it alone."""
    found = [ln for ln in lines if _numbered(ln, face) == number]
    alone = [ln for ln in found if _NUMBERED.fullmatch(ln.text.strip())]
    return found[0] if len(found) == 1 else alone[0] if len(alone) == 1 else None


def _padded(page: Page, box: BBox, pad: float) -> BBox:
    """``box`` grown by ``pad`` on each side, inside the page, but never into a span beside it that it leaves out.

    A span whose middle lies beyond an edge of ``box`` is left out: the edge stops halfway to it, or at its own edge
    where the two overlap (a big bracket's glyph box reaching into the prose line above).
    """
    x0, y0 = max(box.x0 - pad, 0.0), max(box.y0 - pad, 0.0)
    x1, y1 = min(box.x1 + pad, page.width), min(box.y1 + pad, page.height)
    for span in page.spans:
        b = span.provenance.bbox
        (cx, cy), across, level = span.center, b.x0 < x1 and x0 < b.x1, b.y0 < y1 and y0 < b.y1
        if across and cy < box.y0:
            y0 = max(y0, (b.y1 + box.y0) / 2, b.y1)
        elif across and cy > box.y1:
            y1 = min(y1, (b.y0 + box.y1) / 2, b.y0)
        elif level and cx < box.x0:
            x0 = max(x0, (b.x1 + box.x0) / 2, b.x1)
        elif level and cx > box.x1:
            x1 = min(x1, (b.x0 + box.x1) / 2, b.x0)
    return BBox(x0, y0, x1, y1)


class _Columns:
    """How a page is set in columns around one equation's number, and the edges its lines are judged by."""

    def __init__(self, page: Page, lines: list[layout.Line], middle: float, body: float) -> None:
        self.mid, self.body = page.width / 2, body
        text = [ln for ln in lines if len(ln.text.split()) >= TEXT_WORDS and layout.wordy(ln.text)]
        near = [ln for ln in text if abs(_middle(ln) - middle) <= COLUMN_WINDOW * body] or text
        self.two = sum(ln.column != 'F' for ln in near) > sum(ln.column == 'F' for ln in near)
        self.left = min((ln.x0 for ln in text), default=0.0)
        self.right = max((ln.x1 for ln in text), default=page.width)
        self.right_column = min((ln.x0 for ln in text if ln.column == 'R'), default=self.mid)

    def left_of(self, box: BBox) -> bool:
        """Is ``box`` (a number) in the left column?"""
        return box.x1 <= self.mid + layout.GUTTER

    def edge(self, line: layout.Line) -> tuple[float, float]:
        """The left edge and the width of the column ``line`` stands in."""
        if self.two and line.column == 'L':
            return self.left, self.mid - self.left
        if self.two and line.column == 'R':
            return self.right_column, self.right - self.right_column
        return self.left, self.right - self.left

    def holds(self, box: BBox, side: str, number: BBox, *, across: bool) -> bool:
        """May the equation of ``number`` take ``box``, standing in column ``side``: its column's, inside the text?"""
        inside = box.x0 >= self.left - self.body and box.x1 <= self.right + self.body
        if not self.two or across:
            return inside
        return inside and (side == 'L' if self.left_of(number) else side != 'L')


def _images(page: Page) -> list[BBox]:
    """The box of each image placed on the page."""
    return [f.provenance.bbox for f in page.figures if f.provenance.bbox is not None]


def _painted(page: Page) -> list[BBox]:
    """The box of each painted path on the page (all its subpaths: one drawing)."""
    parts: dict[int, list[BBox]] = defaultdict(list)
    for path in page.paths:
        if path.provenance.bbox is not None:
            parts[path.group].append(path.provenance.bbox)
    return [union(boxes) for boxes in parts.values()]


def regions(
    page: Page,
    face: tuple[str, float],
    drawings: Sequence[Drawing] = (),
    *,
    include_hidden: bool = False,
    include_in_background_color: bool = False,
) -> list[Region]:
    """The figures, tables and equations of ``page``, top to bottom.

    ``face``: the font and size the document's prose is set in (`document_lab.layout.body_face`), its size the body
    size every measure here is in. ``drawings``: the page's drawn ink (`document_lab.ink`), when the caller holds its
    picture; without them a region is told by its words alone. ``include_hidden``: what hidden layers draw counts too;
    ``include_in_background_color``: the text filled in the color under it too (see `document_lab.ir`).
    """
    page = page.shown(include_hidden=include_hidden, include_in_background_color=include_in_background_color)
    if not page.spans or face[1] <= 0:
        return []
    judged = _Page(page, face[1])
    found: list[Region] = []
    for line in judged.lines:
        if caption := _caption(line):
            kind, label = caption
            found.append(_figure(judged, line, label) if kind == 'figure' else _table(judged, line, label))
    found += _equations(page, face)
    found += _untitled_tables(judged, found)
    found += _drawn(judged, drawings, found)
    found = [replace(r, table=_cells(judged, r)) if r.kind == 'table' else r for r in found]
    # Clamped to the page, a region its text sets across the page's edge can be left without area (a caption the foot
    # cuts, prose right above it: its figure's top falls below the foot); it shows nothing and is none.
    shown = sorted((r for r in found if r.bbox.x1 > r.bbox.x0 and r.bbox.y1 > r.bbox.y0), key=lambda r: r.bbox.y0)
    counts: dict[str, int] = {}
    out = []
    for r in shown:  # untitled ones numbered in page order, whichever way each was told
        if not r.captioned:
            counts[r.kind] = counts.get(r.kind, 0) + 1
            r = replace(r, label=f'{_UNTITLED}{r.kind} {counts[r.kind]}')  # noqa: PLW2901
        out.append(r)
    return out
