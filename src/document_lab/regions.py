"""A page's figures, tables and equations as regions, found from its text.

Nothing here reads pixels: an OCR'd capture is one picture with words over it, so a region is told by its words.
A figure's caption (``Figure 2 - ...``) is below it: the figure runs up from the caption to the nearest prose above.
A table's caption is above it: the table runs down from the caption to the next prose, clause heading or caption.
Prose is a line that reads as words (`document_lab.layout.wordy`), set near the body size, across half the text
width or :data:`PROSE_WORDS` words long. A mention in prose (``Table 7, item 1d``) or a contents entry (a leader) is
no caption. A figure or table spans the text width; its label is the caption's, as read.

Every display equation is found by its own content, numbered or not (:func:`equations`): a number ending a line
right of :data:`NUMBER_RIGHT` of its column -- alone, or after a formula (``(3)``, ``(A.1)``) -- or a formula's line
that is no prose and states a relation; from either, its formula's lines to the prose around it, in its column on a
page set in two columns there, across both when the formula is, with the bars and small images set among its lines --
and its number when one is printed. A formula OCR'd reads as noise and is taken all the same; prose with a formula in
it is prose. One function finds every equation -- a paper's (`document_lab.paper`) and a captured standard's -- so
each is cut alike; one without a number is labeled ``Untitled equation N``.

A table without a caption (a standard's normative-reference annex, a list set in columns) is told by its rows:
:data:`TABLE_ROWS` or more bands of :data:`ROW_CELLS` or more cells side by side, set near the body size, no prose
among them; a cell's further lines -- words, starting within one of the row's cells -- belong to it, and anything
else (a formula, a note, prose) ends it. It runs from its first row to its last and is
labeled ``Untitled table N`` in page order. Two columns of prose have two cells a band, so they are no table; a
band a captioned region already holds is not found again. A table whose cells OCR read no words from has nothing to
be told by here, and is not found.

The caller that holds the page's picture may also hand over its drawings (`document_lab.ink`): a drawing no caption,
number or row of cells told is a region too -- a ruled table of two columns, a figure without a caption
(``Untitled figure N``), a formula drawn (:func:`_drawn_formula`: its text reads as math -- a sign, a Greek
letter, a math font -- an equation's; a logo is a figure). Regions still read no pixels: a drawing is its box and what
it is drawn with. Drawings are one figure unless prose runs between them or, set apart by white space, each stands
under its own title (a data sheet's dimensional drawing, circuit and reel). A region
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
from document_lab.tables import SHORT_CELL, aligned, tables

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.ink import Drawing
    from document_lab.ir import Page, Table, TextSpan

__all__ = ['Region', 'equations', 'regions']

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
#: A drawing at most this many body sizes high whose text reads as no word, and as math, is a formula drawn; ink
#: at most so high is a piece an equation may take, and a drawing taller and wider is a figure.
EQUATION_HEIGHT: Final = 4.0
#: A plot's legend box is at most this many body sizes wide (and taller than EQUATION_HEIGHT).
LEGEND_WIDTH: Final = 6.0
#: An image covering this share of the page is the page's own picture (a scan, a capture), no figure.
PAGE_PICTURE: Final = 0.9
#: Math: a relation or operator sign, a Greek letter, or a math font (TeX's, a Unicode math font).
_MATH_SIGNS: Final = frozenset(
    '=<>+'
    '\N{MINUS SIGN}\N{PLUS-MINUS SIGN}\N{MULTIPLICATION SIGN}\N{DIVISION SIGN}\N{N-ARY SUMMATION}'
    '\N{N-ARY PRODUCT}\N{INTEGRAL}\N{SQUARE ROOT}\N{PARTIAL DIFFERENTIAL}\N{NABLA}\N{INFINITY}'
    '\N{ALMOST EQUAL TO}\N{NOT EQUAL TO}\N{IDENTICAL TO}\N{LESS-THAN OR EQUAL TO}'
    '\N{GREATER-THAN OR EQUAL TO}\N{PROPORTIONAL TO}'
)
_MATH_FONT: Final = re.compile(r'(?i)math|cmmi|cmsy|cmex|stix|msbm|msam|eufm|rsfs|mtmi|mtsy|txmi|txsy|pxmi|lmmi')
#: A typewriter font: a code listing's, whose ``x = 1;`` is no formula.
_CODE_FONT: Final = re.compile(r'(?i)mono|courier|cmtt|typewriter|consolas|lmtt')
#: A caption's opening: ``Fig. 3.``, ``FIGURE 21.``, ``Table II``, a subfigure's ``(b) ``.
_CAPTION_OPENS: Final = re.compile(r'^(?:Fig\.|FIG\.|Figure|FIGURE|Table|TABLE)\s*[A-Z]?[\dIVX]|^\([a-h]\)\s')
#: A formula line flush with its column within this many body sizes below a line of prose is the paragraph's tail.
TAIL_GAP: Final = 0.4
#: A relation a display states: what tells a formula's line, numbered or not.
_RELATIONS: Final = frozenset('=<>≈≠≡≤≥∝≃≅≜≔≪≫→⇒⇔⟶')
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
    a bare rectangle round words (a link's frame) is no region. Other drawings are joined into figures
    (:func:`_joined`). A formula drawn is an equation's (:func:`equations` finds it first); a cover's logo, drawn
    without text or round its brand's, is a figure.
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
    found += [Region('figure', _UNTITLED + 'figure', box) for box in _joined(page, pieces)]
    return found


def _math(lines: list[layout.Line]) -> bool:
    """Does the text of ``lines`` read as math: a relation or operator sign, a Greek letter, or set in a math font?"""
    text = ''.join(ln.text for ln in lines)
    return _math_text(text) or any(_MATH_FONT.search(s.font) for ln in lines for s in ln.spans)


def _math_text(text: str) -> bool:
    """Does ``text`` hold a relation or operator sign, or a Greek letter?"""
    return any(c in _MATH_SIGNS or c in _RELATIONS or unicodedata.name(c, '').startswith('GREEK') for c in text)


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


def equations(
    page: Page, face: tuple[str, float], drawings: Sequence[Drawing] = (), *, include_hidden: bool = False
) -> list[Region]:
    """Every display equation of ``page``, numbered or not, top to bottom: its formula and its number, nothing else.

    ``face``: the font and size the document's prose is set in (`document_lab.layout.body_face`); ``page`` is best
    handed without its running heads and feet (`document_lab.layout.without_running`), which no equation holds.
    ``drawings``: the page's drawn ink (`document_lab.ink`), when the caller holds its picture: a formula's bars and
    the glyphs OCR read as nothing. ``include_hidden``: what hidden layers draw counts too (see `document_lab.ir`).

    A numbered equation is labeled by its number as read (``(3)``, ``(16a)``), one without ``Untitled equation N``
    in page order. Each is found by what the page states, never guessed:

    - A number: a line ending ``(N)`` alone or after a formula (:func:`_numbered`: a line reading as words, ending in
      a reference, is none), its right end past :data:`NUMBER_RIGHT` of the column it stands in. A number printed
      more than once is the one a formula stands level with (:func:`_number`: a figure's legend repeats ``(1)`` alone).
    - A display without a number: a line of its formula -- no prose (:class:`_Around`), stating a relation
      (:data:`_RELATIONS`: ``=``, ``<``, an arrow) -- or a short drawing whose text reads as math and as no word
      (:func:`_drawn_formula`); none set inside a figure (:func:`_figures`). Prose with a formula in it is prose:
      inline math is no display.
    - From its number or its line, the equation takes its formula's pieces (:func:`_equation_box`).
    """
    page = page.shown(include_hidden=include_hidden)
    if not page.spans or face[1] <= 0:
        return []
    return _in_order(_equations(page, face, drawings))


def _equations(page: Page, face: tuple[str, float], drawings: Sequence[Drawing]) -> list[Region]:
    """The page's display equations (:func:`equations`), the unnumbered ones labeled ``Untitled equation``."""
    body = face[1]
    lines = layout.lines(page, weighted=True)
    ink = [d.box for d in drawings if d.box.height <= EQUATION_HEIGHT * body]
    figures, legends = _figures(page, drawings, body), _legends(page, body)
    labels: list[str] = []
    for line in lines:
        label = _Around(page, lines, _middle(line), face).number(line) if _numbered(line, face) else None
        if label is not None and label not in labels:
            labels.append(label)
    found: list[Region] = []
    for label in labels:
        if (line := _number(page, lines, label, face, legends)) is not None:
            box = _equation_box(page, lines, _spans_box(line), line, face=face, ink=ink, ocr=_ocr(line))
            found.append(Region('equation', f'({label})', box))
    seeds: list[tuple[BBox, layout.Line | None, bool]] = [(_spans_box(ln), ln, _ocr(ln)) for ln in lines if _states(ln)]
    seeds += [(d.box, None, any(map(_ocr, inside))) for d in drawings if (inside := _drawn_formula(d, lines, body))]
    for box, line, ocr in sorted(seeds, key=lambda seed: seed[0].y0):
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        if any(f.contains(cx, cy) for f in [*figures, *(r.bbox for r in found)]):
            continue
        around = _Around(page, lines, cy, face)
        if line is not None and (around.prose(line) or around.inline(line)):
            continue  # prose with a formula in it: inline math
        eq = _equation_box(page, lines, box, line, face=face, ink=ink, ocr=ocr)
        if eq.height <= EQUATION_IMAGE * body:  # taller, and no number: a figure's labels, a listing
            found.append(Region('equation', _UNTITLED + 'equation', eq))
    return sorted(found, key=lambda r: r.bbox.y0)


def _equation_box(
    page: Page,
    lines: list[layout.Line],
    seed: BBox,
    own: layout.Line | None,
    *,
    face: tuple[str, float],
    ink: Sequence[BBox],
    ocr: bool,
) -> BBox:
    """The box of the equation grown from ``seed`` -- its number's box, or its formula's line ``own`` -- on ``page``.

    - The column: where lines of words (:data:`TEXT_WORDS` words or more) within :data:`COLUMN_WINDOW` body sizes of
      the seed are set mostly in two columns, a seed in the left column takes the left column's lines, one in the
      right column the right column's -- and every column's when what it takes crosses the page middle, or the left
      column holds level with it a formula's pieces alone, no prose and no number (an equation set across both
      columns); else the page is set in one column there and the equation may span it all. Never past the page's
      lines of words by more than a body size (a line number in the margin).
    - The pieces: from the seed, every piece -- a line that is no prose (:meth:`_Around.prose`), an image no taller
      than :data:`EQUATION_IMAGE` body sizes, a short drawing of ``ink`` -- within :data:`EQUATION_REACH` body sizes of
      what is taken (:data:`OCR_REACH` where the seed is OCR'd), until none is left; never past the prose above or
      below the seed, and never a piece nearer another equation's number between them (:meth:`_Around.number`: a mark
      ``(1)`` inside a matrix is none). A piece's box is its spans' whole boxes (a sum's limits, a big bracket), not a
      line's median.
    - The drawings whose middle lies in what is taken (a fraction's bar, a root's) are taken too; the box stands
      :data:`EQUATION_PAD` body sizes clear of it all, inside the page -- but never into text it leaves out
      (:func:`_padded`: the next equation set a point below).
    """
    body = face[1]
    middle = (seed.y0 + seed.y1) / 2
    around = _Around(page, lines, middle, face)

    def take(*, across: bool) -> BBox:
        here = [ln for ln in lines if ln is not own and around.holds(_spans_box(ln), ln.column, seed, across=across)]
        bounds = [_spans_box(ln) for ln in here if around.prose(ln)]
        top = max((b.y1 for b in bounds if (b.y0 + b.y1) / 2 < middle), default=0.0)
        bottom = min((b.y0 for b in bounds if (b.y0 + b.y1) / 2 > middle), default=page.height)
        # the other equations' numbers between the prose that bounds it
        numbers = [m for ln in here if around.number(ln) and top <= (m := _middle(ln)) <= bottom]
        pieces = [_spans_box(ln) for ln in here if not around.prose(ln)]
        pieces += [
            b
            for b in [*(i for i in _images(page) if i.height <= EQUATION_IMAGE * body), *ink]
            if around.holds(b, layout.column(b.x0, b.x1, around.mid), seed, across=across)
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
        return bool(level) and not any(around.prose(ln) or around.number(ln) for ln in level)

    box = take(across=False)
    if around.two and not around.left_of(seed) and (box.x0 < around.mid - layout.GUTTER or open_left(box)):
        box = take(across=True)  # an equation set across both columns
    drawn = [
        d
        for d in _painted(page)
        if box.x0 <= (d.x0 + d.x1) / 2 <= box.x1
        and box.y0 <= (d.y0 + d.y1) / 2 <= box.y1
        and d.width <= box.width + body
    ]
    return _padded(page, union([box, *drawn]), EQUATION_PAD * body)


def _in_order(found: list[Region]) -> list[Region]:
    """``found`` top to bottom, its untitled regions numbered in page order within their kind (``Untitled table 2``)."""
    counts: dict[str, int] = {}
    out = []
    for r in sorted(found, key=lambda r: r.bbox.y0):
        if not r.captioned:
            counts[r.kind] = counts.get(r.kind, 0) + 1
            r = replace(r, label=f'{_UNTITLED}{r.kind} {counts[r.kind]}')  # noqa: PLW2901
        out.append(r)
    return out


def _states(line: layout.Line) -> bool:
    """Does ``line`` state a relation (:data:`_RELATIONS`) between terms, as a formula's line does?

    Something besides the signs (an arrow drawn over a vector's letter is none), and not set in a typewriter font
    (:data:`_CODE_FONT`: a listing's ``x = 1;``).
    """
    text = line.text
    return (
        any(c in _RELATIONS for c in text)
        and any(c.isalnum() for c in text)
        and not all(_CODE_FONT.search(s.font) for s in line.spans)
    )


def _ocr(line: layout.Line) -> bool:
    """Is ``line`` OCR'd: every span of it invisible over the picture it was read from?"""
    return all(s.render_mode == INVISIBLE for s in line.spans)


def _drawn_formula(drawing: Drawing, lines: list[layout.Line], body: float) -> list[layout.Line]:
    """The lines of ``drawing`` when it is a formula drawn, else none.

    At least :data:`DRAWN_MIN` and at most :data:`EQUATION_HEIGHT` body sizes high, drawing a stroke
    :data:`DRAWN_STROKE` body sizes long (a bar; words OCR missed draw none), its text reading as math (:func:`_math`)
    and as no word -- a logo drawn round its brand's name is none.
    """
    box = drawing.box
    if not DRAWN_MIN * body <= box.height <= EQUATION_HEIGHT * body or drawing.stroke < DRAWN_STROKE * body:
        return []
    inside = [ln for ln in lines if box.x0 <= ln.x0 and ln.x1 <= box.x1 and box.y0 <= ln.y0 and ln.y1 <= box.y1]
    return inside if not any(layout.wordy(ln.text) for ln in inside) and _math(inside) else []


def _figures(page: Page, drawings: Sequence[Drawing], body: float) -> list[BBox]:
    """The boxes of the page's figures and ruled tables, which no equation is set inside.

    A plot's legend states ``k = 1``, a table's row ``Ldev -> MSE``. An image taller than :data:`EQUATION_IMAGE` body
    sizes, a painted path or a drawing (``drawings``) taller and wider than :data:`EQUATION_HEIGHT` body sizes (a
    plot's frame, a block diagram) -- none the page's own picture or ground, covering :data:`PAGE_PICTURE` of it --
    and a table its rules make (`document_lab.tables.tables`).
    """
    area = page.width * page.height
    images = [b for b in _images(page) if b.height > EQUATION_IMAGE * body]
    drawn = [
        b for b in [*_painted(page), *(d.box for d in drawings)] if min(b.width, b.height) > EQUATION_HEIGHT * body
    ]
    ruled = [t.provenance.bbox for t in tables(page) if t.provenance.bbox is not None]
    return [b for b in images + drawn if b.width * b.height < PAGE_PICTURE * area] + ruled


def _legends(page: Page, body: float) -> list[BBox]:
    """The boxes of a plot's legends on ``page``, whose entries' numbers (``(1)`` .. ``(8)``) are no equation's.

    A painted path taller than :data:`EQUATION_HEIGHT` body sizes, from :data:`DRAWN_MIN` to :data:`LEGEND_WIDTH` wide:
    a frame round a group of displays is as wide as their column, a big bracket drawn thinner.
    """
    return [
        b
        for b in _painted(page)
        if b.height > EQUATION_HEIGHT * body and DRAWN_MIN * body < b.width <= LEGEND_WIDTH * body
    ]


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


def _number(
    page: Page, lines: list[layout.Line], number: str, face: tuple[str, float], legends: list[BBox]
) -> layout.Line | None:
    """The line that ends in equation number ``(number)``, or None.

    A line ending in it (:func:`_numbered`) at the right of its column (:meth:`_Around.number`; ``V (1)`` in a line of
    a formula is a citation) and in no plot's legend (``legends``, :func:`_legends`). Of several, the one a formula
    stands level with (:func:`_level_formula`) -- of several so, the one alone on its line, else none; of none so,
    the one alone on its line, else none.
    """
    found = [
        ln
        for ln in lines
        if _Around(page, lines, _middle(ln), face).number(ln) == number
        and not any(f.contains((ln.x0 + ln.x1) / 2, _middle(ln)) for f in legends)
    ]
    if len(found) <= 1:
        return found[0] if found else None
    level = [ln for ln in found if _level_formula(page, lines, ln, face)]
    if len(level) == 1:
        return level[0]
    alone = [ln for ln in level or found if _NUMBERED.fullmatch(ln.text.strip())]
    return alone[0] if len(alone) == 1 else None


def _level_formula(page: Page, lines: list[layout.Line], number: layout.Line, face: tuple[str, float]) -> bool:
    """Does a formula stand level with the line ending in a number?

    Math before the number on its own line (:func:`_math_text`), or a line of no prose to its left in its column, its
    middle within the number's height.
    """
    found = _NUMBERED.search(number.text)
    if found is not None and _math_text(number.text[: found.start()]):
        return True
    box = _spans_box(number)
    around = _Around(page, lines, (box.y0 + box.y1) / 2, face)
    for line in lines:
        b = _spans_box(line)
        level = line is not number and box.y0 <= (b.y0 + b.y1) / 2 <= box.y1 and b.x1 <= box.x0
        if level and around.holds(b, line.column, box, across=False) and not around.prose(line):
            return True
    return False


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
    # never into the box itself: a span reaching across its edge stops the edge at the box
    return BBox(min(x0, box.x0), min(y0, box.y0), max(x1, box.x1), max(y1, box.y1))


class _Around:
    """A page around one equation's place: its columns there, and which of its lines are prose or a number."""

    def __init__(self, page: Page, lines: list[layout.Line], middle: float, face: tuple[str, float]) -> None:
        self.mid, self.face, self.body, self.lines = page.width / 2, face, face[1], lines
        text = [ln for ln in lines if len(ln.text.split()) >= TEXT_WORDS and layout.wordy(ln.text)]
        near = [ln for ln in text if abs(_middle(ln) - middle) <= COLUMN_WINDOW * self.body] or text
        self.two = sum(ln.column != 'F' for ln in near) > sum(ln.column == 'F' for ln in near)
        self.left = min((ln.x0 for ln in text), default=0.0)
        self.right = max((ln.x1 for ln in text), default=page.width)
        self.right_column = min((ln.x0 for ln in text if ln.column == 'R'), default=self.mid)

    def left_of(self, box: BBox) -> bool:
        """Is ``box`` (a number, a formula's line) in the left column?"""
        return box.x1 <= self.mid + layout.GUTTER

    def edge(self, line: layout.Line) -> tuple[float, float]:
        """The left edge and the width of the column ``line`` stands in."""
        if self.two and line.column == 'L':
            return self.left, self.mid - self.left
        if self.two and line.column == 'R':
            return self.right_column, self.right - self.right_column
        return self.left, self.right - self.left

    def holds(self, box: BBox, side: str, seed: BBox, *, across: bool) -> bool:
        """May the equation of ``seed`` take ``box``, standing in column ``side``: its column's, inside the text?"""
        inside = box.x0 >= self.left - self.body and box.x1 <= self.right + self.body
        if not self.two or across:
            return inside
        return inside and (side == 'L' if self.left_of(seed) else side != 'L')

    def prose(self, line: layout.Line) -> bool:
        """Is ``line`` prose, which bounds an equation and is never a piece of one?

        A line that reads as words (:func:`_reads`: a word of four letters set in the body's font, or two in any font
        -- an italic heading) and starts within :data:`PROSE_INDENT` body sizes of its column's left edge (a
        paragraph's line, ``values [13].`` above a formula), or spans :data:`PROSE_WIDTH` of its column with no math
        in it (:func:`_math`) or reaching its right edge (a justified line, a list item's): a cases row set in from the
        edge (``(x) > 0 Solid material``) and a matrix's row are a formula's. A line that reads as no word is prose
        only at the edge -- reaching across the column (a justified line of inline math) or as a paragraph's tail, in
        the body's font, no math (``[55]:``). A line holding :data:`TEXT_WORDS` words of four letters is prose
        wherever it starts (a list item's second line). A formula's letters (``cos hweTs`` flush left, ``Lvsd =``,
        ``Ie5jwet``) and its short words set inside it (``with``, ``otherwise``) are no prose.
        """
        if _words(line.text) >= TEXT_WORDS:  # words a formula never holds so many of: a list item's second line
            return True
        left, width = self.edge(line)
        flush = line.x0 - left <= PROSE_INDENT * self.body
        full = left + width - line.x1 <= self.body  # it reaches its column's right edge: a justified line
        if not _reads(line.spans, line.text, self.face):
            return flush and (full or (all(s.font == self.face[0] for s in line.spans) and not _math([line])))
        return flush or (line.x1 - line.x0 >= PROSE_WIDTH * width and (full or not _math([line])))

    def inline(self, line: layout.Line) -> bool:
        """Is the formula on ``line`` set in a paragraph, a caption or a table, no display?

        It opens a caption (``Fig. 3.``, ``FIGURE 21.``, a subfigure's ``(b)``); or it stands level with a line of prose
        of its column, beside or across it (a sub- or superscript the reader set apart from its line: ``R>=0``); or it
        is the tail of a paragraph, a list's item or a caption: within :data:`TAIL_GAP` body sizes below a line of prose
        or a caption's, flush with the column or starting where that line starts (``eta2 = 99.15 %.``, ``complex with
        p = 1;``); or :data:`ROW_CELLS` lines more stand level with it in its column, a table's row (``M = 10 [Nm]``).
        """
        if _CAPTION_OPENS.match(line.text):
            return True
        box = _spans_box(line)
        left, _ = self.edge(line)
        flush = line.x0 - left <= PROSE_INDENT * self.body
        cells = 0
        for other in self.lines:
            if other is line or self.edge(other)[0] != left:
                continue
            b = _spans_box(other)
            level = min(b.y1, box.y1) - max(b.y0, box.y0) >= (box.y1 - box.y0) / 2
            cells += level and abs(_middle(other) - _middle(line)) <= self.body / 2
            if not (self.prose(other) or _CAPTION_OPENS.match(other.text)):
                continue
            if level and b.x0 < box.x1 + self.body and box.x0 < b.x1 + self.body:
                return True
            aligned = flush or abs(box.x0 - b.x0) <= TAIL_GAP * self.body
            if aligned and 0 <= box.y0 - b.y1 <= TAIL_GAP * self.body:
                return True
        return cells >= ROW_CELLS

    def number(self, line: layout.Line) -> str | None:
        """The equation number ``line`` ends in (:func:`_numbered`), where it ends right of its column's middle.

        Right of :data:`NUMBER_RIGHT` of its column, else None: a list's ``(1)`` stands at its left, a matrix's mark
        ``(1)`` inside it.
        """
        label = _numbered(line, self.face)
        left, width = self.edge(line)
        return label if label is not None and line.x1 >= left + NUMBER_RIGHT * width else None


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
    page: Page, face: tuple[str, float], drawings: Sequence[Drawing] = (), *, include_hidden: bool = False
) -> list[Region]:
    """The figures, tables and equations of ``page``, top to bottom.

    ``face``: the font and size the document's prose is set in (`document_lab.layout.body_face`), its size the body
    size every measure here is in. ``drawings``: the page's drawn ink (`document_lab.ink`), when the caller holds its
    picture; without them a region is told by its words alone. ``include_hidden``: what hidden layers draw counts too
    (see `document_lab.ir`).
    """
    page = page.shown(include_hidden=include_hidden)
    if not page.spans or face[1] <= 0:
        return []
    judged = _Page(page, face[1])
    found: list[Region] = []
    for line in judged.lines:
        if caption := _caption(line):
            kind, label = caption
            found.append(_figure(judged, line, label) if kind == 'figure' else _table(judged, line, label))
    captioned = [r.bbox for r in found]
    for r in _equations(page, face, drawings):  # one without a number inside a captioned figure is the figure's
        middle = ((r.bbox.x0 + r.bbox.x1) / 2, (r.bbox.y0 + r.bbox.y1) / 2)
        if r.captioned or not any(box.contains(*middle) for box in captioned):
            found.append(r)
    found += _untitled_tables(judged, found)
    found += _drawn(judged, drawings, found)
    found = [replace(r, table=_cells(judged, r)) if r.kind == 'table' else r for r in found]
    # Clamped to the page, a region its text sets across the page's edge can be left without area (a caption the foot
    # cuts, prose right above it: its figure's top falls below the foot); it shows nothing and is none.
    return _in_order([r for r in found if r.bbox.x1 > r.bbox.x0 and r.bbox.y1 > r.bbox.y0])
