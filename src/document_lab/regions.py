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
labeled ``Untitled table N`` in page order. Two columns of prose have two cells a band, so they are no table; a
band a captioned region already holds is not found again. A table whose cells OCR read no words from has nothing to
be told by here, and is not found.

The caller that holds the page's picture may also hand over its drawings (`document_lab.ink`): a drawing no caption,
number or row of cells told is a region too -- a ruled table of two columns, a figure without a caption
(``Untitled figure N``), an equation without a number (``Untitled equation N``). Regions still read no pixels: a
drawing is its box and what it is drawn with.

A tagged PDF states its tables and figures (`document_lab.tags`; the ``tables`` and ``figures`` uses, each its own
option): given its structure, a page the tags cover takes them from its ``Table`` and ``Figure`` elements in place
of the inferred ones (equations are still inferred), each
region the box of what its element covers there; a page they cover nothing on is inferred as an untagged one is,
and so are the tables (figures) of a document whose tags have no ``Table`` (``Figure``) element at all.
A tagged region's label is its caption's (``Fig. 3``, ``Table 2`` -- a ``Caption`` among its kids or next to it), else
an inferred region's of its kind it overlaps, else ``Untitled ...``; a figure keeps its ``/Alt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from document_lab import layout, tags
from document_lab.ir import BBox

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from document_lab.ink import Drawing
    from document_lab.ir import Page, Structure

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
#: A drawing at least this many body sizes high and wide may be a region (`document_lab.ink`); a rule is not.
DRAWN_MIN: Final = 1.5
#: ...and draws a straight stroke at least this many body sizes long: words OCR missed are glyphs, none so long.
DRAWN_STROKE: Final = 2.0
#: A drawing in a text line's band at most this many times the line's height frames its words (a link, a header).
FRAME_LINES: Final = 3.0
FRAME_FILL: Final = 0.8
#: A drawing at most this many body sizes high whose text reads as no word is an equation without a number.
EQUATION_HEIGHT: Final = 4.0
_HEADING: Final = re.compile(r'^\d+(?:\.\d+)*\.?\s+[A-Z]')
#: A tagged caption's label: ``Figure``, ``Fig.`` or ``Table``, then its number.
_TAGGED_CAPTION: Final = re.compile(r'^(Figure|Fig|Table)\.?\s*([A-Z]?\.?\d+(?:\.\d+)*)', re.IGNORECASE)
#: Two regions are one when they share this share of the smaller's area.
SAME_REGION: Final = 0.5

Kind = Literal['figure', 'table', 'equation']


@dataclass(frozen=True)
class Region:
    """A ``figure``, ``table`` or ``equation`` region of a page: its ``label`` as read, its ``bbox`` in points."""

    kind: Kind
    label: str
    bbox: BBox
    #: What a tagged figure shows, in the document's words (its ``/Alt``); None when it states none.
    alt: str | None = None


def overlap(a: BBox, b: BBox) -> float:
    """The share of the smaller box's area the two boxes share (0 when either has none)."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    smaller = min(a.width * a.height, b.width * b.height)
    return max(w, 0.0) * max(h, 0.0) / smaller if smaller > 0 else 0.0


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
    return _bands_of(page, page.lines)


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


def _drawn(page: _Page, drawings: Sequence[Drawing], held: list[Region]) -> list[Region]:
    """The drawings no caption, number or row of cells told, labeled ``Untitled ...``.

    A drawing counts when it lies within the text's column (a margin's sideways license note, a cover's band to the
    page edge do not), is at least :data:`DRAWN_MIN` body sizes high and wide (a rule under a header is not), is no
    box round a line's words (a link's frame), draws a stroke :data:`DRAWN_STROKE` body sizes long (words OCR missed
    draw none), crosses no prose (a box round a note, links over lines do) and lies in no region found.
    A drawing ruled with a frame and a rule inside it, nothing off its rules larger than a body size (words, which
    may be too small for OCR to read), is a table; a bare rectangle round words (a link's frame) is no region. Other
    drawings no prose runs between are one; at most :data:`EQUATION_HEIGHT` body sizes
    high with no word in it (a formula OCR read as noise, its fraction bars drawn) it is an equation, else a figure.
    """
    found: list[Region] = []
    tables = sum(r.label.startswith('Untitled table') for r in held)
    pieces: list[BBox] = []
    for drawing in drawings:
        box = drawing.box
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        if (
            box.x0 < page.left - page.body
            or box.x1 > page.right + page.body
            or min(box.width, box.height) < DRAWN_MIN * page.body
            or any(_round_words(box, ln) for ln in page.lines)
            or any(
                page.prose(ln) and ln.x0 < box.x1 and box.x0 < ln.x1 and ln.y0 < box.y1 and box.y0 < ln.y1
                for ln in page.lines
            )
            or any(r.bbox.x0 <= cx <= r.bbox.x1 and r.bbox.y0 <= cy <= r.bbox.y1 for r in [*held, *found])
            or (drawing.rules == (2, 2) and drawing.loose <= page.body)  # a bare rectangle, words at most
            or drawing.stroke < DRAWN_STROKE * page.body  # words OCR missed: glyphs, no drawn line
        ):
            continue
        pad = PAD * page.body
        grown = BBox(max(box.x0 - pad, 0.0), max(box.y0 - pad, 0.0), box.x1 + pad, box.y1 + pad)
        ruled = min(drawing.rules) >= 2 and max(drawing.rules) >= 3  # noqa: PLR2004 -- a frame and a rule in it
        if ruled and drawing.loose <= page.body:  # off the rules, only words
            tables += 1
            found.append(Region('table', f'Untitled table {tables}', grown))
        else:
            pieces.append(grown)
    equations = figures = 0
    for box in _joined(page, pieces):
        words = [ln for ln in page.lines if box.x0 <= ln.x0 and ln.x1 <= box.x1 and box.y0 <= ln.y0 and ln.y1 <= box.y1]
        if box.height <= EQUATION_HEIGHT * page.body and not any(layout.wordy(ln.text) for ln in words):
            equations += 1
            found.append(Region('equation', f'Untitled equation {equations}', box))
        else:
            figures += 1
            found.append(Region('figure', f'Untitled figure {figures}', box))
    return found


def _joined(page: _Page, boxes: list[BBox]) -> list[BBox]:
    """``boxes`` with any two no prose line runs between made one, until none are: a drawing's parts, one figure.

    Between is in the gap down the page between them; two side by side (a drawing's views) have none.
    """
    out = list(boxes)

    def apart(a: BBox, b: BBox) -> bool:
        top, bottom = (a, b) if a.y0 <= b.y0 else (b, a)
        return any(page.prose(ln) and top.y1 <= ln.y0 and ln.y1 <= bottom.y0 for ln in page.lines)

    merged = True
    while merged:
        merged = False
        pairs = [(i, j) for i in range(len(out)) for j in range(i + 1, len(out))]
        for i, j in pairs:
            if not apart(out[i], out[j]):
                a, b = out[i], out[j]
                out[i] = BBox(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
                del out[j]
                merged = True
                break
    return sorted(out, key=lambda b: (b.y0, b.x0))


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


def _tagged(page: _Page, structure: Structure, inferred: list[Region], uses: frozenset[str]) -> list[Region]:
    """The page's tables and figures the tags give (``uses``), labeled (see the module); the rest inferred.

    A kind the tags never state is inferred: their silence is not a statement.
    """
    stated_kinds = {
        kind
        for kind, role in (('table', 'Table'), ('figure', 'Figure'))
        if f'{kind}s' in uses and tags.states(structure, role)
    }
    found = [r for r in inferred if r.kind not in stated_kinds]
    untitled = {'table': 0, 'figure': 0}
    pad = PAD * page.body
    for stated in tags.regions(structure, page.page.number):
        if stated.kind not in stated_kinds:
            continue
        box = BBox(
            max(stated.bbox.x0 - pad, 0.0),
            max(stated.bbox.y0 - pad, 0.0),
            min(stated.bbox.x1 + pad, page.page.width),
            min(stated.bbox.y1 + pad, page.page.height),
        )
        caption = _TAGGED_CAPTION.match(stated.caption or '')
        near = [r for r in inferred if r.kind == stated.kind and overlap(r.bbox, box) >= SAME_REGION]
        if caption:
            label = f'{"Table" if caption.group(1).lower() == "table" else "Figure"} {caption.group(2)}'
        elif near:
            label = near[0].label
        else:
            untitled[stated.kind] += 1
            label = f'Untitled {stated.kind} {untitled[stated.kind]}'
        found.append(Region(stated.kind, label, box, stated.alt))
    return found


def regions(
    page: Page,
    body: float,
    drawings: Sequence[Drawing] = (),
    *,
    include_hidden: bool = False,
    structure: Structure | None = None,
    tagged: Collection[str] = tags.DEFAULT_USES,
) -> list[Region]:
    """The figures, tables and equations of ``page`` (``body``: the document's body size), top to bottom.

    ``drawings``: the page's drawn ink (`document_lab.ink`), when the caller holds its picture;
    without them a region is told by its words alone. ``include_hidden``: what hidden layers draw counts too (see
    `document_lab.ir`). ``structure``: a tagged document's (:attr:`~document_lab.ir.Document.structure`), whose
    tables and figures -- as ``tagged`` names (``tables``, ``figures``) -- stand in for the inferred ones on a page it
    covers (see the module); None infers them all.
    """
    uses = tags.checked(tagged)
    page = page.shown(include_hidden=include_hidden)
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
    found += _drawn(judged, drawings, found)
    if structure is not None and uses & {'tables', 'figures'} and tags.covers(structure, page.number):
        found = _tagged(judged, structure, found, uses)
    return sorted(found, key=lambda r: r.bbox.y0)
