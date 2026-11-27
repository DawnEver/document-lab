"""A page's spans as lines in reading order: what every layout of a page -- its markdown, its regions -- starts from.

- Furniture: text turned sideways (neither :func:`upright` nor written vertically) wholly outside the box of the
  page's text (upright, or written vertically) -- an arXiv stamp up the margin -- is :func:`furniture` and makes no
  line; turned text inside that box (a figure's axis label) stays. A page without such text has no furniture.
- Frames: a line is built and read in its own frame -- x along its text, y down across its lines -- the page's own
  for upright text; text turned sideways reads along its baseline (a quarter turn exactly within :data:`UPRIGHT`),
  vertical writing down its column. A line's page box is its frame box turned back.
- Runs: a span that ``continues_run`` (:class:`~document_lab.ir.TextSpan`) is one run with the span before it
  (:func:`_runs`): a reader splits a run where its fill changes and says which piece goes on with it, and a line is
  built and measured from runs, so a scan's glyphs each in a grey of their own read as the run their source set.
- Lines: runs on one baseline running one way, each starting within :data:`LINE_GAP` font sizes of the line's
  end, its middle across within 0.4 of a size of the line's own (never the last run's); a column gutter is wider, so
  two columns' lines at one height stay two lines. A line's text spaces two
  neighboring spans apart unless they share an ``unspaced`` number (:class:`~document_lab.ir.TextSpan`): where a
  word gap is, is the spans' source's to say, never measured here. A bare clause number takes the text set apart
  beside it, within :data:`CLAUSE_GAP` font sizes, as a standard sets ``4.1    General`` and a paper
  ``1    Introduction``: in an unweighted document (no bold span: OCR'd text carries no weight) any clause number or
  bullet; in a weighted one a bold number or appendix letter (``A``), and only bold text of its size.
- Columns: a line is FULL width (crossing the page middle) or in the LEFT or RIGHT column -- a turned line its
  block's; :func:`ordered` reads the bands between full-width lines, each its left column, then its right.
- Blocks: :func:`blocks` sets lines into paragraphs -- each next line within :data:`BLOCK_GAP` font sizes below
  the last, in its column, overlapping it across.
- Turned text (vertical writing, text turned sideways): a block is lines of one way side by side -- overlapping
  along their text, within :data:`BLOCK_GAP` font sizes across -- its page column the whole block's. Lines of one way
  overlapping along their text are placed as one, at their top, and read in their frame: line by line down it --
  lines whose middles across stand within half a line of each other are one line of it -- each along its text. So a
  caption up the side of a landscape table reads line by line, left to right, and vertical writing column by column,
  right to left.
- Vertical writing (spans written vertically): a line is a column -- spans down one center line, each starting within
  :data:`DOWN_GAP` ems below the last; columns within :data:`COLUMN_GAP` ems apart are one block. Its em is its box's
  width across (writing mode 1 sets an em across; the size an OCR'd word states is its box's share of the column).
- Words: :func:`wordy` tells text that reads as words from OCR noise (a figure's labels, a garbled leader);
  :func:`body_size` is the size most of a document's text is set in, :func:`body_face` the font and size.
- Running heads and feet: :func:`running` finds the lines at a page's top and foot that recur at the same place
  on :data:`RUNNING_PAGES` pages (a page number its digits aside): a document's furniture, not its body;
  :func:`without_running` is the document without them, the text its body is written from.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING, Final

from document_lab.ir import BBox, spaced

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from document_lab.ir import Document, Page, TextSpan

__all__ = [
    'BLOCK_GAP',
    'CLAUSE_GAP',
    'COLUMN_GAP',
    'DOWN_GAP',
    'EQUATION_NUMBER',
    'GUTTER',
    'LINE_GAP',
    'RUNNING_PAGES',
    'SAME_WAY',
    'UPRIGHT',
    'Block',
    'Line',
    'Running',
    'blocks',
    'body_face',
    'body_size',
    'column',
    'furniture',
    'lines',
    'ordered',
    'running',
    'upright',
    'without_running',
    'wordy',
]

#: A span further right of a line than this many font sizes starts another line (a column gutter).
LINE_GAP: Final = 1.0
#: A bare clause number (``4.1``) takes the text beside it within this many font sizes.
CLAUSE_GAP: Final = 8.0
#: A line this many font sizes or less below a block's last line, overlapping it across, continues the block.
BLOCK_GAP: Final = 0.6
#: Vertical writing: a span this many ems or less below a column continues it (an OCR's word boxes hug their ink,
#: leaving up to an em between words down a column).
DOWN_GAP: Final = 1.5
#: Vertical writing: columns this many ems or less apart are one block (columns are set up to an em apart).
COLUMN_GAP: Final = 1.0
#: Points a column line may cross the page middle.
GUTTER: Final = 5.0
#: A span runs left to right when its baseline's direction has at least this x component (within ~8 degrees).
UPRIGHT: Final = 0.99
#: Spans run one way when their directions differ by at most this much (x and y differences summed).
SAME_WAY: Final = 0.02
#: Words: a word of at least this many letters, and letters (and spaces) this share of the text.
WORD_LETTERS: Final = 4
LETTER_SHARE: Final = 0.6
#: A line runs (a running head or foot) when this many pages -- every page of a shorter document, two at least -- set
#: it at the same place: within one font size of it down the page, its left, right or middle within one across.
RUNNING_PAGES: Final = 3
_CLAUSE: Final = re.compile(r'^(?:\d+(?:\.\d+)*\.?|[-\N{EN DASH}\N{EM DASH}\N{BULLET}])$')  # a clause number, a bullet
_NUMBER: Final = re.compile(r'^(?:\d+|[A-Z])(?:\.\d+)*\.?$')  # a heading's number: 2, 2.1, an appendix's A, A.1
#: An equation's number, in parentheses: ``(3)``, ``(12b)``, an annex's ``(A.1)``, a chapter's ``(4.23)`` -- never a
#: value such as ``(0.167)``, as no number starts with 0. Never a running line (:func:`running`); `document_lab.regions`
#: finds equations by it.
EQUATION_NUMBER: Final = re.compile(r'\(\s*((?:[A-Z]\.?)?[1-9]\d*(?:\.\d+)*[a-z]?)\s*\)')
_UP: Final = (1.0, 0.0)  # the way upright text reads: its frame is the page's
_QUARTERS: Final = (_UP, (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


@dataclass
class Line:
    """Spans on one baseline (or down one column), and the ``column`` they stand in: ``'L'``, ``'R'`` or ``'F'``."""

    spans: list[TextSpan]
    column: str

    @property
    def vertical(self) -> bool:
        """Is it written vertically: its spans down one column, read top to bottom?"""
        return self.spans[0].vertical

    @property
    def way(self) -> tuple[float, float]:
        """The direction it reads along on the page (y down): its first span's (:func:`_way`)."""
        return _way(self.spans[0])

    @property
    def frame(self) -> BBox:
        """Its box in its own frame: x along its text, y down across its lines.

        From the start of its first span to the end of its last, between the median top and bottom of its runs
        (:func:`_runs`, a run however many fills split it): one tall glyph, a big bracket, does not move the line, and a
        column's middle stands where most of its words' boxes do.
        """
        way = self.way
        boxes = [_run_box(run, way) for run in _runs(self.spans)]
        return BBox(
            min(b.x0 for b in boxes), median(b.y0 for b in boxes), max(b.x1 for b in boxes), median(b.y1 for b in boxes)
        )

    @property
    def text(self) -> str:
        """The spans' text along the line (upright, left to right), spaced but where they share ``unspaced``.

        Every space it states is a plain one, a run of them one, a word joiner none (:func:`~document_lab.ir.spaced`).
        """
        parts: list[str] = []
        before = None
        way = self.way
        for span in sorted(self.spans, key=lambda s: _framed(s.provenance.bbox, way).x0):
            if before is not None and (span.unspaced is None or span.unspaced != before.unspaced):
                parts.append(' ')
            parts.append(span.text)
            before = span
        return ' '.join(spaced(''.join(parts)).split())

    @property
    def y0(self) -> float:
        """Its top on the page: its :attr:`frame` turned back -- upright, the median span top; vertically, its top."""
        return _unframed(self.frame, self.way).y0

    @property
    def y1(self) -> float:
        """Its bottom on the page: upright, the median span bottom; vertically, the bottom of its last span."""
        return _unframed(self.frame, self.way).y1

    @property
    def x0(self) -> float:
        """Its left on the page: upright, the left end of its leftmost span; vertically, the median span left."""
        return _unframed(self.frame, self.way).x0

    @property
    def x1(self) -> float:
        """Its right on the page: upright, the right end of its rightmost span; vertically, the median span right."""
        return _unframed(self.frame, self.way).x1

    @property
    def size(self) -> float:
        """Its largest font size."""
        return max(s.size for s in self.spans)

    @property
    def bold(self) -> bool:
        """Is every span with text bold?"""
        return all(s.bold for s in self.spans if spaced(s.text).strip())


def upright(span: TextSpan) -> bool:
    """Does ``span`` run left to right, as a page's text does (neither turned sideways nor upside down)?"""
    return span.direction[0] >= UPRIGHT


def _way(span: TextSpan) -> tuple[float, float]:
    """The direction ``span`` reads along on the page (y down): down its column, or along its baseline.

    A baseline within :data:`UPRIGHT` of a quarter turn reads along the quarter turn exactly, so an upright span's
    frame is the page's.
    """
    if span.vertical:
        return (0.0, 1.0)
    dx, dy = span.direction
    return next((q for q in _QUARTERS if dx * q[0] + dy * q[1] >= UPRIGHT), (dx, dy))


def _framed(box: BBox, way: tuple[float, float]) -> BBox:
    """``box`` in the frame of text reading along ``way``: x along it, y down across its lines."""
    if way == _UP:
        return box
    dx, dy = way
    corners = [(x, y) for x in (box.x0, box.x1) for y in (box.y0, box.y1)]
    along = [x * dx + y * dy for x, y in corners]
    across = [y * dx - x * dy for x, y in corners]
    return BBox(min(along), min(across), max(along), max(across))


def _unframed(box: BBox, way: tuple[float, float]) -> BBox:
    """``box`` of the frame of text reading along ``way`` as the page shows it: :func:`_framed` undone."""
    if way == _UP:
        return box
    dx, dy = way
    corners = [(a, b) for a in (box.x0, box.x1) for b in (box.y0, box.y1)]
    xs = [a * dx - b * dy for a, b in corners]
    ys = [a * dy + b * dx for a, b in corners]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def furniture(page: Page) -> list[TextSpan]:
    """The page's text turned sideways wholly outside the box of its text: a margin's stamp or note.

    Its text is upright or written vertically (a column of CJK writing is no turned text).
    """
    frame = [s.provenance.bbox for s in page.spans if upright(s) or s.vertical]
    if not frame:
        return []
    x0, y0 = min(b.x0 for b in frame), min(b.y0 for b in frame)
    x1, y1 = max(b.x1 for b in frame), max(b.y1 for b in frame)

    def outside(box: BBox) -> bool:
        return box.x1 <= x0 or box.x0 >= x1 or box.y1 <= y0 or box.y0 >= y1

    return [s for s in page.spans if not upright(s) and not s.vertical and outside(s.provenance.bbox)]


def _same_way(a: TextSpan, b: TextSpan) -> bool:
    return abs(a.direction[0] - b.direction[0]) + abs(a.direction[1] - b.direction[1]) <= SAME_WAY


def _reach(line: Line, span: TextSpan, *, weighted: bool) -> float:
    """How many font sizes right of ``line`` the ``span`` may start and still join it."""
    if not weighted:
        return CLAUSE_GAP if _CLAUSE.match(line.text) else LINE_GAP
    heading = line.bold and span.bold and abs(span.size - line.size) <= 0.01 * line.size
    return CLAUSE_GAP if heading and _NUMBER.match(line.text) else LINE_GAP


def _runs(spans: Iterable[TextSpan]) -> list[list[TextSpan]]:
    """``spans``, in their source's order, as the runs it set: a span that ``continues_run`` goes on with the last.

    Read as one run -- its box round its pieces -- a fill change moves no line and splits no paragraph.
    """
    out: list[list[TextSpan]] = []
    for span in spans:
        if out and span.continues_run:
            out[-1].append(span)
        else:
            out.append([span])
    return out


def _run_box(run: list[TextSpan], way: tuple[float, float]) -> BBox:
    """The box round ``run``'s pieces in the frame of text reading along ``way``."""
    boxes = [_framed(s.provenance.bbox, way) for s in run]
    return BBox(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def _joins(line: Line, run: list[TextSpan], *, weighted: bool) -> bool:
    """Does ``run`` continue ``line`` in its frame: starting within reach of its end, its middle on the line's?"""
    span = run[0]
    frame, box = line.frame, _run_box(run, line.way)
    if span.vertical:  # an em across, its box's width
        em, reach = max(box.y1 - box.y0, frame.y1 - frame.y0), DOWN_GAP
    else:
        em, reach = max(span.size, line.size), _reach(line, span, weighted=weighted)
    near = -GUTTER <= box.x0 - frame.x1 <= reach * em
    return near and abs((frame.y0 + frame.y1) / 2 - (box.y0 + box.y1) / 2) <= 0.4 * em


def _start(span: TextSpan) -> tuple[float, float]:
    """Where ``span`` starts in its own frame: along its text, then across."""
    box = _framed(span.provenance.bbox, _way(span))
    return box.x0, box.y0


def lines(page: Page, *, weighted: bool) -> list[Line]:
    """The page's lines (unordered), its :func:`furniture` left out; a bare clause number takes the text beside it.

    Each line is built in its own frame: upright text along the page, text turned sideways along its baseline,
    vertical writing down its column. ``weighted`` False (no bold in the document): any clause number or bullet takes
    its text; True: a bold number or appendix letter, and only bold text of its size.
    """
    mid = page.width / 2
    margin = {id(s) for s in furniture(page)}
    found: list[Line] = []
    runs = [r for run in _runs(page.spans) if (r := [s for s in run if id(s) not in margin])]
    for run in sorted(runs, key=lambda run: _start(run[0])):
        span = run[0]
        for line in found:
            same = line.vertical == span.vertical and (span.vertical or _same_way(line.spans[0], span))
            if same and _joins(line, run, weighted=weighted):
                line.spans.extend(run)
                break
        else:
            found.append(Line(list(run), ''))
    for line in found:
        if line.way == _UP:
            line.column = column(line.x0, line.x1, mid)
    for block in _turned_blocks([ln for ln in found if ln.way != _UP]):
        side = column(min(ln.x0 for ln in block), max(ln.x1 for ln in block), mid)
        for line in block:
            line.column = side
    return [line for line in found if line.text]


def column(x0: float, x1: float, mid: float) -> str:
    """The page column a line (or a turned block) from ``x0`` to ``x1`` stands in: L, R or F (across ``mid``)."""
    return 'L' if x1 <= mid + GUTTER else 'R' if x0 >= mid - GUTTER else 'F'


def _beside(a: Line, b: Line, frames: dict[int, BBox]) -> bool:
    """Are turned lines ``a`` and ``b`` of one block: one way, overlapping along it, near enough across?"""
    if a.way != b.way or a.vertical != b.vertical:
        return False
    fa, fb = frames[id(a)], frames[id(b)]
    if not (fa.x0 < fb.x1 and fb.x0 < fa.x1):
        return False
    gap = max(fa.y0, fb.y0) - min(fa.y1, fb.y1)
    if a.vertical:  # columns: an em across, a column's width
        return gap <= COLUMN_GAP * max(fa.y1 - fa.y0, fb.y1 - fb.y0)
    return gap <= BLOCK_GAP * max(a.size, b.size)


def _turned_blocks(found: list[Line]) -> list[list[Line]]:
    """Turned lines (vertical writing, text turned sideways) as blocks, each read in its frame (:func:`_in_frame`).

    A block is lines linked to one another side by side (:func:`_beside`), however long the chain.
    """
    frames = {id(ln): ln.frame for ln in found}
    out: list[list[Line]] = []
    for line in found:
        linked = [block for block in out if any(_beside(line, other, frames) for other in block)]
        out = [block for block in out if all(block is not other for other in linked)]
        out.append([ln for block in linked for ln in block] + [line])
    return [_in_frame(block) for block in out]


def _in_frame(found: list[Line]) -> list[Line]:
    """Lines of one way as read in their frame: line by line down it, each along its text.

    Lines whose middles across stand within half a line of the first's are on one line of it -- a column's words
    however wide their boxes run, a table row's cells -- read along it; then the next line down.
    """
    frames = {id(ln): ln.frame for ln in found}

    def middle(line: Line) -> float:
        return (frames[id(line)].y0 + frames[id(line)].y1) / 2

    rows: list[list[Line]] = []
    for line in sorted(found, key=middle):
        first = rows[-1][0] if rows else None
        if first is not None and middle(line) - middle(first) <= (frames[id(first)].y1 - frames[id(first)].y0) / 2:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: frames[id(ln)].x0)]


@dataclass
class Block:
    """Lines set as one paragraph, top to bottom -- or turned lines side by side, as read in their frame."""

    lines: list[Line]

    @property
    def bbox(self) -> BBox:
        """The box round its lines."""
        return BBox(
            min(ln.x0 for ln in self.lines),
            min(ln.y0 for ln in self.lines),
            max(ln.x1 for ln in self.lines),
            max(ln.y1 for ln in self.lines),
        )

    @property
    def text(self) -> str:
        """Its lines' text, a line to a line."""
        return '\n'.join(ln.text for ln in self.lines)


def blocks(page: Page) -> list[Block]:
    """The page's lines set into paragraphs, in reading order (:func:`ordered` of their first lines)."""
    found: list[Block] = []
    every = lines(page, weighted=True)
    for line in sorted((ln for ln in every if ln.way == _UP), key=lambda ln: (ln.y0, ln.x0)):
        for block in found:
            last = block.lines[-1]
            gap = line.y0 - last.y1
            across = line.x0 < block.bbox.x1 and block.bbox.x0 < line.x1
            if (
                last.column == line.column
                and across
                and -0.5 * line.size <= gap <= BLOCK_GAP * max(line.size, last.size)
            ):
                block.lines.append(line)
                break
        else:
            found.append(Block([line]))
    found += [Block(block) for block in _turned_blocks([ln for ln in every if ln.way != _UP])]
    first = {id(b.lines[0]): b for b in found}
    return [first[id(ln)] for ln in ordered([b.lines[0] for b in found])]


def ordered(found: list[Line]) -> list[Line]:
    """Bands between full-width lines; each band's left column, then its right; each read by :func:`_top_down`."""
    out: list[Line] = []
    band: list[Line] = []

    def flush() -> None:
        out.extend(_top_down([ln for ln in band if ln.column == 'L']))
        out.extend(_top_down([ln for ln in band if ln.column == 'R']))
        band.clear()

    columns = [ln for ln in found if ln.column != 'F']
    for row in _rows([ln for ln in found if ln.column == 'F']):
        top = min(ln.y0 for ln in row)
        band.extend(ln for ln in columns if ln.y0 < top and ln not in out)
        columns = [ln for ln in columns if ln not in band]
        flush()
        out.extend(row)
    band.extend(columns)
    flush()
    return out


def _top_down(found: list[Line]) -> list[Line]:
    """Lines top to bottom, turned lines side by side as one (:func:`_rows`)."""
    return [ln for row in _rows(found) for ln in row]


def _rows(found: list[Line]) -> list[list[Line]]:
    """Lines top to bottom, each a row of its own -- but turned lines, placed side by side as one.

    Turned lines (vertical writing, text turned sideways) of one way overlapping along their text are one row, at
    their top, read in their frame (:func:`_in_frame`).
    """
    rows = [[ln] for ln in found if ln.way == _UP]
    ways: dict[tuple[tuple[float, float], bool], list[Line]] = {}
    for line in found:
        if line.way != _UP:
            ways.setdefault((line.way, line.vertical), []).append(line)
    for same in ways.values():
        frames = {id(ln): ln.frame for ln in same}
        side: list[list[Line]] = []
        reach = 0.0
        for line in sorted(same, key=lambda ln: frames[id(ln)].x0):
            frame = frames[id(line)]
            if side and frame.x0 < reach:
                side[-1].append(line)
                reach = max(reach, frame.x1)
            else:
                side.append([line])
                reach = frame.x1
        rows += [_in_frame(lines) for lines in side]
    return sorted(rows, key=lambda row: min(ln.y0 for ln in row))


def wordy(text: str) -> bool:
    """Does ``text`` read as words, not OCR noise: a word of :data:`WORD_LETTERS` letters, letters most of it?"""
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters >= LETTER_SHARE * len(text) and any(len(w) >= WORD_LETTERS and w.isalpha() for w in text.split())


def body_face(doc: Document) -> tuple[str, float]:
    """The font and size (to a tenth of a point) most of ``doc``'s text is set in (``('', 0.0)`` when it has none)."""
    faces: Counter[tuple[str, float]] = Counter()
    for span in doc.spans():
        faces[span.font, round(span.size, 1)] += len(span.text)
    return faces.most_common(1)[0][0] if faces else ('', 0.0)


def body_size(doc: Document) -> float:
    """The font size most of ``doc``'s text is set in (0 when it has none)."""
    sizes: Counter[float] = Counter()
    for span in doc.spans():
        sizes[round(span.size, 1)] += len(span.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


@dataclass(frozen=True)
class Running:
    """A page's running ``head`` and ``foot``: its furniture lines at its top and its foot, each top to bottom."""

    head: tuple[Line, ...]
    foot: tuple[Line, ...]


def running(doc: Document) -> list[Running]:
    """Each page's running head and foot: the lines at its top, and at its foot, that other pages repeat.

    A line repeats when :data:`RUNNING_PAGES` pages (every page of a shorter document, two at least) set its text --
    as :func:`_furniture_key` compares it, so a page number repeats -- at its place. The head is the page's lines
    from the top while each repeats, the foot from the bottom: a repeated line inside the body (a heading each page
    opens a table with) is not reached. Only a whole document can tell, and a page alone has none. Built on
    :func:`lines`, it never sees a page's :func:`furniture` (a sideways stamp): a span is one or the other, never both.
    """
    pages = [sorted(lines(page, weighted=True), key=lambda ln: (ln.y0, ln.x0)) for page in doc.pages]
    need = max(2, min(RUNNING_PAGES, len(pages)))
    seen: dict[str, list[tuple[int, Line]]] = {}
    for number, found in enumerate(pages):
        for line in found:
            seen.setdefault(_furniture_key(line.text), []).append((number, line))

    def repeats(number: int, line: Line) -> bool:
        reach = line.size
        here = {
            other
            for other, them in seen[_furniture_key(line.text)]
            if abs((them.y0 + them.y1) / 2 - (line.y0 + line.y1) / 2) <= reach
            and min(abs(them.x0 - line.x0), abs(them.x1 - line.x1), abs(them.x0 + them.x1 - line.x0 - line.x1) / 2)
            <= reach
        }
        return len(here | {number}) >= need

    out = []
    for number, found in enumerate(pages):
        head = list(itertools.takewhile(lambda ln, n=number: repeats(n, ln), found))
        rest = found[len(head) :]
        foot = list(itertools.takewhile(lambda ln, n=number: repeats(n, ln), reversed(rest)))
        out.append(Running(tuple(head), tuple(reversed(foot))))
    return out


def without_running(doc: Document, found: Sequence[Running] | None = None) -> Document:
    """``doc`` without its pages' running heads and feet: :func:`running`'s, or ``found`` when the caller holds them."""
    found = running(doc) if found is None else found
    pages = []
    for page, run in zip(doc.pages, found, strict=True):
        out = {id(span) for line in (*run.head, *run.foot) for span in line.spans}
        pages.append(replace(page, spans=tuple(s for s in page.spans if id(s) not in out)))
    return replace(doc, pages=tuple(pages))


def _furniture_key(text: str) -> str:
    """``text`` as running lines are compared: case and spacing aside.

    In a line without a letter (a page number, ``- 3 -``) each run of digits is one ``#``; a line of words
    (``Section 2``) must repeat as it is, and so must an equation's number (``(12)``): pages ending in a numbered
    equation set its number at the text's last baseline, each its own.
    """
    text = ' '.join(text.casefold().split())
    if any(c.isalpha() for c in text) or EQUATION_NUMBER.fullmatch(text):
        return text
    return re.sub(r'\d+', '#', text)
