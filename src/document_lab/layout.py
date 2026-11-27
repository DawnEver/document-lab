"""A page's spans as lines in reading order: what every layout of a page -- its markdown, its regions -- starts from.

- Furniture: text turned sideways (neither :func:`upright` nor written vertically) wholly outside the box of the
  page's text (upright, or written vertically) -- an arXiv stamp up the margin -- is :func:`furniture` and makes no
  line; turned text inside that box (a figure's axis label) stays. A page without such text has no furniture.
- Lines: spans on one baseline running one way, each starting within :data:`LINE_GAP` font sizes of the line's
  right end; a column gutter is wider, so two columns' lines at one height stay two lines. A line's text spaces two
  neighboring spans apart unless they share an ``unspaced`` number (:class:`~document_lab.ir.TextSpan`): where a
  word gap is, is the spans' source's to say, never measured here. A bare clause number takes the text set apart
  beside it, within :data:`CLAUSE_GAP` font sizes, as a standard sets ``4.1    General`` and a paper
  ``1    Introduction``: in an unweighted document (no bold span: OCR'd text carries no weight) any clause number or
  bullet; in a weighted one a bold number or appendix letter (``A``), and only bold text of its size.
- Columns: a line is FULL width (crossing the page middle) or in the LEFT or RIGHT column; :func:`ordered` reads the
  bands between full-width lines, each its left column, then its right.
- Blocks: :func:`blocks` sets lines into paragraphs -- each next line within :data:`BLOCK_GAP` font sizes below
  the last, in its column, overlapping it across.
- Vertical writing (spans written vertically): a line is a column -- spans down one center line, each starting within
  :data:`LINE_GAP` font sizes below the last; a block is columns side by side, each next within :data:`BLOCK_GAP`
  font sizes to the left, its page column the whole block's; lines side by side read right to left.
- Words: :func:`wordy` tells text that reads as words from OCR noise (a figure's labels, a garbled leader);
  :func:`body_size` is the size most of a document's text is set in.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Final

from document_lab.ir import BBox

if TYPE_CHECKING:
    from collections.abc import Iterable

    from document_lab.ir import Document, Page, TextSpan

__all__ = [
    'BLOCK_GAP',
    'CLAUSE_GAP',
    'GUTTER',
    'LINE_GAP',
    'SAME_WAY',
    'UPRIGHT',
    'Block',
    'Line',
    'blocks',
    'body_size',
    'furniture',
    'lines',
    'ordered',
    'upright',
    'wordy',
]

#: A span further right of a line than this many font sizes starts another line (a column gutter).
LINE_GAP: Final = 1.0
#: A bare clause number (``4.1``) takes the text beside it within this many font sizes.
CLAUSE_GAP: Final = 8.0
#: A line this many font sizes or less below a block's last line, overlapping it across, continues the block.
BLOCK_GAP: Final = 0.6
#: Points a column line may cross the page middle.
GUTTER: Final = 5.0
#: A span runs left to right when its baseline's direction has at least this x component (within ~8 degrees).
UPRIGHT: Final = 0.99
#: Spans run one way when their directions differ by at most this much (x and y differences summed).
SAME_WAY: Final = 0.02
#: Words: a word of at least this many letters, and letters (and spaces) this share of the text.
WORD_LETTERS: Final = 4
LETTER_SHARE: Final = 0.6
_CLAUSE: Final = re.compile(r'^(?:\d+(?:\.\d+)*\.?|[-\N{EN DASH}\N{EM DASH}\N{BULLET}])$')  # a clause number, a bullet
_NUMBER: Final = re.compile(r'^(?:\d+|[A-Z])(?:\.\d+)*\.?$')  # a heading's number: 2, 2.1, an appendix's A, A.1


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
    def text(self) -> str:
        """The spans' text left to right (vertically, top to bottom), spaced but where they share ``unspaced``."""
        parts: list[str] = []
        before = None
        start = (lambda b: b.y0) if self.vertical else (lambda b: b.x0)
        for span in sorted(self.spans, key=lambda s: start(s.provenance.bbox)):
            if before is not None and (span.unspaced is None or span.unspaced != before.unspaced):
                parts.append(' ')
            parts.append(span.text)
            before = span
        return ' '.join(''.join(parts).split())

    @property
    def y0(self) -> float:
        """The median span top (one tall glyph, a big bracket, does not move the line); vertically, its top."""
        tops = [s.provenance.bbox.y0 for s in self.spans]
        return min(tops) if self.vertical else median(tops)

    @property
    def y1(self) -> float:
        """The median span bottom; written vertically, the bottom of its last span."""
        bottoms = [s.provenance.bbox.y1 for s in self.spans]
        return max(bottoms) if self.vertical else median(bottoms)

    @property
    def x0(self) -> float:
        """The left end of its leftmost span; written vertically, the median span left: the column's."""
        lefts = [s.provenance.bbox.x0 for s in self.spans]
        return median(lefts) if self.vertical else min(lefts)

    @property
    def x1(self) -> float:
        """The right end of its rightmost span; written vertically, the median span right."""
        rights = [s.provenance.bbox.x1 for s in self.spans]
        return median(rights) if self.vertical else max(rights)

    @property
    def size(self) -> float:
        """Its largest font size."""
        return max(s.size for s in self.spans)

    @property
    def bold(self) -> bool:
        """Is every span with text bold?"""
        return all(s.bold for s in self.spans if s.text.strip())


def upright(span: TextSpan) -> bool:
    """Does ``span`` run left to right, as a page's text does (neither turned sideways nor upside down)?"""
    return span.direction[0] >= UPRIGHT


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


def lines(page: Page, *, weighted: bool) -> list[Line]:
    """The page's lines (unordered), its :func:`furniture` left out; a bare clause number takes the text beside it.

    ``weighted`` False (no bold in the document): any clause number or bullet takes its text; True: a bold number or
    appendix letter, and only bold text of its size.
    """
    mid = page.width / 2
    margin = {id(s) for s in furniture(page)}
    found: list[Line] = []
    flat = (s for s in page.spans if not s.vertical and id(s) not in margin)
    for span in sorted(flat, key=lambda s: (s.provenance.bbox.x0, s.provenance.bbox.y0)):
        box = span.provenance.bbox
        center = (box.y0 + box.y1) / 2
        for line in found:
            if not _same_way(line.spans[0], span):
                continue
            reach = _reach(line, span, weighted=weighted)
            near = -GUTTER <= box.x0 - line.x1 <= reach * max(span.size, line.size)
            if near and abs((line.y0 + line.y1) / 2 - center) <= 0.4 * max(span.size, line.size):
                line.spans.append(span)
                break
        else:
            found.append(Line([span], ''))
    for line in found:
        line.column = _column(line.x0, line.x1, mid)
    columns = _columns_down(s for s in page.spans if s.vertical)
    for block in _vertical_blocks(columns):
        column = _column(min(ln.x0 for ln in block), max(ln.x1 for ln in block), mid)
        for line in block:
            line.column = column
    return [line for line in found + columns if line.text]


def _column(x0: float, x1: float, mid: float) -> str:
    """The page column a line (or a vertical block) from ``x0`` to ``x1`` stands in: L, R or F (across ``mid``)."""
    return 'L' if x1 <= mid + GUTTER else 'R' if x0 >= mid - GUTTER else 'F'


def _columns_down(spans: Iterable[TextSpan]) -> list[Line]:
    """Vertical spans as lines down their columns: each next span centered on one, within :data:`LINE_GAP` below."""
    found: list[Line] = []
    for span in sorted(spans, key=lambda s: (s.provenance.bbox.y0, s.provenance.bbox.x0)):
        box = span.provenance.bbox
        center = (box.x0 + box.x1) / 2
        for line in found:
            size = max(span.size, line.size)
            near = -GUTTER <= box.y0 - line.y1 <= LINE_GAP * size
            if near and abs((line.x0 + line.x1) / 2 - center) <= 0.4 * size:
                line.spans.append(span)
                break
        else:
            found.append(Line([span], ''))
    return found


def _vertical_blocks(found: list[Line]) -> list[list[Line]]:
    """Vertical lines as paragraphs right to left: each next within :data:`BLOCK_GAP` left, beside it in height."""
    out: list[list[Line]] = []
    for line in sorted(found, key=lambda ln: (-ln.x1, ln.y0)):
        for block in out:
            last = block[-1]
            gap = last.x0 - line.x1
            beside = line.y0 < max(ln.y1 for ln in block) and min(ln.y0 for ln in block) < line.y1
            if beside and -0.5 * line.size <= gap <= BLOCK_GAP * max(line.size, last.size):
                block.append(line)
                break
        else:
            out.append([line])
    return out


@dataclass
class Block:
    """Lines set as one paragraph, top to bottom -- or written vertically, right to left."""

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
    for line in sorted((ln for ln in every if not ln.vertical), key=lambda ln: (ln.y0, ln.x0)):
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
    found += [Block(block) for block in _vertical_blocks([ln for ln in every if ln.vertical])]
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
    for full in _top_down([ln for ln in found if ln.column == 'F']):
        band.extend(ln for ln in columns if ln.y0 < full.y0 and ln not in out)
        columns = [ln for ln in columns if ln not in band]
        flush()
        out.append(full)
    band.extend(columns)
    flush()
    return out


def _top_down(found: list[Line]) -> list[Line]:
    """Lines top to bottom; vertical lines side by side (each above the last one's bottom) right to left."""
    rows: list[list[Line]] = []
    for line in sorted(found, key=lambda ln: ln.y0):
        if line.vertical and rows and rows[-1][-1].vertical and line.y0 < max(ln.y1 for ln in rows[-1]):
            rows[-1].append(line)
        else:
            rows.append([line])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: -ln.x1)]


def wordy(text: str) -> bool:
    """Does ``text`` read as words, not OCR noise: a word of :data:`WORD_LETTERS` letters, letters most of it?"""
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters >= LETTER_SHARE * len(text) and any(len(w) >= WORD_LETTERS and w.isalpha() for w in text.split())


def body_size(doc: Document) -> float:
    """The font size most of ``doc``'s text is set in (0 when it has none)."""
    sizes: Counter[float] = Counter()
    for span in doc.spans():
        sizes[round(span.size, 1)] += len(span.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0
