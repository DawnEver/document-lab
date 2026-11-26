"""A page's spans as lines in reading order: what every layout of a page -- its markdown, its regions -- starts from.

- Lines: spans on one baseline, each starting within :data:`LINE_GAP` font sizes of the line's right end; a column
  gutter is wider, so two columns' lines at one height stay two lines. A line's text spaces two neighboring spans
  apart unless they share an ``unspaced`` number (:class:`~document_lab.ir.TextSpan`): where a word gap is, is
  the spans' source's to say, never measured here. In an unweighted document (no bold span:
  OCR'd text carries no weight) a bare clause number or bullet takes the text set apart beside it, within
  :data:`CLAUSE_GAP` font sizes, as a standard sets ``4.1    General``.
- Columns: a line is FULL width (crossing the page middle) or in the LEFT or RIGHT column; :func:`ordered` reads the
  bands between full-width lines, each its left column, then its right.
- Blocks: :func:`blocks` sets lines into paragraphs -- each next line within :data:`BLOCK_GAP` font sizes below
  the last, in its column, overlapping it across.
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
    from document_lab.ir import Document, Page, TextSpan

__all__ = [
    'BLOCK_GAP',
    'CLAUSE_GAP',
    'GUTTER',
    'LINE_GAP',
    'Block',
    'Line',
    'blocks',
    'body_size',
    'lines',
    'ordered',
    'wordy',
]

#: A span further right of a line than this many font sizes starts another line (a column gutter).
LINE_GAP: Final = 1.0
#: Unweighted: a bare clause number (``4.1``) or bullet takes the text beside it within this many font sizes.
CLAUSE_GAP: Final = 8.0
#: A line this many font sizes or less below a block's last line, overlapping it across, continues the block.
BLOCK_GAP: Final = 0.6
#: Points a column line may cross the page middle.
GUTTER: Final = 5.0
#: Words: a word of at least this many letters, and letters (and spaces) this share of the text.
WORD_LETTERS: Final = 4
LETTER_SHARE: Final = 0.6
_CLAUSE: Final = re.compile(r'^(?:\d+(?:\.\d+)*\.?|[-\N{EN DASH}\N{EM DASH}\N{BULLET}])$')  # a clause number, a bullet


@dataclass
class Line:
    """Spans on one baseline, and the ``column`` they stand in: ``'L'``, ``'R'`` or ``'F'`` (full width)."""

    spans: list[TextSpan]
    column: str

    @property
    def text(self) -> str:
        """The spans' text left to right, a space between two unless they share an ``unspaced`` number."""
        parts: list[str] = []
        before = None
        for span in sorted(self.spans, key=lambda s: s.provenance.bbox.x0):
            if before is not None and (span.unspaced is None or span.unspaced != before.unspaced):
                parts.append(' ')
            parts.append(span.text)
            before = span
        return ' '.join(''.join(parts).split())

    @property
    def y0(self) -> float:
        """The median span top: one tall glyph (a big bracket) does not move the line."""
        return median(s.provenance.bbox.y0 for s in self.spans)

    @property
    def y1(self) -> float:
        """The median span bottom."""
        return median(s.provenance.bbox.y1 for s in self.spans)

    @property
    def x0(self) -> float:
        """The left end of its leftmost span."""
        return min(s.provenance.bbox.x0 for s in self.spans)

    @property
    def x1(self) -> float:
        """The right end of its rightmost span."""
        return max(s.provenance.bbox.x1 for s in self.spans)

    @property
    def size(self) -> float:
        """Its largest font size."""
        return max(s.size for s in self.spans)

    @property
    def bold(self) -> bool:
        """Is every span with text bold?"""
        return all(s.bold for s in self.spans if s.text.strip())


def lines(page: Page, *, weighted: bool) -> list[Line]:
    """The page's lines (unordered); ``weighted`` False lets a bare clause number or bullet take its text."""
    mid = page.width / 2
    found: list[Line] = []
    for span in sorted(page.spans, key=lambda s: (s.provenance.bbox.x0, s.provenance.bbox.y0)):
        box = span.provenance.bbox
        center = (box.y0 + box.y1) / 2
        for line in found:
            reach = CLAUSE_GAP if not weighted and _CLAUSE.match(line.text) else LINE_GAP
            near = -GUTTER <= box.x0 - line.x1 <= reach * max(span.size, line.size)
            if near and abs((line.y0 + line.y1) / 2 - center) <= 0.4 * max(span.size, line.size):
                line.spans.append(span)
                break
        else:
            found.append(Line([span], ''))
    for line in found:
        line.column = 'L' if line.x1 <= mid + GUTTER else 'R' if line.x0 >= mid - GUTTER else 'F'
    return [line for line in found if line.text]


@dataclass
class Block:
    """Lines set as one paragraph, top to bottom."""

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
    for line in sorted(lines(page, weighted=True), key=lambda ln: (ln.y0, ln.x0)):
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
    first = {id(b.lines[0]): b for b in found}
    return [first[id(ln)] for ln in ordered([b.lines[0] for b in found])]


def ordered(found: list[Line]) -> list[Line]:
    """Bands between full-width lines; each band's left column, then its right."""
    out: list[Line] = []
    band: list[Line] = []

    def flush() -> None:
        out.extend(sorted((ln for ln in band if ln.column == 'L'), key=lambda ln: ln.y0))
        out.extend(sorted((ln for ln in band if ln.column == 'R'), key=lambda ln: ln.y0))
        band.clear()

    columns = [ln for ln in found if ln.column != 'F']
    for full in sorted((ln for ln in found if ln.column == 'F'), key=lambda ln: ln.y0):
        band.extend(ln for ln in columns if ln.y0 < full.y0 and ln not in out)
        columns = [ln for ln in columns if ln not in band]
        flush()
        out.append(full)
    band.extend(columns)
    flush()
    return out


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
