"""A page's figures, tables and equations as regions, found from its text alone.

Nothing here reads a drawing or an image: an OCR'd capture is one picture with words over it, so a region is told
by its words. A figure's caption (``Figure 2 - ...``) is below it: the figure runs up from the caption to the
nearest prose above. A table's caption is above it: the table runs down from the caption to the next prose, clause
heading or caption. An equation number standing alone at the right (``(3)``, ``(A.1)``) is an equation: its region
takes in every line beside it that is not prose -- a formula OCR'd reads as noise -- up to the prose around it.
Prose is a line that reads as words (`document_lab.layout.wordy`), set near the body size, across half the text
width. A mention in prose (``Table 7, item 1d``) or a contents entry (a leader) is no caption. Each region spans
the text width; its label is the caption's or the number's, as read.
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
#: A region stops this many body sizes short of the prose that bounds it.
PAD: Final = 0.3
#: An equation takes a line not prose within this many body sizes of it, above or below.
EQUATION_REACH: Final = 2.0

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
        words = [ln for ln in self.lines if layout.wordy(ln.text)]
        self.left = min((ln.x0 for ln in words), default=0.0)
        self.right = max((ln.x1 for ln in words), default=page.width)

    def prose(self, line: layout.Line) -> bool:
        return (
            layout.wordy(line.text)
            and line.size <= PROSE_SIZE * self.body
            and '....' not in line.text
            and line.x1 - line.x0 >= PROSE_WIDTH * (self.right - self.left)
        )

    def box(self, top: float, bottom: float) -> BBox:
        return BBox(self.left, max(top, 0.0), self.right, min(bottom, self.page.height))


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
    return sorted(found, key=lambda r: r.bbox.y0)
