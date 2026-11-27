"""What a tagged PDF's tags state, in the terms layout infers: reading order, paragraphs, headings, tables, figures.

A tagged PDF (ISO 32000-2 14.7-14.8) states what document-lab otherwise infers from where text stands
(`document_lab.layout`). This module reads :attr:`~document_lab.ir.Document.structure` into those terms for the
modules that infer them, each taking it as an option (on by default; the inferred kept as the alternative;
`document_lab.agreement` sets the two side by side). Roles are judged by the standard role each maps to; a title is
also an element whose role as stated is ``Title`` (a word processor maps its title style to ``P``).

- :func:`reading` -- each span the tags cover, its place in the stated order and its block: the innermost element
  of a paragraph's kind holding it (:data:`BLOCKS`: a paragraph, heading, list item, table cell, caption, ...), else
  the innermost element holding it. A line (`document_lab.layout.Line`) takes its first span's in that order.
  :meth:`Reading.ordered` sets a page's lines in the stated order, the lines the tags do not cover (an artifact: a
  running head, a page number) keeping the places layout's order gives them; :meth:`Reading.level` is a line's
  heading level, as markdown counts it: ``Title`` 1, ``Hn`` n + 1 (a document's title above its sections), ``H`` one
  more than the sections (``Sect``) it is in, at most 6.
- :func:`tables` -- a page's tagged tables as IR tables: a ``Table``'s rows (``TR``, under ``THead``, ``TBody`` or
  ``TFoot`` or not), each row's cells (``TH``, ``TD``) set left to right in the first free slot, spanning ``RowSpan``
  rows and ``ColSpan`` columns. A row belongs to the page its content is first drawn on (a row that draws nothing, to
  the page of the row before it): a table running over a page break is two tables, one a page.
- :func:`regions` -- a page's tagged tables and figures (``Table``, ``Figure``) as boxes: what each covers on the
  page, its caption (a ``Caption`` among its kids or next to it) and a figure's ``/Alt``; :func:`covers` says
  whether the tags cover anything on a page at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from document_lab import layout
from document_lab.ir import Element, MarkedContent, Provenance, Table

if TYPE_CHECKING:
    from collections.abc import Iterator

    from document_lab.ir import BBox, Document, Structure, TextSpan

__all__ = ['BLOCKS', 'Reading', 'TaggedRegion', 'covers', 'heading_level', 'reading', 'regions', 'tables']

#: The standard roles a block of text is set in: what a paragraph is, in tags.
BLOCKS: Final = frozenset(
    {
        'P',
        'H',
        'Title',
        'Caption',
        'LI',
        'TH',
        'TD',
        'TOCI',
        'BlockQuote',
        'Formula',
        'Figure',
        'Note',
        'FENote',
        'BibEntry',
    }
)
#: The standard roles a table's rows are grouped in.
_ROW_GROUPS: Final = frozenset({'THead', 'TBody', 'TFoot'})
_CELLS: Final = frozenset({'TH', 'TD'})
#: At most this many heading levels: markdown's.
_LEVELS: Final = 6


def heading_level(element: Element, sections: int = 0) -> int | None:
    """``element``'s heading level as markdown counts it; None for no heading.

    ``sections``: how many ``Sect`` elements it is in.
    """
    role = element.standard or ''
    if role == 'Title' or element.role == 'Title':
        return 1
    if role == 'H':
        return min(_LEVELS, 1 + max(1, sections))
    if role.startswith('H') and role[1:].isdigit():
        return min(_LEVELS, int(role[1:]) + 1)
    return None


def _block(element: Element) -> bool:
    return element.standard in BLOCKS or heading_level(element) is not None


@dataclass(frozen=True)
class Reading:
    """Each span a document's tags cover: its place in the stated order, its block, and each block's heading level."""

    rank: dict[TextSpan, int]
    block: dict[TextSpan, Element]
    levels: dict[Element, int]

    def _first(self, line: layout.Line) -> TextSpan | None:
        ranked = [s for s in line.spans if s in self.rank]
        return min(ranked, key=self.rank.__getitem__) if ranked else None

    def rank_of(self, line: layout.Line) -> int | None:
        """The line's place in the stated order: its first span's; None when the tags cover none of its spans."""
        first = self._first(line)
        return None if first is None else self.rank[first]

    def block_of(self, line: layout.Line) -> Element | None:
        """The block the line's first span in the stated order is set in; None when the tags cover none of them."""
        first = self._first(line)
        return None if first is None else self.block[first]

    def level(self, line: layout.Line) -> int | None:
        """The heading level of the line's block; None when it heads nothing."""
        block = self.block_of(line)
        return None if block is None else self.levels.get(block)

    def ordered(self, lines: list[layout.Line]) -> list[layout.Line]:
        """``lines`` in the stated order; a line the tags do not cover keeps its place in layout's order."""
        inferred = layout.ordered(lines)
        stated = iter(sorted((ln for ln in inferred if self.rank_of(ln) is not None), key=self.rank_of))
        return [next(stated) if self.rank_of(ln) is not None else ln for ln in inferred]


def reading(doc: Document) -> Reading | None:
    """What ``doc``'s tags state of its text; None when it has no tags or they cover no span."""
    if doc.structure is None:
        return None
    rank: dict[TextSpan, int] = {}
    block: dict[TextSpan, Element] = {}
    levels: dict[Element, int] = {}

    def walk(element: Element, holder: Element | None, sections: int) -> None:
        if _block(element):
            holder = element
            if (level := heading_level(element, sections)) is not None:
                levels[element] = level
        sections += element.standard == 'Sect'
        for kid in element.kids:
            if isinstance(kid, Element):
                walk(kid, holder, sections)
            elif isinstance(kid, MarkedContent):
                for span in kid.spans:
                    rank.setdefault(span, len(rank))
                    block.setdefault(span, holder or element)

    for root in doc.structure.roots:
        walk(root, None, 0)
    return Reading(rank, block, levels) if rank else None


def _elements(element: Element, siblings: tuple[Element, ...]) -> Iterator[tuple[Element, tuple[Element, ...]]]:
    """``element`` and every element below it, each with the elements beside it (itself among them)."""
    yield element, siblings
    kids = tuple(k for k in element.kids if isinstance(k, Element))
    for kid in kids:
        yield from _elements(kid, kids)


def _rows(table: Element) -> list[Element]:
    """A table's rows, in order: its ``TR`` kids, those of its row groups too; a table inside a cell is its own."""
    rows = []
    for kid in table.kids:
        if not isinstance(kid, Element):
            continue
        if kid.standard == 'TR':
            rows.append(kid)
        elif kid.standard in _ROW_GROUPS:
            rows += _rows(kid)
    return rows


def _span(cell: Element, name: str) -> int:
    value = cell.attribute(name)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 1 else 1


def _on_page(rows: list[Element], page: int) -> list[Element]:
    """The rows of a table on ``page``: those whose content is first drawn there.

    A row drawing nothing follows the row before it; one before every row that draws, the first of them.
    """
    first = [min(pages) for row in rows if (pages := {i.page for i in row.content() if i.page is not None})]
    out, last = [], first[0] if first else None
    for row in rows:
        pages = {item.page for item in row.content() if item.page is not None}
        last = min(pages) if pages else last
        if last == page:
            out.append(row)
    return out


def _table(element: Element, page: int, provenance: Provenance) -> Table | None:
    rows = _on_page(_rows(element), page)
    if not rows:
        return None
    slots: dict[tuple[int, int], tuple[Element, int, int]] = {}
    taken: set[tuple[int, int]] = set()
    for r, row in enumerate(rows):
        c = 0
        for cell in (k for k in row.kids if isinstance(k, Element) and k.standard in _CELLS):
            while (r, c) in taken:
                c += 1
            spans = min(_span(cell, 'RowSpan'), len(rows) - r), _span(cell, 'ColSpan')
            slots[r, c] = cell, *spans
            taken |= {(r + i, c + j) for i in range(spans[0]) for j in range(spans[1])}
            c += spans[1]
    width = max((c + 1 for _, c in taken), default=0)
    texts = [[''] * width for _ in rows]
    boxes: list[list[BBox | None]] = [[None] * width for _ in rows]
    for (r, c), (cell, _, _) in slots.items():
        texts[r][c] = cell.text
        boxes[r][c] = cell.bbox(page)
    return Table(
        rows=tuple(map(tuple, texts)),
        cell_bboxes=tuple(map(tuple, boxes)),
        provenance=provenance,
        merged=tuple((r, c, rs, cs) for (r, c), (_, rs, cs) in sorted(slots.items()) if rs > 1 or cs > 1),
        heads=tuple((r, c) for (r, c), (cell, _, _) in sorted(slots.items()) if cell.standard == 'TH'),
    )


def tables(doc: Document, page: int) -> list[Table]:
    """The tables ``doc``'s tags state on 1-based ``page``, in the stated order (none for an untagged document)."""
    if doc.structure is None:
        return []
    found = []
    for element in doc.structure.elements():
        if element.standard == 'Table' and (box := element.bbox(page)) is not None:
            table = _table(element, page, Provenance(doc.sha256, doc.uri, page=page, bbox=box))
            if table is not None:
                found.append(table)
    return found


@dataclass(frozen=True)
class TaggedRegion:
    """A ``table`` or ``figure`` the tags state on a page: its box, its caption's text, and a figure's ``/Alt``."""

    kind: Literal['table', 'figure']
    bbox: BBox
    caption: str | None
    alt: str | None


def _caption(element: Element, siblings: tuple[Element, ...]) -> str | None:
    """The text of a ``Caption`` among ``element``'s kids, else of one right before or after it among ``siblings``."""
    for kid in element.kids:
        if isinstance(kid, Element) and kid.standard == 'Caption':
            return kid.text
    at = next(i for i, k in enumerate(siblings) if k is element)
    near = [siblings[i] for i in (at - 1, at + 1) if 0 <= i < len(siblings)]
    return next((k.text for k in near if k.standard == 'Caption'), None)


def covers(structure: Structure, page: int) -> bool:
    """Whether ``structure`` covers anything drawn on 1-based ``page``: whether the tags state that page at all."""
    return any(
        isinstance(item, MarkedContent) and item.page == page and (item.spans or item.paths or item.figures)
        for element in structure.elements()
        for item in element.kids
    )


def regions(structure: Structure, page: int) -> list[TaggedRegion]:
    """The tables and figures ``structure`` states on 1-based ``page``, in the stated order."""
    found = []
    for root in structure.roots:
        for element, siblings in _elements(root, structure.roots):
            kind = {'Table': 'table', 'Figure': 'figure'}.get(element.standard or '')
            if kind is not None and (box := element.bbox(page)) is not None:
                found.append(TaggedRegion(kind, box, _caption(element, siblings), element.alt))
    return found
