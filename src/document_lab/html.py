"""HTML tables read into the IR with the standard library's parser.

A cell is its visible text, whitespace collapsed (``<sup>2</sup>`` reads ``2``; ``<br>`` a space); a cell
spanning rows or columns repeats its text into every grid slot it covers, so each row reads the same width.
A nested table is a table of its own, numbered in document order; the outer cell keeps only its own text.
HTML has no geometry, so a cell cites a locator (``table[i]/row[r]/col[c]``) instead of a box.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from document_lab.ir import Provenance, Table

__all__ = ['read_html', 'tables_from_html']


def read_html(path: Path | str, *, encoding: str = 'utf-8') -> tuple[Table, ...]:
    """Every table in the HTML file at ``path``, citing the file's content hash and URI."""
    path = Path(path)
    data = path.read_bytes()
    return tables_from_html(data.decode(encoding), sha256=hashlib.sha256(data).hexdigest(), uri=path.resolve().as_uri())


def tables_from_html(text: str, *, sha256: str, uri: str) -> tuple[Table, ...]:
    """Every table in the HTML ``text``, in document order."""
    parser = _TableParser()
    parser.feed(text)
    parser.close()
    return tuple(
        Table(_grid(t.rows), None, Provenance(sha256, uri, locator=f'table[{t.index}]'))
        for t in sorted(parser.done, key=lambda t: t.index)
    )


@dataclass
class _Cell:
    parts: list[str] = field(default_factory=list)
    rowspan: int = 1
    colspan: int = 1

    @property
    def text(self) -> str:
        return ' '.join(''.join(self.parts).split())


@dataclass
class _Open:
    index: int
    rows: list[list[_Cell]] = field(default_factory=list)
    cell: _Cell | None = None


def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
    value = dict(attrs).get(name)
    return int(value) if value and value.strip().isdigit() and int(value) > 0 else 1


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Open] = []
        self.done: list[_Open] = []
        self.count = 0
        self.skip = 0  # inside <script>/<style>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {'script', 'style'}:
            self.skip += 1
        elif tag == 'table':
            self.stack.append(_Open(self.count))
            self.count += 1
        elif not self.stack:
            return
        elif tag == 'tr':
            self.stack[-1].rows.append([])
            self.stack[-1].cell = None
        elif tag in {'td', 'th'}:
            top = self.stack[-1]
            if not top.rows:
                top.rows.append([])
            top.cell = _Cell(rowspan=_span(attrs, 'rowspan'), colspan=_span(attrs, 'colspan'))
            top.rows[-1].append(top.cell)
        elif tag == 'br' and self.stack[-1].cell is not None:
            self.stack[-1].cell.parts.append(' ')

    def handle_endtag(self, tag: str) -> None:
        if tag in {'script', 'style'}:
            self.skip = max(0, self.skip - 1)
        elif tag == 'table' and self.stack:
            self.done.append(self.stack.pop())
        elif tag in {'td', 'th'} and self.stack:
            self.stack[-1].cell = None

    def handle_data(self, data: str) -> None:
        if not self.skip and self.stack and self.stack[-1].cell is not None:
            self.stack[-1].cell.parts.append(data)

    def close(self) -> None:
        super().close()
        while self.stack:  # an unclosed table still counts
            self.done.append(self.stack.pop())


def _grid(rows: list[list[_Cell]]) -> tuple[tuple[str, ...], ...]:
    """Lay cells into a grid, repeating a spanning cell into each slot it covers; pad short rows with ''."""
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    grid[(r + dr, c + dc)] = cell.text
            c += cell.colspan
    if not grid:
        return ()
    height = max(r for r, _ in grid) + 1
    width = max(c for _, c in grid) + 1
    return tuple(tuple(grid.get((r, c), '') for c in range(width)) for r in range(height))
