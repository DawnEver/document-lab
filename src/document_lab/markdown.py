"""A document's markdown written from document_lab's own IR: reading order, headings and paragraphs from the spans.

- Reading order: a page's lines are FULL width (crossing the page middle) or in the LEFT or RIGHT column.
  Full-width lines cut the page into bands; each band reads its left column, then its right.
- Headings: the largest text on page 1 is the title (``#``); a line that is a Roman-numbered upper-case heading
  (``II. METHOD``), a standalone ``REFERENCES``/``ACKNOWLEDGMENT``/``APPENDIX``/``CONCLUSION``, or an all-bold
  short line numbered ``2`` / ``2.1`` is a section (``##``, ``###`` per depth). A line without a letter (a page
  number) or a contents entry (a leader then a page number) never is, nor is any line of a table-of-contents page
  (:data:`TOC_LINES` such entries): it names the sections, it does not open them.
- Paragraphs: lines of one column closer than :data:`PARAGRAPH_GAP` line heights join with a space (a hyphen at a
  line end joins the word); a larger gap, a heading or a column change starts a new paragraph.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from document_lab.ir import Document, Page, TextSpan

__all__ = ['page_markdowns', 'to_markdown']

#: A gap between lines larger than this many line heights starts a paragraph.
PARAGRAPH_GAP: Final = 0.6
#: How many blocks back a cut paragraph's continuation may look.
REJOIN_REACH: Final = 3
#: Consecutive lines may overlap by this many font sizes (tight leading).
OVERLAP: Final = 0.3
#: A title is at least this many times the body size.
TITLE_SCALE: Final = 1.6
#: Prefix that keeps body text starting with "#" from reading as a markdown heading.
ESCAPE: Final = chr(92)
#: A gap between spans wider than this many font sizes is a space.
WORD_GAP: Final = 0.15
#: A span further right of a line than this many font sizes starts another line (a column gutter).
LINE_GAP: Final = 1.0
#: Points a column line may cross the page middle.
GUTTER: Final = 5.0
_ROMAN: Final = re.compile(r'^(?:X{0,3})(?:IX|IV|V?I{0,3})\.\s+[A-Z][A-Z0-9 ,:;&/()\-]{3,}$')
_STANDALONE: Final = re.compile(r'^(?:ACKNOWLEDG(?:E)?MENTS?|REFERENCES|APPENDIX|CONCLUSIONS?)$')
_NUMBERED: Final = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+[A-Z]')
#: A table-of-contents entry: words, a leader (dots, ellipses, middle dots, rules), a page number at the end.
_TOC_ENTRY: Final = re.compile(r'^(?=.*[A-Za-z]).*?(?:(?:\.\s?){4,}|\u2026{2,}|\u00b7{3,}|_{4,})\s*[\dixvIXV]{1,5}$')
#: Lines of a page that are contents entries for the page to be a table of contents.
TOC_LINES: Final = 3


@dataclass
class _Line:
    spans: list[TextSpan]
    column: str  # 'L', 'R' or 'F'

    @property
    def text(self) -> str:
        parts: list[str] = []
        right = None
        for span in sorted(self.spans, key=lambda s: s.provenance.bbox.x0):
            box = span.provenance.bbox
            if right is not None and box.x0 - right > WORD_GAP * span.size:
                parts.append(' ')
            parts.append(span.text)
            right = box.x1
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
    def size(self) -> float:
        return max(s.size for s in self.spans)

    @property
    def bold(self) -> bool:
        return all(s.bold for s in self.spans if s.text.strip())


def to_markdown(doc: Document) -> str:
    """The document as markdown, page after page: a paragraph a page break cut is joined again."""
    return '\n\n'.join(_rejoin([block for page in _blocks(doc) for block in page])) + '\n'


def page_markdowns(doc: Document) -> list[str]:
    """Each page's own markdown, in order (empty for a page without text).

    Body size and title are judged over the whole document; nothing is joined across a page break: a page stands alone.
    """
    return ['\n\n'.join(_rejoin(page)) + '\n' if page else '' for page in _blocks(doc)]


def _blocks(doc: Document) -> list[list[str]]:
    body = _body_size(doc)
    title_size = max((s.size for s in doc.pages[0].spans), default=0.0) if doc.pages else 0.0
    return [_page_blocks(page, body, title_size) for page in doc.pages]


def _rejoin(blocks: list[str]) -> list[str]:
    """Paragraphs cut by a column or page break, joined again.

    A block starting lower-case continues the nearest earlier paragraph (within :data:`REJOIN_REACH` blocks,
    skipping page furniture between them) that ends unterminated; a heading stops the search.
    """
    out: list[str] = []
    for block in blocks:
        target = None
        if block[:1].islower():
            for i in range(len(out) - 1, max(-1, len(out) - 1 - REJOIN_REACH), -1):
                if out[i].startswith('#'):
                    break
                if not out[i].rstrip().endswith(('.', ':', '?', '!')):
                    target = i
                    break
        if target is None:
            out.append(block)
            continue
        head = out[target].rstrip()
        out[target] = head[:-1] + block if head.endswith('-') else f'{head} {block}'
    return out


def _body_size(doc: Document) -> float:
    sizes = Counter()
    for span in doc.spans():
        sizes[round(span.size, 1)] += len(span.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _lines(page: Page) -> list[_Line]:
    mid = page.width / 2
    lines: list[_Line] = []
    # A line: spans on one baseline, each starting within LINE_GAP font sizes of the line's right end (a column
    # gutter is wider, so two columns' lines at one height stay two lines).
    for span in sorted(page.spans, key=lambda s: (s.provenance.bbox.x0, s.provenance.bbox.y0)):
        box = span.provenance.bbox
        centre = (box.y0 + box.y1) / 2
        for line in lines:
            right = max(s.provenance.bbox.x1 for s in line.spans)
            near = -GUTTER <= box.x0 - right <= LINE_GAP * max(span.size, line.size)
            if near and abs((line.y0 + line.y1) / 2 - centre) <= 0.4 * max(span.size, line.size):
                line.spans.append(span)
                break
        else:
            lines.append(_Line([span], ''))
    for line in lines:
        x0 = min(s.provenance.bbox.x0 for s in line.spans)
        x1 = max(s.provenance.bbox.x1 for s in line.spans)
        line.column = 'L' if x1 <= mid + GUTTER else 'R' if x0 >= mid - GUTTER else 'F'
    return [line for line in lines if line.text]


def _ordered(lines: list[_Line]) -> list[_Line]:
    """Bands between full-width lines; each band's left column, then its right."""
    out: list[_Line] = []
    band: list[_Line] = []

    def flush() -> None:
        out.extend(sorted((ln for ln in band if ln.column == 'L'), key=lambda ln: ln.y0))
        out.extend(sorted((ln for ln in band if ln.column == 'R'), key=lambda ln: ln.y0))
        band.clear()

    columns = [ln for ln in lines if ln.column != 'F']
    for full in sorted((ln for ln in lines if ln.column == 'F'), key=lambda ln: ln.y0):
        band.extend(ln for ln in columns if ln.y0 < full.y0 and ln not in out)
        columns = [ln for ln in columns if ln not in band]
        flush()
        out.append(full)
    band.extend(columns)
    flush()
    return out


def _heading(line: _Line, body: float, title_size: float | None) -> str | None:
    text = line.text
    if not any(c.isalpha() for c in text) or _TOC_ENTRY.match(text):
        return None
    if title_size is not None and line.size >= TITLE_SCALE * body:
        return f'# {text}'
    if _ROMAN.match(text) or _STANDALONE.match(text):
        return f'## {text}'
    numbered = _NUMBERED.match(text)
    if numbered and line.bold and len(text) <= 80:
        return f'{"#" * (numbered.group(1).count(".") + 2)} {text}'
    return None


def _page_blocks(page: Page, body: float, title_size: float | None) -> list[str]:
    blocks: list[str] = []
    paragraph: list[str] = []
    previous: _Line | None = None
    title: list[str] = []

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(' '.join(paragraph))
            paragraph.clear()

    lines = _ordered(_lines(page))
    contents = sum(bool(_TOC_ENTRY.match(line.text)) for line in lines) >= TOC_LINES
    for line in lines:
        heading = None if contents else _heading(line, body, title_size)
        if heading and heading.startswith('# '):
            title.append(line.text)  # a title wraps over several lines
            previous = line
            continue
        if title:
            end_paragraph()
            blocks.append('# ' + ' '.join(title))
            title.clear()
        if heading:
            end_paragraph()
            blocks.append(heading)
            previous = line
            continue
        if _continues_heading(blocks, previous, line):
            blocks[-1] = f'{blocks[-1]} {line.text}'
            previous = line
            continue
        joined = (
            previous is not None
            and previous.column == line.column
            and -OVERLAP * line.size <= line.y0 - previous.y1 <= PARAGRAPH_GAP * line.size
        )
        if not joined:
            end_paragraph()
        if joined and paragraph and paragraph[-1].endswith('-') and line.text[:1].islower():
            paragraph[-1] = paragraph[-1][:-1] + line.text
        else:
            paragraph.append(ESCAPE + line.text if line.text.startswith('#') else line.text)
        previous = line
    if title:
        end_paragraph()
        blocks.append('# ' + ' '.join(title))
    end_paragraph()
    return blocks


def _continues_heading(blocks: list[str], previous: _Line | None, line: _Line) -> bool:
    """An upper-case line right under a section heading, in its column, is the heading's wrapped second line."""
    if not blocks or not blocks[-1].startswith('## ') or previous is None or previous.column != line.column:
        return False
    text = line.text
    close = -OVERLAP * line.size <= line.y0 - previous.y1 <= PARAGRAPH_GAP * line.size
    letters = sum(c.isalpha() for c in text)
    words = text.isprintable() and letters >= 0.6 * len(text.replace(' ', ''))
    return close and words and text == text.upper()
