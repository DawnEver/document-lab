"""A document's markdown written from document_lab's own IR: headings and paragraphs from its lines.

Lines and their reading order are `document_lab.layout`'s.

- Headings: a line that is a Roman-numbered upper-case heading (``II. METHOD``) or a standalone back-matter word in
  capitals or title case (``REFERENCES``, ``Acknowledgments``, ``Appendix``, ``Conclusion``) is a section (``##``),
  whatever its size; else a line at least :data:`TITLE_SCALE` times the body size is a title (``#``: a paper bound in
  after the first opens with one); else an all-bold line numbered ``2`` / ``2.1`` is a section (``##``, ``###`` per
  depth); else page 1's largest text is its title when at least :data:`PAGE_TITLE_SCALE` times the body; else an
  all-bold line lettered as an appendix's ``A`` / ``A.1`` is a section. No heading is set smaller than
  :data:`HEADING_FLOOR` times the body (a footnote's number is). A document with no bold text at all (OCR'd: it
  carries no weight) cannot be judged by weight: there a short numbered line that is no sentence heads its clause,
  and any heading must read as words (`document_lab.layout.wordy`) -- a figure's labels, OCR'd, do not. A line
  without a letter (a page number), a run of four dots (a contents leader, however garbled) or a contents entry (a
  leader then a page number) never heads, nor does any line of a table-of-contents page (:data:`TOC_LINES` such
  entries): it names the sections, it does not open them.
- Paragraphs: lines of one column closer than :data:`PARAGRAPH_GAP` line heights join with a space (a hyphen at a
  line end joins the word); a larger gap, a heading or a column change starts a new paragraph.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from document_lab import layout

if TYPE_CHECKING:
    from document_lab.ir import Document, Page

__all__ = ['back_matter', 'page_markdowns', 'to_markdown']

#: A gap between lines larger than this many line heights starts a paragraph.
PARAGRAPH_GAP: Final = 0.6
#: How many blocks back a cut paragraph's continuation may look.
REJOIN_REACH: Final = 3
#: Consecutive lines may overlap by this many font sizes (tight leading).
OVERLAP: Final = 0.3
#: A title is at least this many times the body size.
TITLE_SCALE: Final = 1.6
#: Page 1's largest text is its title when at least this many times the body size (a title set smaller).
PAGE_TITLE_SCALE: Final = 1.2
#: A heading is set at least this many times the body size: a footnote, set smaller, heads nothing.
HEADING_FLOOR: Final = 0.9
#: Prefix that keeps body text starting with "#" from reading as a markdown heading.
ESCAPE: Final = chr(92)
#: Unweighted: a numbered line of at most this many words, not ending a sentence, heads its clause.
HEADING_WORDS: Final = 8
_ROMAN: Final = re.compile(r'^(?:X{0,3})(?:IX|IV|V?I{0,3})\.\s+[A-Z][A-Z0-9 ,:;&/()\-]{3,}$')
#: Back matter, matched in capitals; a line heads by it in capitals or in title case.
_STANDALONE: Final = re.compile(r'^(?:ACKNOWLEDG(?:E)?MENTS?|REFERENCES|APPENDIX|CONCLUSIONS?)$')
#: A clause number: one or two digits, then ``.N`` parts -- a year (``2011 Evaluation ...``, a table row) is none.
_NUMBERED: Final = re.compile(r'^(\d{1,2}(?:\.\d+)*)\.?\s+[A-Z]')
#: An appendix's number: a letter, then ``.N`` parts; weighted documents only (``A`` starts sentences too).
_LETTERED: Final = re.compile(r'^([A-Z](?:\.\d+)*)\.?\s+[A-Z]')
#: A table-of-contents entry: words, a leader (dots, ellipses, middle dots, rules), a page number at the end.
_TOC_ENTRY: Final = re.compile(r'^(?=.*[A-Za-z]).*?(?:(?:\.\s?){4,}|\u2026{2,}|\u00b7{3,}|_{4,})\s*[\dixvIXV]{1,5}$')
#: Lines of a page that are contents entries for the page to be a table of contents.
TOC_LINES: Final = 3


def to_markdown(doc: Document) -> str:
    """The document as markdown, page after page: a paragraph a page break cut is joined again."""
    return '\n\n'.join(_rejoin([block for page in _blocks(doc) for block in page])) + '\n'


def page_markdowns(doc: Document) -> list[str]:
    """Each page's own markdown, in order (empty for a page without text).

    Body size and title are judged over the whole document; nothing is joined across a page break: a page stands alone.
    """
    return ['\n\n'.join(_rejoin(page)) + '\n' if page else '' for page in _blocks(doc)]


def _blocks(doc: Document) -> list[list[str]]:
    weighted = any(span.bold for span in doc.spans())
    body = layout.body_size(doc)
    first = layout.lines(doc.pages[0], weighted=weighted) if doc.pages else []
    largest = max((ln.size for ln in first), default=0.0)
    title = largest if largest >= PAGE_TITLE_SCALE * body else None
    return [_page_blocks(page, body, title if i == 0 else None, weighted=weighted) for i, page in enumerate(doc.pages)]


def back_matter(text: str) -> bool:
    """Is ``text`` a back-matter heading in capitals or title case (``REFERENCES``, ``Acknowledgments``, ...)?"""
    return bool(_STANDALONE.match(text.upper())) and text in {text.upper(), text.capitalize()}


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


def _heading(line: layout.Line, body: float, title: float | None, *, weighted: bool) -> str | None:
    """The line as a heading (``#`` a title, ``##`` a section, deeper per clause depth), else None.

    ``title``: the size page 1 sets its title in, on page 1 (None elsewhere, or when page 1 sets none apart).
    """
    text = line.text
    if not any(c.isalpha() for c in text) or _TOC_ENTRY.match(text) or '....' in text:
        return None
    if line.size < HEADING_FLOOR * body or (not weighted and not layout.wordy(text)):
        return None  # a footnote; OCR noise: a figure's labels, a garbled leader
    if _ROMAN.match(text) or back_matter(text):
        return f'## {text}'
    if line.size >= TITLE_SCALE * body:
        return f'# {text}'
    # A clause number states a section more surely than page 1's size states a title; an initial letter less surely.
    unsentenced = len(text.split()) <= HEADING_WORDS and not text.endswith(('.', ',', ';', ':'))
    clause = len(text) <= 80 and (line.bold if weighted else unsentenced)
    numbered = _NUMBERED.match(text)
    if numbered and clause:
        return f'{"#" * (numbered.group(1).count(".") + 2)} {text}'
    if title is not None and line.size >= title:
        return f'# {text}'
    lettered = _LETTERED.match(text) if weighted else None
    if lettered and clause:
        return f'{"#" * (lettered.group(1).count(".") + 2)} {text}'
    return None


def _page_blocks(page: Page, body: float, title_size: float | None, *, weighted: bool) -> list[str]:
    blocks: list[str] = []
    paragraph: list[str] = []
    previous: layout.Line | None = None
    title: list[str] = []

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(' '.join(paragraph))
            paragraph.clear()

    lines = layout.ordered(layout.lines(page, weighted=weighted))
    contents = sum(bool(_TOC_ENTRY.match(line.text)) for line in lines) >= TOC_LINES
    for line in lines:
        heading = None if contents else _heading(line, body, title_size, weighted=weighted)
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


def _continues_heading(blocks: list[str], previous: layout.Line | None, line: layout.Line) -> bool:
    """An upper-case line right under a section heading, in its column, is the heading's wrapped second line."""
    if not blocks or not blocks[-1].startswith('## ') or previous is None or previous.column != line.column:
        return False
    text = line.text
    close = -OVERLAP * line.size <= line.y0 - previous.y1 <= PARAGRAPH_GAP * line.size
    letters = sum(c.isalpha() for c in text)
    words = text.isprintable() and letters >= 0.6 * len(text.replace(' ', ''))
    return close and words and text == text.upper()
