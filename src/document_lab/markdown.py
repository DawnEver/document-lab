"""A document's markdown written from document_lab's own IR: headings and paragraphs from its lines.

Lines and their reading order are `document_lab.layout`'s -- or, for a tagged PDF, what its tags state
(`document_lab.tags`; the ``tagged`` uses, all on by default): ``order``, lines in the stated order; ``headings``, a
line heading as its tag says (``Title`` ``#``, ``H1`` ``##``, ...); ``paragraphs``, the lines of one tagged block (a
paragraph, a list item, a cell) one paragraph, a block rejoined across a break only with its own element. A tagged
heading the rules below make a heading too takes their level (a tag's level follows a word processor's styles: the
IEEE template sets its acknowledgment and references as ``H5``); one only the tags state keeps its tag's level, at
most one below the heading before it (no level skipped). Tags
stating no heading at all leave every line's heading to the rules below, tags stating no title the title, and a
heading those rules find heads no tagged block. A line the tags do not cover -- an artifact, a document tagged in
part -- is read as an untagged one is:

- Headings: the largest text on page 1 is the title (``#``); a line that is a Roman-numbered upper-case heading
  (``II. METHOD``), a standalone ``REFERENCES``/``ACKNOWLEDGMENT``/``APPENDIX``/``CONCLUSION``, or an all-bold
  line numbered ``2`` / ``2.1`` is a section (``##``, ``###`` per depth). A document with no bold text at all (OCR'd:
  it carries no weight) cannot be judged by weight: there a short numbered line that is no sentence, nor all
  capitals, heads its clause, and any heading must read as words (`document_lab.layout.wordy`) -- a figure's
  labels, OCR'd, do not. A line without a letter (a page number), a run of four dots (a contents leader, however
  garbled) or a contents entry (a leader then a page number) never heads, nor does any line of a table-of-contents
  page (:data:`TOC_LINES` such entries): it names the sections, it does not open them.
- Paragraphs: lines of one column closer than :data:`PARAGRAPH_GAP` line heights join with a space (a hyphen at a
  line end joins the word); a larger gap, a heading or a column change starts a new paragraph.

:func:`headings` lists the headings either way, for `document_lab.agreement` to set side by side.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from document_lab import layout, tags

if TYPE_CHECKING:
    from collections.abc import Collection

    from document_lab.ir import Document, Element, Page

#: A block of markdown, and the tagged element it is set from (None: inferred).
type _Block = tuple[str, Element | None]

__all__ = ['headings', 'page_markdowns', 'to_markdown']

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
#: Unweighted: a numbered line of at most this many words, not ending a sentence, heads its clause.
HEADING_WORDS: Final = 8
_ROMAN: Final = re.compile(r'^(?:X{0,3})(?:IX|IV|V?I{0,3})\.\s+[A-Z][A-Z0-9 ,:;&/()\-]{3,}$')
_STANDALONE: Final = re.compile(r'^(?:ACKNOWLEDG(?:E)?MENTS?|REFERENCES|APPENDIX|CONCLUSIONS?)$')
#: A clause number: one or two digits, then ``.N`` parts -- a year (``2011 Evaluation ...``, a table row) is none.
_NUMBERED: Final = re.compile(r'^(\d{1,2}(?:\.\d+)*)\.?\s+[A-Z]')
#: A table-of-contents entry: words, a leader (dots, ellipses, middle dots, rules), a page number at the end.
_TOC_ENTRY: Final = re.compile(r'^(?=.*[A-Za-z]).*?(?:(?:\.\s?){4,}|\u2026{2,}|\u00b7{3,}|_{4,})\s*[\dixvIXV]{1,5}$')
#: Lines of a page that are contents entries for the page to be a table of contents.
TOC_LINES: Final = 3


def to_markdown(doc: Document, *, include_hidden: bool = False, tagged: Collection[str] = tags.USES) -> str:
    """The document as markdown, page after page: a paragraph a page break cut is joined again.

    ``include_hidden``: the text hidden layers draw is written too (see `document_lab.ir`). ``tagged``: what a tagged
    PDF's tags give (``order``, ``headings``, ``paragraphs``; see the module); none infers it all from layout.
    """
    blocks = _blocks(doc, include_hidden=include_hidden, tagged=tagged)
    return '\n\n'.join(_rejoin([block for page in blocks for block in page])) + '\n'


def page_markdowns(doc: Document, *, include_hidden: bool = False, tagged: Collection[str] = tags.USES) -> list[str]:
    """Each page's own markdown, in order (empty for a page without text).

    Body size and title are judged over the whole document; nothing is joined across a page break: a page stands alone.
    ``include_hidden``, ``tagged``: as for :func:`to_markdown`.
    """
    return [
        '\n\n'.join(_rejoin(page)) + '\n' if page else ''
        for page in _blocks(doc, include_hidden=include_hidden, tagged=tagged)
    ]


def headings(
    doc: Document, *, include_hidden: bool = False, tagged: Collection[str] = tags.USES
) -> list[tuple[int, str]]:
    """Each heading :func:`page_markdowns` writes, as ``(page, heading line)`` in order."""
    return [
        (number, line)
        for number, page in enumerate(page_markdowns(doc, include_hidden=include_hidden, tagged=tagged), start=1)
        for line in page.splitlines()
        if line.startswith('#')
    ]


def _blocks(doc: Document, *, include_hidden: bool, tagged: Collection[str]) -> list[list[_Block]]:
    uses = tags.checked(tagged) & {'order', 'headings', 'paragraphs'}
    reading = tags.reading(doc) if uses else None
    doc = doc.shown(include_hidden=include_hidden)
    weighted = any(span.bold for span in doc.spans())
    body = layout.body_size(doc)
    title_size = max((s.size for s in doc.pages[0].spans), default=0.0) if doc.pages else 0.0
    last = [0]  # the level of the last heading written, page after page
    return [
        _page_blocks(page, body, title_size, weighted=weighted, reading=reading, uses=uses, last=last)
        for page in doc.pages
    ]


def _rejoin(blocks: list[_Block]) -> list[str]:
    """Paragraphs cut by a column or page break, joined again.

    A block set from a tagged element continues the nearest earlier block of that same element, within
    :data:`REJOIN_REACH` blocks (a paragraph the tags state over a page break). Any other block starting lower-case
    continues the nearest earlier paragraph (within :data:`REJOIN_REACH` blocks, skipping page furniture between
    them) that ends unterminated; a heading stops the search.
    """
    out: list[str] = []
    owners: list[Element | None] = []
    for block, owner in blocks:
        target = None
        reach = range(len(out) - 1, max(-1, len(out) - 1 - REJOIN_REACH), -1)
        if owner is not None:
            target = next((i for i in reach if owners[i] is owner), None)
        elif block[:1].islower():
            for i in reach:
                if out[i].startswith('#'):
                    break
                if not out[i].rstrip().endswith(('.', ':', '?', '!')):
                    target = i
                    break
        if target is None:
            out.append(block)
            owners.append(owner)
            continue
        head = out[target].rstrip()
        out[target] = head[:-1] + block if head.endswith('-') else f'{head} {block}'
    return out


def _heading(line: layout.Line, body: float, title_size: float | None, *, weighted: bool) -> str | None:
    text = line.text
    if not any(c.isalpha() for c in text) or _TOC_ENTRY.match(text) or '....' in text:
        return None
    if not weighted and not layout.wordy(text):
        return None  # OCR noise: a figure's labels, a garbled leader
    if title_size is not None and line.size >= TITLE_SCALE * body:
        return f'# {text}'
    if _ROMAN.match(text) or _STANDALONE.match(text):
        return f'## {text}'
    numbered = _NUMBERED.match(text)
    clause = len(text.split()) <= HEADING_WORDS and not text.endswith(('.', ',', ';', ':')) and not text.isupper()
    if numbered and len(text) <= 80 and (line.bold if weighted else clause):
        return f'{"#" * (numbered.group(1).count(".") + 2)} {text}'
    return None


def _joined(text: str, line: str) -> str:
    """``line`` set after ``text`` in one paragraph: a hyphen at the end of ``text`` joins the word."""
    return text[:-1] + line if text.endswith('-') and line[:1].islower() else f'{text} {line}'


def _page_blocks(
    page: Page,
    body: float,
    title_size: float | None,
    *,
    weighted: bool,
    reading: tags.Reading | None,
    uses: frozenset[str],
    last: list[int],
) -> list[_Block]:
    blocks: list[str] = []
    owners: list[Element | None] = []  # the tagged element each block is set from
    paragraph: list[str] = []
    title: list[str] = []
    began: dict[str, Element | None] = {'paragraph': None, 'title': None}  # the element each began in
    previous: layout.Line | None = None
    held = None  # the tagged block the previous line is set in

    def put(text: str, owner: Element | None) -> None:
        blocks.append(text)
        owners.append(owner)
        if text.startswith('#'):
            last[0] = len(text) - len(text.lstrip('#'))

    def end_paragraph() -> None:
        if paragraph:
            put(' '.join(paragraph), began['paragraph'])
            paragraph.clear()

    def end_title() -> None:
        if title:
            end_paragraph()
            put('# ' + ' '.join(title), began['title'])
            title.clear()

    found = layout.lines(page, weighted=weighted)
    lines = reading.ordered(found) if reading is not None and 'order' in uses else layout.ordered(found)
    contents = sum(bool(_TOC_ENTRY.match(line.text)) for line in lines) >= TOC_LINES
    stated_heads = reading is not None and 'headings' in uses and reading.heads
    paragraphs = reading is not None and 'paragraphs' in uses
    for line in lines:
        covered = reading is not None and reading.block_of(line) is not None
        block = reading.block_of(line) if paragraphs else None
        before = reading.block_of(previous) if paragraphs and previous is not None else None
        inferred = None if contents else _heading(line, body, title_size, weighted=weighted)
        # A line goes on its tagged block -- unless the tags state no heading and this line heads by the rules.
        goes_on = stated_heads or inferred is None
        if goes_on and block is not None and block is held and (title or paragraph or blocks):
            if title:
                title.append(line.text)
            elif paragraph:
                paragraph[-1] = _joined(paragraph[-1], line.text)
            else:
                blocks[-1] = _joined(blocks[-1], line.text)  # a tagged heading's further line
            previous = line
            continue
        held = block
        level = reading.level(line) if covered and stated_heads else None
        if level:
            # The tags state the line heads. Where the rules make it a heading too, theirs is its level (a tag's
            # follows a word processor's styles); else the tag's, at most one below the heading before it.
            level = inferred.index(' ') if inferred else min(level, max(2, last[0] + 1))
            heading = f'{"#" * level} {line.text}'
        elif covered and stated_heads:
            titled = reading.titled or inferred is None or not inferred.startswith('# ')
            heading = None if titled else inferred  # the tags state no title: the title is inferred
        else:
            heading = inferred
        if heading and not level:
            held = None  # a heading inferred heads no tagged block: the block's next line starts afresh
        if heading and heading.startswith('# '):
            if not title:
                began['title'] = held
            title.append(line.text)  # a title wraps over several lines
            previous = line
            continue
        end_title()
        if heading:
            end_paragraph()
            put(heading, held)
            previous = line
            continue
        if not (covered and stated_heads) and _continues_heading(blocks, previous, line):
            blocks[-1] = f'{blocks[-1]} {line.text}'
            previous = line
            continue
        joined = (
            block is None
            and before is None
            and previous is not None
            and previous.column == line.column
            and -OVERLAP * line.size <= line.y0 - previous.y1 <= PARAGRAPH_GAP * line.size
        )
        if not joined:
            end_paragraph()
        if joined and paragraph and paragraph[-1].endswith('-') and line.text[:1].islower():
            paragraph[-1] = paragraph[-1][:-1] + line.text
        else:
            if not paragraph:
                began['paragraph'] = block
            paragraph.append(ESCAPE + line.text if line.text.startswith('#') else line.text)
        previous = line
    end_title()
    end_paragraph()
    return list(zip(blocks, owners, strict=True))


def _continues_heading(blocks: list[str], previous: layout.Line | None, line: layout.Line) -> bool:
    """An upper-case line right under a section heading, in its column, is the heading's wrapped second line."""
    if not blocks or not blocks[-1].startswith('## ') or previous is None or previous.column != line.column:
        return False
    text = line.text
    close = -OVERLAP * line.size <= line.y0 - previous.y1 <= PARAGRAPH_GAP * line.size
    letters = sum(c.isalpha() for c in text)
    words = text.isprintable() and letters >= 0.6 * len(text.replace(' ', ''))
    return close and words and text == text.upper()
