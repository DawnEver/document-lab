"""A document's markdown written from document_lab's own IR: headings and paragraphs from its lines.

Lines and their reading order are `document_lab.layout`'s.

- Headings: a line that is a Roman-numbered upper-case heading (``II. METHOD``) or a standalone back-matter word in
  capitals or title case (``REFERENCES``, ``Acknowledgments``, ``Appendix``, ``Conclusion``) is a section (``##``),
  or a chapter (``#``) when set at least :data:`TITLE_SCALE` times the body as a thesis sets its chapters; else a
  line at least :data:`TITLE_SCALE` times the body size is a title (``#``: a paper bound in after the first opens
  with one); else an all-bold line numbered ``2`` / ``2.1`` is a section (``##``, ``###`` per depth), and so is a
  numbered line set wholly in a face other than the body's (`document_lab.layout.body_face`: an italic, another font,
  another size) whose parent number already heads, read in order across pages -- ``2.1`` after a heading ``2``: a
  subsection set in italic heads, a data sheet's row in another face (``09 August 2018``, ``13.56 MHz``) is
  top-level or continues nothing; else page 1's
  largest text is its title when at least :data:`PAGE_TITLE_SCALE` times the body and set in at most
  :data:`PAGE_TITLE_LINES` lines (a repository's cover sheet set at one size throughout has none); else an all-bold
  line lettered as an appendix's ``A.1`` is a section, and one lettered ``A`` alone when set at least
  :data:`LETTER_SCALE` times the body (in the body size a bold letter is a table's label), its title read as words
  (not a letter-spaced ``C O N D I T I O N S``). A web address is never a title. No heading is set smaller than
  :data:`HEADING_FLOOR` times the body (a footnote's number is), but for a capital Roman-numbered or back-matter
  heading, which small capitals set below it (a bold title-case ``References`` as small is a table's label). A
  document with no bold text at all (OCR'd: it carries no weight) cannot be judged by weight: there a short numbered
  line that is no sentence heads its clause, and any heading must read as words (`document_lab.layout.wordy`) -- a
  figure's labels, OCR'd, do not. A line without a letter (a page number), a run of four dots (a contents leader,
  however garbled) or a contents entry (a leader then a page number) never heads, nor does any line of a
  table-of-contents page (:data:`TOC_LINES` such entries): it names the sections, it does not open them.
- Paragraphs: lines of one column closer than :data:`PARAGRAPH_GAP` line heights join with a space (a hyphen at a
  line end joins the word); a larger gap, a heading or a column change starts a new paragraph.
- Placed blocks: a caller that knows a page's regions (`document_lab.per_page`) hands boxes with their own markdown
  (:data:`Placed`: a table read from its cells, a figure's link); the lines inside a box leave the paragraphs, and the
  block stands where the box does in reading order. :func:`table` writes rows as a markdown table.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from document_lab import layout
from document_lab.ir import BBox, Provenance, TextSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.ir import Document, Page


#: A box of a page and the markdown that stands for the lines inside it.
type Placed = tuple[BBox, str]
_HOLDER: Final = '\N{OBJECT REPLACEMENT CHARACTER}'  # the text of the line that holds a placed block's place
__all__ = ['Placed', 'back_matter', 'page_markdowns', 'table', 'to_markdown']

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
#: ... and set in at most this many lines: a title is a short block, a cover sheet set at one size throughout is none.
PAGE_TITLE_LINES: Final = 4
#: A heading is set at least this many times the body size: a footnote, set smaller, heads nothing -- but a capital
#: Roman-numbered or back-matter heading may be set in small capitals below it.
HEADING_FLOOR: Final = 0.9
#: An appendix lettered alone (``A``) is set at least this many times the body: a table's bold label is not.
LETTER_SCALE: Final = 1.1
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
#: A web address: never a title, however large.
_URL: Final = re.compile(r'://|\bwww\.', re.IGNORECASE)
#: A table-of-contents entry: words, a leader (dots, ellipses, middle dots, rules), a page number at the end.
_TOC_ENTRY: Final = re.compile(r'^(?=.*[A-Za-z]).*?(?:(?:\.\s?){4,}|\u2026{2,}|\u00b7{3,}|_{4,})\s*[\dixvIXV]{1,5}$')
#: Lines of a page that are contents entries for the page to be a table of contents.
TOC_LINES: Final = 3


def to_markdown(doc: Document) -> str:
    """The document as markdown, page after page: a paragraph a page break cut is joined again."""
    return '\n\n'.join(_rejoin([block for page in _blocks(doc) for block in page])) + '\n'


def page_markdowns(doc: Document, placed: Sequence[Sequence[Placed]] = ()) -> list[str]:
    """Each page's own markdown, in order (empty for a page without text).

    Body size and title are judged over the whole document; nothing is joined across a page break: a page stands alone.
    ``placed``: each page's placed blocks (none for a page past its end).
    """
    return ['\n\n'.join(_rejoin(page)) + '\n' if page else '' for page in _blocks(doc, placed)]


def table(rows: Sequence[Sequence[str]]) -> str:
    """``rows`` as a markdown table, the first its header; a ``|`` in a cell escaped, a short row padded."""
    width = max(len(row) for row in rows)
    cells = [[c.replace('|', ESCAPE + '|') for c in row] + [''] * (width - len(row)) for row in rows]
    lines = [f'| {" | ".join(row)} |' for row in cells]
    return '\n'.join([lines[0], '|' + ' --- |' * width, *lines[1:]])


def _blocks(doc: Document, placed: Sequence[Sequence[Placed]] = ()) -> list[list[str]]:
    weighted = any(span.bold for span in doc.spans())
    body = layout.body_size(doc)
    face = layout.body_face(doc)
    headed: set[str] = set()  # the clause numbers headed so far, page after page
    first = layout.lines(doc.pages[0], weighted=weighted) if doc.pages else []
    largest = max((ln.size for ln in first), default=0.0)
    at_largest = sum(ln.size >= largest for ln in first)
    title = largest if largest >= PAGE_TITLE_SCALE * body and at_largest <= PAGE_TITLE_LINES else None
    return [
        _page_blocks(
            page,
            body,
            title if i == 0 else None,
            placed[i] if i < len(placed) else (),
            weighted=weighted,
            face=face,
            headed=headed,
        )
        for i, page in enumerate(doc.pages)
    ]


def back_matter(text: str) -> bool:
    """Is ``text`` a back-matter heading in capitals or title case (``REFERENCES``, ``Acknowledgments``, ...)?"""
    return bool(_STANDALONE.match(text.upper())) and text in {text.upper(), text.capitalize()}


def _rejoin(blocks: list[str]) -> list[str]:
    """Paragraphs cut by a column or page break, joined again.

    A block starting lower-case continues the nearest earlier paragraph (within :data:`REJOIN_REACH` blocks,
    skipping page furniture between them) that ends unterminated; a heading or a placed block (a region's link
    leads it) stops the search.
    """
    out: list[str] = []
    for block in blocks:
        target = None
        if block[:1].islower():
            for i in range(len(out) - 1, max(-1, len(out) - 1 - REJOIN_REACH), -1):
                if out[i].startswith(('#', '![')):
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


def _heading(
    line: layout.Line,
    body: float,
    title: float | None,
    *,
    weighted: bool,
    face: tuple[str, float],
    headed: set[str],
) -> str | None:
    """The line as a heading (``#`` a title, ``##`` a section, deeper per clause depth), else None.

    ``title``: the size page 1 sets its title in, on page 1 (None elsewhere, or when page 1 sets none apart).
    ``face``: the body's font and size; ``headed``: the clause numbers headed before this line.
    """
    text = line.text
    if not any(c.isalpha() for c in text) or _TOC_ENTRY.match(text) or '....' in text:
        return None
    capital = bool(_ROMAN.match(text)) or (back_matter(text) and text.isupper())
    if (line.size < HEADING_FLOOR * body and not capital) or (not weighted and not layout.wordy(text)):
        return None  # a footnote; OCR noise: a figure's labels, a garbled leader
    titled = line.size >= TITLE_SCALE * body and not _URL.search(text)
    if _ROMAN.match(text) or back_matter(text):
        return f'{"#" if titled else "##"} {text}'  # the level of the headings of its size: a chapter's, a section's
    if titled:
        return f'# {text}'
    # A clause number states a section more surely than page 1's size states a title; an initial letter less surely.
    unsentenced = len(text.split()) <= HEADING_WORDS and not text.endswith(('.', ',', ';', ':'))
    clause = len(text) <= 80 and (line.bold if weighted else unsentenced)
    numbered = _NUMBERED.match(text)
    if numbered and (clause or (weighted and len(text) <= 80 and _continues(numbered.group(1), line, face, headed))):
        return f'{"#" * (numbered.group(1).count(".") + 2)} {text}'
    if title is not None and line.size >= title and not _URL.search(text):
        return f'# {text}'
    lettered = _LETTERED.match(text) if weighted else None
    # Its title reads as words: a letter-spaced word (``C O N``) is no title.
    letter = lettered.group(1) if lettered and layout.wordy(text[lettered.end(1) :]) else None
    if letter and clause and ('.' in letter or line.size >= LETTER_SCALE * body):
        return f'{"#" * (letter.count(".") + 2)} {text}'
    return None


def _with_placed(page: Page, lines: list[layout.Line], placed: Sequence[Placed]) -> dict[int, str]:
    """``lines`` less those inside a placed box, a line holding each block's place added; the blocks by holder id."""
    held: dict[int, str] = {}

    def inside(line: layout.Line) -> bool:
        cx, cy = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
        return any(box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1 for box, _ in placed)

    lines[:] = [ln for ln in lines if not inside(ln)]
    for box, text in placed:
        here = Provenance('', '', page=page.number, bbox=box)
        holder = layout.Line([TextSpan(_HOLDER, 0.0, '', here)], layout.column(box.x0, box.x1, page.width / 2))
        lines.append(holder)
        held[id(holder)] = text
    return held


def _continues(number: str, line: layout.Line, face: tuple[str, float], headed: set[str]) -> bool:
    """Is the numbered line set wholly in a face other than the body's ``face``, its parent number ``headed``?"""
    parent, dot, _ = number.rpartition('.')
    other = all((s.font, round(s.size, 1)) != face for s in line.spans if s.text.strip())
    return bool(dot) and parent in headed and other


def _page_blocks(
    page: Page,
    body: float,
    title_size: float | None,
    placed: Sequence[Placed] = (),
    *,
    weighted: bool,
    face: tuple[str, float],
    headed: set[str],
) -> list[str]:
    """The page's blocks; ``headed`` gains the clause number of each numbered heading it states."""
    blocks: list[str] = []
    paragraph: list[str] = []
    previous: layout.Line | None = None
    title: list[str] = []
    beside: list[str] = []  # placed blocks met while a title wraps on: they follow it, not cut it

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(' '.join(paragraph))
            paragraph.clear()

    def end_title() -> None:
        end_paragraph()
        blocks.append('# ' + ' '.join(title))
        blocks.extend(beside)
        title.clear()
        beside.clear()

    found = layout.lines(page, weighted=weighted)
    held = _with_placed(page, found, placed)
    lines = layout.ordered(found)
    contents = sum(bool(_TOC_ENTRY.match(line.text)) for line in lines) >= TOC_LINES
    for line in lines:
        heading = None if contents else _heading(line, body, title_size, weighted=weighted, face=face, headed=headed)
        if heading and (number := _NUMBERED.match(line.text)):
            headed.add(number.group(1))
        if heading and heading.startswith('# '):
            title.append(line.text)  # a title wraps over several lines
            previous = line
            continue
        if id(line) in held:
            if title:
                beside.append(held[id(line)])
            else:
                end_paragraph()
                blocks.append(held[id(line)])
                previous = None
            continue
        if title:
            end_title()
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
        end_title()
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
