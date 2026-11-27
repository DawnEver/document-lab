"""A document's markdown written from document_lab's own IR: headings and paragraphs from its lines.

Lines and their reading order are `document_lab.layout`'s.

- Headings (:func:`headings`): a document stating an outline (bookmarks) has its sections named by it -- each item is
  looked for on the page its destination names (or the next, where a section starts at the top of a page), its title
  matched to a line (or two or three wrapped lines) up to case, punctuation and numbering, the nearest to the place
  the destination states; a top-level item heads ``##``, a child ``###`` and so on (a lone top-level item holding
  all others is the title, ``#``; an item without a title names no heading and is reported). The outline is used
  when every titled item is found on its page and it names at least one of the headings the rules below find (when
  they find any: an outline of a cover sheet's title and terms of use is no section structure). Then the outline
  alone heads sections, and the headings the rules would add are reported as ``unstated``; else what does not match
  is reported and the rules decide. The rules -- the inferred structure, the alternative: the largest
  text on page 1 is the title (``#``, kept beside an outline too); a line that is a Roman-numbered upper-case heading
  (``II. METHOD``), a standalone ``REFERENCES``/``ACKNOWLEDGMENT``/``APPENDIX``/``CONCLUSION``, or an all-bold
  line numbered ``2`` / ``2.1`` is a section (``##``, ``###`` per depth). A document with no bold text at all (OCR'd:
  it carries no weight) cannot be judged by weight: there a short numbered line that is no sentence, nor all
  capitals, heads its clause, and any heading must read as words (`document_lab.layout.wordy`) -- a figure's
  labels, OCR'd, do not. A line without a letter (a page number), a run of four dots (a contents leader, however
  garbled) or a contents entry (a leader then a page number) never heads, nor does any line of a table-of-contents
  page (:data:`TOC_LINES` such entries): it names the sections, it does not open them.
- Paragraphs: lines of one column closer than :data:`PARAGRAPH_GAP` line heights join with a space (a hyphen at a
  line end joins the word); a larger gap, a heading or a column change starts a new paragraph.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from document_lab import layout

if TYPE_CHECKING:
    from collections.abc import Iterator

    from document_lab.ir import Document, OutlineItem

__all__ = ['Headings', 'headings', 'page_markdowns', 'to_markdown']

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
#: Lines an outline item's title may wrap over.
TITLE_LINES: Final = 3
#: A line opening with an outline item's title, then more (a run-in heading), is its heading only this many of its
#: font sizes from the place the item's destination states.
RUN_IN_REACH: Final = 3.0
#: A section number before a title, ignored when titles are matched: ``2``, ``2.1.``, ``IV.``, ``III-A``, ``(a)``.
_NUMBERING: Final = re.compile(r'^\(?(?:\d+|(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})|[a-z])(?:[.-](?:\d+|[a-z]))*[.):]?\s+')


@dataclass(frozen=True, slots=True)
class Headings:
    """Which lines head a document's sections, and where that comes from: what :func:`headings` judged."""

    #: ``'outline'``: the document's stated outline, found on its pages; ``'inferred'``: the rules.
    source: Literal['outline', 'inferred']
    #: Each page's heading lines, by page: (index of the line in reading order, its markdown heading, the lines its
    #: text wraps over).
    marks: tuple[tuple[tuple[int, str, int], ...], ...]
    #: How many items the document's outline states (0: it states none, or the outline was not asked for).
    stated: int = 0
    #: Outline items not found on their page, each as ``'title' (page N)``: why a stated outline was not used.
    unlocated: tuple[str, ...] = ()
    #: Headings the rules find that the outline does not name (beside an outline found on its pages).
    unstated: tuple[str, ...] = ()
    #: The headings the outline put in the markdown, as written there (without their ``#``).
    named: tuple[str, ...] = ()
    #: Outline items stating no title, each as ``page N``: they name no heading.
    untitled: tuple[str, ...] = ()
    #: How many of the rules' headings the outline names (beside an outline found on its pages).
    shared: int = 0

    @property
    def disagreements(self) -> tuple[str, ...]:
        """Where the stated outline and the inferred structure differ, in words; empty when they agree or no outline."""
        return (
            *(f'outline item {u} is not found on its page' for u in self.unlocated),
            *(f'an outline item ({u}) states no title: it names no heading' for u in self.untitled),
            *(f'heading {u!r} is found by the rules but named by no outline item' for u in self.unstated),
        )


def headings(doc: Document, *, outline: bool = True, include_hidden: bool = False) -> Headings:
    """Which lines of ``doc`` head its sections: its stated outline, else the rules (the module says how).

    ``outline``: the outline is used when the document states one that matches its pages (the module says when);
    False keeps to the rules. ``include_hidden``: as for :func:`to_markdown`.
    """
    shown = doc.shown(include_hidden=include_hidden)
    weighted = any(span.bold for span in shown.spans())
    body = layout.body_size(shown)
    title_size = max((s.size for s in shown.pages[0].spans), default=0.0) if shown.pages else 0.0
    pages = [layout.ordered(layout.lines(page, weighted=weighted)) for page in shown.pages]
    inferred = [_inferred(lines, body, title_size, weighted=weighted) for lines in pages]
    items = list(_flatten(doc.outline)) if outline else []
    if not items:
        return Headings('inferred', inferred)
    located, unlocated, untitled = _locate(doc.outline, items, pages)
    if unlocated:
        return Headings('inferred', inferred, len(items), tuple(unlocated), untitled=tuple(untitled))
    covered = {(n, i + k) for (n, i), (_, span) in located.items() for k in range(span)}
    rules = {(n, i): h for n, page in enumerate(inferred, start=1) for i, h, _ in page if not h.startswith('# ')}
    shared = len(rules.keys() & covered)
    unstated = tuple(h.lstrip('#').strip() for key, h in sorted(rules.items()) if key not in covered)
    if rules and not shared:
        return Headings('inferred', inferred, len(items), (), unstated, untitled=tuple(untitled))
    marks = []
    named = []
    for number, (lines, page_rules) in enumerate(zip(pages, inferred, strict=True), start=1):
        page = [(i, h, span) for i, h, span in page_rules if h.startswith('# ') and (number, i) not in covered]
        for (n, i), (level, span) in located.items():
            if n == number:
                text = ' '.join(line.text for line in lines[i : i + span])
                page.append((i, f'{"#" * level} {text}', span))
                named.append(text)
        marks.append(tuple(sorted(page)))  # the rules' title kept: an outline rarely names it
    return Headings('outline', tuple(marks), len(items), (), unstated, tuple(named), tuple(untitled), shared)


def to_markdown(doc: Document, *, include_hidden: bool = False, sections: Headings | None = None) -> str:
    """The document as markdown, page after page: a paragraph a page break cut is joined again.

    ``include_hidden``: the text hidden layers draw is written too (see `document_lab.ir`). ``sections``: which lines
    head sections, as :func:`headings` judged them (None: judged here, a stated outline first).
    """
    return '\n\n'.join(_rejoin([block for page in _blocks(doc, include_hidden, sections) for block in page])) + '\n'


def page_markdowns(doc: Document, *, include_hidden: bool = False, sections: Headings | None = None) -> list[str]:
    """Each page's own markdown, in order (empty for a page without text).

    Body size and title are judged over the whole document; nothing is joined across a page break: a page stands alone.
    ``include_hidden`` and ``sections``: as for :func:`to_markdown`.
    """
    return ['\n\n'.join(_rejoin(page)) + '\n' if page else '' for page in _blocks(doc, include_hidden, sections)]


def _blocks(doc: Document, include_hidden: bool, sections: Headings | None) -> list[list[str]]:
    shown = doc.shown(include_hidden=include_hidden)
    sections = sections or headings(doc, include_hidden=include_hidden)
    weighted = any(span.bold for span in shown.spans())
    inferred = sections.source == 'inferred'
    return [
        _page_blocks(layout.ordered(layout.lines(page, weighted=weighted)), marks, inferred=inferred)
        for page, marks in zip(shown.pages, sections.marks, strict=True)
    ]


def _inferred(lines: list[layout.Line], body: float, title_size: float, *, weighted: bool) -> tuple:
    """The page's heading lines by the rules: (index, markdown heading, 1) each; none on a contents page."""
    if sum(bool(_TOC_ENTRY.match(line.text)) for line in lines) >= TOC_LINES:
        return ()
    marks = ((i, _heading(line, body, title_size, weighted=weighted)) for i, line in enumerate(lines))
    return tuple((i, heading, 1) for i, heading in marks if heading)


def _flatten(items: tuple[OutlineItem, ...], depth: int = 0) -> Iterator[tuple[int, OutlineItem]]:
    for item in items:
        yield depth, item
        yield from _flatten(item.children, depth + 1)


def _normal(text: str) -> str:
    """``text`` as titles are matched: folded (compatibility, case), its section numbers dropped, letters and digits.

    Numbers are dropped one after another (``4.3. A Comparison`` reads as ``A Comparison`` reads), never to nothing.
    """
    folded = unicodedata.normalize('NFKC', text).casefold().strip()
    while (number := _NUMBERING.match(folded)) and folded[number.end() :].strip():
        folded = folded[number.end() :]
    return ''.join(c for c in folded if c.isalnum())


def _locate(
    outline: tuple[OutlineItem, ...], items: list[tuple[int, OutlineItem]], pages: list[list[layout.Line]]
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[str], list[str]]:
    """Where each outline item heads, the items not found on their page, and the items without a title.

    Where: ``{(page, line index): (heading level, lines its title wraps over)}``. Unfound: each as
    ``'title' (page N)``; untitled: each as ``page N``.
    """
    shift = 1 if len(outline) == 1 and outline[0].children else 2  # a lone top-level item is the title
    located: dict[tuple[int, int], tuple[int, int]] = {}
    unlocated = []
    untitled = []
    for depth, item in items:
        dest = item.destination
        where = f'page {dest.page}' if dest is not None else 'no destination in this document'
        title = _normal(item.title or '')
        if not title:
            untitled.append(where)
            continue
        found = _find(title, dest.page, dest.y, pages, located) if dest is not None else None
        if found is None:
            unlocated.append(f'{item.title!r} ({where})')
            continue
        number, i, span = found
        located[number, i] = (depth + shift, span)
    return located, unlocated, untitled


def _find(
    title: str, page: int, y: float | None, pages: list[list[layout.Line]], taken: dict
) -> tuple[int, int, int] | None:
    """The page, first line and line count of the lines reading ``title`` (normalized), not yet ``taken``.

    Looked for on ``page``, else the next; the nearest to ``y`` (on ``page``) when several read it. A line opening
    with the title (a run-in heading) counts only within :data:`RUN_IN_REACH` font sizes of ``y``.
    """
    for number in (page, page + 1):
        if not 1 <= number <= len(pages):
            continue
        lines = pages[number - 1]
        near = y if number == page else None
        exact, run_in = [], []
        for i in range(len(lines)):
            if (number, i) in taken:
                continue
            for span in range(1, TITLE_LINES + 1):
                text = _normal(' '.join(line.text for line in lines[i : i + span]))
                if text == title:
                    exact.append((i, span))
                    break
                close = near is not None and abs(lines[i].y0 - near) <= RUN_IN_REACH * lines[i].size
                if span == 1 and close and text.startswith(title):
                    run_in.append((i, 1))
                if not title.startswith(text) or i + span >= len(lines):
                    break
        for found in (exact, run_in):
            if found:
                i, span = min(found, key=lambda f: 0.0 if near is None else abs(lines[f[0]].y0 - near))
                return number, i, span
    return None


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


def _page_blocks(lines: list[layout.Line], marks: tuple, *, inferred: bool) -> list[str]:
    """The page's blocks: its ``marks`` (index, heading, lines it wraps over) heading, its other lines in paragraphs.

    An upper-case line under an ``inferred`` heading continues it (the rules judge a heading line by line).
    """
    blocks: list[str] = []
    paragraph: list[str] = []
    previous: layout.Line | None = None
    title: list[str] = []
    heads = {i: (heading, span) for i, heading, span in marks}
    skip = 0

    def end_paragraph() -> None:
        if paragraph:
            blocks.append(' '.join(paragraph))
            paragraph.clear()

    for i, line in enumerate(lines):
        if skip:
            skip -= 1  # a wrapped line of the heading before it
            previous = line
            continue
        heading, span = heads.get(i, (None, 1))
        skip = span - 1
        if heading and heading.startswith('# '):
            title.append(heading[2:])  # a title wraps over several lines
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
        if inferred and _continues_heading(blocks, previous, line):
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
