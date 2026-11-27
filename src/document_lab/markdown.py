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
part -- is read as an untagged one is.

- Headings (:func:`headings`): a document stating an outline (bookmarks) has its sections named by it -- each item is
  looked for on the page its destination names (or the next, where a section starts at the top of a page), its title
  matched to a line (or two or three wrapped lines) up to case, punctuation and numbering, the nearest to the place
  the destination states; a top-level item heads ``##``, a child ``###`` and so on (a lone top-level item holding
  all others is the title, ``#``; an item without a title names no heading and is reported). The outline is used
  when every titled item is found on its page and it names at least one of the headings the tags and rules below
  find (when they find any: an outline of a cover sheet's title and terms of use is no section structure). Then the
  outline heads sections, beside their title and back matter (an outline often leaves an unnumbered ``References``
  or ``Acknowledgments`` out), and the headings they would add are reported as ``unstated``; else what does not
  match is reported and the tags and rules decide. The rules -- the inferred structure, the alternative: a line that
  is a Roman-numbered upper-case heading (``II. METHOD``) or a standalone back-matter word in
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
  table-of-contents page (:data:`TOC_LINES` such entries): it names the sections, it does not open them. An
  upper-case line right under a section heading the rules find is its wrapped line.
- Paragraphs: lines of one column and one way closer than :data:`PARAGRAPH_GAP` line heights -- measured down their
  own frame (`document_lab.layout.Line.frame`), so a caption turned sideways joins its next line beside it -- join
  with a space (a hyphen at a line end joins the word); a larger gap, a heading or a column change starts a new
  paragraph.
- Placed blocks: a caller that knows a page's regions (`document_lab.per_page`) hands boxes with their own markdown
  (:data:`Placed`: a table read from its cells, a figure's link); the lines inside a box leave the paragraphs, and the
  block stands where the box does in reading order -- read in the page's frame (`document_lab.layout.holder`), so a
  table printed sideways follows the caption beside it. :func:`table` writes rows as a markdown table.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from document_lab import layout, tags
from document_lab.ir import BBox

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

    from document_lab.ir import Document, Element, OutlineItem, Page


#: A box of a page and the markdown that stands for the lines inside it.
type Placed = tuple[BBox, str]
_HOLDER: Final = '\N{OBJECT REPLACEMENT CHARACTER}'  # the text of the line that holds a placed block's place
#: A block of markdown, and the tagged element it is set from (None: inferred).
type _Block = tuple[str, Element | None]
#: A page's heading lines: (index of the line in reading order, its markdown heading, the lines its text spans).
type _Marks = tuple[tuple[int, str, int], ...]
__all__ = ['MARKDOWN_USES', 'Headings', 'Placed', 'back_matter', 'headings', 'page_markdowns', 'table', 'to_markdown']

#: The uses of a tagged PDF's tags (`document_lab.tags.USES`) markdown reads; the others are `document_lab.regions`'.
MARKDOWN_USES: Final = frozenset({'order', 'headings', 'paragraphs'})
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
_TOC_ENTRY: Final = re.compile(r'^(?=.*[A-Za-z]).*?(?:(?:\.\s?){4,}|…{2,}|·{3,}|_{4,})\s*[\dixvIXV]{1,5}$')
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

    #: ``'outline'``: the document's stated outline, found on its pages; ``'tags'``: a tagged PDF's tags, the rules
    #: beside them; ``'inferred'``: the rules.
    source: Literal['outline', 'tags', 'inferred']
    #: Each page's heading lines, by page: (index of the line in reading order, its markdown heading, the lines its
    #: text spans).
    marks: tuple[_Marks, ...]
    #: How many items the document's outline states (0: it states none, or the outline was not asked for).
    stated: int = 0
    #: Outline items not found on their page, each as ``'title' (page N)``: why a stated outline was not used.
    unlocated: tuple[str, ...] = ()
    #: Headings the tags and rules find that the outline does not name (beside an outline found on its pages).
    unstated: tuple[str, ...] = ()
    #: The headings the outline put in the markdown, as written there (without their ``#``).
    named: tuple[str, ...] = ()
    #: Outline items stating no title, each as ``page N``: they name no heading.
    untitled: tuple[str, ...] = ()
    #: How many of the tags' and rules' headings the outline names (beside an outline found on its pages).
    shared: int = 0
    #: The uses of the tags they were judged with (of :data:`MARKDOWN_USES`): with ``order`` the lines' reading order
    #: is the tags'.
    uses: frozenset[str] = frozenset()
    #: Whether the text hidden layers draw was judged too.
    include_hidden: bool = False
    #: Whether the text filled in the color under it was judged too.
    include_in_background_color: bool = False

    @property
    def written(self) -> tuple[tuple[int, str], ...]:
        """Each heading as the markdown writes it, ``(page, heading line)`` in order.

        A title (``#``) set over consecutive lines is one heading, its lines' text joined, as :func:`page_markdowns`
        writes it.
        """
        out: list[tuple[int, str]] = []
        for number, marks in enumerate(self.marks, start=1):
            end = -1  # where the title being written ends, in line indexes
            for i, heading, span in marks:
                if heading.startswith('# ') and i == end and out[-1][1].startswith('# '):
                    out[-1] = (number, f'{out[-1][1]} {heading[2:]}')
                else:
                    out.append((number, heading))
                end = i + span
        return tuple(out)

    @property
    def disagreements(self) -> tuple[str, ...]:
        """Where the stated outline and the inferred structure differ, in words; empty when they agree or no outline."""
        return (
            *(f'outline item {u} is not found on its page' for u in self.unlocated),
            *(f'an outline item ({u}) states no title: it names no heading' for u in self.untitled),
            *(f'heading {u!r} is found by the rules but named by no outline item' for u in self.unstated),
        )


def headings(
    doc: Document,
    *,
    outline: bool = True,
    include_hidden: bool = False,
    include_in_background_color: bool = False,
    tagged: Collection[str] = tags.DEFAULT_USES,
) -> Headings:
    """Which lines of ``doc`` head its sections: its stated outline, else its tags and the rules (the module says how).

    ``outline``: the outline is used when the document states one that matches its pages (the module says when);
    False keeps to the tags and rules. ``include_hidden``, ``include_in_background_color``, ``tagged``: as for
    :func:`to_markdown`.
    """
    uses = tags.checked(tagged) & MARKDOWN_USES
    reading = tags.reading(doc) if uses else None
    shown = doc.shown(include_hidden=include_hidden, include_in_background_color=include_in_background_color)
    weighted = any(span.bold for span in shown.spans())
    pages = [_ordered(layout.lines(page, weighted=weighted), reading, uses) for page in shown.pages]
    judged = _judged(shown, pages, weighted=weighted, reading=reading, uses=uses)
    base: Literal['tags', 'inferred'] = 'tags' if _tag_heads(reading, uses) else 'inferred'
    options = {
        'uses': uses,
        'include_hidden': include_hidden,
        'include_in_background_color': include_in_background_color,
    }
    items = list(_flatten(doc.outline)) if outline else []
    if not items:
        return Headings(base, judged, **options)
    located, unlocated, untitled = _locate(doc.outline, items, pages)
    if unlocated:
        return Headings(base, judged, len(items), tuple(unlocated), untitled=tuple(untitled), **options)
    covered = {(n, i + k) for (n, i), (_, span) in located.items() for k in range(span)}
    found = {(n, i): h for n, page in enumerate(judged, start=1) for i, h, _ in page if not h.startswith('# ')}
    shared = len(found.keys() & covered)
    unstated = tuple(h.lstrip('#').strip() for key, h in sorted(found.items()) if key not in covered)
    if found and not shared:
        return Headings(base, judged, len(items), (), unstated, untitled=tuple(untitled), **options)
    marks = []
    named = []
    for number, (lines, page_judged) in enumerate(zip(pages, judged, strict=True), start=1):
        page = [
            (i, h, span)
            for i, h, span in page_judged
            if (h.startswith('# ') or back_matter(h.lstrip('#').strip()))
            and all((number, i + k) not in covered for k in range(span))
        ]
        for (n, i), (level, span) in located.items():
            if n == number:
                text = ' '.join(line.text for line in lines[i : i + span])
                page.append((i, f'{"#" * level} {text}', span))
                named.append(text)
        marks.append(tuple(sorted(page)))  # the title and back matter kept: an outline rarely names them
    return Headings('outline', tuple(marks), len(items), (), unstated, tuple(named), tuple(untitled), shared, **options)


def to_markdown(
    doc: Document,
    *,
    include_hidden: bool = False,
    include_in_background_color: bool = False,
    tagged: Collection[str] = tags.DEFAULT_USES,
    sections: Headings | None = None,
) -> str:
    """The document as markdown, page after page: a paragraph a page break cut is joined again.

    ``include_hidden``: the text hidden layers draw is written too; ``include_in_background_color``: the text filled
    in the color under it too (see `document_lab.ir`). ``tagged``: what a tagged PDF's tags give (``order``,
    ``headings``, ``paragraphs``; see the module); none infers it all from layout. ``sections``: which lines head
    sections, as :func:`headings` judged them with the same ``include_hidden``, ``include_in_background_color`` and
    ``tagged`` (None: judged here, a stated outline first).
    """
    shown = _Shown(include_hidden, include_in_background_color)
    blocks = _blocks(doc, _sections(doc, sections, shown, tagged=tagged))
    return '\n\n'.join(_rejoin([block for page in blocks for block in page])) + '\n'


def page_markdowns(
    doc: Document,
    placed: Sequence[Sequence[Placed]] = (),
    *,
    include_hidden: bool = False,
    include_in_background_color: bool = False,
    tagged: Collection[str] = tags.DEFAULT_USES,
    sections: Headings | None = None,
) -> list[str]:
    """Each page's own markdown, in order (empty for a page without text).

    Body size and title are judged over the whole document; nothing is joined across a page break: a page stands alone.
    ``placed``: each page's placed blocks (none for a page past its end). ``include_hidden``,
    ``include_in_background_color``, ``tagged`` and ``sections``: as for :func:`to_markdown`.
    """
    sections = _sections(doc, sections, _Shown(include_hidden, include_in_background_color), tagged=tagged)
    return ['\n\n'.join(_rejoin(page)) + '\n' if page else '' for page in _blocks(doc, sections, placed)]


def table(rows: Sequence[Sequence[str]], boxes: Sequence[Sequence[BBox | None]] | None = None) -> str:
    """``rows`` as a markdown table, the first its header; a ``|`` in a cell escaped, a short row padded.

    ``boxes``: each slot's cell box (`document_lab.ir.Table.cell_bboxes`), when known: a cell merged over several
    slots repeats its box, and is written once, in its first slot -- the leftmost of its top row -- its other slots
    left empty, as a reader sees a merged cell.
    """
    width = max(len(row) for row in rows)

    def merged(r: int, c: int) -> bool:
        box = None if boxes is None else boxes[r][c]
        return box is not None and ((c > 0 and boxes[r][c - 1] == box) or (r > 0 and boxes[r - 1][c] == box))  # type: ignore[index]

    cells = [
        ['' if merged(r, c) else ' '.join(text.split()).replace('|', ESCAPE + '|') for c, text in enumerate(row)]
        + [''] * (width - len(row))
        for r, row in enumerate(rows)
    ]
    lines = [f'| {" | ".join(row)} |' for row in cells]
    return '\n'.join([lines[0], '|' + ' --- |' * width, *lines[1:]])


@dataclass(frozen=True, slots=True)
class _Shown:
    """What :meth:`~document_lab.ir.Document.shown` keeps beside what a document shows: the options it is asked with."""

    include_hidden: bool
    include_in_background_color: bool


def _sections(doc: Document, sections: Headings | None, shown: _Shown, *, tagged: Collection[str]) -> Headings:
    """``sections``, else judged here; refused when judged with other options: its marks index other lines."""
    if sections is None:
        return headings(
            doc,
            include_hidden=shown.include_hidden,
            include_in_background_color=shown.include_in_background_color,
            tagged=tagged,
        )
    uses = tags.checked(tagged) & MARKDOWN_USES
    judged = (sections.uses, sections.include_hidden, sections.include_in_background_color)
    if judged != (uses, shown.include_hidden, shown.include_in_background_color):
        msg = (
            f'the sections were judged with tagged={sorted(sections.uses)}, include_hidden={sections.include_hidden}, '
            f'include_in_background_color={sections.include_in_background_color}; the markdown is asked with '
            f'tagged={sorted(uses)}, include_hidden={shown.include_hidden}, '
            f"include_in_background_color={shown.include_in_background_color}: judge them with the markdown's own "
            '(headings(doc, include_hidden=..., include_in_background_color=..., tagged=...))'
        )
        raise ValueError(msg)
    return sections


def _blocks(doc: Document, sections: Headings, placed: Sequence[Sequence[Placed]] = ()) -> list[list[_Block]]:
    reading = tags.reading(doc) if sections.uses else None
    shown = doc.shown(
        include_hidden=sections.include_hidden, include_in_background_color=sections.include_in_background_color
    )
    weighted = any(span.bold for span in shown.spans())
    return [
        _page_blocks(
            page, marks, placed[i] if i < len(placed) else (), weighted=weighted, reading=reading, uses=sections.uses
        )
        for i, (page, marks) in enumerate(zip(shown.pages, sections.marks, strict=True))
    ]


def _ordered(lines: list[layout.Line], reading: tags.Reading | None, uses: frozenset[str]) -> list[layout.Line]:
    """``lines`` in reading order: the order the tags state, where ``order`` is used, else layout's."""
    return reading.ordered(lines) if reading is not None and 'order' in uses else layout.ordered(lines)


def _tag_heads(reading: tags.Reading | None, uses: frozenset[str]) -> bool:
    """Do the tags, used for ``headings``, state any heading?"""
    return reading is not None and 'headings' in uses and reading.heads


def _judged(
    doc: Document,
    pages: list[list[layout.Line]],
    *,
    weighted: bool,
    reading: tags.Reading | None,
    uses: frozenset[str],
) -> tuple[_Marks, ...]:
    """Each page's heading lines by the tags and the rules, page after page (the module says how).

    ``pages``: each page's lines in reading order. A contents page has none. A numbered line heads by its face only
    where its parent number heads before it (`_continues`), so the pages are judged in order. A heading spans its
    tagged block's further lines, and an upper-case line wrapping a section heading the rules find
    (`_continues_heading`).
    """
    body = layout.body_size(doc)
    face = layout.body_face(doc)
    headed: set[str] = set()  # the clause numbers headed so far, page after page
    first = pages[0] if pages else []
    largest = max((ln.size for ln in first), default=0.0)
    at_largest = sum(ln.size >= largest for ln in first)
    title = largest if largest >= PAGE_TITLE_SCALE * body and at_largest <= PAGE_TITLE_LINES else None
    stated = _tag_heads(reading, uses)
    paragraphs = reading is not None and 'paragraphs' in uses
    last = 0  # the level of the last heading, page after page
    out = []
    for n, lines in enumerate(pages):
        contents = sum(bool(_TOC_ENTRY.match(line.text)) for line in lines) >= TOC_LINES
        marks: list[tuple[int, str, int]] = []
        current = None  # the tagged block the previous line is set in
        for i, line in enumerate(lines):
            covered = reading is not None and reading.block_of(line) is not None
            block = reading.block_of(line) if paragraphs else None
            ends = bool(marks) and marks[-1][0] + marks[-1][2] == i  # the previous line ends a heading
            inferred = (
                None
                if contents
                else _heading(line, body, title if n == 0 else None, weighted=weighted, face=face, headed=headed)
            )
            # A line goes on its tagged block -- unless the tags state no heading and this line heads by the rules.
            if (stated or inferred is None) and block is not None and block is current:
                if ends:
                    at, heading, span = marks[-1]
                    text = f'{heading} {line.text}' if heading.startswith('# ') else _joined(heading, line.text)
                    marks[-1] = (at, text, span + 1)
                continue
            current = block
            level = reading.level(line) if covered and stated else None
            if level:
                # The tags state the line heads. Where the rules make it a heading too, theirs is its level (a tag's
                # follows a word processor's styles); else the tag's, at most one below the heading before it.
                level = inferred.index(' ') if inferred else min(level, max(2, last + 1))
                heading = f'{"#" * level} {line.text}'
            elif covered and stated:
                titled = reading.titled or inferred is None or not inferred.startswith('# ')
                heading = None if titled else inferred  # the tags state no title: the title is inferred
            else:
                heading = inferred
            if heading and (number := _NUMBERED.match(line.text)):
                headed.add(number.group(1))
            if heading:
                if not level:
                    current = None  # a heading inferred heads no tagged block: the block's next line starts afresh
                marks.append((i, heading, 1))
                last = heading.index(' ')
            elif (
                not (covered and stated)
                and ends
                and marks[-1][1].startswith('## ')
                and _continues_heading(lines[i - 1], line)
            ):
                at, heading, span = marks[-1]
                marks[-1] = (at, f'{heading} {line.text}', span + 1)
        out.append(tuple(marks))
    return tuple(out)


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


def back_matter(text: str) -> bool:
    """Is ``text`` a back-matter heading in capitals or title case (``REFERENCES``, ``Acknowledgments``, ...)?"""
    return bool(_STANDALONE.match(text.upper())) and text in {text.upper(), text.capitalize()}


def _rejoin(blocks: list[_Block]) -> list[str]:
    """Paragraphs cut by a column or page break, joined again.

    A block set from a tagged element continues the nearest earlier block of that same element, within
    :data:`REJOIN_REACH` blocks (a paragraph the tags state over a page break). Any other block starting lower-case
    continues the nearest earlier paragraph (within :data:`REJOIN_REACH` blocks, skipping page furniture between
    them) that ends unterminated; a heading or a placed block (a region's link leads it) stops the search.
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
                if out[i].startswith(('#', '![')):
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
        holder = layout.holder(page, box, _HOLDER)  # read in the page's frame: a sideways table after its caption
        lines.append(holder)
        held[id(holder)] = text
    return held


def _continues(number: str, line: layout.Line, face: tuple[str, float], headed: set[str]) -> bool:
    """Is the numbered line set wholly in a face other than the body's ``face``, its parent number ``headed``?"""
    parent, dot, _ = number.rpartition('.')
    other = all((s.font, round(s.size, 1)) != face for s in line.spans if s.text.strip())
    return bool(dot) and parent in headed and other


def _joined(text: str, line: str) -> str:
    """``line`` set after ``text`` in one paragraph: a hyphen at the end of ``text`` joins the word."""
    return text[:-1] + line if text.endswith('-') and line[:1].islower() else f'{text} {line}'


def _page_blocks(
    page: Page,
    marks: _Marks,
    placed: Sequence[Placed] = (),
    *,
    weighted: bool,
    reading: tags.Reading | None,
    uses: frozenset[str],
) -> list[_Block]:
    """The page's blocks: its ``marks`` (index, heading, lines it spans) heading, its other lines in paragraphs.

    Each block comes with the tagged element it is set from (None: inferred, or a placed block). The marks index the
    page's lines in reading order; a placed block takes the lines inside its box out of them.
    """
    blocks: list[str] = []
    owners: list[Element | None] = []  # the tagged element each block is set from
    paragraph: list[str] = []
    title: list[str] = []
    beside: list[str] = []  # placed blocks met while a title wraps on: they follow it, not cut it
    began: dict[str, Element | None] = {'paragraph': None, 'title': None}  # the element each began in
    previous: layout.Line | None = None
    current = None  # the tagged block the previous line is set in

    def put(text: str, owner: Element | None) -> None:
        blocks.append(text)
        owners.append(owner)

    def end_paragraph() -> None:
        if paragraph:
            put(' '.join(paragraph), began['paragraph'])
            paragraph.clear()

    def end_title() -> None:
        if title:
            end_paragraph()
            put('# ' + ' '.join(title), began['title'])
            for text in beside:
                put(text, None)
            title.clear()
            beside.clear()

    found = _ordered(layout.lines(page, weighted=weighted), reading, uses)
    heads = {id(found[i]): heading for i, heading, _ in marks}
    wrapped = {id(found[i + k]) for i, _, span in marks for k in range(1, span)}  # a heading's further lines
    held = _with_placed(page, found, placed)
    lines = _ordered(found, reading, uses)
    stated = _tag_heads(reading, uses)
    paragraphs = reading is not None and 'paragraphs' in uses
    for line in lines:
        if id(line) in wrapped:
            previous = line
            continue
        if id(line) in held:
            if title:
                beside.append(held[id(line)])
            else:
                end_paragraph()
                put(held[id(line)], None)
                previous = None
            current = None
            continue
        heading = heads.get(id(line))
        block = reading.block_of(line) if paragraphs else None
        if heading is None and block is not None and block is current and paragraph:
            paragraph[-1] = _joined(paragraph[-1], line.text)  # the tagged block's further line
            previous = line
            continue
        if heading:
            # A heading the tags state is set from its block: the block's part past a break rejoins it.
            owner = block if stated and reading.level(line) else None
            current = None
            if heading.startswith('# '):
                if not title:
                    began['title'] = owner
                title.append(heading[2:])  # a title wraps over several lines
                previous = line
                continue
            end_title()
            end_paragraph()
            put(heading, owner)
            previous = line
            continue
        end_title()
        before = reading.block_of(previous) if paragraphs and previous is not None else None
        current = block
        joined = (
            block is None
            and before is None
            and previous is not None
            and previous.column == line.column
            and previous.way == line.way  # measured down their own frame: a turned line's next line stands beside it
            and -OVERLAP * line.size <= line.frame.y0 - previous.frame.y1 <= PARAGRAPH_GAP * line.size
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


def _continues_heading(previous: layout.Line, line: layout.Line) -> bool:
    """Is ``line``, upper-case right under the section heading line ``previous`` in its column, its wrapped line?"""
    if previous.column != line.column:
        return False
    text = line.text
    close = -OVERLAP * line.size <= line.y0 - previous.y1 <= PARAGRAPH_GAP * line.size
    letters = sum(c.isalpha() for c in text)
    words = text.isprintable() and letters >= 0.6 * len(text.replace(' ', ''))
    return close and words and text == text.upper()
