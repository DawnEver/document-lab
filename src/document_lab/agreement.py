"""How a tagged PDF's tags agree with what document-lab infers from where its text stands.

Each use of the tags -- `document_lab.markdown`'s reading order, headings and paragraphs, `document_lab.regions`'
tables and figures -- is an option, on by default, the inferred kept as the alternative. :func:`compare` sets the two
side by side, so a disagreement is reported, never silent:

- headings: the heading lines markdown writes from the tags and from layout alone, matched page by page by their
  text (``#`` marks, case, spacing and closing punctuation aside);
- reading order: of every two lines the tags order on a page, the share layout's order puts the same way round;
- tables and figures: the regions each finds on each page the tags cover, one matching another of its kind when
  they share half the smaller's area (:data:`~document_lab.regions.SAME_REGION`).

:attr:`Comparison.disagreements` names what one finds and the other does not, at most :data:`EXAMPLES` of each.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from document_lab import layout, tags
from document_lab.markdown import headings
from document_lab.regions import SAME_REGION, Region, overlap, regions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.ink import Drawing
    from document_lab.ir import Document

__all__ = ['EXAMPLES', 'Comparison', 'Tally', 'compare']

#: At most this many disagreements of each kind are named.
EXAMPLES: Final = 5


@dataclass(frozen=True)
class Tally:
    """What the tags state, what layout infers, and how many of each the other finds too."""

    tagged: int
    inferred: int
    both: int

    @property
    def agreement(self) -> float:
        """The matched over the larger side: 1 when neither finds any."""
        whole = max(self.tagged, self.inferred)
        return 1.0 if whole == 0 else self.both / whole


@dataclass(frozen=True)
class Comparison:
    """A tagged document's tags against layout's inference."""

    headings: Tally
    #: Of every two lines the tags order on a page, the share layout orders alike; None when the tags order no two.
    order: float | None
    tables: Tally
    figures: Tally
    #: What one finds and the other does not, in words.
    disagreements: tuple[str, ...]


def _heading_text(line: str) -> str:
    return re.sub(r'[\s.:;,]+$', '', ' '.join(line.lstrip('#').split()).casefold())


def _headings(doc: Document, notes: list[str]) -> Tally:
    tagged = {(p, _heading_text(h)): h for p, h in headings(doc, tagged=True)}
    inferred = {(p, _heading_text(h)): h for p, h in headings(doc, tagged=False)}
    for keys, other, what in ((tagged, inferred, 'tagged, not inferred'), (inferred, tagged, 'inferred, not tagged')):
        missing = [k for k in keys if k not in other]
        notes += [f'heading {what}: page {p} {keys[p, t]!r}' for p, t in missing[:EXAMPLES]]
        if len(missing) > EXAMPLES:
            notes.append(f'heading {what}: {len(missing) - EXAMPLES} more')
    return Tally(len(tagged), len(inferred), len(tagged.keys() & inferred.keys()))


def _order(doc: Document, reading: tags.Reading) -> float | None:
    alike = pairs = 0
    weighted = any(span.bold for span in doc.spans())
    for page in doc.shown().pages:
        inferred = layout.ordered(layout.lines(page, weighted=weighted))
        ranked = [(reading.rank_of(ln), i) for i, ln in enumerate(inferred) if reading.rank_of(ln) is not None]
        ranked.sort()
        positions = [i for _, i in ranked]
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                pairs += 1
                alike += positions[a] < positions[b]
    return alike / pairs if pairs else None


def _matched(found: list[Region], others: list[Region]) -> int:
    return sum(any(overlap(r.bbox, o.bbox) >= SAME_REGION for o in others) for r in found)


def compare(doc: Document, drawings: Sequence[Sequence[Drawing]] | None = None) -> Comparison | None:
    """``doc``'s tags against layout's inference; None when it has no tags.

    ``drawings``: each page's, as `document_lab.regions` takes them.
    """
    reading = tags.reading(doc)
    if doc.structure is None:
        return None
    notes: list[str] = []
    heads = _headings(doc, notes)
    body = layout.body_size(doc)
    counts = {'table': [0, 0, 0], 'figure': [0, 0, 0]}
    for i, page in enumerate(doc.pages):
        if not tags.covers(doc.structure, page.number):
            continue
        drawn = drawings[i] if drawings is not None else ()
        inferred = regions(page, body, drawn)
        stated = regions(page, body, drawn, structure=doc.structure)
        for kind, count in counts.items():
            mine = [r for r in stated if r.kind == kind]
            theirs = [r for r in inferred if r.kind == kind]
            count[0] += len(mine)
            count[1] += len(theirs)
            count[2] += _matched(mine, theirs)
            notes += [
                f'{kind} tagged, not inferred: page {page.number} {r.label!r}'
                for r in mine
                if not any(overlap(r.bbox, o.bbox) >= SAME_REGION for o in theirs)
            ][:EXAMPLES]
            notes += [
                f'{kind} inferred, not tagged: page {page.number} {r.label!r}'
                for r in theirs
                if not any(overlap(r.bbox, o.bbox) >= SAME_REGION for o in mine)
            ][:EXAMPLES]
    return Comparison(
        headings=heads,
        order=_order(doc, reading) if reading is not None else None,
        tables=Tally(*counts['table']),
        figures=Tally(*counts['figure']),
        disagreements=tuple(notes),
    )
