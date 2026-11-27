"""One document read twice -- by `document_lab.pdf` and by the previous implementation, PyMuPDF -- and the difference.

Text is compared as each page's characters, whitespace dropped and order ignored (reading order is layout's, not the
reader's): the share of the characters we state that PyMuPDF states too. A glyph no font source names reads as
U+FFFD here (never a guess) while PyMuPDF prints something for it -- its glyph id, a byte -- so those characters are
counted apart (``unknown``), out of the agreement. A symbol font's code PyMuPDF prints as the private-use code point
its ``ToUnicode`` or ``cmap`` states (U+F0xx) reads here as the symbol shown (`pdf::font`), so each private-use
character of PyMuPDF's we do not print pairs with one of ours it does not match, and both are counted apart
(``private``), out of the agreement.

Images are compared by where they are placed. PyMuPDF lists every placement, seen or not; the reader leaves out what
its clip hides, and its clip is a box holding the true one (`pdf::interpret`), so it leaves out only what cannot show.
An image of ours PyMuPDF does not place there is ``invented``; PyMuPDF's placements we do not report are ``clipped``.
Rendered ink cannot judge a clipped one: a Form showing part of an embedded page hides images that another Form,
showing another part of the same page, draws in the very place.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pymupdf

from document_lab.pdf import Locked, read_pdf

if TYPE_CHECKING:
    from pathlib import Path

UNKNOWN = '\N{REPLACEMENT CHARACTER}'
#: Points two placements of an image may differ by and be one.
PLACED = 0.5


@dataclass(frozen=True)
class Page:
    """One page compared: its size off PyMuPDF's (points), its characters, and its images placed."""

    number: int
    size_off: float
    matched: int
    stated: int  # characters we state, U+FFFD not counted
    reference: int  # characters PyMuPDF states
    unknown: int  # U+FFFD we state
    private: int  # PyMuPDF's private-use characters we state as another character
    images: int  # ours
    invented: int  # ours PyMuPDF does not place
    clipped: int  # PyMuPDF's placements we do not report

    @property
    def whole(self) -> int:
        """The larger side's characters, the unknown taken from PyMuPDF's and the private from both."""
        return max(self.stated - self.private, self.reference - self.unknown - self.private)

    @property
    def agreement(self) -> float:
        """Matched over :attr:`whole`: 1 when neither states a character."""
        return 1.0 if self.whole <= 0 else self.matched / self.whole


@dataclass(frozen=True)
class Compared:
    """A document compared: both refused it for a password (``locked``), or its pages side by side."""

    path: Path
    locked: bool
    pages: tuple[Page, ...]
    reference_pages: int

    @property
    def agreement(self) -> float:
        """The document's characters matched over all its pages' (as :attr:`Page.agreement`, summed first)."""
        whole = sum(p.whole for p in self.pages)
        return 1.0 if whole <= 0 else sum(p.matched for p in self.pages) / whole

    @property
    def unknown_share(self) -> float:
        """The share of PyMuPDF's characters we state as U+FFFD."""
        reference = sum(p.reference for p in self.pages)
        return 0.0 if reference == 0 else sum(p.unknown for p in self.pages) / reference


def _characters(text: str) -> Counter[str]:
    return Counter(''.join(text.split()))


def _private(c: str) -> bool:
    """Whether ``c`` is in the Basic Multilingual Plane's private-use area."""
    return unicodedata.category(c) == 'Co'


def _same(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(abs(x - y) <= PLACED for x, y in zip(a, b, strict=True))


def _images(ours: list, theirs: pymupdf.Page) -> dict[str, int]:
    """Our placements against PyMuPDF's: how many, invented and clipped (as the module says)."""
    unmatched = [tuple(i['bbox']) for i in theirs.get_image_info()]
    invented = 0
    for box in (tuple(b) for b in ours):
        match = next((p for p in unmatched if _same(box, p)), None)
        if match is None:
            invented += 1
        else:
            unmatched.remove(match)
    return {'images': len(ours), 'invented': invented, 'clipped': len(unmatched)}


def compare(path: Path, reference_path: Path | None = None) -> Compared:
    """``path`` read by both, page by page; refused by both for a password, ``locked``.

    ``reference_path``: PyMuPDF reads this document instead -- the planted control, a wrong truth the check must report.
    """
    with pymupdf.open(reference_path or path) as reference:
        if reference.needs_pass:
            try:
                read_pdf(path)
            except Locked:
                return Compared(path, locked=True, pages=(), reference_pages=reference.page_count)
            msg = f'{path.name}: PyMuPDF asks a password, the reader did not'
            raise AssertionError(msg)
        doc = read_pdf(path)
        pages = []
        for ours, theirs in zip(doc.pages, reference, strict=False):
            mine = _characters(''.join(s.text for s in ours.spans))
            unknown = mine.pop(UNKNOWN, 0)
            other = _characters(theirs.get_text('text'))
            matched = sum((mine & other).values())
            stated = sum(mine.values())
            private = sum(n for c, n in (other - mine).items() if _private(c))
            pages.append(
                Page(
                    number=ours.number,
                    size_off=max(abs(ours.width - theirs.rect.width), abs(ours.height - theirs.rect.height)),
                    matched=matched,
                    stated=stated,
                    reference=sum(other.values()),
                    unknown=unknown,
                    private=min(private, stated - matched),
                    **_images([f.provenance.bbox for f in ours.figures], theirs),
                )
            )
        return Compared(path, locked=False, pages=tuple(pages), reference_pages=reference.page_count)
