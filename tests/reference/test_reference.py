"""The ``reference`` part: the reader accepted against the previous implementation, PyMuPDF, on real PDFs.

``DOCUMENT_LAB_REFERENCE_PDFS`` names PDFs, or directories searched for them, ``os.pathsep``-separated (real
documents are never committed); the part FAILS without one rather than skip. Every document must read whole, refuse
a password exactly when PyMuPDF does, match PyMuPDF's page count and page sizes, and place no image PyMuPDF does not;
its text must agree with PyMuPDF's within the tolerances below (`tests.reference.compare` says how text and images are
compared). The planted control: each document's text checked against another document's must be reported.

The tolerances were measured on 283 local PDFs (6499 pages: papers, standards, theses) on 2026-11-26: page sizes
within 3e-5 pt, no image invented, text agreeing 0.99994 on average and 0.9939 at worst -- that document's own
hyphens, which the reader then read as soft hyphens (fixed since).
"""

import hashlib
import os
from functools import cache
from pathlib import Path

import pytest

from tests.reference.compare import Compared, compare

pytestmark = pytest.mark.reference

_ROOTS = [Path(p) for p in os.environ.get('DOCUMENT_LAB_REFERENCE_PDFS', '').split(os.pathsep) if p]
PDFS = sorted({pdf for root in _ROOTS for pdf in ([root] if root.is_file() else root.rglob('*.pdf'))})

#: Points a page's size may differ from PyMuPDF's (both read the same boxes: measured 3e-5).
SIZE_TOLERANCE = 0.01
#: A document's characters that must agree with PyMuPDF's.
DOCUMENT_AGREEMENT = 0.99
#: Characters a page may state apart from PyMuPDF's before the page is named: pages of a few characters are noise.
PAGE_SLACK = 40


def _needs_documents() -> None:
    if not _ROOTS:
        pytest.fail('the reference part needs DOCUMENT_LAB_REFERENCE_PDFS naming real PDFs or directories of them')


def test_the_part_has_documents():
    _needs_documents()
    assert PDFS, f'no PDF under {_ROOTS}'


def _checked(pdf: Path | None) -> Compared:
    _needs_documents()
    assert pdf is not None, f'no PDF under {_ROOTS}'
    return compare(pdf)


def problems(found: Compared) -> list[str]:
    """What in ``found`` is outside the tolerances, in words; empty when the document is accepted."""
    if found.locked:
        return []
    out = []
    if len(found.pages) != found.reference_pages:
        out.append(f'{len(found.pages)} pages, PyMuPDF {found.reference_pages}')
    out += [f'page {p.number} is {p.size_off:.3f} pt off in size' for p in found.pages if p.size_off > SIZE_TOLERANCE]
    out += [f'page {p.number} places {p.invented} image(s) PyMuPDF does not' for p in found.pages if p.invented]
    if found.agreement < DOCUMENT_AGREEMENT:
        pages = [f'{p.number} ({p.agreement:.2f})' for p in found.pages if p.whole - p.matched > PAGE_SLACK]
        out.append(f'text agrees {found.agreement:.4f} < {DOCUMENT_AGREEMENT}; pages {", ".join(pages[:10])}')
    return out


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_a_document_reads_as_the_reference_does(pdf):
    assert problems(_checked(pdf)) == []


@cache
def _digest(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_another_documents_text_is_reported(pdf):
    """The planted control: the next other document of the corpus taken as this one's reference.

    A document stating no text, or with no other document to plant, has nothing to tell apart and passes.
    """
    found = _checked(pdf)
    if found.locked or sum(p.stated for p in found.pages) == 0:
        return
    after = PDFS[PDFS.index(pdf) + 1 :] + PDFS[: PDFS.index(pdf)]
    other = next((p for p in after if _digest(p) != _digest(pdf)), None)
    if other is None:
        return
    planted = compare(pdf, other)
    if sum(p.reference for p in planted.pages) == 0:
        return
    assert any('text agrees' in p for p in problems(planted)), f'{other.name} passed as {pdf.name}'
