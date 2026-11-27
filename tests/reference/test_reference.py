"""The ``reference`` part: the reader accepted against the previous implementation, PyMuPDF, on real PDFs.

``DOCUMENT_LAB_REFERENCE_PDFS`` names PDFs, or directories searched for them, ``os.pathsep``-separated (real
documents are never committed); the part FAILS without one rather than skip. Every document must read whole, refuse
a password exactly when PyMuPDF does, match PyMuPDF's page count and page sizes, place no image PyMuPDF does not, and
state the outline and page labels PyMuPDF states; its text must agree with PyMuPDF's, stand where PyMuPDF's stands,
and its painted paths be PyMuPDF's drawings, within the tolerances below (`tests.reference.compare` says how each is
compared). The planted controls, each of which must be reported: another document's reading (its text and outline)
taken as this one's, PyMuPDF's geometry shifted, every second line of its text risen, its colors turned.

The tolerances were measured on 283 local PDFs (6499 pages: papers, standards, theses) on 2026-11-26. Page sizes
agree within 3e-5 pt and no image is invented. Text agrees 0.9986 at worst. Of 4,250,325 characters stated, 0.99984
are located, a document 0.9970 at worst but for one whose predefined CJK `CMap` (`GBK-EUC-H`) the reader maps to the
wrong CIDs for one-byte codes, measuring them full-width (0.9914: a known defect, which this part reports). Of
283,472 measured spans 99.76 % start and end within 0.1 pt of PyMuPDF's characters -- PyMuPDF's own advances drift
up to 0.06 pt along a justified line, which ours end on its margin to 0.005 pt -- and 99.82 % cut PyMuPDF's baseline
within 0.01 of their font's share; a document's spans off either way are 8 % at most (an OCR text layer of
overlapping runs; synthetic-oblique symbols, whose sheared boxes the two bound apart), 9.6 % for the CJK one. Of
590,099 painted paths compared 4 go unmatched (one figure, 4 of a paper's 281) and PyMuPDF draws 1 fill or stroke
we do not report. The plants are reported on every document: shifted 2 pt, 86 % of a document's spans or more stand
off and 34 % of its paths or more go unmatched (a mesh of 2 pt cells matches itself); risen 1 pt, 20 % of a
document's spans or more stand off their baseline; turned, 12 % of its paths or more go unmatched.
"""

import hashlib
import os
from functools import cache
from pathlib import Path

import pytest

from tests.reference.compare import Compared, Plant, compare

pytestmark = pytest.mark.reference

_ROOTS = [Path(p) for p in os.environ.get('DOCUMENT_LAB_REFERENCE_PDFS', '').split(os.pathsep) if p]
PDFS = sorted({pdf for root in _ROOTS for pdf in ([root] if root.is_file() else root.rglob('*.pdf'))})

#: Points a page's size may differ from PyMuPDF's (both read the same boxes: measured 3e-5).
SIZE_TOLERANCE = 0.01
#: A document's characters that must agree with PyMuPDF's.
DOCUMENT_AGREEMENT = 0.99
#: Characters a page may state apart from PyMuPDF's before the page is named: pages of a few characters are noise.
PAGE_SLACK = 40
#: A document's characters that must stand where PyMuPDF states them.
DOCUMENT_LOCATION = 0.995
#: Points a measured span's start or end may stand off its characters'.
ALONG = 0.1
#: How far, in shares of a measured span's height, PyMuPDF's baseline may cut it off its font's median.
BASELINE = 0.01
#: The share of a document's measured spans that may stand off, along (`ALONG`) or on their baseline (`BASELINE`).
SPANS_OFF = 0.1
#: The share of a document's painted paths that may go unmatched; PyMuPDF's unreported fills and strokes, likewise.
PATHS_OFF = 0.05
#: The planted controls: PyMuPDF's geometry moved by this many points, every second line of its text by this.
SHIFT = (2.0, 2.0)
RISE = 1.0


def _needs_documents() -> None:
    if not _ROOTS:
        pytest.fail('the reference part needs DOCUMENT_LAB_REFERENCE_PDFS naming real PDFs or directories of them')


def test_the_part_has_documents():
    _needs_documents()
    assert PDFS, f'no PDF under {_ROOTS}'


@cache
def _checked(pdf: Path | None) -> Compared:
    _needs_documents()
    assert pdf is not None, f'no PDF under {_ROOTS}'
    return compare(pdf)


def _pages(numbers: list[int]) -> str:
    return ', '.join(str(n) for n in sorted(set(numbers))[:10])


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
        pages = [
            f'{p.number} ({p.agreement:.2f})'
            for p in found.pages
            if max(p.stated, p.reference - p.unknown) - p.matched > PAGE_SLACK
        ]
        out.append(f'text agrees {found.agreement:.4f} < {DOCUMENT_AGREEMENT}; pages {", ".join(pages[:10])}')
    out += _stated(found)
    if found.location < DOCUMENT_LOCATION:
        worst = sorted((p for p in found.pages if p.stated > p.located), key=lambda p: p.located - p.stated)
        pages = ', '.join(f'{p.number} ({p.stated - p.located} not located)' for p in worst[:10])
        out.append(f'text located {found.location:.4f} < {DOCUMENT_LOCATION}; pages {pages}')
    spans = [(p.number, e.along) for p in found.pages for e in p.edges]
    off = [n for n, along in spans if along > ALONG]
    if len(off) > SPANS_OFF * len(spans):
        out.append(f'{len(off)} of {len(spans)} spans start or end over {ALONG} pt off; pages {_pages(off)}')
    baselines = found.baselines()
    off = [n for n, share in baselines if share > BASELINE]
    if len(off) > SPANS_OFF * len(baselines):
        out.append(f'{len(off)} of {len(baselines)} spans stand off their baseline; pages {_pages(off)}')
    paths = sum(p.paths for p in found.pages)
    unmatched = [p.number for p in found.pages for _ in range(p.unmatched)]
    if len(unmatched) > PATHS_OFF * paths:
        out.append(f'{len(unmatched)} of {paths} paths PyMuPDF does not draw; pages {_pages(unmatched)}')
    unreported = [p.number for p in found.pages for _ in range(p.unreported)]
    if len(unreported) > PATHS_OFF * paths:
        out.append(f'{len(unreported)} fills and strokes PyMuPDF draws are not reported; pages {_pages(unreported)}')
    return out


def _stated(found: Compared) -> list[str]:
    """Where ``found``'s outline and page labels are apart from PyMuPDF's, in words."""
    out = []
    if found.outline != found.reference_outline:
        first = next(
            (i for i, (a, b) in enumerate(zip(found.outline, found.reference_outline, strict=False)) if a != b),
            min(len(found.outline), len(found.reference_outline)),
        )
        ours, theirs = len(found.outline), len(found.reference_outline)
        out.append(f'outline of {ours} items, PyMuPDF {theirs}; first apart at item {first}')
    if found.labels != found.reference_labels:
        apart = [i + 1 for i, (a, b) in enumerate(zip(found.labels, found.reference_labels, strict=False)) if a != b]
        out.append(f'page labels apart from PyMuPDF on pages {apart[:10]}')
    return out


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_a_document_reads_as_the_reference_does(pdf):
    assert problems(_checked(pdf)) == []


@cache
def _digest(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_another_documents_reading_is_reported(pdf):
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
    if planted.outline != planted.reference_outline:  # either states an outline: the other's must be told apart
        assert any(p.startswith('outline') for p in problems(planted)), f"{other.name}'s outline passed as {pdf.name}'s"
    if sum(p.reference for p in planted.pages) == 0:
        return
    found = problems(planted)
    assert any('text agrees' in p for p in found), f'{other.name} passed as {pdf.name}'
    assert any('text located' in p for p in found), f'{other.name} stood where {pdf.name} does'


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_shifted_geometry_is_reported(pdf):
    """The planted control: PyMuPDF's characters and drawings moved by `SHIFT`, which spans and paths must report."""
    found = _checked(pdf)
    if found.locked:
        return
    planted = problems(compare(pdf, plant=Plant(shift=SHIFT)))
    if any(p.edges for p in found.pages):
        assert any('spans start or end' in p for p in planted), f'{pdf.name}: its spans passed shifted'
    if any(p.paths for p in found.pages):
        assert any('paths PyMuPDF does not draw' in p for p in planted), f'{pdf.name}: its paths passed shifted'


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_risen_lines_are_reported(pdf):
    """The planted control: every second line of PyMuPDF's text risen by `RISE`, which baselines must report."""
    found = _checked(pdf)
    if found.locked or not found.baselines():
        return
    planted = problems(compare(pdf, plant=Plant(rise=RISE)))
    assert any('off their baseline' in p for p in planted), f'{pdf.name}: its baselines passed risen'


@pytest.mark.parametrize('pdf', PDFS or [None], ids=lambda p: p.name if p else 'none')
def test_turned_colors_are_reported(pdf):
    """The planted control: every color PyMuPDF states turned by half the scale, which paths in RGB must report.

    A document whose paths are colored in RGB too few to report (`PATHS_OFF`) -- the rest keep a tint -- passes.
    """
    found = _checked(pdf)
    if found.locked or sum(p.colored for p in found.pages) <= PATHS_OFF * sum(p.paths for p in found.pages):
        return
    planted = problems(compare(pdf, plant=Plant(recolor=True)))
    assert any('paths PyMuPDF does not draw' in p for p in planted), f'{pdf.name}: its colors passed turned'
