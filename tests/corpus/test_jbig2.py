"""The ``corpus`` part's JBIG2 files (:data:`tests.corpus.sources.JBIG2`, pdf.js's test suite): each drawn by the own
renderer, held to PyMuPDF's drawing (the D80 tolerance, :func:`tests.rendering.agrees`) where MuPDF decodes it.

MuPDF's jbig2dec fails on :data:`MUPDF_FAILS` (refinement and some Huffman and halftone codings: it says 'cannot
decode jbig2 image' and draws black or blank); there the truth is the suite's own: each of those files encodes the
picture ``bitmap-mmr.pdf`` encodes, which MuPDF does decode. Every check is paired with a planted wrong truth it must
report.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tests.corpus.sources import JBIG2, fetch
from tests.rendering import agrees, disagreement, own, reference

pytestmark = pytest.mark.corpus

#: The resolution every file is drawn at (at 72 dpi a downsampled perf_codecs_jbig2 page parts from MuPDF's).
DPI = 150
#: The one file the own renderer refuses, and the cause it names: an embedded stream that starts with a file header.
REFUSED = {'jbig2_file_header.pdf': 'MissingPageInfo'}
#: The file whose picture each of :data:`MUPDF_FAILS` encodes.
LIKE = 'bitmap-mmr.pdf'
#: The files MuPDF does not decode (measured 2026-11-28, PyMuPDF of the dev extra).
MUPDF_FAILS = frozenset(
    {
        'bitmap-composite-and-xnor-refine.pdf',
        'bitmap-composite-or-xor-replace-refine.pdf',
        'bitmap-halftone-composite.pdf',
        'bitmap-halftone-refine.pdf',
        'bitmap-refine-customat-tpgron.pdf',
        'bitmap-refine-customat.pdf',
        'bitmap-refine-lossless.pdf',
        'bitmap-refine-page-subrect.pdf',
        'bitmap-refine-refine.pdf',
        'bitmap-refine-template1-tpgron.pdf',
        'bitmap-refine-template1.pdf',
        'bitmap-refine-tpgron.pdf',
        'bitmap-refine.pdf',
        'bitmap-symbol-context-reuse.pdf',
        'bitmap-symbol-symhuffrefine-textrefine.pdf',
        'bitmap-symbol-symhuffrefineone.pdf',
        'bitmap-symbol-symhuffrefineseveral.pdf',
        'bitmap-symbol-texthuffrefine.pdf',
        'bitmap-symbol-texthuffrefineB15.pdf',
        'bitmap-symbol-texthuffrefinecustom.pdf',
        'bitmap-symbol-texthuffrefinecustomdims.pdf',
        'bitmap-symbol-texthuffrefinecustompos.pdf',
        'bitmap-symbol-texthuffrefinecustomposdims.pdf',
        'bitmap-symbol-texthuffrefinecustomsize.pdf',
        'bitmap-trailing-7fff-stripped-harder-refine.pdf',
    }
)
NAMES = sorted(JBIG2)
DECODED = sorted(set(JBIG2) - MUPDF_FAILS - set(REFUSED))


@cache
def _own(pdf: Path) -> tuple[np.ndarray, dict[str, int]]:
    return own(pdf.read_bytes(), DPI)


def drawn(pdf: Path, refused: str | None) -> list[str]:
    """The own renderer draws page 1 whole, or (``refused`` a cause) leaves it undrawn naming that cause alone."""
    _pixels, undrawn = _own(pdf)
    if refused is None:
        return [f'undrawn {undrawn}'] if undrawn else []
    return [] if undrawn and all(refused in why for why in undrawn) else [f'undrawn {undrawn}, not for {refused}']


def as_reference(pdf: Path, turn: int) -> list[str]:
    """Page 1 drawn by the own renderer agrees with PyMuPDF's drawing turned ``turn`` degrees (not 0: planted)."""
    pixels, _undrawn = _own(pdf)
    truth = np.rot90(reference(pdf.read_bytes(), DPI), turn // 90)
    if pixels.shape != truth.shape:
        return [f'drawn {pixels.shape}, the reference {truth.shape}']
    mean, share = disagreement(pixels, truth)
    return [] if agrees(mean, share) else [f'{share:.2%} of inked pixels off, mean {mean:.1f} levels']


def as_like(pdf: Path, like: Path, turn: int) -> list[str]:
    """Page 1 drawn by the own renderer is ``like``'s drawing turned ``turn`` degrees (not 0: planted), and PyMuPDF's
    drawing of it parts from that -- MuPDF still fails on it."""
    pixels, _undrawn = _own(pdf)
    truth = np.rot90(_own(like)[0], turn // 90)
    out = [] if np.array_equal(pixels, truth) else [f'not the picture of {like.name}']
    reference_pixels = reference(pdf.read_bytes(), DPI)
    if reference_pixels.shape == pixels.shape and agrees(*disagreement(pixels, reference_pixels)):
        out.append('PyMuPDF now decodes it: move it out of MUPDF_FAILS')
    return out


@pytest.mark.parametrize('pdf', NAMES, indirect=True)
def test_a_jbig2_file_is_drawn_or_refused_by_name(pdf: Path, request: pytest.FixtureRequest):
    name = request.node.callspec.params['pdf']
    assert drawn(pdf, REFUSED.get(name)) == []
    assert drawn(pdf, None if name in REFUSED else 'MissingPageInfo'), 'the planted wrong truth was not reported'


@pytest.mark.parametrize('pdf', DECODED, indirect=True)
def test_a_jbig2_file_is_drawn_as_pymupdf_draws_it(pdf: Path):
    assert as_reference(pdf, 0) == []
    assert as_reference(pdf, 180), 'the planted wrong truth (the reference turned) was not reported'


@pytest.mark.parametrize('pdf', sorted(MUPDF_FAILS), indirect=True)
def test_a_jbig2_file_mupdf_fails_on_is_drawn_as_its_like(pdf: Path):
    like = fetch(LIKE)
    assert as_like(pdf, like, 0) == []
    assert as_like(pdf, like, 180), 'the planted wrong truth (the picture turned) was not reported'
