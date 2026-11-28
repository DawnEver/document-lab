"""What every test shares: the fonts :mod:`document_lab.fonts` fetches are kept in this checkout's ``output/cache``,
and never downloaded.

A test never writes the user's cache nor reaches the internet: fonts are kept in ``output/cache/fonts`` (git-ignored,
declared in ``layout.toml``), downloading off -- set in the environment, as the command line reads it
(``DOCUMENT_LAB_CACHE``, ``DOCUMENT_LAB_OFFLINE``), so a test running the command line keeps them so too. The
tests of downloading serve their fonts from a local server (``tests/unit/test_fonts.py``). A test that reads glyph
shapes asks for :func:`reference_font`: the font they are read against, kept from PyMuPDF's own Helvetica program --
the bytes pinned, which MuPDF ships -- so that a PyMuPDF whose program is other bytes fails as an ERROR at setup,
naming them, never as the test's FAILED.
"""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest

from document_lab import fonts
from document_lab.glyphs import REFERENCE_FONT

os.environ['DOCUMENT_LAB_CACHE'] = str(Path(__file__).resolve().parents[1] / 'output' / 'cache')
os.environ['DOCUMENT_LAB_OFFLINE'] = '1'
fonts.use(fonts.Fonts.of(os.environ))


@pytest.fixture(scope='session')
def reference_font() -> Path:
    """The font :mod:`document_lab.glyphs` reads against, kept from PyMuPDF's Helvetica when the cache lacks it."""
    if not fonts.held(REFERENCE_FONT):
        fonts.keep(REFERENCE_FONT, bytes(pymupdf.Font('helv').buffer))
    return fonts.fetch(REFERENCE_FONT)
