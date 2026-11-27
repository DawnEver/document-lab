"""What every test shares: the fonts :mod:`document_lab.fonts` fetches are kept in this checkout's ``output/cache``.

A test never writes the user's cache: fonts are kept in ``output/cache/fonts`` (git-ignored, declared in
``layout.toml``), downloads allowed. A test that reads glyph shapes asks for :func:`reference_font`, so
that fetching the font they are read against fails as an ERROR at setup -- the network, not the code -- never as the
test's FAILED.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from document_lab import fonts
from document_lab.glyphs import REFERENCE_FONT

fonts.use(fonts.Fonts(Path(__file__).resolve().parents[1] / 'output' / 'cache' / 'fonts'))


@pytest.fixture(scope='session')
def reference_font() -> Path:
    """The font :mod:`document_lab.glyphs` reads against, fetched (once, when the cache lacks it)."""
    return fonts.fetch(REFERENCE_FONT)
