"""Shared by the paper tests: a body long enough to pass the noise filters, and the real-paper runner.

The real papers are never committed (they are not ours to publish): ``DOCUMENT_LAB_PAPERS`` names the local directory
holding them, and the ``papers`` part FAILS without it -- never skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from document_lab.paper.ingest import ingest

LONG_BODY = 'Long body text that passes all thresholds. ' * 20  # well over 400 chars


def paper(name: str) -> Path:
    """The local real paper ``name`` in the directory ``DOCUMENT_LAB_PAPERS`` names; a failure says what is missing."""
    where = os.environ.get('DOCUMENT_LAB_PAPERS')
    if not where:
        pytest.fail('the papers part needs DOCUMENT_LAB_PAPERS naming the directory that holds the real papers')
    path = Path(where) / name
    if not path.is_file():
        pytest.fail(f'{path} does not exist: DOCUMENT_LAB_PAPERS must hold {name}')
    return path


_cache: dict[tuple[str, str], Path] = {}


def ingested(pdf: Path, tool: str, factory: pytest.TempPathFactory) -> Path:
    """The paper layout of ``pdf`` by ``tool``, ingested once per session."""
    key = (str(pdf), tool)
    if key not in _cache:
        out = factory.mktemp(f'{pdf.stem}_{tool}')
        done = ingest(pdf, out, tool=tool)
        assert done.tool == tool
        _cache[key] = out
    return _cache[key]
