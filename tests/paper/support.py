"""Shared by the paper tests: a body long enough to pass the noise filters, and the real-paper runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from document_lab.paper.ingest import ingest

LONG_BODY = 'Long body text that passes all thresholds. ' * 20  # well over 400 chars
DATA = Path(__file__).resolve().parent / 'data'


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
