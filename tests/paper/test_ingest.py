"""``ingest`` on a document that is not a paper: a one-page drawing or data sheet still exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymupdf
import pytest

from document_lab.paper.ingest import NoSections, ingest

if TYPE_CHECKING:
    from pathlib import Path


def _pdf(path: Path, lines: list[str]) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((72, 72 + 14 * i), line, fontsize=10)
    doc.save(path)
    return path


def test_a_sheet_whose_every_fragment_is_noise_exports_as_one_section(tmp_path: Path) -> None:
    """A production drawing: short title-block fragments the paper filters discard one by one."""
    pdf = _pdf(tmp_path / 'RM-8S.pdf', ['Detail K', 'Scale 3:1', 'Drawing No. PY230225M28', 'Rev 00'])
    done = ingest(pdf, tmp_path / 'out', tool='pymupdf')
    assert done.sections == 1
    body = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'out' / 'md').glob('*.md'))
    assert 'PY230225M28' in body


def test_planted_control_a_page_with_no_text_still_refuses(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / 'blank.pdf', [])
    with pytest.raises(NoSections):
        ingest(pdf, tmp_path / 'out', tool='pymupdf')
