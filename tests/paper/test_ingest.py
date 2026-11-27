"""``ingest`` on a document that is not a paper: a one-page drawing or data sheet still exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymupdf
import pytest

from document_lab.__main__ import main
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
    done = ingest(pdf, tmp_path / 'out', tool='ir')
    assert done.sections == 1
    body = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'out' / 'md').glob('*.md'))
    assert 'PY230225M28' in body


def test_planted_control_a_blank_page_still_refuses(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / 'blank.pdf', [])
    with pytest.raises(NoSections):
        ingest(pdf, tmp_path / 'out', tool='ir')


def test_a_scanned_sheet_with_no_text_layer_exports_its_pages_as_images(tmp_path: Path) -> None:
    """A catalogue printed as images or outlined glyphs has ink but no text layer: each page becomes an image the
    reader reads, one section per page, instead of a refusal that leaves nothing to read."""
    pdf = tmp_path / 'catalogue.pdf'
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(pdf)
    done = ingest(pdf, tmp_path / 'out', tool='ir')
    assert done.sections == 2
    pages = sorted((tmp_path / 'out' / 'img' / 'pages').glob('*.png'))
    assert [p.name for p in pages] == ['page-001.png', 'page-002.png']
    body = (tmp_path / 'out' / 'md' / '01-page-1.md').read_text(encoding='utf-8')
    assert '../img/pages/page-001.png' in body and 'no text layer' in body


def _outlined_paper(path: Path) -> Path:
    """A paper whose section headings are plain words the rules cannot tell from text; its outline names them."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), 'A Paper Headed By Its Outline', fontsize=20)
    body = 'This paragraph carries enough words to be a section body and not a stray fragment of a figure label.'
    toc = []
    for i, title in enumerate(('Background', 'Approach', 'Findings')):
        y = 120 + 120 * i
        page.insert_text((72, y), title, fontsize=11)
        page.insert_text((72, y + 20), body, fontsize=8)
        page.insert_text((72, y + 32), body, fontsize=8)
        toc.append([1, title, 1, {'kind': pymupdf.LINK_GOTO, 'page': 0, 'to': pymupdf.Point(72, y - 12)}])
    doc.set_toc(toc)
    doc.save(path)
    return path


def test_a_stated_outline_names_the_sections_and_inferring_is_the_option(tmp_path: Path) -> None:
    pdf = _outlined_paper(tmp_path / 'outlined.pdf')
    done = ingest(pdf, tmp_path / 'outline', tool='ir')
    assert (done.structure.source, done.structure.disagreements) == ('outline', ())
    names = sorted(p.name for p in (tmp_path / 'outline' / 'md').glob('*.md'))
    assert [n.split('-', 1)[1] for n in names] == ['background.md', 'approach.md', 'findings.md']
    inferred = ingest(pdf, tmp_path / 'inferred', tool='ir', outline=False)
    assert inferred.structure.source == 'inferred'
    assert inferred.sections < done.sections  # the planted control: the rules alone miss the plain-word headings


def test_text_filled_in_the_color_under_it_is_written_only_when_the_command_asks(tmp_path: Path) -> None:
    doc = pymupdf.open(_outlined_paper(tmp_path / 'outlined.pdf'))
    words = 'Words set white on the white page, which a reader of the printed paper never sees.'
    doc[0].insert_text((72, 172), words, fontsize=8, color=(1, 1, 1))
    doc.save(tmp_path / 'white.pdf')
    for asked in (False, True):
        out = tmp_path / f'asked-{asked}'
        flag = ['--include-in-background-color'] if asked else []
        assert main(['paper', str(tmp_path / 'white.pdf'), str(out), *flag]) == 0
        body = ''.join(p.read_text(encoding='utf-8') for p in (out / 'md').glob('*.md'))
        assert 'This paragraph carries enough words' in body
        assert ('Words set white' in body) is asked
