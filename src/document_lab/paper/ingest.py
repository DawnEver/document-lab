"""One paper PDF to a directory of per-section markdown and cropped figures, tables and equations.

Layout written under ``out_dir``::

    paper.md      title, abstract, links to the sections
    INDEX.md      figure/table label -> image file -> section
    md/           01-<slug>.md ... one file per section
    img/secNN/    images owned by section NN; img/orphan/ the unreferenced
    appended/     NN-<slug>/ the same layout for each further paper bound into the same PDF
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_lab.paper.convert import convert
from document_lab.paper.output import write_paper_output
from document_lab.paper.sections import clean_sections, split_sections
from document_lab.paper.text import slug

if TYPE_CHECKING:
    from pathlib import Path

    from document_lab.paper.convert import Tool

__all__ = ['Ingested', 'NoSections', 'ingest']


class NoSections(ValueError):
    """The converter's markdown split into no section of the main paper."""


@dataclass(frozen=True, slots=True)
class Ingested:
    """What one ingest wrote: the tool, the counts of the main paper, and one directory per appended paper."""

    tool: Tool
    sections: int
    figures: int
    tables: int
    appended: tuple[Path, ...]


def ingest(pdf: Path, out_dir: Path, *, tool: Tool | None = None) -> Ingested:
    """Convert ``pdf`` (with ``tool``, else the best available) and write its paper layout into ``out_dir``."""
    pdf = pdf.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_text, used = convert(pdf, out_dir, tool=tool)
    main, appended = clean_sections(split_sections(md_text))
    if not main:
        msg = f'{pdf.name}: the converted markdown has no section'
        raise NoSections(msg)
    n_sec, n_fig, n_tbl = write_paper_output(main, out_dir, md_text, pdf_path=pdf)

    appended_root = out_dir / 'appended'
    shutil.rmtree(appended_root, ignore_errors=True)
    written = []
    for i, (title, sections) in enumerate(appended, 1):
        target = appended_root / f'{i:02d}-{slug(title) or f"paper-{i}"}'
        flat = target / 'img' / 'flat'
        flat.mkdir(parents=True, exist_ok=True)
        for img in (out_dir / 'img' / 'flat').iterdir():
            if img.is_file():
                shutil.copy2(img, flat / img.name)
        write_paper_output(sections, target, sections[0][1], title_override=title, pdf_path=pdf)
        shutil.rmtree(flat, ignore_errors=True)
        written.append(target)

    shutil.rmtree(out_dir / 'img' / 'flat', ignore_errors=True)
    return Ingested(used, n_sec, n_fig, n_tbl, tuple(written))
