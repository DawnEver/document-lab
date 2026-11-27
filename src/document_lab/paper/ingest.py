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

from document_lab import formula, png
from document_lab.agreement import compare
from document_lab.ir import BBox
from document_lab.paper.convert import convert
from document_lab.paper.output import write_paper_output
from document_lab.paper.sections import clean_sections, split_sections
from document_lab.paper.text import slug
from document_lab.pdf import read_pdf, read_raster
from document_lab.tags import DEFAULT_USES

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from document_lab.formula import Formulas, Model
    from document_lab.markdown import Headings
    from document_lab.paper.convert import Tool, ToolChoice
    from document_lab.pdf import Renderer

__all__ = ['Ingested', 'NoSections', 'ingest']


class NoSections(ValueError):
    """The converter's markdown split into no section of the main paper."""


@dataclass(frozen=True, slots=True)
class Ingested:
    """What one ingest wrote: the converter's choice, the main paper's counts, one directory per appended paper.

    ``structure``: how the sections were headed -- by the PDF's outline or inferred, and where the two disagree
    (`document_lab.markdown.Headings`); None when marker headed them.
    """

    choice: ToolChoice
    sections: int
    figures: int
    tables: int
    appended: tuple[Path, ...]
    structure: Headings | None
    #: Where a tagged PDF's tags and layout's inference disagree, in words (`document_lab.agreement`); empty for an
    #: untagged PDF or marker's conversion.
    disagreements: tuple[str, ...] = ()


#: Render resolution of a scanned page: 3x (216 dpi) keeps a catalogue's 6 pt table digits legible.
_SCAN_DPI = 216.0


def _scanned_pages(pdf: Path, out_dir: Path, renderer: Renderer) -> int:
    """Write a document with ink but no text layer as page images; the count written (0: nothing inked).

    Raster scans and outlined glyphs: each inked page becomes ``img/pages/page-NNN.png`` with one section
    ``md/NN-page-N.md`` linking it.
    """
    pages = read_pdf(pdf).pages
    inked = [i for i, page in enumerate(pages) if page.figures or page.paths]
    if not inked:
        return 0
    img_dir, md_dir = out_dir / 'img' / 'pages', out_dir / 'md'
    img_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for k, i in enumerate(inked, 1):
        name = f'page-{i + 1:03d}.png'
        whole = BBox(0.0, 0.0, pages[i].width, pages[i].height)
        pixels = read_raster(pdf, i + 1, whole, dpi=_SCAN_DPI, renderer=renderer).pixels
        (img_dir / name).write_bytes(png.encode(pixels))
        section = f'{k:02d}-page-{i + 1}.md'
        (md_dir / section).write_text(
            f'# {pdf.stem} page {i + 1}\n\nno text layer: the page is the image\n\n'
            f'![page {i + 1}](../img/pages/{name})\n',
            encoding='utf-8',
        )
        links.append(f'- [page {i + 1}](md/{section})')
    (out_dir / 'paper.md').write_text(f'# {pdf.stem}\n\n' + '\n'.join(links) + '\n', encoding='utf-8')
    return len(inked)


def ingest(
    pdf: Path,
    out_dir: Path,
    *,
    tool: Tool | None = None,
    renderer: Renderer = 'pymupdf',
    outline: bool = True,
    tagged: Collection[str] = DEFAULT_USES,
    formulas: Formulas | None = None,
    formula_model: Model = 'texteller',
    include_hidden: bool = False,
    include_in_background_color: bool = False,
) -> Ingested:
    """Convert ``pdf`` (with ``tool``, else the IR path) and write its paper layout into ``out_dir``.

    Its figures, equations and scanned pages are rendered by ``renderer`` (`document_lab.pdf.read_raster`).
    ``outline``: the PDF's stated outline names the sections when it matches the pages; False keeps to its tags and
    the rules. ``tagged``: what a tagged PDF's tags give its markdown, and so its sections
    (`document_lab.paper.convert.convert`). ``formulas``: ``latex`` writes each equation crop's LaTeX under its image,
    read by ``formula_model`` (`document_lab.formula`); refused by :class:`document_lab.formula.FormulaUnavailable`
    before anything is written when that model cannot run. Every step reading text reads the paper as its default
    view shows it (`document_lab.ir.Page.shown`): its markdown and running heads, its tags and how they agree with
    layout (`document_lab.agreement`), its captions and equations; ``include_hidden``: what hidden layers draw is
    read too, ``include_in_background_color``: the text filled in the color under it.
    """
    transcribe = None if formulas is None else formula.transcriber(formulas, model=formula_model)
    pdf = pdf.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_text, choice, structure = convert(
        pdf,
        out_dir,
        tool=tool,
        outline=outline,
        tagged=tagged,
        include_hidden=include_hidden,
        include_in_background_color=include_in_background_color,
    )
    compared = (
        compare(read_pdf(pdf), include_hidden=include_hidden, include_in_background_color=include_in_background_color)
        if choice.tool == 'ir'
        else None
    )
    disagreements = compared.disagreements if compared is not None else ()
    stated = frozenset(structure.named) if structure is not None else frozenset()
    main, appended = clean_sections(split_sections(md_text), stated=stated)
    if not main and md_text.strip():
        # A drawing or data sheet, not a paper: every fragment is too short for the paper filters, so the
        # whole text is its one section rather than nothing.
        main, appended = [(f'# {pdf.stem}', md_text)], []
    if not main:
        scanned = _scanned_pages(pdf, out_dir, renderer)
        if scanned:
            return Ingested(choice, scanned, scanned, 0, (), structure, disagreements)
        msg = f'{pdf.name}: the converted markdown has no section'
        raise NoSections(msg)
    n_sec, n_fig, n_tbl = write_paper_output(
        main,
        out_dir,
        md_text,
        pdf_path=pdf,
        renderer=renderer,
        transcribe=transcribe,
        include_hidden=include_hidden,
        include_in_background_color=include_in_background_color,
    )

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
        write_paper_output(
            sections,
            target,
            sections[0][1],
            title_override=title,
            pdf_path=pdf,
            renderer=renderer,
            include_hidden=include_hidden,
            include_in_background_color=include_in_background_color,
            transcribe=transcribe,
        )
        shutil.rmtree(flat, ignore_errors=True)
        written.append(target)

    shutil.rmtree(out_dir / 'img' / 'flat', ignore_errors=True)
    return Ingested(choice, n_sec, n_fig, n_tbl, tuple(written), structure, disagreements)
