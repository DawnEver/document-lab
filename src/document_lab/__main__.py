"""The one ``document-lab`` command line.

``capture`` and ``capture-merge`` make a PDF of page images from the screen, ``capture-files`` one for each PDF its
viewer shows; ``ocr`` lays a text layer over one;
``pages`` writes any PDF as a markdown file and an image per page (standards, manuals); ``paper`` writes a paper as
sections with its figures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from lab_commons.log import emit

from document_lab.capture import cli as capture_cli
from document_lab.formula import MODELS
from document_lab.regions import UNNUMBERED

if TYPE_CHECKING:
    from document_lab.markdown import Headings

__all__ = ['main']

_NO_OUTLINE = "infer the section headings even where the PDF's outline names them on its pages"
#: `--include-in-background-color`: the text filled in the color under it is read too (`document_lab.ir`).
_IN_BACKGROUND_COLOR = (
    'read the text filled in the color under it too (white on the white page), which shows nothing; left out by default'
)
#: `--include-hidden`: what a layer turned off in the default view draws is read too (`document_lab.ir`).
_HIDDEN = "read what the PDF's layers turned off in its default view draw too; left out by default"

#: `--renderer`: what draws a page or region's pixels (`document_lab.pdf.read_raster`).
_RENDERER = {
    'choices': ('pymupdf', 'own'),
    'default': 'pymupdf',
    'help': "what draws the pictures: pymupdf (default) or own (document-lab's renderer; refuses what it cannot draw)",
}
#: `--unnumbered-equations`: how displays without a number are found (`document_lab.regions.Unnumbered`).
_UNNUMBERED = {
    'choices': UNNUMBERED,
    'default': 'strict',
    'help': 'displays without a number: strict (default: those set apart above, below and across, in no table or '
    'figure, showing a formula beyond one relation sign), '
    'loose (every formula line that is no prose: more found, more taken in error) or off (numbered ones only)',
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='document-lab', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture_cli.add_parser(sub)
    capture_cli.add_merge_parser(sub)
    capture_cli.add_files_parser(sub)
    ocr = sub.add_parser('ocr', help='a text layer over a PDF of page images (Tesseract); the pixels kept exactly')
    ocr.add_argument('pdf', type=Path)
    ocr.add_argument('-o', '--out', type=Path, help='the PDF to write (default: the PDF itself, replaced when whole)')
    ocr.add_argument(
        '--lang',
        default='eng',
        help="Tesseract's languages, '+'-joined (default eng; e.g. chi_sim+eng; jpn_vert, chi_sim_vert for columns)",
    )
    ocr.add_argument(
        '--orientation-confidence',
        type=_off_or_float,
        default=argparse.SUPPRESS,
        help="OSD's confidence at which a page is read turned (default: document_lab.ocr's measured one; 'off': never)",
    )
    ocr.add_argument(
        '--deskew-above',
        type=_off_or_float,
        default=argparse.SUPPRESS,
        help="degrees of skew above which a page is read turned back (default: document_lab.ocr's; 'off': never)",
    )
    pages = sub.add_parser(
        'pages',
        help='a PDF as page-NNN.md and page-NNN.png per page, its regions cut out, and INDEX.md (a folder or a .zip)',
    )
    pages.add_argument('pdf', type=Path)
    pages.add_argument('out', type=Path, help='the directory to write the pages into, or a .zip archive')
    pages.add_argument('--renderer', **_RENDERER)
    pages.add_argument('--no-outline', action='store_true', help=_NO_OUTLINE)
    _formulas_option(pages)
    pages.add_argument('--unnumbered-equations', **_UNNUMBERED)
    pages.add_argument('--include-hidden', action='store_true', help=_HIDDEN)
    pages.add_argument('--include-in-background-color', action='store_true', help=_IN_BACKGROUND_COLOR)
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument(
        '--tool',
        choices=('marker', 'ir'),
        help="the converter (default: ir; marker needs the gpu extra and surya's model server, and refuses without)",
    )
    paper.add_argument('--renderer', **_RENDERER)
    paper.add_argument('--no-outline', action='store_true', help=_NO_OUTLINE)
    _formulas_option(paper)
    paper.add_argument('--unnumbered-equations', **_UNNUMBERED)
    paper.add_argument('--include-hidden', action='store_true', help=_HIDDEN)
    paper.add_argument('--include-in-background-color', action='store_true', help=_IN_BACKGROUND_COLOR)
    return parser


def _formulas_option(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        '--formulas',
        choices=('latex',),
        help="write each equation crop's LaTeX beside it, read in-process ('formula' extra; default: the crop alone)",
    )
    command.add_argument(
        '--formula-model',
        choices=tuple(MODELS),
        help='the model --formulas reads with: texteller (the default; Apache-2.0, fastest on a GPU) or codeformula-v2 '
        "(docling's CodeFormulaV2; CDLA-Permissive-2.0 weights)",
    )


def _formulas(args: argparse.Namespace) -> dict[str, str]:
    """The formula keywords of ``write_pages`` / ``ingest`` the command line asked for."""
    asked = {'formulas': args.formulas}
    if args.formula_model is not None:
        asked['formula_model'] = args.formula_model
    return asked


def _refused(error: Exception) -> int:
    emit(f'refused: {error}')
    return 2


def _off_or_float(value: str) -> float | None:
    """A command-line number, or ``off`` for None."""
    return None if value == 'off' else float(value)


def _ocr(args: argparse.Namespace) -> int:
    from document_lab import ocr  # noqa: PLC0415 -- a subcommand loads only what it runs

    out = args.out or args.pdf
    asked = {k: v for k, v in vars(args).items() if k in {'orientation_confidence', 'deskew_above'}}  # those given
    laid = ocr.text_layer(args.pdf, out, args.lang, places=ocr.TesseractPlaces.of(os.environ), **asked)
    emit(f'{out}: text layer on {len(laid.readings)} page(s)')
    for note in filter(None, (reading.note() for reading in laid.readings)):
        emit(f'  {note}')
    return 0


def _pages(args: argparse.Namespace) -> int:
    from document_lab.formula import FormulaUnavailable  # noqa: PLC0415
    from document_lab.per_page import write_pages  # noqa: PLC0415

    try:
        done = write_pages(
            args.pdf,
            args.out,
            renderer=args.renderer,
            outline=not args.no_outline,
            unnumbered=args.unnumbered_equations,
            include_hidden=args.include_hidden,
            include_in_background_color=args.include_in_background_color,
            **_formulas(args),
        )
    except FormulaUnavailable as refused:
        return _refused(refused)
    emit(f'{done.out}: {done.pages} page(s)')
    _structure(done.structure)
    if len(done.textless) == done.pages:
        emit('no page has text: a capture needs `document-lab ocr` first')
        return 1
    if done.unread:
        emit(f'pictures without a text layer: pages {list(done.unread)}; `document-lab ocr` lays one')
    if blank := [n for n in done.textless if n not in done.unread]:
        emit(f'pages without text (blank, or drawings only): {blank}')
    return 0


def _paper(args: argparse.Namespace) -> int:
    from document_lab.formula import FormulaUnavailable  # noqa: PLC0415
    from document_lab.paper.convert import MarkerUnavailable  # noqa: PLC0415
    from document_lab.paper.ingest import ingest  # noqa: PLC0415

    try:
        done = ingest(
            args.pdf,
            args.out,
            tool=args.tool,
            renderer=args.renderer,
            outline=not args.no_outline,
            unnumbered=args.unnumbered_equations,
            include_hidden=args.include_hidden,
            include_in_background_color=args.include_in_background_color,
            **_formulas(args),
        )
    except (MarkerUnavailable, FormulaUnavailable) as refused:
        return _refused(refused)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.choice.tool}: {done.choice.why}); {len(done.appended)} appended paper(s)'
    )
    if done.structure is not None:
        _structure(done.structure)
    return 0


#: Disagreements between a stated outline and the inferred headings named one by one before the rest are counted.
_SHOWN_DISAGREEMENTS = 5


def _structure(found: Headings) -> None:
    """Say where the section headings came from and where the PDF's outline and the inferred headings disagree."""
    if found.source == 'outline':
        emit(f'sections: from the PDF outline ({found.stated} items)')
    elif found.unlocated:
        emit(f'sections: inferred -- the PDF outline ({found.stated} items) is not found on its pages')
    elif found.stated:
        emit(f'sections: inferred -- the PDF outline ({found.stated} items) names none of the headings the rules find')
    else:
        emit('sections: inferred (the PDF states no outline, or it was not asked for)')
    lines = found.disagreements
    for line in lines[:_SHOWN_DISAGREEMENTS]:
        emit(f'  {line}')
    if len(lines) > _SHOWN_DISAGREEMENTS:
        emit(f'  ... and {len(lines) - _SHOWN_DISAGREEMENTS} more')


_RUN = {
    'capture': capture_cli.run,
    'capture-merge': capture_cli.run_merge,
    'capture-files': capture_cli.run_files,
    'ocr': _ocr,
    'pages': _pages,
    'paper': _paper,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run its subcommand; the exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, 'formula_model', None) is not None and args.formulas is None:
        parser.error('--formula-model names the model --formulas reads with: add --formulas latex')
    return _RUN[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
