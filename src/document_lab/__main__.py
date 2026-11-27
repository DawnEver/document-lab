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

from lab_commons.log import emit

from document_lab.capture import cli as capture_cli

__all__ = ['main']


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='document-lab', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture_cli.add_parser(sub)
    capture_cli.add_merge_parser(sub)
    capture_cli.add_files_parser(sub)
    ocr = sub.add_parser('ocr', help='a text layer over a PDF of page images (Tesseract); the pixels kept exactly')
    ocr.add_argument('pdf', type=Path)
    ocr.add_argument('-o', '--out', type=Path, help='the PDF to write (default: the PDF itself, replaced when whole)')
    ocr.add_argument('--lang', default='eng', help="Tesseract's languages, '+'-joined (default eng; e.g. chi_sim+eng)")
    pages = sub.add_parser(
        'pages',
        help='a PDF as page-NNN.md and page-NNN.png per page, its regions cut out, and INDEX.md (a folder or a .zip)',
    )
    pages.add_argument('pdf', type=Path)
    pages.add_argument('out', type=Path, help='the directory to write the pages into, or a .zip archive')
    _formulas_option(pages)
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument('--tool', choices=('marker', 'ir'), help='force a converter (default: best available)')
    _formulas_option(paper)
    return parser


def _formulas_option(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        '--formulas',
        choices=('latex',),
        help="write each equation crop's LaTeX beside it, read in-process by TexTeller (the 'formula' extra; "
        'default: the crop alone)',
    )


def _refused(error: Exception) -> int:
    emit(f'refused: {error}')
    return 2


def _ocr(args: argparse.Namespace) -> int:
    from document_lab.ocr import TesseractPlaces, text_layer  # noqa: PLC0415 -- a subcommand loads only what it runs

    out = args.out or args.pdf
    laid = text_layer(args.pdf, out, args.lang, places=TesseractPlaces.of(os.environ))
    emit(f'{out}: text layer on {laid} page(s)')
    return 0


def _pages(args: argparse.Namespace) -> int:
    from document_lab.formula import FormulaUnavailable  # noqa: PLC0415
    from document_lab.per_page import write_pages  # noqa: PLC0415

    try:
        done = write_pages(args.pdf, args.out, formulas=args.formulas)
    except FormulaUnavailable as refused:
        return _refused(refused)
    emit(f'{done.out}: {done.pages} page(s)')
    if len(done.textless) == done.pages:
        emit('no page has text: a capture needs `document-lab ocr` first')
        return 1
    if done.textless:
        emit(f'pages without text (blank, or a picture only): {list(done.textless)}')
    return 0


def _paper(args: argparse.Namespace) -> int:
    from document_lab.formula import FormulaUnavailable  # noqa: PLC0415
    from document_lab.paper.ingest import ingest  # noqa: PLC0415

    try:
        done = ingest(args.pdf, args.out, tool=args.tool, formulas=args.formulas)
    except FormulaUnavailable as refused:
        return _refused(refused)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.tool}); {len(done.appended)} appended paper(s)'
    )
    return 0


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
    args = _parser().parse_args(argv)
    return _RUN[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
