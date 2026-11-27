"""The one ``document-lab`` command line.

``capture`` and ``capture-merge`` make a PDF of page images from the screen, ``capture-files`` one for each PDF its
viewer shows; ``ocr`` lays a text layer over one;
``pages`` writes any PDF as a markdown file and an image per page (standards, manuals); ``paper`` writes a paper as
sections with its figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from lab_commons.log import emit

from document_lab.capture import cli as capture_cli

if TYPE_CHECKING:
    from document_lab.markdown import Headings

__all__ = ['main']

_NO_OUTLINE = "infer the section headings even where the PDF's outline names them on its pages"


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
    pages.add_argument('--no-outline', action='store_true', help=_NO_OUTLINE)
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument('--tool', choices=('marker', 'ir'), help='force a converter (default: best available)')
    paper.add_argument('--no-outline', action='store_true', help=_NO_OUTLINE)
    return parser


def _ocr(args: argparse.Namespace) -> int:
    from document_lab.ocr import text_layer  # noqa: PLC0415 -- a subcommand loads only what it runs

    out = args.out or args.pdf
    emit(f'{out}: text layer on {text_layer(args.pdf, out, args.lang)} page(s)')
    return 0


def _pages(args: argparse.Namespace) -> int:
    from document_lab.per_page import write_pages  # noqa: PLC0415

    done = write_pages(args.pdf, args.out, outline=not args.no_outline)
    emit(f'{done.out}: {done.pages} page(s)')
    _structure(done.structure)
    if len(done.textless) == done.pages:
        emit('no page has text: a capture needs `document-lab ocr` first')
        return 1
    if done.textless:
        emit(f'pages without text (blank, or a picture only): {list(done.textless)}')
    return 0


def _paper(args: argparse.Namespace) -> int:
    from document_lab.paper.ingest import ingest  # noqa: PLC0415

    done = ingest(args.pdf, args.out, tool=args.tool, outline=not args.no_outline)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.tool}); {len(done.appended)} appended paper(s)'
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
    args = _parser().parse_args(argv)
    return _RUN[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
