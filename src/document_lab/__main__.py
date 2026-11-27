"""The one ``document-lab`` command line.

``capture`` and ``capture-merge`` make a PDF of page images from the screen, ``capture-files`` one for each PDF its
viewer shows; ``ocr`` lays a text layer over one;
``pages`` writes any PDF as a markdown file and an image per page (standards, manuals); ``paper`` writes a paper as
sections with its figures.
"""

from __future__ import annotations

import argparse
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
    pages.add_argument('--untagged', action='store_true', help="infer a tagged PDF's structure as an untagged one's")
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument('--tool', choices=('marker', 'ir'), help='force a converter (default: best available)')
    paper.add_argument('--untagged', action='store_true', help="infer a tagged PDF's sections as an untagged one's")
    return parser


def _ocr(args: argparse.Namespace) -> int:
    from document_lab.ocr import text_layer  # noqa: PLC0415 -- a subcommand loads only what it runs

    out = args.out or args.pdf
    emit(f'{out}: text layer on {text_layer(args.pdf, out, args.lang)} page(s)')
    return 0


def _pages(args: argparse.Namespace) -> int:
    from document_lab.per_page import write_pages  # noqa: PLC0415

    done = write_pages(args.pdf, args.out, tagged=not args.untagged)
    emit(f'{done.out}: {done.pages} page(s)')
    _disagreements(done.disagreements)
    if len(done.textless) == done.pages:
        emit('no page has text: a capture needs `document-lab ocr` first')
        return 1
    if done.textless:
        emit(f'pages without text (blank, or a picture only): {list(done.textless)}')
    return 0


def _paper(args: argparse.Namespace) -> int:
    from document_lab.paper.ingest import ingest  # noqa: PLC0415

    done = ingest(args.pdf, args.out, tool=args.tool, tagged=not args.untagged)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.tool}); {len(done.appended)} appended paper(s)'
    )
    _disagreements(done.disagreements)
    return 0


def _disagreements(notes: tuple[str, ...]) -> None:
    """Where a tagged PDF's tags and the inference disagree, each in a line: the tags were used (``--untagged`` not)."""
    if notes:
        emit(f'the tags and the inference disagree ({len(notes)}; --untagged infers instead):')
        for note in notes:
            emit(f'  {note}')


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
