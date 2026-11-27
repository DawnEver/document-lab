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


_DEFAULT = 'order,headings,paragraphs,tables'
_TAGS_HELP = (
    f"what a tagged PDF's tags give in place of inference, comma-separated (default {_DEFAULT}; also figures; "
    "'none' infers all)"
)


def _uses(text: str) -> frozenset[str]:
    """``--tags``: the uses named, ``none`` for none; an unknown one is refused by name."""
    from document_lab.tags import checked  # noqa: PLC0415 -- the parser loads no reader

    names = () if text == 'none' else [t.strip() for t in text.split(',') if t.strip()]
    try:
        return checked(names)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


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
    pages.add_argument('--tags', type=_uses, default=_DEFAULT, help=_TAGS_HELP)
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument('--tool', choices=('marker', 'ir'), help='force a converter (default: best available)')
    paper.add_argument('--tags', type=_uses, default=_DEFAULT, help=_TAGS_HELP)
    return parser


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
    from document_lab.per_page import write_pages  # noqa: PLC0415

    done = write_pages(args.pdf, args.out, tagged=args.tags)
    emit(f'{done.out}: {done.pages} page(s)')
    _disagreements(done.disagreements)
    if len(done.textless) == done.pages:
        emit('no page has text: a capture needs `document-lab ocr` first')
        return 1
    if done.unread:
        emit(f'pictures without a text layer: pages {list(done.unread)}; `document-lab ocr` lays one')
    if blank := [n for n in done.textless if n not in done.unread]:
        emit(f'pages without text (blank, or drawings only): {blank}')
    return 0


def _paper(args: argparse.Namespace) -> int:
    from document_lab.paper.ingest import ingest  # noqa: PLC0415

    done = ingest(args.pdf, args.out, tool=args.tool, tagged=args.tags)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.tool}); {len(done.appended)} appended paper(s)'
    )
    _disagreements(done.disagreements)
    return 0


def _disagreements(notes: tuple[str, ...]) -> None:
    """Where a tagged PDF's tags and the inference disagree, each in a line (``--tags`` chooses which is used)."""
    if notes:
        emit(f'the tags and the inference disagree ({len(notes)}; --tags chooses which is used):')
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
