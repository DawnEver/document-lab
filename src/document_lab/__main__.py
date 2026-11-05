"""The one ``document-lab`` command line: ``document-lab paper <pdf> <out>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_commons.log import emit

__all__ = ['main']


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run its subcommand; the exit status."""
    parser = argparse.ArgumentParser(prog='document-lab', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument('--tool', choices=('marker', 'pymupdf'), help='force a converter (default: best available)')
    args = parser.parse_args(argv)

    from document_lab.paper.ingest import ingest  # noqa: PLC0415 -- a subcommand loads only what it runs

    done = ingest(args.pdf, args.out, tool=args.tool)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.tool}); {len(done.appended)} appended paper(s)'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
