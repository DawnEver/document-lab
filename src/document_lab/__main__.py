"""The one ``document-lab`` command line: ``paper <pdf> <out>`` and ``capture <out.pdf>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_commons.log import emit

__all__ = ['main']


def _rect(text: str) -> tuple[int, int, int, int]:
    left, top, width, height = (int(v) for v in text.split(','))
    return left, top, width, height


def _capture_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    capture = sub.add_parser(
        'capture',
        help='a window or screen region, scrolled to its end and stitched, to a PDF (Windows)',
        description='Esc stops scrolling and keeps what was captured.',
    )
    capture.add_argument('out', type=Path, nargs='?', help='the PDF to write (not with --list)')
    target = capture.add_mutually_exclusive_group(required=True)
    target.add_argument('--window', metavar='TITLE', help='the one window whose title contains TITLE')
    target.add_argument('--hwnd', type=lambda v: int(v, 0), help='the window with this handle (see --list)')
    target.add_argument('--pick', action='store_true', help='the window you click next')
    target.add_argument('--region', action='store_true', help='a rectangle you drag on the screen')
    target.add_argument('--rect', type=_rect, metavar='L,T,W,H', help='a screen rectangle in physical pixels')
    target.add_argument('--list', action='store_true', help='print the capturable windows and exit')
    scroll = capture.add_argument_group('scrolling')
    scroll.add_argument('--no-scroll', action='store_true', help='one frame: what the target shows now')
    scroll.add_argument('--method', choices=('wheel', 'message', 'keys'), default='wheel')
    scroll.add_argument('--notches', type=int, default=3, help='wheel notches (or Down presses) per step')
    scroll.add_argument('--delay', type=float, default=1.0, help='seconds to wait before the first frame')
    scroll.add_argument('--settle', type=float, default=0.12, help='seconds between grabs while the view settles')
    scroll.add_argument('--max-frames', type=int, default=500)
    scroll.add_argument('--max-height', type=int, help='stop once the image is this many pixels tall')
    match = capture.add_argument_group('stitching')
    match.add_argument('--ignore-top', type=int, default=0, help='fixed rows at the top (a header), kept once')
    match.add_argument('--ignore-bottom', type=int, default=0, help='fixed rows at the bottom, kept once')
    match.add_argument('--min-overlap', type=int, help='fewest rows two frames must share (default: 1/4 view)')
    match.add_argument('--tolerance', type=float, default=0, help='allowed mean grey difference 0-255 (0: exact)')
    pdf = capture.add_argument_group('pdf')
    pdf.add_argument('--page-height', type=int, help='pixels per page (default: A4-shaped; 0: one page)')
    pdf.add_argument('--dpi', type=float, default=96)
    pdf.add_argument('--ocr', action='store_true', help='add a searchable text layer (needs Tesseract)')
    pdf.add_argument('--png', type=Path, help='also write the stitched image here')
    return capture


def _run_capture(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    import time  # noqa: PLC0415 -- a subcommand loads only what it runs

    import pymupdf  # noqa: PLC0415

    from document_lab.capture import desktop  # noqa: PLC0415
    from document_lab.capture.export import save_png, to_pdf  # noqa: PLC0415
    from document_lab.capture.session import Limits, Settling, Stitching, capture  # noqa: PLC0415

    if args.list:
        for window in desktop.windows():
            area = window.client()
            emit(f'{window.hwnd:#010x}  {area.width}x{area.height}  {window.title}')
        return 0
    if args.out is None:
        parser.error('the PDF to write is required')
    if args.window:
        target = desktop.find_window(args.window)
    elif args.hwnd is not None:
        target = desktop.window(args.hwnd)
    elif args.pick:
        emit('click the window to capture (Esc cancels)')
        target = desktop.pick_window()
    elif args.region:
        target = desktop.select_region()
    else:
        target = desktop.Region(*args.rect)
    source = desktop.Source(target, args.notches, args.method)
    time.sleep(args.delay)
    image = capture(
        source.grab,
        None if args.no_scroll else source.scroll,
        limits=Limits(args.max_frames, args.max_height),
        stitching=Stitching(args.ignore_top, args.ignore_bottom, args.min_overlap, args.tolerance),
        settling=Settling(interval=args.settle),
        aborted=desktop.escape_pressed,
    )
    if args.png:
        save_png(image, args.png)
    with pymupdf.open(to_pdf(image, args.out, page_height=args.page_height, dpi=args.dpi, ocr=args.ocr)) as doc:
        emit(f'{args.out}: {image.shape[1]}x{image.shape[0]} px on {doc.page_count} page(s)')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run its subcommand; the exit status."""
    parser = argparse.ArgumentParser(prog='document-lab', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    paper = sub.add_parser('paper', help='a paper PDF to per-section markdown and cropped figures')
    paper.add_argument('pdf', type=Path)
    paper.add_argument('out', type=Path, help='the directory to write the paper layout into')
    paper.add_argument('--tool', choices=('marker', 'pymupdf'), help='force a converter (default: best available)')
    capture = _capture_parser(sub)
    args = parser.parse_args(argv)
    if args.command == 'capture':
        return _run_capture(args, capture)

    from document_lab.paper.ingest import ingest  # noqa: PLC0415 -- a subcommand loads only what it runs

    done = ingest(args.pdf, args.out, tool=args.tool)
    emit(
        f'{args.out}: {done.sections} sections, {done.figures} figures, {done.tables} tables '
        f'({done.tool}); {len(done.appended)} appended paper(s)'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
