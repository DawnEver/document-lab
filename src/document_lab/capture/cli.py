"""``document-lab capture``: its arguments and its run. Only the run loads the desktop, numpy and PyMuPDF."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from lab_commons.log import emit

if TYPE_CHECKING:
    import numpy as np

    from document_lab.capture.desktop import Region, Window
    from document_lab.capture.session import Capture
    from document_lab.capture.stitch import Stitcher

__all__ = ['add_parser', 'run']


def _rect(text: str) -> tuple[int, int, int, int]:
    left, top, width, height = (int(v) for v in text.split(','))
    return left, top, width, height


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the ``capture`` subcommand to ``sub`` and return its parser."""
    capture = sub.add_parser(
        'capture',
        help='a window or screen region, scrolled to its end and stitched, to a PDF (Windows)',
        description='With no target, mark one on the screen: click a window or drag a rectangle, then Enter or Start. '
        'Esc stops scrolling and keeps what was captured.',
    )
    capture.add_argument('out', type=Path, nargs='?', help='the PDF to write (default: capture-<date>-<time>.pdf here)')
    target = capture.add_mutually_exclusive_group()  # none: mark it on the screen (a click or a drag), then confirm
    target.add_argument('--window', metavar='TITLE', help='the one window whose title contains TITLE')
    target.add_argument('--hwnd', type=lambda v: int(v, 0), help='the window with this handle (see --list)')
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
    match.add_argument(
        '--drop-overlays',
        action='store_true',
        help='paint out what stays put while the page scrolls (floating buttons, sticky bars) from where it recurs',
    )
    match.add_argument('--ignore-top', type=int, default=0, help='rows at the top never matched (changing chrome)')
    match.add_argument('--ignore-bottom', type=int, default=0, help='rows at the bottom never matched')
    match.add_argument('--min-overlap', type=int, help='fewest rows two frames must share (default: 1/4 view)')
    match.add_argument('--tolerance', type=float, default=0, help='grey levels a cell may differ by (0: exact)')
    match.add_argument(
        '--outliers',
        type=float,
        default=0.05,
        help='share of cells that may disagree at the true scroll (a scrollbar thumb, a caret); 0: none',
    )
    pdf = capture.add_argument_group('pdf')
    pdf.add_argument('--page-height', type=int, help='pixels per page (default: A4-shaped; 0: one page)')
    pdf.add_argument('--dpi', type=float, default=96)
    pdf.add_argument(
        '--ocr',
        nargs='?',
        const='eng',
        metavar='LANGS',
        help="add a searchable text layer in Tesseract's LANGS (default eng; e.g. chi_sim+eng)",
    )
    pdf.add_argument('--png', type=Path, help='also write the stitched image here')
    return capture


def _target(args: argparse.Namespace) -> Window | Region:
    from document_lab.capture import desktop  # noqa: PLC0415 -- the run alone touches the desktop

    if args.window:
        return desktop.find_window(args.window)
    if args.hwnd is not None:
        return desktop.window(args.hwnd)
    if args.rect is not None:
        return desktop.Region(*args.rect)
    emit('click a window or drag a rectangle, then Enter or Start (Esc cancels)')
    return desktop.select()


def _progress(stitcher: Stitcher) -> None:
    sys.stderr.write(f'\rcapturing: {stitcher.height} px tall (Esc stops and keeps it)')
    sys.stderr.flush()


_SHORT: Final = {
    'aborted': 'stopped by Esc',
    'max_frames': 'stopped at --max-frames',
    'max_height': 'stopped at --max-height',
}


def _report(result: Capture) -> int:
    """Say how the capture ended; the exit status (1: it stopped short of the end on its own)."""
    sys.stderr.write('\n')
    if result.stopped.startswith('unstitchable: '):
        reason = result.stopped.removeprefix('unstitchable: ')
        emit(f'stopped after {result.frames} frames, short of the end: {reason}')
        emit('what was stitched is written; try --tolerance 8 or fewer --notches')
        return 1
    if result.stopped in _SHORT:
        emit(f'{_SHORT[result.stopped]} after {result.frames} frames')
    return 0


def _write(image: np.ndarray, args: argparse.Namespace) -> None:
    import pymupdf  # noqa: PLC0415

    from document_lab.capture.export import save_png, to_pdf  # noqa: PLC0415

    if args.png:
        save_png(image, args.png)
    try:
        to_pdf(image, args.out, page_height=args.page_height, dpi=args.dpi, ocr=args.ocr)
    except (OSError, ValueError):
        kept = save_png(image, args.png or args.out.with_suffix('.png'))
        emit(f'the PDF was not written; the capture is kept in {kept}')
        raise
    with pymupdf.open(args.out) as doc:
        emit(f'{args.out.resolve()}: {image.shape[1]}x{image.shape[0]} px on {doc.page_count} page(s)')


def run(args: argparse.Namespace) -> int:
    """Run ``document-lab capture`` with the parsed ``args``; the exit status."""
    from document_lab.capture import desktop  # noqa: PLC0415
    from document_lab.capture.export import check_ocr, check_page  # noqa: PLC0415
    from document_lab.capture.session import Limits, Settling, capture  # noqa: PLC0415
    from document_lab.capture.stitch import Match  # noqa: PLC0415

    if args.list:
        for window in desktop.windows():
            area = window.client()
            emit(f'{window.hwnd:#010x}  {area.width}x{area.height}  {window.title}')
        return 0
    if args.out is None:
        args.out = Path(f'capture-{datetime.now().astimezone():%Y%m%d-%H%M%S}.pdf')
    target = _target(args)
    source = desktop.Source(target, args.notches, args.method)
    area = target.client() if isinstance(target, desktop.Window) else target
    if args.page_height != 0:  # refuse what cannot be written before the capture, not after it
        check_page(area.width, args.page_height, args.dpi)
    if args.ocr:
        check_ocr(args.ocr)
    time.sleep(args.delay)
    result = capture(
        source.grab,
        None if args.no_scroll else source.scroll,
        limits=Limits(args.max_frames, args.max_height),
        match=Match(args.ignore_top, args.ignore_bottom, args.min_overlap, args.tolerance, args.outliers),
        settling=Settling(interval=args.settle),
        drop_overlays=args.drop_overlays,
        aborted=desktop.escape_pressed,
        progress=_progress,
    )
    status = _report(result)
    _write(result.image, args)
    return status
