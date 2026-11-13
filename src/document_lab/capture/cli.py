"""``document-lab capture``: its arguments and its run. Only the run loads the desktop, numpy and PyMuPDF."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from lab_commons.log import emit

from document_lab.capture.join import JOIN_TOLERANCE
from document_lab.capture.session import Pace, Settling
from document_lab.capture.stitch import Match

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from document_lab.capture.desktop import Hud, Region, Window
    from document_lab.capture.export import PdfWriter
    from document_lab.capture.pages import Rhythm
    from document_lab.capture.session import Capture, Lost
    from document_lab.capture.stitch import Stitcher

__all__ = ['add_merge_parser', 'add_parser', 'run', 'run_merge']


def _rect(text: str) -> tuple[int, int, int, int]:
    left, top, width, height = (int(v) for v in text.split(','))
    return left, top, width, height


def _page_height(text: str) -> int | str:
    return 'auto' if text == 'auto' else int(text)


def add_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the ``capture`` subcommand to ``sub`` and return its parser."""
    capture = sub.add_parser(
        'capture',
        help='a window or screen region, scrolled to its end and stitched, to a PDF (Windows)',
        description='With no target, mark one on the screen: click a window or drag a rectangle, then Enter or '
        'Confirm; the desktop stays usable until Start on the panel beside the area begins. '
        'Finish on the panel beside the area ends it and keeps what was captured.',
    )
    capture.add_argument(
        'out',
        type=Path,
        nargs='?',
        help='the PDF to write (default: the --resume PDF, else capture-<date>-<time>.pdf here)',
    )
    capture.add_argument(
        '--resume',
        type=Path,
        metavar='PDF',
        help='go on from the end of this earlier capture: bring the view to anywhere overlapping where it ended, '
        'then Start; the two are joined where they overlap (as capture-merge does)',
    )
    target = capture.add_mutually_exclusive_group()  # none: mark it on the screen (a click or a drag), then confirm
    target.add_argument('--window', metavar='TITLE', help='the one window whose title contains TITLE')
    target.add_argument('--hwnd', type=lambda v: int(v, 0), help='the window with this handle (see --list)')
    target.add_argument('--rect', type=_rect, metavar='L,T,W,H', help='a screen rectangle in physical pixels')
    target.add_argument('--list', action='store_true', help='print the capturable windows and exit')
    scroll = capture.add_argument_group(
        'scrolling', 'By default you scroll (wheel, scrollbar, keys) at your own pace and it follows; Finish ends it.'
    )
    scroll.add_argument('--no-scroll', action='store_true', help='one frame: what the target shows now')
    scroll.add_argument('--auto', action='store_true', help='it scrolls for you, to the end of the view')
    scroll.add_argument('--method', choices=('wheel', 'message', 'keys'), default='wheel', help='how --auto scrolls')
    scroll.add_argument(
        '--notches', type=int, default=Pace.first, help='--auto: wheel notches (or Down presses) of the first step'
    )
    scroll.add_argument(
        '--stride',
        type=float,
        default=Pace.stride,
        help='--auto: the share of the view each step scrolls, its notches measured and grown to it (0: --notches '
        'every step; nearer 1 is faster, and riskier where a notch varies)',
    )
    scroll.add_argument(
        '--stride-max',
        type=float,
        default=Pace.most,
        help='--auto: the share of the view a step grows to while steps keep placing; dropped back to --stride when '
        'one loses the view (at or below --stride: a fixed --stride)',
    )
    scroll.add_argument(
        '--streak',
        type=int,
        default=Pace.streak,
        help='--auto: steps placed in a row before the step grows and the settle gap shrinks (0: never)',
    )
    scroll.add_argument('--delay', type=float, default=1.0, help='--auto: seconds to wait before the first frame')
    scroll.add_argument(
        '--settle',
        type=float,
        default=Settling.interval,
        help='--auto: the least seconds between the two equal grabs that make a view settled; doubled, up to '
        '--settle-max, each time a view taken for settled was still rendering',
    )
    scroll.add_argument(
        '--settle-max',
        type=float,
        default=Settling.max_interval,
        help='--auto: the most seconds --settle grows to for a slow renderer',
    )
    scroll.add_argument(
        '--every', type=float, default=0.0, help='seconds between grabs while you scroll (0: as fast as it can match)'
    )
    scroll.add_argument('--max-frames', type=int, help='stop after this many frames (default: no cap)')
    scroll.add_argument(
        '--keep-lost',
        type=Path,
        metavar='DIR',
        help='save each view it could not place (and the last one it did) here, with the reason: for a report',
    )
    scroll.add_argument(
        '--end-wait',
        type=float,
        default=0.8,
        help='seconds a view that stopped moving gets to move again before it is the end (a feed loading more)',
    )
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
    match.add_argument(
        '--tolerance',
        type=float,
        default=Match.tolerance,
        help=f'mean grey levels the shared rows may differ by (default {Match.tolerance:g}; 0: exact)',
    )
    match.add_argument(
        '--join-tolerance',
        type=float,
        default=JOIN_TOLERANCE,
        help=f'--resume: as --tolerance, where this capture meets the earlier one (default {JOIN_TOLERANCE:g}: the '
        'document reopened renders the same text a little differently)',
    )
    match.add_argument(
        '--outliers',
        type=float,
        default=0.05,
        help='share of cells that may disagree at the true scroll (a scrollbar thumb, a caret); 0: none',
    )
    pdf = capture.add_argument_group('pdf')
    pdf.add_argument(
        '--page-height',
        type=_page_height,
        help="pixels per page (default: A4-shaped; 0: one page, held whole; 'auto': cut on the reader's page "
        'separators, A4-shaped if it finds none)',
    )
    pdf.add_argument('--dpi', type=float, default=96)
    pdf.add_argument(
        '--ocr',
        nargs='?',
        const='eng',
        metavar='LANGS',
        help="add a searchable text layer in Tesseract's LANGS (default eng; e.g. chi_sim+eng)",
    )
    pdf.add_argument('--png', type=Path, help='also write the stitched image here')
    pdf.add_argument('--open', action='store_true', help='open the PDF once it is written')
    pdf.add_argument(
        '--overwrite', action='store_true', help='replace an existing PDF or PNG (default: refuse, before anything)'
    )
    return capture


def add_merge_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the ``capture-merge`` subcommand to ``sub`` and return its parser."""
    merge = sub.add_parser(
        'capture-merge',
        help='captures of one document, in order, joined where each overlaps the one before',
        description='Each capture goes on from where its start is found in the end of the one before; the rows both '
        'show are kept once. Captures that share nothing, or of different widths, are refused.',
    )
    merge.add_argument('pdfs', type=Path, nargs='+', metavar='PDF', help='two or more captures, top to bottom')
    merge.add_argument('-o', '--out', type=Path, required=True, help='the joined PDF to write')
    merge.add_argument('--min-overlap', type=int, help='fewest rows two captures must share (default: 1/6 of 1000)')
    merge.add_argument(
        '--tolerance',
        type=float,
        default=JOIN_TOLERANCE,
        help=f'mean grey levels the shared rows may differ by (default {JOIN_TOLERANCE:g}: two captures are two '
        'renderings of the same text)',
    )
    merge.add_argument('--outliers', type=float, default=Match.outliers, help='as for capture')
    merge.add_argument('--overwrite', action='store_true', help='replace an existing output')
    return merge


def run_merge(args: argparse.Namespace) -> int:
    """Run ``document-lab capture-merge``; the exit status (1: two captures would not join, nothing written)."""
    import shutil  # noqa: PLC0415

    from document_lab.capture.export import check_name  # noqa: PLC0415
    from document_lab.capture.join import join  # noqa: PLC0415
    from document_lab.capture.stitch import Unstitchable  # noqa: PLC0415

    if len(args.pdfs) < 2:  # noqa: PLR2004
        emit('capture-merge needs two captures or more')
        return 1
    check_name(args.out)
    if args.out.exists() and not args.overwrite:
        msg = f'{args.out} exists: name another output, or add --overwrite to replace it'
        raise FileExistsError(msg)
    match = Match(min_overlap=args.min_overlap, tolerance=args.tolerance, outliers=args.outliers)
    work = args.out.with_name(args.out.name + '.merge.pdf')
    shutil.copyfile(args.pdfs[0], work)
    try:
        for pdf in args.pdfs[1:]:
            try:
                joined = join(work, pdf, work, match)
            except Unstitchable as refused:
                emit(f'{pdf} does not join the captures before it: {refused}')
                return 1
            emit(f'{pdf.name}: {joined.shared} px shared')
        work.replace(args.out)
    finally:
        work.unlink(missing_ok=True)
    emit(f'{args.out.resolve()}: {len(args.pdfs)} captures joined')
    return 0


def _target(args: argparse.Namespace) -> Window | Region:
    from document_lab.capture import desktop  # noqa: PLC0415 -- the run alone touches the desktop

    if args.window:
        return desktop.find_window(args.window)
    if args.hwnd is not None:
        return desktop.window(args.hwnd)
    if args.rect is not None:
        return desktop.Region(*args.rect)
    emit('click a window or drag a rectangle, then Enter or Confirm (Esc cancels); then Start on the panel')
    return desktop.select()


_BACK: Final = {
    'up': 'Lost: scrolled too far at once. Scroll back UP a little, to anything already in the picture below.',
    'down': 'Lost: scrolled back above what is held. Scroll DOWN to anything already in the picture below.',
}


def _say(text: str) -> None:
    sys.stderr.write(chr(13) + text[:100].ljust(100))
    sys.stderr.flush()


class _Shown:
    """The capture's callbacks, each also shown on the terminal line and the on-screen :class:`desktop.Hud`."""

    def __init__(self, hud: Hud, keep: Path | None = None) -> None:
        self.hud, self.keep, self.kept = hud, keep, 0

    def stopped(self) -> bool:
        self.hud.pump()  # once per grab: the panel stays drawn, and its Finish button is heard, in this one thread
        return self.hud.finished

    def paused(self) -> bool:
        return self.hud.paused  # the panel is pumped by stopped(), asked in the same loop

    def progress(self, stitcher: Stitcher) -> None:
        text = f'Capturing: {stitcher.height} px. Keep scrolling; click Finish when you are done.'
        _say(text)
        self.hud.status(text, ok=True)

    def lost(self, where: Lost) -> None:
        _say(_BACK[where.back])
        self.hud.status(_BACK[where.back], ok=False)
        if self.keep is not None and self.kept < _KEPT:
            from document_lab.capture.export import save_png  # noqa: PLC0415

            self.keep.mkdir(parents=True, exist_ok=True)
            self.kept += 1
            save_png(where.last, self.keep / f'lost-{self.kept:02d}-last.png')
            save_png(where.frame, self.keep / f'lost-{self.kept:02d}-view.png')
            (self.keep / f'lost-{self.kept:02d}.txt').write_text(
                f'{where.back}: {where.reason}' + chr(10), encoding='utf-8'
            )


_KEPT: Final = 10  # lost views --keep-lost saves at most

_SHORT: Final = {
    'aborted': 'finished',
    'max_frames': 'stopped at --max-frames',
    'max_height': 'stopped at --max-height',
}


_APART: Final = 3  # times the tolerance a best match may miss by before the frames are taken to share nothing


def _report(result: Capture, tolerance: float = Match.tolerance) -> int:
    """Say how the capture ended; the exit status (1: it stopped short of the end on its own)."""
    sys.stderr.write('\n')
    if result.stopped.startswith('blocked: '):
        emit(f'stopped after {result.frames} frames, short of the end: {result.stopped.removeprefix("blocked: ")}')
        emit('what was stitched is written. Windows dropped the scroll: a UAC prompt, a program running as')
        emit('administrator in front, or the lock screen / screen saver. Clear it, then capture the rest.')
        return 1
    if result.stopped.startswith('unstitchable: '):
        reason = result.stopped.removeprefix('unstitchable: ')
        emit(f'stopped after {result.frames} frames, short of the end: {reason}')
        if result.error is not None and result.error > _APART * max(tolerance, 1.0):
            emit('what was stitched is written; the frames before and after the scroll share nothing at all: the')
            emit('reader is probably in single-page / page-flip mode. Switch it to continuous scrolling and capture')
            emit('again (a larger --tolerance will not help)')
        else:
            emit('what was stitched is written; try a larger --tolerance, or scroll yourself (leave out --auto)')
        return 1
    if result.stuck:
        emit('the first scroll moved nothing, so only one screen was captured: what is under the middle of the area')
        emit('did not scroll. Scroll it yourself (leave out --auto), or try --method keys')
        return 1
    if result.stopped in _SHORT:
        emit(f'{_SHORT[result.stopped]} after {result.frames} frames')
    return 0


def _report_pages(page_height: int | str | None, rhythm: Rhythm | None) -> None:
    """Say where ``--page-height auto`` cut the pages."""
    if page_height != 'auto':
        return
    if rhythm is None:
        emit('no page separators found in the image: pages cut A4-shaped')
    else:
        emit(f'pages cut on the reader separators, {rhythm.pitch:.1f} px apart')


def _no_reason(_target: Window | Region, _area: Region) -> str | None:
    return None


def _started(
    target: Window | Region, refuse: Callable[[Window | Region, Region], str | None] = _no_reason
) -> tuple[Window | Region, Region, Hud, bool] | None:
    """The area set, its HUD and whether to scroll for the user (Auto scroll) once a start is clicked; None on Cancel.

    The desktop is free and nothing is grabbed until then; Adjust reopens the selection on the area. A start that
    ``refuse`` gives a reason for is not one: the reason is shown, and the panel waits again.
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    while True:
        area = target.client() if isinstance(target, desktop.Window) else target
        hud = desktop.Hud(area)
        answer = hud.ready()
        while answer in {'start', 'auto'} and (reason := refuse(target, area)):
            hud.status(reason, ok=False)
            answer = hud.ready()
        if answer in {'start', 'auto'}:
            hud.begin(answer)
            return target, area, hud, answer == 'auto'
        hud.close()
        if answer == 'cancel':
            return None
        target = desktop.select(target)


def _prepare(args: argparse.Namespace) -> str | None:
    """Name and check the output; why the capture's windows cannot open here, None when they can.

    An unnamed output is named for the time; a name Windows cannot hold (`export.check_name`)
    and one that exists (:func:`_refuse_existing`) are refused.
    """
    if args.resume is not None:
        if args.png is not None:
            msg = '--png would hold only what this capture adds: leave it out with --resume'
            raise ValueError(msg)
        if not args.resume.is_file():
            msg = f'{args.resume}: no such capture to resume'
            raise FileNotFoundError(msg)
    if args.out is None:
        args.out = args.resume or Path(f'capture-{datetime.now().astimezone():%Y%m%d-%H%M%S}.pdf')
    from document_lab.capture import desktop  # noqa: PLC0415
    from document_lab.capture.export import check_name  # noqa: PLC0415

    for path in (args.out, args.png):
        if path is not None:
            check_name(path)
    _refuse_existing(args)

    return desktop.tk_problem()  # the selection and the panel are tkinter windows


def _refuse_existing(args: argparse.Namespace) -> None:
    """Refuse an output that exists, unless ``--overwrite``: it may be the very document on screen."""
    for path in (args.out, args.png):
        if path is not None and path.exists() and not args.overwrite and path != args.resume:
            msg = f'{path} exists: name another output, or add --overwrite to replace it'
            raise FileExistsError(msg)


_RERENDER: Final = 3  # times the tolerance a resumed view may miss by and still be the same text rendered anew


def _wider(error: float) -> int:
    """A tolerance with room above ``error``: the next multiple of 5 past one and a half times it."""
    return 5 * -(-round(1.5 * error) // 5)


def _match(args: argparse.Namespace) -> Match:
    return Match(args.ignore_top, args.ignore_bottom, args.min_overlap, args.tolerance, args.outliers)


def _seam_match(args: argparse.Namespace) -> Match:
    """How this capture meets the earlier one (``--resume``): as within it, at ``--join-tolerance``."""
    return Match(args.ignore_top, args.ignore_bottom, args.min_overlap, args.join_tolerance, args.outliers)


def _resumable(args: argparse.Namespace) -> Callable[[Window | Region, Region], str | None]:
    """Why a start cannot go on from the end of ``--resume``: the area's width, or a view showing nothing of it."""
    if args.resume is None:
        return _no_reason

    def refuse(target: Window | Region, area: Region) -> str | None:
        import pymupdf  # noqa: PLC0415

        from document_lab.capture import desktop  # noqa: PLC0415
        from document_lab.capture.join import tail_rows, width  # noqa: PLC0415
        from document_lab.capture.stitch import NoOverlap, Unstitchable, locate  # noqa: PLC0415

        name = args.resume.name
        with pymupdf.open(args.resume) as doc:
            wide = width(doc)
        if area.width != wide:
            return f'The area is {area.width} px wide and {name} {wide} px: Adjust it to {wide} px, then Start.'
        frame = desktop.Source(target, args.notches, args.method).grab()
        try:
            locate(tail_rows(args.resume, 3 * len(frame)), frame, _seam_match(args))
        except NoOverlap as missed:
            if missed.error is not None and missed.error <= _RERENDER * max(args.join_tolerance, 1.0):
                return (
                    f'The nearest place in {name} differs by {missed.error:.1f} grey levels (--join-tolerance '
                    f'{args.join_tolerance:g}). If it is the same text, Cancel and run again with --join-tolerance '
                    f'{_wider(missed.error)}; else scroll to where it ends.'
                )
            return f'The view shows nothing of where {name} ends: scroll back to it, then Start.'
        except Unstitchable as unsure:
            return f'The view fits {name} in more than one place ({unsure}): scroll to something unique, then Start.'
        return None

    return refuse


def _joined(args: argparse.Namespace, written: Path) -> int:
    """Join what was captured to ``--resume`` into the output; 1 when they did not join (both kept, and said)."""
    from document_lab.capture.join import join  # noqa: PLC0415
    from document_lab.capture.stitch import Unstitchable  # noqa: PLC0415

    if not written.exists():
        emit('nothing was captured: the output is as it was')
        return 1
    try:
        joined = join(args.resume, written, args.out, _seam_match(args))
    except Unstitchable as refused:
        emit(f'what was captured did not join {args.resume.name} ({refused}); it is kept as {written}')
        return 1
    written.unlink()
    emit(f'{args.out.resolve()}: {args.resume.name} continued, {joined.shared} px shared')
    return 0


def run(args: argparse.Namespace) -> int:
    """Run ``document-lab capture`` with the parsed ``args``; the exit status."""
    from document_lab.capture import desktop  # noqa: PLC0415
    from document_lab.capture.export import PdfWriter, PngWriter  # noqa: PLC0415
    from document_lab.capture.session import Limits, capture, follow  # noqa: PLC0415

    if args.list:
        for window in desktop.windows():
            emit(f'{window.hwnd:#010x}  {window.client().width}x{window.client().height}  {window.title}')
        return 0
    if problem := _prepare(args):
        emit(problem)
        return 1
    target = _target(args)
    started = _started(target, _resumable(args))
    if started is None:
        emit('cancelled')
        return 1
    target, area, hud, auto = started
    source = desktop.Source(target, args.notches, args.method)
    width = area.width
    shown = _Shown(hud, args.keep_lost)
    written = args.out if args.resume is None else args.out.with_name(args.out.name + '.new.pdf')

    # The writers check page size and OCR before the capture starts, and keep what was written whatever stops it.
    with (
        contextlib.closing(hud),
        PdfWriter(written, width, page_height=args.page_height, dpi=args.dpi, ocr=args.ocr) as pdf,
        PngWriter(args.png, width) if args.png else contextlib.nullcontext() as png,
    ):

        def sink(rows: np.ndarray) -> None:
            pdf.write(rows)
            if png is not None:
                png.write(rows)
            hud.add(rows)

        limits = Limits(args.max_frames, args.max_height)
        match = _match(args)
        if not (args.auto or auto or args.no_scroll):
            emit('now scroll the area down at your own pace; click Finish on the panel when you are done')
            result = follow(
                source.grab,
                sink,
                aborted=shown.stopped,
                paused=shown.paused,
                interval=args.every,
                limits=limits,
                match=match,
                drop_overlays=args.drop_overlays,
                progress=shown.progress,
                lost=shown.lost,
            )
        else:
            time.sleep(args.delay)
            result = capture(
                source.grab,
                None if args.no_scroll else source.scroll,
                sink,
                limits=limits,
                match=match,
                settling=Settling(interval=args.settle, end_wait=args.end_wait, max_interval=args.settle_max),
                pace=Pace(args.notches, args.stride, args.stride_max, args.streak),
                drop_overlays=args.drop_overlays,
                aborted=shown.stopped,
                paused=shown.paused,
                progress=shown.progress,
                lost=shown.lost,
            )
    return _finished(args, result, pdf, written)


def _finished(args: argparse.Namespace, result: Capture, pdf: PdfWriter, written: Path) -> int:
    """Say how it ended and what was written -- joined to ``--resume`` -- and open it if asked; the exit status."""
    from document_lab.capture import desktop  # noqa: PLC0415

    status = _report(result, args.tolerance)
    _report_pages(args.page_height, pdf.rhythm)
    if args.resume is not None:
        status = _joined(args, written) or status
    else:
        emit(f'{args.out.resolve()}: {pdf.width}x{result.height} px on {pdf.pages} page(s)')
        if args.png:
            emit(f'{args.png.resolve()}: the same, as one image')
    if args.open and args.out.exists():
        desktop.open_file(args.out)
    return status
