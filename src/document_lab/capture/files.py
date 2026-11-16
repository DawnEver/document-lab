"""PDFs captured page by page as their viewer shows them: ``document-lab capture-files`` (Windows).

A PDF only its viewer may open (a secured standard) is captured from the screen, as a person would: each is opened
in the viewer (:class:`Viewer`, Adobe Acrobat by default), the window it opens found (:func:`document_window`: the
one that was not there before, or titled with the file's name -- a secured standard is titled with its own, ``BS EN
IEC 60072-1:2022 (SECURED)``), maximised, cleared of its panes and toolbars
-- shown full screen (Acrobat's ``ctrl+l``: one page at a time, fitted to the monitor, on black) at its first page.
Then each page is taken as it shows -- the part it fills (:func:`page_box`) or the area the user marked -- and
the viewer paged on (``pagedown``): one image a page, written as one PDF page each, no stitching to lose its place
and no blank stretch to pass for the end. It stops after the PDF's own page count when the PDF opens here
(:func:`page_shapes`); a secured one that does not stops when paging on changes nothing. The run is the caller's: it
names what this module opens and closes.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from lab_commons.log import emit

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from document_lab.capture.desktop import Region, Window

    Gate = Callable[[str], None]  # between pages: shows what is in hand, waits while paused, raises Stopped

__all__ = [
    'ACROBAT',
    'MAX_PAGES',
    'PAGE_CONTRAST',
    'Stopped',
    'Viewer',
    'document_window',
    'listed',
    'open_in',
    'page_box',
    'page_shapes',
    'page_through',
    'same_page',
]

#: Where Adobe Acrobat (or Reader) is installed by default, the first present one taken.
ACROBAT: Final = (
    Path(r'C:\Program Files\Adobe\Acrobat DC\Acrobat\Acrobat.exe'),
    Path(r'C:\Program Files (x86)\Adobe\Acrobat Reader DC\Reader\AcroRd32.exe'),
    Path(r'C:\Program Files\Adobe\Acrobat Reader DC\Reader\AcroRd32.exe'),
)
#: Seconds a viewer takes to redraw after a key: one pressed mid-redraw is lost, a page grabbed mid-redraw is torn.
REDRAW: Final = 1.0
#: Times ``next`` is pressed again when the page did not turn (the viewer was drawing) before it is the end.
TURN_TRIES: Final = 2
#: A take whose height over width is off the page's by more than this share is taken again, this many times at most.
SHAPE_TOL: Final = 0.02
TAKE_TRIES: Final = 3
#: A page is taken when two grabs this many seconds apart are alike (a floating toolbar faded), or after SETTLE.
STILL: Final = 0.15
SETTLE: Final = 5.0
#: Seconds a viewer may take to show a page at all: a secured document is decrypted and laid out before the first.
PAGE_WAIT: Final = 30.0
#: Seconds between looks at the view while a viewer turns to the next page.
TURN_POLL: Final = 0.1
#: A cleared window's page view is at least this share of it wide; the keys are pressed this many times to get there.
CLEARED: Final = 0.95
CLEAR_TRIES: Final = 2
#: The page against its viewer's backdrop: corners this many pixels square are the backdrop, a pixel this many off
#: it is the page's, a column is the page's when this share of its rows is, and a page is at least this share wide.
PAGE_EDGE: Final = 3
PAGE_CONTRAST: Final = 3  # a backdrop near the page's own white (248 to 255) is still told from it
PAGE_ROWS: Final = 0.5
PAGE_SHARE: Final = 0.2
#: A run off the backdrop this short (a share of the view, or pixels) inside the page is a rule, not the backdrop.
PAGE_GAP: Final = 0.01
PAGE_GAP_PX: Final = 3
#: Mean grey levels two takes may differ by and be the same page: paging on past the last page changes nothing.
SAME_PAGE: Final = 1.0
#: The most pages taken of a PDF whose page count is not known here: a viewer that never stops is not followed.
MAX_PAGES: Final = 2000
#: Seconds without input that say the user has left the machine and the capture may take the foreground again.
USER_IDLE: Final = 2.0


class Stopped(Exception):
    """The run was told to stop (``--stop-file`` appeared): the pages taken so far are kept, and the run ends."""


class Park:
    """The mouse while a run goes on: off the viewer's page, and back where the user left it.

    A viewer floats its toolbar over a page while the mouse rests on it, so a page is taken with the mouse off the
    window (:func:`desktop.park_cursor`). Where the user last put it is remembered across the whole run -- not given
    back after each document only to be taken away again -- and given back when the run ends or waits for the user.
    A mouse the user takes back while a run goes on teaches the new place: it is where it will be given back.
    """

    def __init__(self) -> None:
        """Remember where the mouse is now: the user's place until they put it somewhere else."""
        from document_lab.capture import desktop  # noqa: PLC0415

        self.where = desktop.cursor()
        self.at: tuple[int, int] | None = None  # where the run parked it; None when it is the user's again

    def go(self, window: Window) -> None:
        """Take the mouse off ``window`` when it is on it, keeping the user's place for :meth:`restore`."""
        from document_lab.capture import desktop  # noqa: PLC0415

        was = desktop.park_cursor(window)
        if was is not None:  # it was on the window: that is the user's place, and the mouse is parked now
            self.where, self.at = was, desktop.cursor()

    def restore(self) -> None:
        """Give the mouse back where the user left it, unless they have moved it since it was parked."""
        from document_lab.capture import desktop  # noqa: PLC0415

        if self.at is not None and desktop.cursor() == self.at:
            desktop.move_cursor(*self.where)
        self.at = None


@dataclass(frozen=True)
class Viewer:
    """A PDF viewer: how it opens a file, where it shows the pages, and the keys that clear it and page it.

    ``program`` with ``options`` opens a file in a window of its own; ``view`` titles the child window that shows
    the pages; the key ``chords``, pressed in order, show a document full screen (Acrobat's ``ctrl+l``: one page at a
    time, fitted to the monitor on a black backdrop, nothing else) at its first page -- ``ctrl+0`` (fit page) because
    a viewer remembers the zoom a document was last read at, and fit width cuts the page's head and foot off;
    ``next`` shows the next page.
    """

    program: Path
    options: tuple[str, ...] = ('/n',)
    view: str = 'AVPageView'
    chords: tuple[str, ...] = ('ctrl+l', 'ctrl+0', 'home')
    next: str = 'pagedown'
    close: str | None = 'ctrl+w'  # closing the document rather than its window: a window killed marks a crash

    @classmethod
    def acrobat(cls) -> Viewer:
        """Adobe Acrobat where it is installed; refused when it is nowhere it installs to."""
        program = next((p for p in ACROBAT if p.is_file()), None)
        if program is None:
            msg = f'no Adobe Acrobat at {", ".join(map(str, ACROBAT))}: name the viewer (--viewer)'
            raise FileNotFoundError(msg)
        return cls(program)


def listed(sources: Sequence[Path]) -> list[Path]:
    """The PDFs ``sources`` name, in order, each once.

    A source is a PDF itself, a directory's ``*.pdf`` (sorted), or a text file listing one path a line (blank lines
    and ``#`` comments aside; relative to the list). Refused when one names nothing that exists -- never skipped.
    """
    out: list[Path] = []
    for source in sources:
        if source.is_dir():
            found = sorted(source.glob('*.pdf'), key=lambda p: p.name.casefold())
        elif source.suffix.lower() == '.pdf' and source.is_file():
            found = [source]
        elif source.is_file():
            lines = (line.strip() for line in source.read_text(encoding='utf-8').splitlines())
            found = [(source.parent / line) for line in lines if line and not line.startswith('#')]
            if missing := [str(p) for p in found if not p.is_file()]:
                msg = f'{source} lists PDFs that do not exist: {missing}'
                raise FileNotFoundError(msg)
        else:
            msg = f'{source} is no PDF, directory of PDFs or list of them'
            raise FileNotFoundError(msg)
        out += [p.resolve() for p in found if p.resolve() not in out]
    return out


def page_shapes(pdf: Path) -> list[float] | None:
    """Each page's height over its width, when ``pdf`` opens here; None when only its viewer may open it (secured)."""
    import pymupdf  # noqa: PLC0415

    try:
        with pymupdf.open(pdf) as doc:
            return None if doc.needs_pass else [page.rect.height / page.rect.width for page in doc]
    except (RuntimeError, ValueError):
        return None


def page_box(pixels: np.ndarray) -> tuple[int, int, int, int]:
    """``(top, bottom, left, right)``: the part of a viewer's view (rows of RGB) its page fills, no backdrop round it.

    The backdrop is the colour of the view's corners -- a page fitted whole leaves them bare, beside it or above and
    below it; a pixel differing from it by :data:`PAGE_CONTRAST` is the page's, a column (then a row, within those
    columns) is when :data:`PAGE_ROWS` of it is, and the page is the widest run of those -- the view's whole width
    when the page fits it across (a portrait screen). Refused when no page stands apart (no backdrop, or a run under
    :data:`PAGE_SHARE` of the view), or when it runs off the top or the foot of the view but not both: a page fitted
    to the view's height fills it from top to foot, while one cut by a page break -- a ``pagedown`` that scrolled
    rather than paged -- reaches only one.
    """
    rgb = pixels[..., :3].astype(float)
    e = PAGE_EDGE
    corners = [rgb[:e, :e], rgb[:e, -e:], rgb[-e:, :e], rgb[-e:, -e:]]  # beside or above the page, as it fits the view
    backdrop = np.median(np.concatenate(corners).reshape(-1, 3), axis=0)
    differs = np.abs(rgb - backdrop).max(axis=2) > PAGE_CONTRAST
    left, right = _widest(differs.mean(axis=0) > PAGE_ROWS, 'columns')
    top, bottom = _widest(differs[:, left:right].mean(axis=1) > PAGE_ROWS, 'rows')
    if (top == 0) != (bottom == len(rgb)):  # cut by one edge: two pages across a page break, the gap in view
        msg = (
            f'the page runs off the view (rows {top}..{bottom} of {len(rgb)}): the viewer is not showing one whole page'
        )
        raise LookupError(msg)
    return top, bottom, left, right


def _widest(page: np.ndarray, what: str) -> tuple[int, int]:
    """The ``[start, end)`` of the widest run of True in ``page``; refused when it is no page (see :func:`page_box`)."""
    bounds = np.flatnonzero(np.diff(np.concatenate(([0], page.astype(np.int8), [0]))))
    runs: list[tuple[int, int]] = []
    for start, end in zip(bounds[::2].tolist(), bounds[1::2].tolist(), strict=True):
        if runs and start - runs[-1][1] <= max(PAGE_GAP_PX, PAGE_GAP * len(page)):
            runs[-1] = (runs[-1][0], end)  # a rule as dark as the backdrop (a table's) is the page's, not a gap
        else:
            runs.append((start, end))
    start, end = max(runs, key=lambda run: run[1] - run[0], default=(0, 0))
    if end - start < PAGE_SHARE * len(page):
        msg = f'no page stands apart from the backdrop ({what} {start}..{end} of {len(page)}): is a whole page in view?'
        raise LookupError(msg)
    return start, end


def same_page(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two grabs show the same thing: paging on past the last page, or a view holding still."""
    return a.shape == b.shape and float(np.abs(a.astype(float) - b.astype(float)).mean()) <= SAME_PAGE


def document_window(shown: Sequence[Window], before: Collection[int], name: str) -> Window:
    """The one window the viewer opened for ``name`` among ``shown`` (each holding the viewer's page view).

    A viewer may title a document by its own title instead of the file's name -- a secured standard opens as ``BS EN
    IEC 60072-1:2022 (SECURED)``, not ``BS EN IEC 60072-1_2022.pdf`` -- so the window that was not there before
    ``before`` (the handles shown before the viewer was started) is the document's too, and taken first. One titled
    with the file's name is, for a viewer that opens each file in the window it already has. None or several is
    refused, naming what the viewer shows.
    """
    found = [w for w in shown if w.hwnd not in before] or [w for w in shown if name.casefold() in w.title.casefold()]
    if len(found) == 1:
        return found[0]
    listing = '; '.join(repr(w.title) for w in found) or 'none'
    msg = (
        f"{len(found)} of the viewer's {len(shown)} window(s) show {name!r} ({listing}): one document in one window "
        'each, or none opened'
    )
    raise LookupError(msg)


def shut(window: Window, viewer: Viewer) -> None:
    """Close the document ``window`` shows, the way its viewer takes it, and its window with it.

    ``window``'s close button (a window message) is what Windows does when a viewer is ended from outside: Acrobat
    counts that as closed abruptly and opens the next document with a recovery bar over the first page -- which a
    capture would take for the page. The viewer's own close key (Acrobat's ``ctrl+w``) leaves it shut down cleanly.
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    if viewer.close is not None:
        with contextlib.suppress(LookupError, OSError, ValueError):
            desktop.press(window, viewer.close)
            end = time.monotonic() + REDRAW * 5
            while time.monotonic() < end and desktop.windows() and window.hwnd in {w.hwnd for w in desktop.windows()}:
                time.sleep(STILL)
    if window.hwnd in {w.hwnd for w in desktop.windows()}:
        desktop.close(window)


def open_in(viewer: Viewer, pdf: Path, timeout: float, monitor: Region | None = None) -> Window:
    """``pdf`` opened in ``viewer``, its window maximised and cleared, at its first page with the whole page in view.

    The window is the one that was not there before the viewer was started, or titled with the file's name
    (:func:`document_window`: a secured PDF is titled with its own). It is waited for up to ``timeout`` seconds (the
    viewer starting); windows it opened are closed again when it is not found. Maximised on ``monitor`` (default:
    the primary). Cleared is the page view filling the
    window (:data:`CLEARED`): a key the viewer missed (it was still drawing) is pressed again; still not, it is
    refused. A window this opened is closed again when it cannot be brought so far.
    """
    from document_lab.capture import desktop  # noqa: PLC0415 -- the run alone touches the desktop

    before = {w.hwnd for w in desktop.windows()}
    subprocess.Popen([str(viewer.program), *viewer.options, str(pdf)])
    window = _wait_for_document(viewer, pdf.stem, before, timeout)
    try:
        desktop.maximize(window, monitor)
        desktop.view(window, viewer.view)  # the page view is there: the document is open, not a dialog
        for _ in range(CLEAR_TRIES):
            for chord in viewer.chords:
                desktop.press(window, chord)
                time.sleep(REDRAW)
            if _cleared(window, viewer):
                break
        else:
            _not_cleared(window, viewer)
    except (OSError, LookupError, ValueError):
        desktop.close(window)
        raise
    return window


def _cleared(window: Window, viewer: Viewer) -> bool:
    """Whether ``window``'s page view fills it within :data:`SETTLE` seconds.

    A viewer takes a moment to apply a chord -- a big secured document longer -- and judging it after one
    :data:`REDRAW` reads a slow viewer as an uncleared one: the chord pressed again would take the full screen back
    off. So the width is read until it fills the window, or the viewer is given up on.
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    end = time.monotonic() + SETTLE
    while True:
        if desktop.view(window, viewer.view).width >= CLEARED * window.client().width:
            return True
        if time.monotonic() > end:
            return False
        time.sleep(STILL)


def _wait_for_document(viewer: Viewer, name: str, before: Collection[int], timeout: float) -> Window:
    """The window the viewer opened for ``name``, waited for up to ``timeout`` seconds (the viewer starting).

    Only windows holding the viewer's page view (:attr:`Viewer.view`) are the document's -- a dialog it puts up is
    not. One it opened and cannot settle on by ``timeout`` is closed, so a refused run leaves none behind.
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    end = time.monotonic() + timeout
    while True:
        shown = []
        for candidate in desktop.windows():
            with contextlib.suppress(LookupError):
                desktop.view(candidate, viewer.view)
                shown.append(candidate)
        try:
            return document_window(shown, before, name)
        except LookupError:
            if time.monotonic() > end:
                for window in shown:
                    if window.hwnd not in before:
                        with contextlib.suppress(OSError):
                            desktop.close(window)
                raise
        time.sleep(0.5)


def _settled(area: Region, seen: np.ndarray | None = None) -> np.ndarray:
    """``area`` once it holds still: two grabs :data:`STILL` seconds apart alike.

    The page a viewer was just told to turn to is drawn over a moment: the first frame after ``next`` can be a page
    half drawn, so it is taken only when a second look is the same. ``seen`` is a grab of ``area`` already made (the
    page turn's own, :func:`_turned`), taken as the first look rather than grabbed again; a viewer's floating toolbar
    fading in or out is caught by the same rule. After :data:`SETTLE` seconds the last grab is taken as it is.
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    end = time.monotonic() + SETTLE
    last = desktop.grab(area) if seen is None else seen
    while time.monotonic() < end:
        time.sleep(STILL)
        now = desktop.grab(area)
        if same_page(last, now):
            return now
        last = now
    return last


def _take(
    window: Window,
    viewer: Viewer,
    area: Region | None,
    shape: float | None,
    *,
    park: Park | None = None,
    seen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The page ``window`` shows and the view it was taken from: ``area``, else the part it fills of the view.

    A take of another shape than the PDF's page -- the page still being drawn, a viewer still turning to full screen
    -- is taken again; so is a view no page stands apart from yet (a viewer still laying its first page out, a bare
    backdrop), until :data:`PAGE_WAIT` seconds have passed, then refused. The view goes back with the page so the
    next page turn can be waited for as a change of it (:func:`_turned`).
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    end = time.monotonic() + PAGE_WAIT
    shown = np.zeros((1, 1, 3), dtype=np.uint8)
    view = shown
    tries = 0
    while tries < TAKE_TRIES:
        (park.go(window) if park is not None else desktop.park_cursor(window))
        view = _settled(area or desktop.view(window, viewer.view), seen)
        seen = None  # the turn's grab is spent: a later try looks twice
        shown = view
        if area is None:
            try:
                top, bottom, left, right = page_box(shown)
            except LookupError:  # no page in view yet: the viewer is still laying one out
                if time.monotonic() > end:
                    raise
                time.sleep(STILL)
                continue
            shown = shown[top:bottom, left:right]
        off = shape is not None and abs(len(shown) / shown.shape[1] - shape) > SHAPE_TOL * shape
        if area is not None or not off:
            return shown, view
        tries += 1
        time.sleep(SETTLE)
    msg = f'the page shows {shown.shape[1]}x{len(shown)} px, not the {shape:.3f} height to width the PDF states'
    raise LookupError(msg)


def _not_cleared(window: Window, viewer: Viewer) -> None:
    msg = f'{window.title!r}: its panes stay open after {viewer.chords}: the pages do not fill the window'
    raise LookupError(msg)


def page_through(
    window: Window,
    viewer: Viewer,
    out: Path,
    *,
    area: Region | None,
    shapes: list[float] | None,
    dpi: float,
    gate: Gate | None = None,
    park: Park | None = None,
) -> int:
    """Take each page ``window`` shows into ``out``, paging on to the last; how many were taken.

    A page is ``area`` when given (the user's mark), else the part it fills of the viewer's view. As many pages
    are taken when the PDF's own page ``shapes`` are known -- each take checked against its page's (:func:`_take`) --
    and a page that shows what the one before did, ``next`` pressed again :data:`TURN_TRIES` times, is refused (the
    viewer did not page on);
    else up to the page that paging on leaves unchanged (at most :data:`MAX_PAGES`), said so -- two blank pages in a
    row end it there.
    """
    pages = len(shapes) if shapes is not None else None
    return _through(window, viewer, out, area=area, shapes=shapes, pages=pages, dpi=dpi, gate=gate, park=park)


def _through(
    window: Window,
    viewer: Viewer,
    out: Path,
    *,
    area: Region | None,
    shapes: list[float] | None,
    pages: int | None,
    dpi: float,
    gate: Gate | None,
    park: Park | None,
) -> int:
    """The pages of :func:`page_through`'s run, the writer and the loop."""
    from document_lab.capture import desktop  # noqa: PLC0415
    from document_lab.capture.export import PageWriter  # noqa: PLC0415

    last, view, seen = None, None, None
    with PageWriter(out, dpi) as pdf:
        for number in range(pages or MAX_PAGES):
            if gate is not None:
                gate(f'{out.stem}: page {number + 1} of {pages or "?"}')
            shape = None if shapes is None else shapes[number]
            take, view = _take(window, viewer, area, shape, park=park, seen=seen)
            for _ in range(TURN_TRIES if last is not None else 0):  # a key the viewer missed mid-draw: press it again
                if not same_page(last, take):
                    break
                desktop.press(window, viewer.next)
                seen = _turned(window, viewer, view)
                take, view = _take(window, viewer, area, shape, park=park, seen=seen)
            if last is not None and same_page(last, take):
                if pages is None:  # paged past the last page: the end of a PDF whose count is not known here
                    emit(f'  stopped after page {number}: paging on changed nothing (its count is not known here)')
                    break
                msg = f'page {number + 1} of {pages} shows what page {number} did: the viewer did not page on'
                raise LookupError(msg)
            pdf.add(take)
            last = take
            seen = None
            if number + 1 < (pages or MAX_PAGES):
                _yield_to_user(window, gate, park)
                desktop.press(window, viewer.next)
                seen = _turned(window, viewer, view)
        return pdf.pages


def held(*, front: bool, idle: float) -> bool:
    """Whether the capture holds its page: another window in front, and the machine used within :data:`USER_IDLE`."""
    return not front and idle < USER_IDLE


def _yield_to_user(window: Window, gate: Gate | None, park: Park | None = None) -> None:
    """Hold the capture while the user is working in another window, and say so.

    Every key the capture sends takes the foreground for the viewer (:func:`desktop.press`), so a page turned while
    the user types takes their keystrokes and their window loses the front. While another window is in front and
    the machine has been used within :data:`USER_IDLE` seconds, the page is held; the user away from the keyboard,
    or back in the viewer, and the run goes on -- or ``stop`` is asked, and the run ends.
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    told = False
    while held(front=desktop.in_front(window), idle=desktop.idle_seconds()):
        if park is not None:
            park.restore()  # the mouse is the user's while they are at the machine
        if gate is not None:
            gate('held while you use the machine')
        if not told:
            emit('  held while you use the machine')
            told = True
        time.sleep(STILL)
    if told:
        emit('  going on')


def _turned(window: Window, viewer: Viewer, last: np.ndarray) -> np.ndarray | None:
    """The page view once it differs from ``last`` within :data:`REDRAW` seconds of the page turn, or None.

    A viewer draws the next page in well under :data:`REDRAW`; waiting the whole of it on every page is most of a
    capture's time given away. The view is looked at until it changes -- that grab is handed to :func:`_settled` as
    its first look, the page then being taken only once it holds still -- and None is said when it does not change
    (a key lost mid-draw, or the last page).
    """
    from document_lab.capture import desktop  # noqa: PLC0415

    end = time.monotonic() + REDRAW
    while time.monotonic() < end:
        time.sleep(TURN_POLL)
        seen = desktop.grab(desktop.view(window, viewer.view))
        if not same_page(last, seen):
            return seen
    return None
