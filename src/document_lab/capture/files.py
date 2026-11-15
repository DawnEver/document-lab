"""PDFs captured page by page as their viewer shows them: ``document-lab capture-files`` (Windows).

A PDF only its viewer may open (a secured standard) is captured from the screen, as a person would: each is opened
in the viewer (:class:`Viewer`, Adobe Acrobat by default), its window maximised, cleared of its panes and toolbars
(Acrobat's read mode) and brought to its first page with the whole page in view -- sharp enough at a screen's height.
Then each page is taken as it shows -- the part it fills (:func:`page_box`) or the area the user marked -- and
the viewer paged on (``pagedown``): one image a page, written as one PDF page each, no stitching to lose its place
and no blank stretch to pass for the end. It stops after the PDF's own page count when the PDF opens here
(:func:`page_count`); a secured one that does not stops when paging on changes nothing. The run is the caller's: it
names what this module opens and closes.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from lab_commons.log import emit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_lab.capture.desktop import Region, Window

__all__ = [
    'ACROBAT',
    'MAX_PAGES',
    'PAGE_CONTRAST',
    'Viewer',
    'listed',
    'open_in',
    'page_box',
    'page_count',
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
#: The page against its viewer's backdrop: the outermost columns are the backdrop, a pixel this many grey levels off
#: it is the page's, a column is the page's when this share of its rows is, and a page is at least this share wide.
PAGE_EDGE: Final = 3
PAGE_CONTRAST: Final = 8
PAGE_ROWS: Final = 0.5
PAGE_SHARE: Final = 0.2
#: Mean grey levels two takes may differ by and be the same page: paging on past the last page changes nothing.
SAME_PAGE: Final = 1.0
#: The most pages taken of a PDF whose page count is not known here: a viewer that never stops is not followed.
MAX_PAGES: Final = 2000


@dataclass(frozen=True)
class Viewer:
    """A PDF viewer: how it opens a file, where it shows the pages, and the keys that clear it and page it.

    ``program`` with ``options`` opens a file in a window of its own; ``view`` titles the child window that shows
    the pages; the key ``chords``, pressed in order, clear the window of everything but the pages (Acrobat's read
    mode, ``ctrl+h``: no tools pane, toolbars or assistant) and bring a document to its first page with the whole page
    in view (``ctrl+0``); ``next`` shows the next page.
    """

    program: Path
    options: tuple[str, ...] = ('/n',)
    view: str = 'AVPageView'
    chords: tuple[str, ...] = ('ctrl+h', 'ctrl+0', 'home')
    next: str = 'pagedown'

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


def page_count(pdf: Path) -> int | None:
    """How many pages ``pdf`` has, when it opens here; None when only its viewer may open it (secured)."""
    import pymupdf  # noqa: PLC0415

    try:
        with pymupdf.open(pdf) as doc:
            return None if doc.needs_pass else len(doc)
    except (RuntimeError, ValueError):
        return None


def page_box(pixels: np.ndarray) -> tuple[int, int, int, int]:
    """``(top, bottom, left, right)``: the part of a viewer's view (rows of RGB) its page fills, no backdrop round it.

    The backdrop is the colour of the outermost columns; a pixel differing from it by :data:`PAGE_CONTRAST` is the
    page's, a column (then a row, within those columns) is when :data:`PAGE_ROWS` of it is, and the page is the widest
    run of those. Refused when no page stands apart (no backdrop, or a run under :data:`PAGE_SHARE` of the view), or
    when it runs off the top or the foot of the view: a whole page, backdrop above and below, is all that is taken --
    never a part, or two halves across a page break (a ``pagedown`` that scrolled rather than paged).
    """
    rgb = pixels[..., :3].astype(float)
    edges = np.concatenate([rgb[:, :PAGE_EDGE], rgb[:, -PAGE_EDGE:]], axis=1).reshape(-1, 3)
    differs = np.abs(rgb - np.median(edges, axis=0)).max(axis=2) > PAGE_CONTRAST
    left, right = _widest(differs.mean(axis=0) > PAGE_ROWS, 'columns')
    top, bottom = _widest(differs[:, left:right].mean(axis=1) > PAGE_ROWS, 'rows')
    if top == 0 or bottom == len(rgb):  # the page runs off the view: a part of it, or two pages across a page break
        msg = (
            f'the page runs off the view (rows {top}..{bottom} of {len(rgb)}): the viewer is not showing one whole page'
        )
        raise LookupError(msg)
    return top, bottom, left, right


def _widest(page: np.ndarray, what: str) -> tuple[int, int]:
    """The ``[start, end)`` of the widest run of True in ``page``; refused when it is no page (see :func:`page_box`)."""
    bounds = np.flatnonzero(np.diff(np.concatenate(([0], page.astype(np.int8), [0]))))
    runs = list(zip(bounds[::2].tolist(), bounds[1::2].tolist(), strict=True))
    start, end = max(runs, key=lambda run: run[1] - run[0], default=(0, 0))
    if end - start < PAGE_SHARE * len(page) or (what == 'columns' and end - start == len(page)):
        msg = f'no page stands apart from the backdrop ({what} {start}..{end} of {len(page)}): is a whole page in view?'
        raise LookupError(msg)
    return start, end


def same_page(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two takes show the same page: paging on past the last page changes nothing."""
    return a.shape == b.shape and float(np.abs(a.astype(float) - b.astype(float)).mean()) <= SAME_PAGE


def open_in(viewer: Viewer, pdf: Path, timeout: float) -> Window:
    """``pdf`` opened in ``viewer``, its window maximised and cleared, at its first page with the whole page in view.

    The window is waited for up to ``timeout`` seconds (the viewer starting); one this opened is closed again when
    it cannot be brought so far.
    """
    from document_lab.capture import desktop  # noqa: PLC0415 -- the run alone touches the desktop

    subprocess.Popen([str(viewer.program), *viewer.options, str(pdf)])
    window = desktop.wait_for_window(pdf.name, timeout)
    try:
        desktop.maximize(window)
        desktop.view(window, viewer.view)  # the page view is there: the document is open, not a dialog
        for chord in viewer.chords:
            desktop.press(window, chord)
            time.sleep(REDRAW)
    except (OSError, LookupError, ValueError):
        desktop.close(window)
        raise
    return window


def page_through(
    window: Window, viewer: Viewer, out: Path, *, area: Region | None, pages: int | None, dpi: float
) -> int:
    """Take each page ``window`` shows into ``out``, paging on to the last; how many were taken.

    A page is ``area`` when given (the user's mark), else the part it fills of the viewer's view. ``pages`` pages
    are taken when known, and a page that shows what the one before did is refused (the viewer did not page on);
    else up to the page that paging on leaves unchanged (at most :data:`MAX_PAGES`), said so -- two blank pages in a
    row end it there.
    """
    from document_lab.capture import desktop  # noqa: PLC0415
    from document_lab.capture.export import PageWriter  # noqa: PLC0415

    last = None
    with PageWriter(out, dpi) as pdf:
        for number in range(pages or MAX_PAGES):
            if area is None:
                shown = desktop.grab(desktop.view(window, viewer.view))
                top, bottom, left, right = page_box(shown)
                take = shown[top:bottom, left:right]
            else:
                take = desktop.grab(area)
            if last is not None and same_page(last, take):
                if pages is None:  # paged past the last page: the end of a PDF whose count is not known here
                    emit(f'  stopped after page {number}: paging on changed nothing (its count is not known here)')
                    break
                msg = f'page {number + 1} of {pages} shows what page {number} did: the viewer did not page on'
                raise LookupError(msg)
            pdf.add(take)
            last = take
            if number + 1 < (pages or MAX_PAGES):
                desktop.press(window, viewer.next)
                time.sleep(REDRAW)
        return pdf.pages
