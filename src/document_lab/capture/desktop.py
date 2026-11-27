"""The desktop as a capture source, whatever the OS: the target, its selection overlay, and the panel beside it.

The OS half is one module per OS behind `document_lab.capture.screen.System` -- `win32` (Windows), `quartz`
(macOS) and `x11` (Linux under X11) -- and :func:`system` names the one for this OS. Everything here is drawn with
tkinter and asks that system alone: the overlay a target is marked on (:func:`select`), the frame and panel a
capture shows around it (:class:`Hud`), the batch panel of ``capture-files`` (:class:`Panel`), and the grab/scroll
pair a capture takes (:class:`Source`).
"""

from __future__ import annotations

import functools
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from document_lab.capture.screen import SCROLL_METHODS, Region, System, Window, bounds

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    'OVERLAY_TITLE',
    'Hud',
    'Panel',
    'Source',
    'area',
    'find_window',
    'recall',
    'remember',
    'select',
    'system',
    'tk_problem',
    'virtual_screen',
    'window_at',
]

#: The module filling `screen.System` on each ``sys.platform``.
SYSTEMS: Final = {'win32': 'win32', 'darwin': 'quartz', 'linux': 'x11'}


@functools.cache
def system() -> System:
    """The desktop of this OS (its module, imported on first use); refused on an OS no module drives."""
    name = SYSTEMS.get(sys.platform)
    if name is None:
        msg = f'capture drives the Windows, macOS and Linux (X11) desktops; this is {sys.platform}'
        raise OSError(msg)
    return importlib.import_module(f'document_lab.capture.{name}')


def area(target: Window | Region) -> Region:
    """``target``'s rectangle on the screen: a window's content area read fresh, a region as it is."""
    return system().client(target) if isinstance(target, Window) else target


def virtual_screen() -> Region:
    """The rectangle covering every monitor."""
    return bounds(system().monitors())


def find_window(title: str) -> Window:
    """The one window whose title contains ``title`` (case-insensitive); none or several is refused."""
    matches = [w for w in system().windows() if title.casefold() in w.title.casefold()]
    if len(matches) != 1:
        listing = '; '.join(repr(w.title) for w in matches) or 'none'
        msg = f'{len(matches)} windows match {title!r} ({listing}): name one exactly'
        raise LookupError(msg)
    return matches[0]


def window_at(found: list[Window], x: int, y: int) -> Window | None:
    """The front-most of ``found`` (front to back, as `System.windows` lists them) whose content area holds ``x, y``."""
    for candidate in found:
        shown = area(candidate)
        if shown.left <= x < shown.left + shown.width and shown.top <= y < shown.top + shown.height:
            return candidate
    return None


CONFIRM_LABEL: Final = 'Confirm area (Enter)'
HINT: Final = (
    'Click a window or drag an area (drag again, anywhere, to redo)  |  '
    'drag its edges, corners or top handle; arrows nudge  |  Enter or Confirm: set the area  |  Esc: cancel'
)
OVERLAY_TITLE: Final = 'document-lab capture selection'
_JITTER: Final = 4  # a press that moves no further than this is a click, not a drag
_GRIP: Final = 12  # pixels outside the box's edge that take hold of it (inside it, clicks pass to the app below)
_KNOB: Final = 5  # half the side of a handle drawn on an edge or a corner
_HANDLE: Final = 22  # half the width of the move handle above the top edge
_CURSORS: Final = {
    'n': 'sb_v_double_arrow',
    's': 'sb_v_double_arrow',
    'e': 'sb_h_double_arrow',
    'w': 'sb_h_double_arrow',
    'move': 'fleur',
}
#: The corner cursors: Windows' own diagonal arrows; elsewhere the X cursor font's corners, which every Tk has.
_CORNERS: Final = {
    'win32': {'nw': 'size_nw_se', 'se': 'size_nw_se', 'ne': 'size_ne_sw', 'sw': 'size_ne_sw'},
    'other': {
        'nw': 'top_left_corner',
        'se': 'bottom_right_corner',
        'ne': 'top_right_corner',
        'sw': 'bottom_left_corner',
    },
}
_DIM: Final = 0.45  # how dark the screen outside the mark goes: the live overlay's opacity, the picture's darkening
_HOLE: Final = '#ff00fe'  # the color Windows leaves fully transparent in the live overlay: the mark is painted in it
_GLASS: Final = 0.01  # the opacity of the pane over the mark: invisible, yet it takes the mouse (0 would not)
_PANEL: Final = '#202020'
_SHOWN_MS: Final = 150  # how long the overlay takes to be shown and active, before a remembered mark


def tk_problem(
    version: tuple[int, ...],
    prefix: str,
    base_prefix: str,
    environ: Mapping[str, str],
) -> str | None:
    """Why tkinter cannot start here, with the remedy; None when it can.

    Python 3.13.0 in a venv looks for Tcl under the venv rather than the base install (cpython #125235, fixed in
    3.13.1): ``Can't find a usable init.tcl`` -- unless the venv holds a ``tcl`` folder or ``TCL_LIBRARY`` names one.
    """
    if version != (3, 13, 0) or prefix == base_prefix or 'TCL_LIBRARY' in environ or (Path(prefix) / 'tcl').is_dir():
        return None
    return (
        'Python 3.13.0 in a virtual environment cannot start tkinter (cpython #125235: it looks for Tcl in the venv). '
        f'Use Python 3.13.1 or later, or copy {Path(base_prefix) / "tcl"} to {Path(prefix) / "tcl"}.'
    )


def _photo(image: np.ndarray) -> bytes:
    """``image`` (RGB uint8) as the PPM bytes a Tk photo reads."""
    height, width, _ = image.shape
    return f'P6 {width} {height} 255\n'.encode() + np.ascontiguousarray(image).tobytes()


_LABEL_GAP: Final = 26  # px above the mark its size label takes; nearer the screen's top, it goes inside


def size_label(box: Region, screen: Region) -> tuple[str, int, int]:
    """The mark's size as shown beside it -- ``'W x H'`` -- and the screen point of the label's top-left corner.

    Above the mark's top-left corner, as screenshot tools show it; inside, when the mark is at the screen's top.
    """
    top = box.top - _LABEL_GAP if box.top - _LABEL_GAP >= screen.top else box.top + 4
    return f'{box.width} x {box.height}', box.left, top


def _edited(held: str, box: Region, dx: int, dy: int) -> Region:
    """``box`` with the part ``held`` (an edge, a corner, or ``move``) dragged by ``dx, dy``; never under 8 pixels."""
    left, top, right, bottom = box.left, box.top, box.left + box.width, box.top + box.height
    if held == 'move':
        return Region(left + dx, top + dy, box.width, box.height)
    least = 8
    if 'w' in held:
        left = min(left + dx, right - least)
    if 'e' in held:
        right = max(right + dx, left + least)
    if 'n' in held:
        top = min(top + dy, bottom - least)
    if 's' in held:
        bottom = max(bottom + dy, top + least)
    return Region(left, top, right - left, bottom - top)


def _crop(picture: np.ndarray, screen: Region, box: Region) -> tuple[np.ndarray, int, int]:
    """The part of ``picture`` (``screen``'s pixels) under ``box``, and where it is; a box off the screen is cut."""
    x0, y0 = max(box.left, screen.left), max(box.top, screen.top)
    x1 = min(box.left + box.width, screen.left + screen.width)
    y1 = min(box.top + box.height, screen.top + screen.height)
    if x1 <= x0 or y1 <= y0:
        return picture[:1, :1], x0, y0
    return picture[y0 - screen.top : y1 - screen.top, x0 - screen.left : x1 - screen.left], x0, y0


class _Overlay:
    """A dark window over every monitor on which a target is marked, then confirmed.

    On Windows it dims the LIVE screen -- nothing is grabbed or drawn, so it opens at once -- and the mark is painted
    in :data:`_HOLE`, a color Windows leaves fully transparent: the marked area shows clear, the rest dim. Such a hole
    also lets clicks through, so a pane of glass -- a window over it at :data:`_GLASS` opacity, unseen -- takes them:
    a drag inside the mark marks again. Tk on macOS and X11 has no transparent color (nor, on X11 without a
    compositor, any translucency), so there the screen is grabbed once as the overlay opens and shown as a picture:
    darkened outside the mark, as it is inside it. The hint and the confirm button are their own small opaque windows.
    """

    def __init__(self, found: list[Window], screen: Region, marked: Window | Region | None = None) -> None:
        """Open over ``screen``, ``marked`` (an area set before, to adjust) already marked."""
        import tkinter as tk  # noqa: PLC0415 -- only selection opens a window

        self.found, self.screen = found, screen
        self.drag: tuple[int, int] | None = None
        self.marked: Window | Region | None = None
        self.hovered: Window | None = None
        self.box: Region | None = None  # the mark as a rectangle, edited in place
        self.cursor = 'crosshair'  # the canvas's, as last set
        self.edit: tuple[str, Region] | None = None  # the part held and the box when the press took it
        self.confirmed: Window | Region | None = None
        self.answered = False  # confirmed or cancelled: set by :meth:`answer`
        self.root = tk.Tk()
        self.root.withdraw()  # unmapped until it is set up: the picture is grabbed without it
        self.live = self.root.tk.call('tk', 'windowingsystem') == 'win32'
        self.cursors = _CURSORS | _CORNERS['win32' if self.live else 'other']
        self.picture = None if self.live else system().grab(screen)
        self.root.title(OVERLAY_TITLE)
        self.root.overrideredirect(boolean=True)
        self.root.configure(bg='black')
        self.root.attributes('-topmost', True)  # noqa: FBT003 -- Tk's own spelling
        if self.live:
            self.root.attributes('-alpha', _DIM, '-transparentcolor', _HOLE)
        # A Tk offset of -N counts from the far edge: a monitor left of or above the primary needs +-N.
        self.root.geometry(f'{screen.width}x{screen.height}+{screen.left}+{screen.top}')
        self.canvas = tk.Canvas(self.root, cursor='crosshair', bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        for sequence, handler in (
            ('<Motion>', self.hover),
            ('<ButtonPress-1>', self.press),
            ('<B1-Motion>', self.move),
            ('<ButtonRelease-1>', self.release),
        ):
            self.canvas.bind(sequence, handler)
        self.root.bind('<Return>', self.confirm)
        self.root.bind('<Escape>', lambda _: self.answer())
        for key, (dx, dy) in {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}.items():
            self.root.bind(f'<{key}>', lambda _, dx=dx, dy=dy: self.nudge(dx, dy))
            self.root.bind(f'<Shift-{key}>', lambda _, dx=dx, dy=dy: self.nudge(10 * dx, 10 * dy))
            self.root.bind(f'<Control-{key}>', lambda _, dx=dx, dy=dy: self.nudge(dx, dy, grow=True))
        if self.picture is not None:
            self._backdrop(tk, self.picture)
        hint = self._window(tk)
        tk.Label(hint, text=HINT, fg='white', bg=_PANEL, font=('Segoe UI', 13), padx=14, pady=8).pack()
        hint.update_idletasks()
        primary = system().monitors()[0]
        hint.geometry(f'+{primary.left + (primary.width - hint.winfo_reqwidth()) // 2}+{primary.top + 12}')
        self.size = self._window(tk)  # the mark's W x H beside it: its own opaque window, so it is not dimmed
        self.size_text = tk.Label(self.size, fg='white', bg=_PANEL, font=('Segoe UI', 10, 'bold'), padx=6, pady=2)
        self.size_text.pack()
        self.size.withdraw()
        self.panel = self._window(tk)
        self.confirm_button = tk.Button(
            self.panel,
            text=CONFIRM_LABEL,
            command=self.confirm,
            fg='white',
            bg='#d93025',
            activebackground='#b3261e',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=16,
            pady=6,
        )
        self.confirm_button.pack()
        self.panel.withdraw()
        self.glass = self._glass(tk) if self.live else None
        self.root.deiconify()
        self.premarked = marked  # marked in run(), once the overlay is up: marked now, it would cover the button

    def _backdrop(self, tk: object, picture: np.ndarray) -> None:
        """The screen's picture, darkened, under the whole canvas; and the photo the mark shows it undimmed in."""
        self.backdrop = tk.PhotoImage(
            master=self.root, data=_photo((picture * (1 - _DIM)).astype(np.uint8)), format='ppm'
        )
        self.clear = tk.PhotoImage(master=self.root)
        self.canvas.create_image(0, 0, image=self.backdrop, anchor='nw', tags='backdrop')

    def _glass(self, tk: object) -> object:
        """The pane over the live overlay's hole: unseen, it hears the mouse there."""
        glass = tk.Toplevel(self.root, bg='black')
        glass.overrideredirect(boolean=True)
        glass.attributes('-topmost', True, '-alpha', _GLASS)  # noqa: FBT003 -- Tk's own spelling
        glass.configure(cursor='crosshair')
        for sequence, handler in (
            ('<Motion>', self.hover),
            ('<ButtonPress-1>', self.press),
            ('<B1-Motion>', self.move),
            ('<ButtonRelease-1>', self.release),
        ):
            glass.bind(sequence, handler)
        glass.bind('<Return>', self.confirm)
        glass.bind('<Escape>', lambda _: self.answer())
        glass.withdraw()
        return glass

    def _window(self, tk: object) -> object:
        window = tk.Toplevel(self.root, bg=_PANEL)
        window.overrideredirect(boolean=True)
        window.attributes('-topmost', True)  # noqa: FBT003 -- Tk's own spelling
        window.bind('<Return>', self.confirm)
        window.bind('<Escape>', lambda _: self.answer())
        return window

    def run(self) -> Window | Region | None:
        self.root.focus_force()  # raises the overlay: before the mark, whose button and glass must end above it
        if self.picture is not None:
            self.root.update()  # mapped and placed: the picture goes where the screen is, wherever the OS put it
            self.canvas.coords('backdrop', *self.local(self.screen.left, self.screen.top))
        if self.premarked is not None:
            self.root.after(_SHOWN_MS, self.mark, self.premarked)  # once the overlay is up and active
        if not self.answered:  # an answer given while the overlay was set up stands (:meth:`answer`)
            self.root.mainloop()
        # Destroyed only once the loop is left: destroying it inside a handler frees the canvas while the same event
        # still runs its other bindings on it (Tcl: "alloc: invalid block").
        self.root.destroy()
        return self.confirmed

    def local(self, x: int, y: int) -> tuple[int, int]:
        """The canvas point of the screen point ``x, y``, from where the overlay truly is (an OS may move it)."""
        return x - self.canvas.winfo_rootx(), y - self.canvas.winfo_rooty()

    def outline(self, area: Region, *, tag: str, width: int = 3) -> None:
        """A red outline just outside ``area`` (screen pixels)."""
        x0, y0 = self.local(area.left, area.top)
        x1, y1 = x0 + area.width, y0 + area.height
        edge = (x0 - width, y0 - width, x1 + width, y1 + width)
        self.canvas.create_rectangle(*edge, outline='red', width=width, tags=tag)

    def draw(self, box: Region) -> None:
        """Show ``box`` as the mark: clear inside, a red frame, handles on its edges and corners, a move handle above.

        The shapes are made once and moved after (``coords``), so a drag redraws nothing -- but, over a picture, the
        clear part, which is cut from it anew.
        """
        x0, y0 = self.local(box.left, box.top)
        x1, y1 = x0 + box.width, y0 + box.height
        xm, ym, k = (x0 + x1) // 2, (y0 + y1) // 2, _KNOB
        knobs = [(x0, y0), (xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym)]
        shapes = [
            ('frame', (x0 - 2, y0 - 2, x1 + 2, y1 + 2)),
            *((f'knob{i}', (x - k, y - k, x + k, y + k)) for i, (x, y) in enumerate(knobs)),
            ('grip', (xm - _HANDLE, y0 - _GRIP - 4, xm + _HANDLE, y0 - 6)),
        ]
        if not self.canvas.find_withtag('frame'):
            if self.picture is None:
                self.canvas.create_rectangle(0, 0, 0, 0, fill=_HOLE, width=0, tags=('mark', 'hole'))
            else:
                self.canvas.create_image(0, 0, image=self.clear, anchor='nw', tags=('mark', 'hole'))
            self.canvas.create_rectangle(0, 0, 0, 0, outline='red', width=3, tags=('mark', 'frame'))
            for i in range(len(knobs)):
                self.canvas.create_rectangle(0, 0, 0, 0, fill='white', outline='red', tags=('mark', f'knob{i}'))
            self.canvas.create_rectangle(0, 0, 0, 0, fill='red', outline='white', tags=('mark', 'grip'))
        if self.picture is None:
            self.canvas.coords('hole', x0, y0, x1, y1)
        else:
            shown, left, top = _crop(self.picture, self.screen, box)
            self.clear.configure(width=shown.shape[1], height=len(shown))
            self.clear.configure(data=_photo(shown), format='ppm')
            self.canvas.coords('hole', *self.local(left, top))
        for tag, corners in shapes:
            self.canvas.coords(tag, *corners)
        if self.glass is not None:
            self.glass.geometry(f'{max(box.width, 1)}x{max(box.height, 1)}+{box.left}+{box.top}')
            self.glass.deiconify()
            self.glass.lift()
        text, x, y = size_label(box, self.screen)
        self.size_text.configure(text=text)
        self.size.geometry(f'+{x}+{y}')
        self.size.deiconify()
        self.size.lift()

    def grip(self, x: int, y: int) -> str | None:
        """Which part of the mark the screen point ``x, y`` takes hold of: an edge, a corner, ``move``, or None.

        Only just outside the box: inside it, clicks pass through the hole to the application below.
        """
        if self.box is None:
            return None
        b, g = self.box, _GRIP
        left, right, top, bottom = b.left, b.left + b.width, b.top, b.top + b.height
        if abs(x - (left + b.width // 2)) <= _HANDLE and top - g - 6 <= y < top - 4:
            return 'move'
        if not (left - g <= x <= right + g and top - g <= y <= bottom + g):
            return None
        vertical = 'n' if y < top else 's' if y > bottom else ''
        horizontal = 'w' if x < left else 'e' if x > right else ''
        return (vertical + horizontal) or None

    def hover(self, event: object) -> None:
        """Show what a press would do: a resize or move cursor on the mark; before one, outline the window below."""
        held = self.grip(event.x_root, event.y_root)
        cursor = self.cursors.get(held, 'crosshair')
        if cursor != self.cursor:  # setting it costs ~50 ms on a window this size, the same value or not
            self.cursor = cursor
            self.canvas.configure(cursor=cursor)
        if self.marked is not None:
            return
        under = window_at(self.found, event.x_root, event.y_root)
        if under == self.hovered:
            return
        self.hovered = under
        self.canvas.delete('hover')
        if under is not None:
            self.outline(area(under), tag='hover', width=2)

    def press(self, event: object) -> None:
        held = self.grip(event.x_root, event.y_root)
        self.drag = (event.x_root, event.y_root)
        self.edit = (held, self.box) if held else None

    def move(self, event: object) -> None:
        if self.drag is None:
            return
        self.panel.withdraw()
        x, y = event.x_root, event.y_root
        if self.edit is not None:
            self.box = _edited(*self.edit, x - self.drag[0], y - self.drag[1])
            self.draw(self.box)
            return
        self.canvas.delete('hover')
        (x0, x1), (y0, y1) = sorted((self.drag[0], x)), sorted((self.drag[1], y))
        if x1 - x0 > _JITTER or y1 - y0 > _JITTER:
            self.box = Region(x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))
            self.draw(self.box)

    def release(self, event: object) -> None:
        """End a press: an edit lands where the release is, a drag marks the rectangle, a click marks the window.

        All from the press and the release points alone: a motion event in between may never have come.
        """
        if self.drag is None:
            return
        (px, py), x, y = self.drag, event.x_root, event.y_root
        held, self.drag, self.edit = self.edit, None, None
        if held is not None:
            self.mark(_edited(held[0], held[1], x - px, y - py))
            return
        (x0, x1), (y0, y1) = sorted((px, x)), sorted((py, y))
        if x1 - x0 > _JITTER and y1 - y0 > _JITTER:
            self.mark(Region(x0, y0, x1 - x0, y1 - y0))
        elif (under := window_at(self.found, x0, y0)) is not None:
            self.mark(under)

    def nudge(self, dx: int, dy: int, *, grow: bool = False) -> None:
        """Move the mark by ``dx, dy`` pixels -- or, with ``grow``, move its right and bottom edges."""
        if self.box is not None:
            self.mark(_edited('se' if grow else 'move', self.box, dx, dy))

    def mark(self, target: Window | Region) -> None:
        self.marked = target
        self.box = area(target)
        self.canvas.delete('hover')
        self.draw(self.box)
        shown, screen = self.box, self.screen
        self.panel.deiconify()
        self.panel.update_idletasks()
        w, h = self.panel.winfo_reqwidth(), self.panel.winfo_reqheight()
        x = min(max(shown.left + shown.width - w, screen.left), screen.left + screen.width - w)
        below = shown.top + shown.height + 8
        y = below if below + h <= screen.top + screen.height else max(shown.top - 8 - h, screen.top)
        self.panel.geometry(f'+{x}+{y}')
        self.panel.lift()

    def confirm(self, _: object = None) -> None:
        if self.marked is not None:
            self.confirmed = self.marked
            self.answer()

    def answer(self) -> None:
        """End the selection -- confirmed, or (nothing confirmed) cancelled -- even before :meth:`run` enters its loop.

        Tk's ``quit`` only leaves a loop already running: one an ``update`` meets first is undone when ``mainloop``
        starts, which would then wait for nothing.
        """
        self.answered = True
        self.root.quit()


def select(memory: Path, marked: Window | Region | None = None) -> Window | Region:
    """What the user marks over every monitor (the rest dimmed, the mark clear), confirmed before it is returned.

    ``memory``: the file the last area confirmed is kept in (:func:`remember`). ``marked``: an area set before,
    shown marked to adjust; Esc then keeps it rather than cancelling. Without it the overlay
    opens on the last area confirmed (:func:`recall`), if it is still on the screen; Esc cancels as ever.

    A click marks the window under it, a drag marks a rectangle; marking again replaces the mark. Enter, or the
    confirm button by the mark, confirms it; Esc cancels (``KeyboardInterrupt``). The overlay then goes: the desktop
    is the user's again, the capture waiting for :meth:`Hud.ready`.
    """
    found, screen = system().windows(), virtual_screen()  # the windows first: the overlay is never a candidate
    chosen = _Overlay(found, screen, marked or recall(screen, memory)).run() or marked
    if chosen is None:
        msg = 'selection cancelled'
        raise KeyboardInterrupt(msg)
    remember(area(chosen), memory)
    time.sleep(0.2)  # the overlay is repainted away before the first grab
    return chosen


def remember(shown: Region, path: Path) -> None:
    """Keep ``shown`` in ``path`` as the last area confirmed, for the next selection to open on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.part')
    part.write_text(json.dumps(asdict(shown)), encoding='utf-8')
    part.replace(path)  # whole or not at all: two captures at once never leave half a file


def recall(screen: Region, path: Path) -> Region | None:
    """The last area confirmed, if one was and it still lies wholly on ``screen`` (monitors change); else None."""
    try:
        shown = Region(**json.loads(path.read_text(encoding='utf-8')))
    except OSError, ValueError, TypeError:  # none yet, or not one this code wrote: nothing to open on
        return None
    inside = (
        shown.width > 0
        and shown.height > 0
        and screen.left <= shown.left
        and screen.top <= shown.top
        and shown.left + shown.width <= screen.left + screen.width
        and shown.top + shown.height <= screen.top + screen.height
    )
    return shown if inside else None


_EDGE: Final = 3  # the frame around a capture: its thickness, drawn just outside the area so it is never grabbed
_MINI: Final = (180, 360)  # the panel's miniature of the capture: width, and the height of the newest part it shows
OK, LOST, WAIT = '#1aa34a', '#d93025', '#1a73e8'
Answer = Literal['start', 'auto', 'adjust', 'cancel']  # the waiting panel: you scroll, it scrolls, re-mark, quit


class Hud:
    """What a capture shows while it runs, around the area and never inside it.

    A frame just outside the area (blue: waiting for Start, green: following, red: lost) and, beside it, a panel with
    its buttons, a line of status and a live miniature of everything captured so far, its newest rows at the bottom.
    Nothing of it covers the desktop: before Start, and while it runs, every other window is the user's to use.

    It lives in the capture's own thread: :meth:`pump` (called once per grab) keeps it drawn, so nothing is shared.
    Nothing of it is ever inside the area, so it is never in the capture.
    """

    def __init__(self, area: Region) -> None:
        """Open the frame and the panel around ``area``."""
        import tkinter as tk  # noqa: PLC0415 -- only a running capture shows this

        screen = virtual_screen()
        self.root = tk.Tk()
        self.root.withdraw()
        self._edges = []
        e = _EDGE
        for x, y, w, h in (
            (area.left - e, area.top - e, area.width + 2 * e, e),
            (area.left - e, area.top + area.height, area.width + 2 * e, e),
            (area.left - e, area.top, e, area.height),
            (area.left + area.width, area.top, e, area.height),
        ):
            self._edges.append(self._window(tk, x, y, (w, h)))
        width = _MINI[0] + 16
        self.panel = self._window(tk, 0, 0, None, color='#202020')
        self.finished = False  # set by the Finish button: a deliberate click ends the capture, a stray key does not
        self.paused = False  # toggled by Pause / Resume: nothing is grabbed or scrolled while it is set
        self.answer: Answer | None = None  # what the panel answered before the capture
        starts = tk.Frame(self.panel, bg='#202020')  # first, at the top: always in view, whatever the text wraps to
        starts.pack(fill='x', padx=8, pady=(8, 4))
        self.others = tk.Frame(self.panel, bg='#202020')  # what only waiting needs: gone once the capture starts
        self.others.pack(fill='x', padx=8, pady=(0, 4))
        style = {'fg': 'white', 'font': ('Segoe UI', 10, 'bold'), 'relief': 'flat', 'pady': 5}
        self.button = tk.Button(
            starts,
            text='Start (I scroll)',
            command=lambda: self._answer('start'),
            bg=OK,
            activebackground='#137d39',
            **style,
        )
        self.button.pack(side='left', fill='x', expand=True)
        self.pause = tk.Button(starts, text='Pause', command=self.toggle_pause, bg=WAIT, **style)  # packed on begin
        self.auto = tk.Button(starts, text='Auto scroll', command=lambda: self._answer('auto'), bg=OK, **style)
        self.auto.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.adjust = tk.Button(self.others, text='Adjust', command=lambda: self._answer('adjust'), bg=WAIT, **style)
        self.adjust.pack(side='left', fill='x', expand=True)
        self.cancel = tk.Button(
            self.others, text='Cancel', command=lambda: self._answer('cancel'), bg='#5f6368', **style
        )
        self.cancel.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.text = tk.Label(
            self.panel, fg='white', bg='#202020', font=('Segoe UI', 10), wraplength=width - 12, justify='left'
        )
        self.text.pack(fill='x', padx=6, pady=(2, 4))
        self.canvas = tk.Canvas(self.panel, width=_MINI[0], height=_MINI[1], bg='black', highlightthickness=0)
        self.canvas.pack(padx=8, pady=(0, 8))
        self.panel.update_idletasks()  # the panel sizes itself to what it holds; only its place is set
        tall = self.panel.winfo_reqheight()
        right = area.left + area.width + 2 * e
        x = right if right + width <= screen.left + screen.width else max(area.left - 2 * e - width, screen.left)
        y = min(max(area.top, screen.top), screen.top + screen.height - tall)
        self.panel.geometry(f'+{x}+{y}')
        self.mini: list[np.ndarray] = []
        self.step = max(1, -(-area.width // _MINI[0]))  # every step-th pixel each way: the miniature's scale
        self.photo = tk.PhotoImage(master=self.root)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.status(
            'Area set. Use the desktop freely; bring the area to where to begin, then Start (you scroll) or Auto '
            'scroll (it scrolls). Adjust: change the area.',
            ok=None,
        )

    def _window(self, tk: object, x: int, y: int, size: tuple[int, int] | None, color: str = OK) -> object:
        """A borderless topmost window at ``x, y``: ``size`` pixels, or as large as what it holds when None."""
        window = tk.Toplevel(self.root, bg=color)
        window.overrideredirect(boolean=True)
        window.attributes('-topmost', True)  # noqa: FBT003 -- Tk's own spelling
        shape = '' if size is None else f'{size[0]}x{size[1]}'
        window.geometry(f'{shape}+{x}+{y}')  # +-N: Tk reads a bare -N from the far edge
        return window

    def status(self, text: str, *, ok: bool | None) -> None:
        """Say ``text`` on the panel; the frame turns green when ``ok``, red when not, blue when None (waiting)."""
        self.text.configure(text=text)
        for edge in self._edges:
            edge.configure(bg=WAIT if ok is None else OK if ok else LOST)
        self.pump()

    def ready(self) -> Answer:
        """Wait, the desktop free, for the panel's answer: Start (you scroll), Auto scroll, Adjust or Cancel.

        It may be asked again -- a start the caller refused, its reason shown with :meth:`status` -- until
        :meth:`begin`.
        """
        self.root.mainloop()  # left by any button: Tk sleeps meanwhile, nothing polls
        return self.answer

    def begin(self, answer: Answer) -> None:
        """The capture starts, as ``answer`` (``start`` or ``auto``) asked: the first button becomes Finish."""
        self.auto.pack_forget()
        self.others.pack_forget()
        self.button.configure(text='Finish and save', command=self.finish)
        self.pause.pack(side='left', fill='x', expand=True, padx=(6, 0))
        say = {
            'start': 'Capturing. Scroll down at your own pace; click Finish when you are done.',
            'auto': 'Scrolling for you to the end; click Finish to stop early.',
        }
        self.status(say[answer], ok=True)

    def _answer(self, answer: Answer) -> None:
        self.answer = answer
        self.root.quit()

    def add(self, rows: np.ndarray) -> None:
        """Grow the miniature by ``rows``, the capture's newest rows."""
        if not len(rows):
            return
        self.mini.append(np.ascontiguousarray(rows[:: self.step, :: self.step]))
        shown = np.concatenate(self.mini)[-_MINI[1] :]
        self.mini = [shown]  # only what is shown is kept: the miniature never grows with the capture
        self.photo.configure(width=shown.shape[1], height=len(shown))
        self.photo.configure(data=_photo(shown), format='ppm')
        self.canvas.coords(self.canvas.find_all()[0], 0, _MINI[1] - len(shown))

    def toggle_pause(self) -> None:
        """The Pause / Resume button: while paused the desktop is the user's -- scroll anywhere, use any window."""
        self.paused = not self.paused
        self.pause.configure(text='Resume' if self.paused else 'Pause')
        if self.paused:
            self.status(
                'Paused: nothing is captured. Scroll or use other windows freely; Resume picks up wherever the view '
                'overlaps what is captured.',
                ok=None,
            )
        else:
            self.status('Capturing again.', ok=True)

    def finish(self) -> None:
        """The Finish button: the capture ends after the frame in hand."""
        self.finished = True

    def pump(self) -> None:
        """Draw what changed; called once per grab."""
        self.root.update()

    def close(self) -> None:
        """Take the frame and the panel away."""
        self.root.destroy()


@dataclass(frozen=True)
class Source:
    """``target`` as the grab/scroll pair `document_lab.capture.session.capture` takes."""

    target: Window | Region
    notches: int = 3
    method: str = 'wheel'

    def __post_init__(self) -> None:
        """Refuse an unknown scroll method at once, not at the first scroll."""
        if self.method not in SCROLL_METHODS:
            msg = f'scroll method {self.method!r} is not one of {SCROLL_METHODS}'
            raise ValueError(msg)

    def grab(self) -> np.ndarray:
        """The target's pixels now."""
        return system().grab(self.target)

    def scroll(self, notches: int | None = None) -> None:
        """Scroll the target down ``notches`` notches (or Down presses); ``self.notches`` when None."""
        system().scroll(self.target, self.notches if notches is None else notches, self.method)


class Panel:
    """A small window beside the viewer while a batch runs: Pause / Resume, Stop, and what is being captured.

    Out of the viewer's monitor, always on top, in the capture's own thread: :meth:`pump` (called between pages, and
    while it waits) keeps it drawn, so nothing is shared and nothing of it can be captured.
    """

    def __init__(self, monitor: Region | None = None) -> None:
        """Open the panel at the bottom left of ``monitor`` (default: the primary, away from the viewer)."""
        import tkinter as tk  # noqa: PLC0415 -- only a running batch shows this

        self.root = tk.Tk()
        self.root.withdraw()
        self.window = tk.Toplevel(self.root)
        self.window.overrideredirect(True)  # noqa: FBT003 -- tk's own call signature
        self.window.attributes('-topmost', True)  # noqa: FBT003
        self.window.configure(bg=_PANEL)
        self.paused = False  # toggled by Pause / Resume: nothing is grabbed or keyed while it is set
        self.stopped = False  # set by Stop: the pages taken are kept and the run ends (files.Stopped)
        style = {'fg': 'white', 'font': ('Segoe UI', 10, 'bold'), 'relief': 'flat', 'pady': 4, 'padx': 10}
        row = tk.Frame(self.window, bg=_PANEL)
        row.pack(fill='x', padx=8, pady=(8, 4))
        self.pause = tk.Button(row, text='Pause', command=self.toggle_pause, **style)
        self.pause.configure(bg='#8a6d00', activebackground='#a88700')
        self.pause.pack(side='left', fill='x', expand=True)
        self.stop = tk.Button(row, text='Stop', command=self.halt, **style)
        self.stop.configure(bg='#8c2f2f', activebackground='#a63a3a')
        self.stop.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.text = tk.Label(self.window, fg='white', bg=_PANEL, font=('Segoe UI', 9), justify='left', wraplength=260)
        self.text.pack(fill='x', padx=10, pady=(0, 8))
        self.text.configure(text='capture-files')
        self.window.update_idletasks()
        self._place(monitor or system().monitors()[0])

    def _place(self, monitor: Region) -> None:
        height = self.window.winfo_reqheight()
        self.window.geometry(f'+{monitor.left + 24}+{monitor.top + monitor.height - height - 96}')

    def toggle_pause(self) -> None:
        """Hold the run, or let it go on."""
        self.paused = not self.paused
        self.pause.configure(text='Resume' if self.paused else 'Pause')
        self.pause.configure(bg='#1f6f3f' if self.paused else '#8a6d00')
        self.pump()

    def halt(self) -> None:
        """End the run after the page in hand."""
        self.stopped = True
        self.pump()

    def say(self, what: str) -> None:
        """Show what the run is doing."""
        self.text.configure(text=what)
        self.pump()

    def pump(self) -> None:
        """Draw what changed; called between pages and while waiting."""
        self.root.update()

    def close(self) -> None:
        """Take the panel away."""
        self.root.destroy()
