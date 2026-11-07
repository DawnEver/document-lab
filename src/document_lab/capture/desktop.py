"""The Windows desktop as a capture source: windows and screen regions, their pixels, and their scrolling.

Plain ``ctypes`` over user32/gdi32 -- no third-party package. The process is made per-monitor DPI aware on first use,
so every coordinate here is a physical pixel. A :class:`Window` is grabbed through ``PrintWindow`` (its client area,
even when covered by another window); a :class:`Region` is grabbed from the screen as it shows.

Scrolling (:data:`SCROLL_METHODS`): ``wheel`` moves the cursor over the target, sends real wheel input and puts the
cursor back (works for nearly every application; the target must be on screen); ``message`` posts ``WM_MOUSEWHEEL``
to the window under the target (no cursor movement; some applications ignore it); ``keys`` focuses the window and
presses the Down arrow (for views that ignore the wheel).
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    'OVERLAY_TITLE',
    'SCROLL_METHODS',
    'Hud',
    'Region',
    'Source',
    'Window',
    'escape_pressed',
    'find_window',
    'grab',
    'scroll',
    'select',
    'virtual_screen',
    'window',
    'window_at',
    'windows',
]

SCROLL_METHODS: Final = ('wheel', 'message', 'keys')

_WHEEL_DELTA: Final = 120
_WM_MOUSEWHEEL: Final = 0x020A
_MOUSEEVENTF_WHEEL: Final = 0x0800
_KEYEVENTF_KEYUP: Final = 0x0002
_VK_DOWN: Final = 0x28
_VK_ESCAPE: Final = 0x1B
_VK_LBUTTON: Final = 0x01
_GA_ROOT: Final = 2
_PW_CLIENT_FULL: Final = 0x1 | 0x2  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
_SRCCOPY_CAPTUREBLT: Final = 0x00CC0020 | 0x40000000
_SM_VIRTUAL: Final = (76, 77, 78, 79)  # SM_X/Y/CX/CYVIRTUALSCREEN
_PER_MONITOR_AWARE_V2: Final = -4
_DWMWA_CLOAKED: Final = 14


def _require_windows() -> None:
    if sys.platform != 'win32':
        msg = f'document_lab.capture.desktop drives the Windows desktop; this is {sys.platform}'
        raise OSError(msg)


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ('dx', wintypes.LONG),
        ('dy', wintypes.LONG),
        ('mouseData', wintypes.DWORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.c_size_t),
    )


class _KeybdInput(ctypes.Structure):
    _fields_ = (
        ('wVk', wintypes.WORD),
        ('wScan', wintypes.WORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.c_size_t),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (('mi', _MouseInput), ('ki', _KeybdInput))


class _Input(ctypes.Structure):
    _fields_ = (('type', wintypes.DWORD), ('u', _InputUnion))


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = (
        ('biSize', wintypes.DWORD),
        ('biWidth', wintypes.LONG),
        ('biHeight', wintypes.LONG),
        ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD),
        ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD),
        ('biXPelsPerMeter', wintypes.LONG),
        ('biYPelsPerMeter', wintypes.LONG),
        ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD),
    )


_api: dict[str, ctypes.WinDLL] = {}


def _dll(name: str) -> ctypes.WinDLL:
    """user32 or gdi32 with handle-safe signatures, loaded (and the process made DPI aware) on first use."""
    _require_windows()
    if not _api:
        user32, gdi32 = ctypes.WinDLL('user32', use_last_error=True), ctypes.WinDLL('gdi32', use_last_error=True)
        h, b, i, u = wintypes.HANDLE, wintypes.BOOL, ctypes.c_int, wintypes.UINT
        sigs = {
            user32: {
                'SetProcessDpiAwarenessContext': (b, [h]),
                'EnumWindows': (b, [ctypes.WINFUNCTYPE(b, h, wintypes.LPARAM), wintypes.LPARAM]),
                'IsWindowVisible': (b, [h]),
                'IsIconic': (b, [h]),
                'GetWindowTextLengthW': (i, [h]),
                'GetWindowTextW': (i, [h, wintypes.LPWSTR, i]),
                'GetClientRect': (b, [h, ctypes.POINTER(wintypes.RECT)]),
                'ClientToScreen': (b, [h, ctypes.POINTER(wintypes.POINT)]),
                'WindowFromPoint': (h, [wintypes.POINT]),
                'GetAncestor': (h, [h, u]),
                'GetCursorPos': (b, [ctypes.POINTER(wintypes.POINT)]),
                'SetCursorPos': (b, [i, i]),
                'SendInput': (u, [u, ctypes.POINTER(_Input), i]),
                'PostMessageW': (b, [h, u, wintypes.WPARAM, wintypes.LPARAM]),
                'SetForegroundWindow': (b, [h]),
                'GetForegroundWindow': (h, []),
                'GetWindowThreadProcessId': (wintypes.DWORD, [h, ctypes.c_void_p]),
                'AttachThreadInput': (b, [wintypes.DWORD, wintypes.DWORD, b]),
                'GetAsyncKeyState': (wintypes.SHORT, [i]),
                'GetSystemMetrics': (i, [i]),
                'GetDC': (h, [h]),
                'ReleaseDC': (i, [h, h]),
                'PrintWindow': (b, [h, h, u]),
            },
            gdi32: {
                'CreateCompatibleDC': (h, [h]),
                'CreateCompatibleBitmap': (h, [h, i, i]),
                'SelectObject': (h, [h, h]),
                'BitBlt': (b, [h, i, i, i, i, h, i, i, wintypes.DWORD]),
                'GetDIBits': (i, [h, h, u, u, ctypes.c_void_p, ctypes.POINTER(_BitmapInfoHeader), u]),
                'DeleteObject': (b, [h]),
                'DeleteDC': (b, [h]),
            },
        }
        for lib, table in sigs.items():
            for fn, (restype, argtypes) in table.items():
                getattr(lib, fn).restype, getattr(lib, fn).argtypes = restype, argtypes
        user32.SetProcessDpiAwarenessContext(_PER_MONITOR_AWARE_V2)
        dwmapi = ctypes.WinDLL('dwmapi')
        dwmapi.DwmGetWindowAttribute.restype = ctypes.HRESULT
        dwmapi.DwmGetWindowAttribute.argtypes = [h, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        _api.update(user32=user32, gdi32=gdi32, dwmapi=dwmapi)
    return _api[name]


@dataclass(frozen=True)
class Region:
    """A rectangle of the virtual screen in physical pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def centre(self) -> tuple[int, int]:
        """The middle pixel."""
        return self.left + self.width // 2, self.top + self.height // 2


@dataclass(frozen=True)
class Window:
    """A top-level window: its handle and title. Its client area is read fresh on every use."""

    hwnd: int
    title: str

    def client(self) -> Region:
        """The client area in screen pixels."""
        user32 = _dll('user32')
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not (user32.GetClientRect(self.hwnd, ctypes.byref(rect)) and user32.ClientToScreen(self.hwnd, origin)):
            msg = f'window {self.hwnd:#x} ({self.title!r}) is gone'
            raise OSError(msg)
        return Region(origin.x, origin.y, rect.right, rect.bottom)


def _title(hwnd: int) -> str:
    user32 = _dll('user32')
    buffer = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _cloaked(hwnd: int) -> bool:
    """Hidden by the compositor though "visible" (a suspended app, another virtual desktop's window)."""
    flag = wintypes.DWORD()
    _dll('dwmapi').DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(flag), ctypes.sizeof(flag))
    return bool(flag.value)


def windows() -> list[Window]:
    """Visible, titled, non-minimised, uncloaked top-level windows, front to back."""
    user32 = _dll('user32')
    found: list[Window] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.LPARAM)
    def visit(hwnd: int, _: int) -> bool:
        shown = user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and not _cloaked(hwnd)
        if shown and (title := _title(hwnd)):
            found.append(Window(hwnd, title))
        return True

    user32.EnumWindows(visit, 0)
    return [w for w in found if w.client().width and w.client().height]


def find_window(title: str) -> Window:
    """The one window whose title contains ``title`` (case-insensitive); none or several is refused."""
    matches = [w for w in windows() if title.casefold() in w.title.casefold()]
    if len(matches) != 1:
        listing = '; '.join(repr(w.title) for w in matches) or 'none'
        msg = f'{len(matches)} windows match {title!r} ({listing}): name one exactly'
        raise LookupError(msg)
    return matches[0]


def window(hwnd: int) -> Window:
    """The window with handle ``hwnd``."""
    Window(hwnd, '').client()  # refuses a handle that names no window
    return Window(hwnd, _title(hwnd))


def virtual_screen() -> Region:
    """The rectangle covering every monitor."""
    user32 = _dll('user32')
    return Region(*(user32.GetSystemMetrics(m) for m in _SM_VIRTUAL))


def window_at(found: list[Window], x: int, y: int) -> Window | None:
    """The front-most of ``found`` (front to back, as :func:`windows` lists them) whose client area holds ``x, y``."""
    for candidate in found:
        area = candidate.client()
        if area.left <= x < area.left + area.width and area.top <= y < area.top + area.height:
            return candidate
    return None


START_LABEL: Final = 'Start (Enter)'
HINT: Final = (
    'Click a window or drag an area  |  drag its edges, corners or top handle to adjust; arrows nudge  |  '
    'Enter or Start: capture  |  Esc: cancel'
)
OVERLAY_TITLE: Final = 'document-lab capture selection'
_BUTTON: Final = (160, 40)  # the start button's size, beside the mark's lower right corner
_JITTER: Final = 4  # a press that moves no further than this is a click, not a drag
_GRIP: Final = 12  # pixels outside the box's edge that take hold of it (inside it, clicks pass to the app below)
_KNOB: Final = 5  # half the side of a handle drawn on an edge or a corner
_HANDLE: Final = 22  # half the width of the move handle above the top edge
_CURSORS: Final = {
    'n': 'sb_v_double_arrow',
    's': 'sb_v_double_arrow',
    'e': 'sb_h_double_arrow',
    'w': 'sb_h_double_arrow',
    'nw': 'size_nw_se',
    'se': 'size_nw_se',
    'ne': 'size_ne_sw',
    'sw': 'size_ne_sw',
    'move': 'fleur',
}
_DIM: Final = 0.45  # the overlay's opacity: how dark the screen outside the mark goes
_HOLE: Final = '#ff00fe'  # the colour Windows leaves fully transparent in the overlay: the mark is painted in it
_PANEL: Final = '#202020'


def _photo(image: np.ndarray) -> bytes:
    """``image`` (RGB uint8) as the PPM bytes a Tk photo reads."""
    height, width, _ = image.shape
    return f'P6 {width} {height} 255\n'.encode() + np.ascontiguousarray(image).tobytes()


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


class _Overlay:
    """A dark, translucent window over every monitor on which a target is marked, then confirmed.

    It dims the LIVE screen -- nothing is grabbed or drawn, so it opens at once -- and the mark is painted in
    :data:`_HOLE`, a colour Windows leaves fully transparent: the marked area shows clear, the rest dim. A click inside
    that hole reaches the window under it, so a hover only outlines a window, and a new mark is dragged from outside
    the old one. The hint and the start button are their own small opaque windows, readable above the dim.
    """

    def __init__(self, found: list[Window], screen: Region) -> None:
        import tkinter as tk  # noqa: PLC0415 -- only selection opens a window

        self.found, self.screen = found, screen
        self.drag: tuple[int, int] | None = None
        self.marked: Window | Region | None = None
        self.hovered: Window | None = None
        self.box: Region | None = None  # the mark as a rectangle, edited in place
        self.edit: tuple[str, Region] | None = None  # the part held and the box when the press took it
        self.confirmed: Window | Region | None = None
        self.root = tk.Tk()
        self.root.title(OVERLAY_TITLE)
        self.root.overrideredirect(boolean=True)
        self.root.configure(bg='black')
        self.root.attributes('-topmost', True, '-alpha', _DIM, '-transparentcolor', _HOLE)  # noqa: FBT003
        # A Tk offset of -N counts from the far edge: a monitor left of or above the primary needs +-N.
        self.root.geometry(f'{screen.width}x{screen.height}+{screen.left}+{screen.top}')
        self.canvas = tk.Canvas(self.root, cursor='crosshair', bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Motion>', self.hover)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.root.bind('<Return>', self.confirm)
        self.root.bind('<Escape>', lambda _: self.root.quit())
        for key, (dx, dy) in {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, -1), 'Down': (0, 1)}.items():
            self.root.bind(f'<{key}>', lambda _, dx=dx, dy=dy: self.nudge(dx, dy))
            self.root.bind(f'<Shift-{key}>', lambda _, dx=dx, dy=dy: self.nudge(10 * dx, 10 * dy))
            self.root.bind(f'<Control-{key}>', lambda _, dx=dx, dy=dy: self.nudge(dx, dy, grow=True))
        hint = self._window(tk)
        tk.Label(hint, text=HINT, fg='white', bg=_PANEL, font=('Segoe UI', 13), padx=14, pady=8).pack()
        hint.update_idletasks()
        primary = _dll('user32').GetSystemMetrics(0)
        hint.geometry(f'+{(primary - hint.winfo_reqwidth()) // 2}+12')
        self.panel = self._window(tk)
        self.start = tk.Button(
            self.panel,
            text=START_LABEL,
            command=self.confirm,
            fg='white',
            bg='#d93025',
            activebackground='#b3261e',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=16,
            pady=6,
        )
        self.start.pack()
        self.panel.withdraw()

    def _window(self, tk: object) -> object:
        window = tk.Toplevel(self.root, bg=_PANEL)
        window.overrideredirect(boolean=True)
        window.attributes('-topmost', True)  # noqa: FBT003 -- Tk's own spelling
        window.bind('<Return>', self.confirm)
        window.bind('<Escape>', lambda _: self.root.quit())
        return window

    def run(self) -> Window | Region | None:
        self.root.focus_force()
        self.root.mainloop()
        # Destroyed only once the loop is left: destroying it inside a handler frees the canvas while the same event
        # still runs its other bindings on it (Tcl: "alloc: invalid block").
        self.root.destroy()
        return self.confirmed

    def local(self, x: int, y: int) -> tuple[int, int]:
        return x - self.screen.left, y - self.screen.top

    def outline(self, area: Region, *, tag: str, width: int = 3) -> None:
        """A red outline just outside ``area`` (screen pixels)."""
        x0, y0 = self.local(area.left, area.top)
        x1, y1 = x0 + area.width, y0 + area.height
        edge = (x0 - width, y0 - width, x1 + width, y1 + width)
        self.canvas.create_rectangle(*edge, outline='red', width=width, tags=tag)

    def draw(self, box: Region) -> None:
        """Show ``box`` as the mark: a clear hole, a red frame, handles on its edges and corners, a move handle above.

        The shapes are made once and moved after (``coords``), so a drag redraws nothing.
        """
        x0, y0 = self.local(box.left, box.top)
        x1, y1 = x0 + box.width, y0 + box.height
        xm, ym, k = (x0 + x1) // 2, (y0 + y1) // 2, _KNOB
        knobs = [(x0, y0), (xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym)]
        shapes = [
            ('hole', (x0, y0, x1, y1)),
            ('frame', (x0 - 2, y0 - 2, x1 + 2, y1 + 2)),
            *((f'knob{i}', (x - k, y - k, x + k, y + k)) for i, (x, y) in enumerate(knobs)),
            ('grip', (xm - _HANDLE, y0 - _GRIP - 4, xm + _HANDLE, y0 - 6)),
        ]
        if not self.canvas.find_withtag('hole'):
            self.canvas.create_rectangle(0, 0, 0, 0, fill=_HOLE, width=0, tags=('mark', 'hole'))
            self.canvas.create_rectangle(0, 0, 0, 0, outline='red', width=3, tags=('mark', 'frame'))
            for i in range(len(knobs)):
                self.canvas.create_rectangle(0, 0, 0, 0, fill='white', outline='red', tags=('mark', f'knob{i}'))
            self.canvas.create_rectangle(0, 0, 0, 0, fill='red', outline='white', tags=('mark', 'grip'))
        for tag, corners in shapes:
            self.canvas.coords(tag, *corners)

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
        self.canvas.configure(cursor=_CURSORS.get(held, 'crosshair'))
        if self.marked is not None:
            return
        under = window_at(self.found, event.x_root, event.y_root)
        if under == self.hovered:
            return
        self.hovered = under
        self.canvas.delete('hover')
        if under is not None:
            self.outline(under.client(), tag='hover', width=2)

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
        self.box = target.client() if isinstance(target, Window) else target
        self.canvas.delete('hover')
        self.draw(self.box)
        area, screen = self.box, self.screen
        self.panel.deiconify()
        self.panel.update_idletasks()
        w, h = self.panel.winfo_reqwidth(), self.panel.winfo_reqheight()
        x = min(max(area.left + area.width - w, screen.left), screen.left + screen.width - w)
        below = area.top + area.height + 8
        y = below if below + h <= screen.top + screen.height else max(area.top - 8 - h, screen.top)
        self.panel.geometry(f'+{x}+{y}')
        self.panel.lift()

    def confirm(self, _: object = None) -> None:
        if self.marked is not None:
            self.confirmed = self.marked
            self.root.quit()


def select() -> Window | Region:
    """What the user marks over every monitor (the rest dimmed, the mark clear), confirmed before it is returned.

    A click marks the window under it, a drag marks a rectangle; marking again replaces the mark. Enter, or the start
    button by the mark, confirms it; Esc cancels (``KeyboardInterrupt``).
    """
    chosen = _Overlay(windows(), virtual_screen()).run()  # the windows first: the overlay is never a candidate
    if chosen is None:
        msg = 'selection cancelled'
        raise KeyboardInterrupt(msg)
    time.sleep(0.2)  # the overlay is repainted away before the first grab
    return chosen


def _pixels(memdc: int, bitmap: int, width: int, height: int) -> np.ndarray:
    header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
    buffer = (ctypes.c_ubyte * (width * height * 4))()
    if _dll('gdi32').GetDIBits(memdc, bitmap, 0, height, buffer, ctypes.byref(header), 0) != height:
        msg = 'GetDIBits read no pixels'
        raise OSError(msg)
    bgra = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(bgra[..., 2::-1])


def grab(target: Window | Region) -> np.ndarray:
    """The pixels of ``target`` as an ``(h, w, 3)`` RGB uint8 array."""
    user32, gdi32 = _dll('user32'), _dll('gdi32')
    area = target.client() if isinstance(target, Window) else target
    screen = user32.GetDC(None)
    memdc = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, area.width, area.height)
    previous = gdi32.SelectObject(memdc, bitmap)
    try:
        if isinstance(target, Window):
            ok = user32.PrintWindow(target.hwnd, memdc, _PW_CLIENT_FULL)
        else:
            ok = gdi32.BitBlt(memdc, 0, 0, area.width, area.height, screen, area.left, area.top, _SRCCOPY_CAPTUREBLT)
        if not ok:
            msg = f'could not read the pixels of {target}'
            raise OSError(msg)
        return _pixels(memdc, bitmap, area.width, area.height)
    finally:
        gdi32.SelectObject(memdc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(None, screen)


def _send(*inputs: _Input) -> None:
    array = (_Input * len(inputs))(*inputs)
    if _dll('user32').SendInput(len(inputs), array, ctypes.sizeof(_Input)) != len(inputs):
        msg = f'SendInput was blocked (error {ctypes.get_last_error()}): is a higher-integrity window in front?'
        raise OSError(msg)


def _require_on_top(target: Window | Region, x: int, y: int) -> None:
    """Refuse a wheel that would land on another window: ``target`` was grabbed even if covered, the wheel is not."""
    if isinstance(target, Window):
        user32 = _dll('user32')
        under = user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), _GA_ROOT)
        if under != target.hwnd:
            msg = f'{_title(under)!r} covers the middle of {target.title!r}: bring it to the front (or method message)'
            raise OSError(msg)


def _bring_to_front(window: int) -> None:
    """Make ``window`` the keyboard target, or refuse: keys sent elsewhere would end the capture short, silently."""
    user32 = _dll('user32')
    front = user32.GetForegroundWindow()
    if front != window:
        # Windows refuses the foreground to a background caller; sharing the front window's input state lifts that
        # without sending a key that the front window could act on.
        mine = ctypes.WinDLL('kernel32').GetCurrentThreadId()
        theirs = user32.GetWindowThreadProcessId(front, None)
        attached = theirs != mine and user32.AttachThreadInput(mine, theirs, 1)
        try:
            user32.SetForegroundWindow(window)
        finally:
            if attached:
                user32.AttachThreadInput(mine, theirs, 0)
        time.sleep(0.05)
    if user32.GetForegroundWindow() != window:
        msg = f'{_title(window)!r} could not be brought to the front for method keys: click it first (or method wheel)'
        raise OSError(msg)


def scroll(target: Window | Region, notches: int, method: str = 'wheel') -> None:
    """Scroll the view under ``target``'s centre down by ``notches`` wheel notches (or Down presses)."""
    user32 = _dll('user32')
    area = target.client() if isinstance(target, Window) else target
    x, y = area.centre
    if method == 'wheel':
        _require_on_top(target, x, y)
        cursor = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(cursor))
        user32.SetCursorPos(x, y)
        wheel = _Input(
            0,
            _InputUnion(mi=_MouseInput(0, 0, ctypes.c_uint32(-_WHEEL_DELTA * notches).value, _MOUSEEVENTF_WHEEL, 0, 0)),
        )
        _send(wheel)
        time.sleep(0.05)  # the wheel is routed to the window under the cursor when it is processed
        user32.SetCursorPos(cursor.x, cursor.y)
    elif method == 'message':
        under = user32.WindowFromPoint(wintypes.POINT(x, y))
        wparam = (ctypes.c_uint16(-_WHEEL_DELTA * notches).value) << 16
        user32.PostMessageW(under, _WM_MOUSEWHEEL, wparam, ((y & 0xFFFF) << 16) | (x & 0xFFFF))
    elif method == 'keys':
        window = (
            target.hwnd
            if isinstance(target, Window)
            else user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), _GA_ROOT)
        )
        _bring_to_front(window)
        down = _Input(1, _InputUnion(ki=_KeybdInput(_VK_DOWN, 0, 0, 0, 0)))
        up = _Input(1, _InputUnion(ki=_KeybdInput(_VK_DOWN, 0, _KEYEVENTF_KEYUP, 0, 0)))
        _send(*[event for _ in range(notches) for event in (down, up)])
    else:
        msg = f'scroll method {method!r} is not one of {SCROLL_METHODS}'
        raise ValueError(msg)


_EDGE: Final = 3  # the frame around a capture: its thickness, drawn just outside the area so it is never grabbed
_MINI: Final = (180, 360)  # the panel's miniature of the capture: width, and the height of the newest part it shows
OK, LOST = '#1aa34a', '#d93025'


class Hud:
    """What a capture shows while it runs, around the area and never inside it.

    A frame just outside the area (green: following, red: lost) and, beside it, a panel with a line of status and a
    live miniature of everything captured so far, its newest rows at the bottom.

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
        width, tall = _MINI[0] + 16, _MINI[1] + 110
        right = area.left + area.width + 2 * e
        x = right if right + width <= screen.left + screen.width else max(area.left - 2 * e - width, screen.left)
        y = min(max(area.top, screen.top), screen.top + screen.height - tall)
        self.panel = self._window(tk, x, y, (width, tall), colour='#202020')
        self.text = tk.Label(
            self.panel, fg='white', bg='#202020', font=('Segoe UI', 10), wraplength=width - 12, justify='left'
        )
        self.text.pack(fill='x', padx=6, pady=(6, 2))
        self.canvas = tk.Canvas(self.panel, width=_MINI[0], height=_MINI[1], bg='black', highlightthickness=0)
        self.canvas.pack(padx=8, pady=(0, 6))
        self.finished = False  # set by the Finish button: a deliberate click ends the capture, a stray key does not
        tk.Button(
            self.panel,
            text='Finish and save',
            command=self.finish,
            fg='white',
            bg=OK,
            activebackground='#137d39',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            pady=4,
        ).pack(fill='x', padx=8, pady=(0, 8))
        self.mini: list[np.ndarray] = []
        self.step = max(1, -(-area.width // _MINI[0]))  # every step-th pixel each way: the miniature's scale
        self.photo = tk.PhotoImage(master=self.root)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.status('Capturing. Scroll down at your own pace; click Finish when you are done.', ok=True)

    def _window(self, tk: object, x: int, y: int, size: tuple[int, int], colour: str = OK) -> object:
        window = tk.Toplevel(self.root, bg=colour)
        window.overrideredirect(boolean=True)
        window.attributes('-topmost', True)  # noqa: FBT003 -- Tk's own spelling
        window.geometry(f'{size[0]}x{size[1]}+{x}+{y}')  # +-N: Tk reads a bare -N from the far edge
        return window

    def status(self, text: str, *, ok: bool) -> None:
        """Say ``text`` on the panel; the frame turns green when ``ok``, else red."""
        self.text.configure(text=text)
        for edge in self._edges:
            edge.configure(bg=OK if ok else LOST)
        self.pump()

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

    def finish(self) -> None:
        """The Finish button: the capture ends after the frame in hand."""
        self.finished = True

    def pump(self) -> None:
        """Draw what changed; called once per grab."""
        self.root.update()

    def close(self) -> None:
        """Take the frame and the panel away."""
        self.root.destroy()


def escape_pressed() -> bool:
    """Is Esc held down right now (anywhere)?"""
    return bool(_dll('user32').GetAsyncKeyState(_VK_ESCAPE) & 0x8000)


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
        return grab(self.target)

    def scroll(self) -> None:
        """Scroll the target down one step."""
        scroll(self.target, self.notches, self.method)
