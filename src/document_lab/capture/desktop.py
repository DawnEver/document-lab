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
    'SCROLL_METHODS',
    'Region',
    'Source',
    'Window',
    'escape_pressed',
    'find_window',
    'grab',
    'pick_window',
    'scroll',
    'select_region',
    'virtual_screen',
    'window',
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


def pick_window(timeout: float = 30) -> Window:
    """The top-level window the user clicks next."""
    user32 = _dll('user32')
    deadline = time.monotonic() + timeout
    while user32.GetAsyncKeyState(_VK_LBUTTON) & 0x8000:  # a button already down is not the pick
        time.sleep(0.02)
    while time.monotonic() < deadline:
        if user32.GetAsyncKeyState(_VK_LBUTTON) & 0x8000:
            point = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(point))
            root = user32.GetAncestor(user32.WindowFromPoint(point), _GA_ROOT)
            return Window(root, _title(root))
        if escape_pressed():
            break
        time.sleep(0.02)
    msg = 'no window was clicked'
    raise TimeoutError(msg)


def virtual_screen() -> Region:
    """The rectangle covering every monitor."""
    user32 = _dll('user32')
    return Region(*(user32.GetSystemMetrics(m) for m in _SM_VIRTUAL))


def select_region() -> Region:
    """A rectangle the user drags on a dimmed overlay of every monitor (Esc cancels)."""
    import tkinter as tk  # noqa: PLC0415 -- only region selection opens a window

    screen = virtual_screen()
    root = tk.Tk()
    root.overrideredirect(boolean=True)
    root.attributes('-topmost', True, '-alpha', 0.3)  # noqa: FBT003 -- Tk's own spelling
    root.geometry(f'{screen.width}x{screen.height}{screen.left:+d}{screen.top:+d}')
    canvas = tk.Canvas(root, cursor='crosshair', bg='grey', highlightthickness=0)
    canvas.pack(fill='both', expand=True)
    drag: dict[str, int] = {}
    chosen: list[Region] = []

    def press(event: tk.Event) -> None:
        drag.update(x=event.x, y=event.y)
        canvas.delete('box')

    def move(event: tk.Event) -> None:
        canvas.delete('box')
        canvas.create_rectangle(drag['x'], drag['y'], event.x, event.y, outline='red', width=2, tags='box')

    def release(event: tk.Event) -> None:
        x0, x1 = sorted((drag['x'], event.x))
        y0, y1 = sorted((drag['y'], event.y))
        if x1 - x0 > 1 and y1 - y0 > 1:
            chosen.append(Region(screen.left + x0, screen.top + y0, x1 - x0, y1 - y0))
            root.destroy()

    canvas.bind('<ButtonPress-1>', press)
    canvas.bind('<B1-Motion>', move)
    canvas.bind('<ButtonRelease-1>', release)
    root.bind('<Escape>', lambda _: root.destroy())
    root.focus_force()
    root.mainloop()
    if not chosen:
        msg = 'region selection cancelled'
        raise KeyboardInterrupt(msg)
    return chosen[0]


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
