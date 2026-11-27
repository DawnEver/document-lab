"""The Windows desktop as a capture source (`document_lab.capture.screen.System`), and what ``capture-files`` drives.

Plain ``ctypes`` over user32/gdi32 -- no third-party package. The process is made per-monitor DPI aware on first use,
so every coordinate here is a physical pixel. A window is grabbed through ``PrintWindow`` (its client area, even
when covered by another window); a region is grabbed from the screen as it shows.

Scrolling (`screen.SCROLL_METHODS`): ``wheel`` moves the cursor over the target, sends real wheel input and puts the
cursor back; ``message`` posts ``WM_MOUSEWHEEL`` to the window under the target; ``keys`` focuses the window and
presses the Down arrow.

Beyond the interface, ``capture-files`` (Windows alone: its viewer is found by its child page view) drives windows
here: :func:`view`, :func:`maximize`, :func:`close`, :func:`press`, :func:`key`, the mouse (:func:`park_cursor`)
and whether the user is at the machine (:func:`idle_seconds`, :func:`in_front`).
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.capture.screen import SCROLL_METHODS, Region, Window, bounds

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    'FRONT_TRIES',
    'client',
    'close',
    'cursor',
    'grab',
    'idle_seconds',
    'in_front',
    'key',
    'maximize',
    'monitors',
    'move_cursor',
    'open_file',
    'park_cursor',
    'parking_spot',
    'press',
    'problem',
    'scroll',
    'view',
    'window',
    'windows',
]

_WHEEL_DELTA: Final = 120
_WM_MOUSEWHEEL: Final = 0x020A
_MOUSEEVENTF_WHEEL: Final = 0x0800
_KEYEVENTF_KEYUP: Final = 0x0002
_VK_DOWN: Final = 0x28
_VK_UP: Final = 0x26
_GA_ROOT: Final = 2
_WM_CLOSE: Final = 0x0010
_SW_MAXIMIZE: Final = 3
_SW_RESTORE: Final = 9
_SWP_NOZORDER: Final = 0x0004
FRONT_TRIES: Final = 5  # tries to bring a window to the front before it is refused
#: The named keys a chord (:func:`press`) may hold; a letter or digit is its own code.
_VK_NAMED: Final = {
    'ctrl': 0x11,
    'shift': 0x10,
    'alt': 0x12,
    'home': 0x24,
    'end': 0x23,
    'pageup': 0x21,
    'pagedown': 0x22,
    'up': 0x26,
    'down': 0x28,
    'left': 0x25,
    'right': 0x27,
    'enter': 0x0D,
    'escape': 0x1B,
    'tab': 0x09,
} | {f'f{n}': 0x6F + n for n in range(1, 13)}  # F1..F12
_PW_CLIENT_FULL: Final = 0x1 | 0x2  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
_SRCCOPY_CAPTUREBLT: Final = 0x00CC0020 | 0x40000000
_PER_MONITOR_AWARE_V2: Final = -4
_DWMWA_CLOAKED: Final = 14


def _require_windows() -> None:
    if sys.platform != 'win32':
        msg = f'document_lab.capture.win32 drives the Windows desktop; this is {sys.platform}'
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


class _LastInput(ctypes.Structure):
    _fields_ = (('cbSize', wintypes.UINT), ('dwTime', wintypes.DWORD))


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
                'GetLastInputInfo': (b, [ctypes.POINTER(_LastInput)]),
                'GetWindowThreadProcessId': (wintypes.DWORD, [h, ctypes.c_void_p]),
                'AttachThreadInput': (b, [wintypes.DWORD, wintypes.DWORD, b]),
                'GetDC': (h, [h]),
                'ReleaseDC': (i, [h, h]),
                'PrintWindow': (b, [h, h, u]),
                'EnumChildWindows': (b, [h, ctypes.WINFUNCTYPE(b, h, wintypes.LPARAM), wintypes.LPARAM]),
                'GetWindowRect': (b, [h, ctypes.POINTER(wintypes.RECT)]),
                'IsWindow': (b, [h]),
                'ShowWindow': (b, [h, i]),
                'BringWindowToTop': (b, [h]),
                'SetWindowPos': (b, [h, h, i, i, i, i, u]),
                'EnumDisplayMonitors': (b, [h, ctypes.c_void_p, ctypes.c_void_p, wintypes.LPARAM]),
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


def problem(*, sends_input: bool) -> str | None:  # noqa: ARG001 -- the interface's; Windows asks no permission
    """Why a capture cannot run here: never, on a Windows desktop (input is refused, if at all, per scroll)."""
    _require_windows()
    return None


def client(window: Window) -> Region:
    """The client area of ``window`` in screen pixels."""
    user32 = _dll('user32')
    rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
    if not (user32.GetClientRect(window.handle, ctypes.byref(rect)) and user32.ClientToScreen(window.handle, origin)):
        msg = f'window {window.handle:#x} ({window.title!r}) is gone'
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
    """Visible, titled, non-minimized, uncloaked top-level windows, front to back."""
    user32 = _dll('user32')
    found: list[Window] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.LPARAM)
    def visit(hwnd: int, _: int) -> bool:
        shown = user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and not _cloaked(hwnd)
        if shown and (title := _title(hwnd)):
            found.append(Window(hwnd, title))
        return True

    user32.EnumWindows(visit, 0)
    return [w for w in found if client(w).width and client(w).height]


def window(handle: int) -> Window:
    """The window with handle ``handle``."""
    client(Window(handle, ''))  # refuses a handle that names no window
    return Window(handle, _title(handle))


def view(window: Window, title: str) -> Region:
    """The screen rectangle of ``window``'s largest visible descendant titled ``title`` (a viewer's page view).

    Refused when it has none: the program's layout is not the one named.
    """
    user32 = _dll('user32')
    found: list[Region] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.LPARAM)
    def visit(hwnd: int, _: int) -> bool:
        rect = wintypes.RECT()
        if user32.IsWindowVisible(hwnd) and _title(hwnd) == title and user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            found.append(Region(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
        return True

    user32.EnumChildWindows(window.handle, visit, 0)
    if not found:
        msg = f'{window.title!r} holds no visible view titled {title!r}'
        raise LookupError(msg)
    return max(found, key=lambda r: r.width * r.height)


def _codes(chord: str) -> list[int]:
    """The virtual-key codes of ``chord``: keys ``+``-joined, held in order and let go in reverse."""
    codes = []
    for name in chord.lower().split('+'):
        code = _VK_NAMED.get(name) or (ord(name.upper()) if len(name) == 1 and name.isalnum() else None)
        if code is None:
            msg = f'{name!r} in {chord!r} is no key: one of {sorted(_VK_NAMED)}, a letter or a digit'
            raise ValueError(msg)
        codes.append(code)
    return codes


def _type(codes: list[int]) -> None:
    """Hold ``codes`` in order, let them go in reverse: what a press is."""
    downs = [_Input(1, _InputUnion(ki=_KeybdInput(code, 0, 0, 0, 0))) for code in codes]
    ups = [_Input(1, _InputUnion(ki=_KeybdInput(code, 0, _KEYEVENTF_KEYUP, 0, 0))) for code in reversed(codes)]
    _send(*downs, *ups)
    time.sleep(0.2)


def press(window: Window, chord: str) -> None:
    """Press ``chord`` in ``window``, brought to the front first.

    A chord is keys ``+``-joined, held in order and let go in reverse: ``ctrl+2``, ``home``; a key is a letter, a
    digit or one of the named keys.
    """
    codes = _codes(chord)
    _bring_to_front(window.handle)
    _type(codes)


def key(chord: str) -> None:
    """Press ``chord`` wherever the keyboard is, bringing nothing to the front.

    A dialog of a viewer's own -- Acrobat's missing-font notice -- holds the keyboard and answers its own button;
    the window behind it takes no key until it is answered.
    """
    _type(_codes(chord))


def monitors() -> list[Region]:
    """Each monitor's rectangle on the virtual screen, the primary first, then left to right."""
    user32 = _dll('user32')
    found: list[tuple[bool, Region]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    def visit(_monitor: int, _dc: int, rect: ctypes.POINTER(wintypes.RECT), _: int) -> bool:
        r = rect.contents
        found.append((r.left == 0 and r.top == 0, Region(r.left, r.top, r.right - r.left, r.bottom - r.top)))
        return True

    user32.EnumDisplayMonitors(None, None, visit, 0)
    return [region for _, region in sorted(found, key=lambda f: (not f[0], f[1].left, f[1].top))]


def maximize(window: Window, monitor: Region | None = None) -> None:
    """Maximize ``window`` on ``monitor`` (default: the primary), so every window shows its view at the same place.

    A program reopens where it last was; the window is restored and moved to the monitor's corner first, so it
    maximizes there.
    """
    user32 = _dll('user32')
    where = monitor or monitors()[0]
    user32.ShowWindow(window.handle, _SW_RESTORE)
    user32.SetWindowPos(window.handle, 0, where.left, where.top, 800, 600, _SWP_NOZORDER)
    user32.ShowWindow(window.handle, _SW_MAXIMIZE)
    time.sleep(0.5)


_PARK_INSET: Final = 40  # pixels a parked mouse keeps off a screen's edges: a hot corner sits in the last pixel


def parking_spot(screen: Region, window: Region) -> tuple[int, int] | None:
    """Where to park the mouse: an inset corner of ``screen`` that ``window`` does not cover, bottom ones first.

    A screen's top right corner is the close button of the window that is there -- a mouse parked on it closes the
    very window being captured -- so the bottom corners come first, and every corner is taken an inset from the
    screen's edge. None when the window covers the whole screen (nowhere to park).
    """
    corners = [
        (screen.left + _PARK_INSET, screen.top + screen.height - _PARK_INSET),
        (screen.left + screen.width - _PARK_INSET, screen.top + screen.height - _PARK_INSET),
        (screen.left + _PARK_INSET, screen.top + _PARK_INSET),
        (screen.left + screen.width - _PARK_INSET, screen.top + _PARK_INSET),
    ]
    return next(
        (
            (x, y)
            for x, y in corners
            if not (window.left <= x < window.left + window.width and window.top <= y < window.top + window.height)
        ),
        None,
    )


def cursor() -> tuple[int, int]:
    """Where the mouse is, in physical pixels."""
    point = wintypes.POINT()
    _dll('user32').GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def move_cursor(x: int, y: int) -> None:
    """Put the mouse at ``x, y``."""
    _dll('user32').SetCursorPos(x, y)


def park_cursor(window: Window) -> tuple[int, int] | None:
    """Move the mouse off ``window`` when it is on it, to :func:`parking_spot`, and give back where it was.

    A viewer floats its toolbar over a page while the mouse rests on it, and shows its hidden panes when it nears the
    top -- but a mouse already off the window is the user's, and is not touched: None says it was left alone (as does
    a window that covers the whole screen, with nowhere to park).
    """
    user32 = _dll('user32')
    was = cursor()
    rect = wintypes.RECT()
    if not user32.GetWindowRect(window.handle, ctypes.byref(rect)):
        return None
    covered = Region(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    on_it = covered.left <= was[0] < covered.left + covered.width
    on_it = on_it and covered.top <= was[1] < covered.top + covered.height
    if not on_it:
        return None
    spot = parking_spot(bounds(monitors()), covered)
    if spot is None:
        return None
    move_cursor(*spot)
    return was


def close(window: Window, timeout: float = 10.0) -> None:
    """Ask ``window`` to close and wait until it has; refused when it is still there after ``timeout`` seconds."""
    user32 = _dll('user32')
    user32.PostMessageW(window.handle, _WM_CLOSE, 0, 0)
    end = time.monotonic() + timeout
    while user32.IsWindow(window.handle):
        if time.monotonic() > end:
            msg = f'{window.title!r} did not close within {timeout:g} s'
            raise OSError(msg)
        time.sleep(0.2)


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
    area = client(target) if isinstance(target, Window) else target
    screen = user32.GetDC(None)
    memdc = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, area.width, area.height)
    previous = gdi32.SelectObject(memdc, bitmap)
    try:
        if isinstance(target, Window):
            ok = user32.PrintWindow(target.handle, memdc, _PW_CLIENT_FULL)
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
        msg = (
            f'SendInput was blocked (error {ctypes.get_last_error()}): a UAC prompt, a program running as '
            'administrator in front, or the lock screen / screen saver -- clear it'
        )
        raise OSError(msg)


def _require_on_top(target: Window | Region, x: int, y: int) -> None:
    """Refuse a wheel that would land on another window: ``target`` was grabbed even if covered, the wheel is not."""
    if isinstance(target, Window):
        user32 = _dll('user32')
        under = user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), _GA_ROOT)
        if under != target.handle:
            msg = f'{_title(under)!r} covers the middle of {target.title!r}: bring it to the front (or method message)'
            raise OSError(msg)


def _bring_to_front(window: int) -> None:
    """Make ``window`` the keyboard target, or refuse: keys sent elsewhere would end the capture short, silently."""
    user32 = _dll('user32')
    for _ in range(FRONT_TRIES):
        front = user32.GetForegroundWindow()
        if front == window:
            return
        # Windows refuses the foreground to a background caller; sharing the input state of the front window and of
        # the target lifts that without sending a key that the front window could act on.
        mine = ctypes.WinDLL('kernel32').GetCurrentThreadId()
        threads = {user32.GetWindowThreadProcessId(hwnd, None) for hwnd in (front, window)} - {mine}
        attached = [thread for thread in threads if user32.AttachThreadInput(mine, thread, 1)]
        try:
            user32.BringWindowToTop(window)
            user32.SetForegroundWindow(window)
        finally:
            for thread in attached:
                user32.AttachThreadInput(mine, thread, 0)
        time.sleep(0.2)
    if user32.GetForegroundWindow() != window:
        msg = f'{_title(window)!r} could not be brought to the front for method keys: click it first (or method wheel)'
        raise OSError(msg)


def scroll(target: Window | Region, notches: int, method: str) -> None:
    """Scroll the view under ``target``'s center by ``notches`` wheel notches (or arrow presses): down, up if < 0."""
    user32 = _dll('user32')
    area = client(target) if isinstance(target, Window) else target
    x, y = area.center
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
            target.handle
            if isinstance(target, Window)
            else user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), _GA_ROOT)
        )
        _bring_to_front(window)
        key = _VK_DOWN if notches > 0 else _VK_UP
        down = _Input(1, _InputUnion(ki=_KeybdInput(key, 0, 0, 0, 0)))
        up = _Input(1, _InputUnion(ki=_KeybdInput(key, 0, _KEYEVENTF_KEYUP, 0, 0)))
        _send(*[event for _ in range(abs(notches)) for event in (down, up)])
    else:
        msg = f'scroll method {method!r} is not one of {SCROLL_METHODS}'
        raise ValueError(msg)


def open_file(path: Path) -> None:
    """Open ``path`` in the application Windows associates with it (a PDF in its reader)."""
    _require_windows()
    os.startfile(path)  # noqa: S606 -- a file this capture wrote, opened as the user asked (--open)


def idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input anywhere on this machine."""
    user32 = _dll('user32')
    info = _LastInput(ctypes.sizeof(_LastInput), 0)
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        msg = 'GetLastInputInfo was refused: the idle time of the machine is unknown'
        raise OSError(msg)
    ticks = ctypes.WinDLL('kernel32').GetTickCount()  # the same clock GetLastInputInfo counts on; kernel32, not user32
    return ((ticks - info.dwTime) & 0xFFFFFFFF) / 1000.0


def in_front(window: Window) -> bool:
    """Whether ``window``, or a window of its own, holds the keyboard: what the user types goes to it."""
    user32 = _dll('user32')
    front = user32.GetForegroundWindow()
    return bool(front) and user32.GetAncestor(front, _GA_ROOT) == window.handle
