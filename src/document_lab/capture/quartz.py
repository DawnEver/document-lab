"""The macOS desktop as a capture source (`document_lab.capture.screen.System`): Quartz through plain ``ctypes``.

No third-party package, as on Windows: CoreGraphics lists the windows (``CGWindowListCopyWindowInfo``), reads their
pixels (``CGWindowListCreateImage``: a window's content even when covered, a region as the screen shows it) and posts
the wheel and the keys (``CGEvent``); the Accessibility API brings a window to the front for ``keys``.

Coordinates are Quartz points, the space Tk places its windows in; a grab is taken one pixel a point
(``kCGWindowImageNominalResolution``), so a grab of a region is the region's size on a Retina display too. A
window's content area is its frame less the title bar AppKit draws on a titled window: a program that draws into its
own title bar loses that strip from a window capture -- mark a region for it.

Two privacy permissions gate it, each refused by name with where to grant it (:func:`problem`): Screen Recording to
read other programs' windows, Accessibility to send them input (asked only when the capture scrolls for the user).
"""

from __future__ import annotations

import ctypes
import functools
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.capture.screen import SCROLL_METHODS, Region, Window

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['client', 'grab', 'monitors', 'open_file', 'problem', 'scroll', 'window', 'windows']

_FRAMEWORK: Final = '/System/Library/Frameworks/{0}.framework/{0}'
_ON_SCREEN_ONLY: Final = 1 << 0  # kCGWindowListOptionOnScreenOnly: front to back
_INCLUDING_WINDOW: Final = 1 << 3  # kCGWindowListOptionIncludingWindow
_EXCLUDE_DESKTOP: Final = 1 << 4  # kCGWindowListExcludeDesktopElements
_NOMINAL_RESOLUTION: Final = 1 << 4  # kCGWindowImageNominalResolution: one pixel a point
_FLOATING_LAYER: Final = 3  # NSFloatingWindowLevel: a window kept on top is a window still
_SINT64: Final = 4  # kCFNumberSInt64Type
_DOUBLE: Final = 13  # kCFNumberDoubleType
_UTF8: Final = 0x08000100  # kCFStringEncodingUTF8
_LINE: Final = 1  # kCGScrollEventUnitLine
_HID_TAP: Final = 0  # kCGHIDEventTap
_DOWN, _UP = 125, 126  # kVK_DownArrow, kVK_UpArrow
_TITLED: Final = 1  # NSWindowStyleMaskTitled
_ALPHA_MASK, _ORDER_MASK, _ORDER_32_LITTLE = 0x1F, 0x7000, 0x2000
_ALPHA_FIRST: Final = frozenset({2, 4, 6})  # premultiplied first, first, none-skip-first: A (or X) leads R, G, B
FRONT_TRIES: Final = 5  # tries to bring a window to the front before it is refused
_SCREEN_RECORDING: Final = (
    'macOS has not let this program record the screen: in System Settings > Privacy & Security > Screen & System '
    'Audio Recording (Screen Recording before macOS 15), allow the app that runs Python -- the terminal or the IDE '
    '-- then quit and reopen it'
)
_ACCESSIBILITY: Final = (
    'macOS has not let this program send input (the scroll, the keys): in System Settings > Privacy & Security > '
    'Accessibility, allow the app that runs Python -- the terminal or the IDE -- then quit and reopen it; or scroll '
    'yourself (leave out --auto)'
)


class _Point(ctypes.Structure):
    _fields_ = (('x', ctypes.c_double), ('y', ctypes.c_double))


class _Size(ctypes.Structure):
    _fields_ = (('width', ctypes.c_double), ('height', ctypes.c_double))


class _Rect(ctypes.Structure):
    _fields_ = (('origin', _Point), ('size', _Size))


def _require_macos() -> None:
    if sys.platform != 'darwin':
        msg = f'document_lab.capture.quartz drives the macOS desktop; this is {sys.platform}'
        raise OSError(msg)


@functools.cache
def _lib(name: str) -> ctypes.CDLL:
    """A system framework with the signatures used here, loaded on first use."""
    _require_macos()
    lib = ctypes.cdll.LoadLibrary(_FRAMEWORK.format(name))
    p, b, n, u32, i32, sz = (
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_size_t,
    )
    table = {
        'CoreFoundation': {
            'CFRelease': (None, [p]),
            'CFArrayGetCount': (n, [p]),
            'CFArrayGetValueAtIndex': (p, [p, n]),
            'CFDictionaryGetValue': (p, [p, p]),
            'CFNumberGetValue': (b, [p, ctypes.c_int, p]),
            'CFStringGetLength': (n, [p]),
            'CFStringGetMaximumSizeForEncoding': (n, [n, u32]),
            'CFStringGetCString': (b, [p, ctypes.c_char_p, n, u32]),
            'CFStringCreateWithCString': (p, [p, ctypes.c_char_p, u32]),
            'CFDataGetLength': (n, [p]),
            'CFDataGetBytePtr': (p, [p]),
        },
        'CoreGraphics': {
            'CGWindowListCopyWindowInfo': (p, [u32, u32]),
            'CGWindowListCreateImage': (p, [_Rect, u32, u32, u32]),
            'CGRectMakeWithDictionaryRepresentation': (b, [p, ctypes.POINTER(_Rect)]),
            'CGImageGetWidth': (sz, [p]),
            'CGImageGetHeight': (sz, [p]),
            'CGImageGetBitsPerPixel': (sz, [p]),
            'CGImageGetBitsPerComponent': (sz, [p]),
            'CGImageGetBytesPerRow': (sz, [p]),
            'CGImageGetBitmapInfo': (u32, [p]),
            'CGImageGetDataProvider': (p, [p]),
            'CGDataProviderCopyData': (p, [p]),
            'CGImageRelease': (None, [p]),
            'CGGetActiveDisplayList': (i32, [u32, ctypes.POINTER(u32), ctypes.POINTER(u32)]),
            'CGDisplayBounds': (_Rect, [u32]),
            'CGMainDisplayID': (u32, []),
            'CGEventCreate': (p, [p]),
            'CGEventGetLocation': (_Point, [p]),
            'CGEventCreateScrollWheelEvent2': (p, [p, u32, u32, i32, i32, i32]),
            'CGEventCreateKeyboardEvent': (p, [p, ctypes.c_uint16, b]),
            'CGEventSetLocation': (None, [p, _Point]),
            'CGEventPost': (None, [u32, p]),
            'CGEventPostToPid': (None, [ctypes.c_int, p]),
            'CGWarpMouseCursorPosition': (i32, [_Point]),
            'CGPreflightScreenCaptureAccess': (b, []),
            'CGRequestScreenCaptureAccess': (b, []),
            'CGPreflightPostEventAccess': (b, []),
        },
        'ApplicationServices': {
            'AXIsProcessTrusted': (b, []),
            'AXUIElementCreateApplication': (p, [ctypes.c_int]),
            'AXUIElementCreateSystemWide': (p, []),
            'AXUIElementCopyAttributeValue': (i32, [p, p, ctypes.POINTER(p)]),
            'AXUIElementSetAttributeValue': (i32, [p, p, p]),
            'AXUIElementPerformAction': (i32, [p, p]),
            'AXUIElementGetPid': (i32, [p, ctypes.POINTER(ctypes.c_int)]),
        },
    }
    for fn, (restype, argtypes) in table.get(name, {}).items():
        getattr(lib, fn).restype, getattr(lib, fn).argtypes = restype, argtypes
    return lib


def _cf() -> ctypes.CDLL:
    return _lib('CoreFoundation')


def _cg() -> ctypes.CDLL:
    return _lib('CoreGraphics')


def _ax() -> ctypes.CDLL:
    return _lib('ApplicationServices')


@functools.cache
def _key(name: str) -> int:
    """The CFString constant ``name`` CoreGraphics exports (``kCGWindowNumber``...)."""
    return ctypes.c_void_p.in_dll(_cg(), name).value


@functools.cache
def _cfstr(text: str) -> int:
    """``text`` as a CFString, made once and kept (an Accessibility attribute's name)."""
    return _cf().CFStringCreateWithCString(None, text.encode(), _UTF8)


def _string(ref: int | None) -> str:
    if not ref:
        return ''
    size = _cf().CFStringGetMaximumSizeForEncoding(_cf().CFStringGetLength(ref), _UTF8) + 1
    buffer = ctypes.create_string_buffer(size)
    return buffer.value.decode() if _cf().CFStringGetCString(ref, buffer, size, _UTF8) else ''


def _number(ref: int | None, kind: int = _SINT64) -> float:
    value = ctypes.c_double() if kind == _DOUBLE else ctypes.c_int64()
    if ref:
        _cf().CFNumberGetValue(ref, kind, ctypes.byref(value))
    return value.value


@dataclass(frozen=True)
class _Info:
    """What the window server says of one window."""

    number: int
    pid: int
    layer: int
    alpha: float
    title: str
    owner: str
    bounds: Region

    def holds(self, x: int, y: int) -> bool:
        b = self.bounds
        return b.left <= x < b.left + b.width and b.top <= y < b.top + b.height


def _infos(option: int, number: int = 0) -> list[_Info]:
    """The windows ``option`` lists (front to back for the on-screen ones), relative to window ``number``."""
    cf, cg = _cf(), _cg()
    listed = cg.CGWindowListCopyWindowInfo(option, number)
    if not listed:
        return []
    try:
        found = []
        for i in range(cf.CFArrayGetCount(listed)):
            entry = cf.CFArrayGetValueAtIndex(listed, i)
            rect = _Rect()
            cg.CGRectMakeWithDictionaryRepresentation(cf.CFDictionaryGetValue(entry, _key('kCGWindowBounds')), rect)
            found.append(
                _Info(
                    int(_number(cf.CFDictionaryGetValue(entry, _key('kCGWindowNumber')))),
                    int(_number(cf.CFDictionaryGetValue(entry, _key('kCGWindowOwnerPID')))),
                    int(_number(cf.CFDictionaryGetValue(entry, _key('kCGWindowLayer')))),
                    _number(cf.CFDictionaryGetValue(entry, _key('kCGWindowAlpha')), _DOUBLE),
                    _string(cf.CFDictionaryGetValue(entry, _key('kCGWindowName'))),
                    _string(cf.CFDictionaryGetValue(entry, _key('kCGWindowOwnerName'))),
                    Region(round(rect.origin.x), round(rect.origin.y), round(rect.size.width), round(rect.size.height)),
                )
            )
        return found
    finally:
        cf.CFRelease(listed)


def _info(window: Window) -> _Info:
    found = [i for i in _infos(_INCLUDING_WINDOW, window.handle) if i.number == window.handle]
    if not found:
        msg = f'window {window.handle} ({window.title!r}) is gone'
        raise OSError(msg)
    return found[0]


@functools.cache
def _title_bar() -> int:
    """The height in points of the title bar AppKit draws on a titled window (``NSWindow frameRectForContentRect``)."""
    _lib('AppKit')  # NSWindow's class
    objc = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.restype, objc.objc_getClass.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    objc.sel_registerName.restype, objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    # A struct this size comes back through objc_msgSend on arm64, through objc_msgSend_stret on x86_64.
    send = objc.objc_msgSend if platform.machine() == 'arm64' else objc.objc_msgSend_stret
    frame_for = ctypes.CFUNCTYPE(_Rect, ctypes.c_void_p, ctypes.c_void_p, _Rect, ctypes.c_ulong)(
        ctypes.cast(send, ctypes.c_void_p).value
    )
    content = _Rect(_Point(0, 0), _Size(400, 300))
    frame = frame_for(
        objc.objc_getClass(b'NSWindow'), objc.sel_registerName(b'frameRectForContentRect:styleMask:'), content, _TITLED
    )
    return round(frame.size.height - content.size.height)


def _may_post() -> bool:
    return bool(_ax().AXIsProcessTrusted() or _cg().CGPreflightPostEventAccess())


def problem(*, sends_input: bool) -> str | None:
    """Why a capture cannot run here: Screen Recording not granted, or Accessibility when it is to send input.

    Screen Recording not yet granted, macOS is asked to put its own request up (once); the grant takes effect when
    the program running Python is reopened.
    """
    if not _cg().CGPreflightScreenCaptureAccess():
        _cg().CGRequestScreenCaptureAccess()
        return _SCREEN_RECORDING
    if sends_input and not _may_post():
        return _ACCESSIBILITY
    return None


def client(window: Window) -> Region:
    """``window``'s content area in points: its frame less the title bar."""
    frame, bar = _info(window).bounds, _title_bar()
    return Region(frame.left, frame.top + bar, frame.width, max(frame.height - bar, 0))


def windows() -> list[Window]:
    """The on-screen, titled windows of programs (normal or kept on top), front to back."""
    listed = [
        i
        for i in _infos(_ON_SCREEN_ONLY | _EXCLUDE_DESKTOP)
        if 0 <= i.layer <= _FLOATING_LAYER and i.alpha > 0 and i.title
    ]
    found = [Window(i.number, i.title) for i in listed]
    return [w for w in found if client(w).width and client(w).height]


def window(handle: int) -> Window:
    """The window numbered ``handle``."""
    return Window(handle, _info(Window(handle, '')).title)


def monitors() -> list[Region]:
    """Each display's rectangle in points, the main one first, then left to right."""
    cg = _cg()
    ids, count = (ctypes.c_uint32 * 32)(), ctypes.c_uint32()
    if cg.CGGetActiveDisplayList(32, ids, ctypes.byref(count)):
        msg = 'CGGetActiveDisplayList failed: the displays are unknown'
        raise OSError(msg)
    main = cg.CGMainDisplayID()
    found = []
    for display in ids[: count.value]:
        r = cg.CGDisplayBounds(display)
        found.append(
            (display != main, Region(round(r.origin.x), round(r.origin.y), round(r.size.width), round(r.size.height)))
        )
    return [region for _, region in sorted(found, key=lambda f: (f[0], f[1].left, f[1].top))]


def _rgb(image: int, width: int, height: int) -> np.ndarray:
    """The pixels of a CGImage as ``(h, w, 3)`` RGB: 8 bits a component, 32 a pixel, in either byte order."""
    cf, cg = _cf(), _cg()
    got = cg.CGImageGetWidth(image), cg.CGImageGetHeight(image)
    if got != (width, height):
        msg = f'macOS gave a {got[0]}x{got[1]} picture of a {width}x{height} area'
        raise OSError(msg)
    info, row = cg.CGImageGetBitmapInfo(image), cg.CGImageGetBytesPerRow(image)
    if (cg.CGImageGetBitsPerPixel(image), cg.CGImageGetBitsPerComponent(image)) != (32, 8):
        msg = f'a {cg.CGImageGetBitsPerPixel(image)}-bit picture is not one this reads (8-bit RGBA)'
        raise OSError(msg)
    data = cg.CGDataProviderCopyData(cg.CGImageGetDataProvider(image))
    try:
        raw = ctypes.string_at(cf.CFDataGetBytePtr(data), cf.CFDataGetLength(data))
    finally:
        cf.CFRelease(data)
    pixels = np.frombuffer(raw, dtype=np.uint8)[: row * height].reshape(height, row)[:, : 4 * width]
    pixels = pixels.reshape(height, width, 4)
    first, little = (info & _ALPHA_MASK) in _ALPHA_FIRST, (info & _ORDER_MASK) == _ORDER_32_LITTLE
    order = {(True, True): [2, 1, 0], (True, False): [1, 2, 3], (False, True): [3, 2, 1], (False, False): [0, 1, 2]}
    return np.ascontiguousarray(pixels[..., order[first, little]])


def grab(target: Window | Region) -> np.ndarray:
    """The pixels of ``target``: a window's content area even when covered, a region as the screen shows it."""
    shown = client(target) if isinstance(target, Window) else target
    rect = _Rect(_Point(shown.left, shown.top), _Size(shown.width, shown.height))
    if isinstance(target, Window):
        image = _cg().CGWindowListCreateImage(rect, _INCLUDING_WINDOW, target.handle, _NOMINAL_RESOLUTION)
    else:
        image = _cg().CGWindowListCreateImage(rect, _ON_SCREEN_ONLY, 0, _NOMINAL_RESOLUTION)
    if not image:
        msg = f'macOS gave no picture of {target}: {_SCREEN_RECORDING}'
        raise OSError(msg)
    try:
        return _rgb(image, shown.width, shown.height)
    finally:
        _cg().CGImageRelease(image)


def _under(x: int, y: int) -> _Info | None:
    """The front-most window holding the point ``x, y`` (the window server's own overlays aside)."""
    return next(
        (
            i
            for i in _infos(_ON_SCREEN_ONLY | _EXCLUDE_DESKTOP)
            if i.layer >= 0 and i.alpha > 0 and i.owner != 'Window Server' and i.holds(x, y)
        ),
        None,
    )


def _require_input() -> None:
    if not _may_post():
        raise OSError(_ACCESSIBILITY)


def _require_on_top(target: Window | Region, x: int, y: int) -> None:
    """Refuse a wheel that would land on another window: ``target`` was grabbed even if covered, the wheel is not."""
    if isinstance(target, Window):
        under = _under(x, y)
        if under is None or under.number != target.handle:
            name = 'nothing' if under is None else repr(under.title or under.owner)
            msg = f'{name} covers the middle of {target.title!r}: bring it to the front (or method message)'
            raise OSError(msg)


def _cursor() -> _Point:
    event = _cg().CGEventCreate(None)
    try:
        return _cg().CGEventGetLocation(event)
    finally:
        _cf().CFRelease(event)


def _wheel(notches: int, x: int, y: int) -> int:
    """A wheel event of ``notches`` lines (down when > 0) at ``x, y``; the caller releases it."""
    event = _cg().CGEventCreateScrollWheelEvent2(None, _LINE, 1, -notches, 0, 0)
    _cg().CGEventSetLocation(event, _Point(x, y))
    return event


def _copy(element: int, attribute: str) -> int | None:
    """``element``'s Accessibility ``attribute`` (a reference the caller releases), None when it has none."""
    value = ctypes.c_void_p()
    if _ax().AXUIElementCopyAttributeValue(element, _cfstr(attribute), ctypes.byref(value)):
        return None
    return value.value


def _in_front(window: Window, pid: int) -> bool:
    """Whether ``window`` holds the keyboard: its program is the focused one and the window its focused one."""
    ax, cf = _ax(), _cf()
    system = ax.AXUIElementCreateSystemWide()
    app = _copy(system, 'AXFocusedApplication')
    cf.CFRelease(system)
    if app is None:
        return False
    try:
        focused = ctypes.c_int()
        ax.AXUIElementGetPid(app, ctypes.byref(focused))
        shown = _copy(app, 'AXFocusedWindow')
        if shown is None:
            return False
        title = _copy(shown, 'AXTitle')
        cf.CFRelease(shown)
        same = _string(title) == window.title
        if title:
            cf.CFRelease(title)
        return focused.value == pid and same
    finally:
        cf.CFRelease(app)


def _raise(window: Window, pid: int) -> None:
    """Make ``window``'s program the front one and raise ``window`` in it (by its title), through Accessibility."""
    ax, cf = _ax(), _cf()
    app = ax.AXUIElementCreateApplication(pid)
    try:
        ax.AXUIElementSetAttributeValue(app, _cfstr('AXFrontmost'), ctypes.c_void_p.in_dll(cf, 'kCFBooleanTrue'))
        listed = _copy(app, 'AXWindows')
        if listed is None:
            return
        try:
            for i in range(cf.CFArrayGetCount(listed)):
                element = cf.CFArrayGetValueAtIndex(listed, i)
                title = _copy(element, 'AXTitle')
                if _string(title) == window.title:
                    ax.AXUIElementPerformAction(element, _cfstr('AXRaise'))
                if title:
                    cf.CFRelease(title)
        finally:
            cf.CFRelease(listed)
    finally:
        cf.CFRelease(app)


def _bring_to_front(window: Window) -> None:
    """Make ``window`` the keyboard target, or refuse: keys sent elsewhere would end the capture short, silently."""
    pid = _info(window).pid
    for _ in range(FRONT_TRIES):
        if _in_front(window, pid):
            return
        _raise(window, pid)
        time.sleep(0.2)
    if not _in_front(window, pid):
        msg = f'{window.title!r} could not be brought to the front for method keys: click it first (or method wheel)'
        raise OSError(msg)


def scroll(target: Window | Region, notches: int, method: str) -> None:
    """Scroll the view under ``target``'s center ``notches`` lines (or arrow presses): down, up if < 0."""
    if method not in SCROLL_METHODS:
        msg = f'scroll method {method!r} is not one of {SCROLL_METHODS}'
        raise ValueError(msg)
    _require_input()
    cg, cf = _cg(), _cf()
    x, y = (client(target) if isinstance(target, Window) else target).center
    if method == 'wheel':
        _require_on_top(target, x, y)
        was = _cursor()
        cg.CGWarpMouseCursorPosition(_Point(x, y))
        event = _wheel(notches, x, y)
        cg.CGEventPost(_HID_TAP, event)
        cf.CFRelease(event)
        time.sleep(0.05)  # the wheel goes to the window under the pointer when it is processed
        cg.CGWarpMouseCursorPosition(was)
    elif method == 'message':
        under = _under(x, y)
        if under is None:
            msg = f'no window is under the middle of {target}'
            raise OSError(msg)
        event = _wheel(notches, x, y)
        cg.CGEventPostToPid(under.pid, event)
        cf.CFRelease(event)
    else:
        under = _under(x, y)
        if not isinstance(target, Window) and under is None:
            msg = f'no window is under the middle of {target}'
            raise OSError(msg)
        _bring_to_front(target if isinstance(target, Window) else Window(under.number, under.title))
        code = _DOWN if notches > 0 else _UP
        for _ in range(abs(notches)):
            for down in (True, False):
                event = cg.CGEventCreateKeyboardEvent(None, code, down)
                cg.CGEventPost(_HID_TAP, event)
                cf.CFRelease(event)


def open_file(path: Path) -> None:
    """Open ``path`` in the application macOS associates with it (a PDF in Preview)."""
    _require_macos()
    subprocess.run(['/usr/bin/open', str(path)], check=True)
