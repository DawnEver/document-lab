"""The macOS desktop as a capture source (`document_lab.capture.screen.System`): Quartz through plain ``ctypes``.

No third-party package, as on Windows: CoreGraphics lists the windows (``CGWindowListCopyWindowInfo``), reads their
pixels (``CGWindowListCreateImage``: a window's content even when covered, a region as the screen shows it) and posts
the wheel and the keys (``CGEvent``); the window server says which window a point's input lands on (``NSWindow
windowNumberAtPoint``); the Accessibility API brings a window to the front for ``keys``.

Coordinates are Quartz points, the space Tk places its windows in; a grab is taken one pixel a point
(``kCGWindowImageNominalResolution``), so a grab of a region is the region's size on a Retina display too, and is
color-matched from the color space macOS gives it in to sRGB, the space a PDF's RGB is read in. A window's content
area is what the program draws and macOS shows untouched: its frame less the title bar AppKit draws on a titled
window, less the edge the window server draws over -- a ring along its sides and bottom, and its rounded bottom
corners. A program that draws into its own title bar loses that strip from a window capture, and every window its
outermost pixels -- mark a region for them.

Two privacy permissions gate it, each refused by name with where to grant it (:func:`problem`): Screen Recording to
read other programs' windows, Accessibility to send them input (asked only when the capture scrolls for the user).
"""

from __future__ import annotations

import ctypes
import functools
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.capture.screen import SCROLL_METHODS, Region, Window

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = ['census', 'client', 'grab', 'monitors', 'open_file', 'owned', 'problem', 'scroll', 'window', 'windows']

_FRAMEWORK: Final = '/System/Library/Frameworks/{0}.framework/{0}'
_ON_SCREEN_ONLY: Final = 1 << 0  # kCGWindowListOptionOnScreenOnly: front to back
_INCLUDING_WINDOW: Final = 1 << 3  # kCGWindowListOptionIncludingWindow
_EXCLUDE_DESKTOP: Final = 1 << 4  # kCGWindowListExcludeDesktopElements
_NOMINAL_RESOLUTION: Final = 1 << 4  # kCGWindowImageNominalResolution: one pixel a point
_KEPT_ON_TOP_LAYER: Final = 19  # kCGUtilityWindowLevel, where Tk keeps a -topmost window; the Dock is 20
_SINT64: Final = 4  # kCFNumberSInt64Type
_DOUBLE: Final = 13  # kCFNumberDoubleType
_UTF8: Final = 0x08000100  # kCFStringEncodingUTF8
_LINE: Final = 1  # kCGScrollEventUnitLine
_HID_TAP: Final = 0  # kCGHIDEventTap
_DOWN, _UP = 125, 126  # kVK_DownArrow, kVK_UpArrow
_TITLED: Final = 1  # NSWindowStyleMaskTitled
_RGBX: Final = 5  # kCGImageAlphaNoneSkipLast, default byte order: R, G, B, then a byte unused, in memory
_RGBA: Final = 1  # kCGImageAlphaPremultipliedLast, default byte order: R, G, B (times A), then A, in memory
_COPY: Final = 17  # kCGBlendModeCopy: the picture's pixels replace the bitmap's, never blended onto them
_NO_INTERPOLATION: Final = 1  # kCGInterpolationNone
#: Points: the border the window server draws just inside a window's left, right and bottom edges, over the
#: program's pixels. It is opaque, so the window's alpha does not show it (measured on macOS 26: one point, tinted).
_RING: Final = 1
_PROBE: Final = 64  # points: the bottom of a window read for its rounded corners, at most
_NOTCH_GAP: Final = 0.01  # seconds between a wheel's notches: each its own event, none merged with the next
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
            'CFCopyDescription': (p, [p]),
        },
        'CoreGraphics': {
            'CGWindowListCopyWindowInfo': (p, [u32, u32]),
            'CGWindowListCreateImage': (p, [_Rect, u32, u32, u32]),
            'CGRectMakeWithDictionaryRepresentation': (b, [p, ctypes.POINTER(_Rect)]),
            'CGImageGetWidth': (sz, [p]),
            'CGImageGetHeight': (sz, [p]),
            'CGImageGetColorSpace': (p, [p]),
            'CGImageRelease': (None, [p]),
            'CGColorSpaceCreateWithName': (p, [p]),
            'CGBitmapContextCreate': (p, [p, sz, sz, sz, sz, p, u32]),
            'CGContextSetBlendMode': (None, [p, ctypes.c_int]),
            'CGContextSetInterpolationQuality': (None, [p, ctypes.c_int]),
            'CGContextDrawImage': (None, [p, _Rect, p]),
            'CGContextRelease': (None, [p]),
            'CGGetActiveDisplayList': (i32, [u32, ctypes.POINTER(u32), ctypes.POINTER(u32)]),
            'CGDisplayBounds': (_Rect, [u32]),
            'CGDisplayPixelsWide': (sz, [u32]),
            'CGDisplayCopyColorSpace': (p, [u32]),
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
            'AXUIElementCopyElementAtPosition': (i32, [p, ctypes.c_float, ctypes.c_float, ctypes.POINTER(p)]),
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


def _described(ref: int | None) -> str:
    """What CoreFoundation says of the object ``ref`` (of a color space: its kind and its profile's name)."""
    if not ref:
        return 'none'
    text = _cf().CFCopyDescription(ref)
    try:
        return _string(text)
    finally:
        if text:
            _cf().CFRelease(text)


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
def _objc() -> ctypes.CDLL:
    """The Objective-C runtime, AppKit loaded: ``NSWindow``'s class methods are sent through it."""
    _lib('AppKit')
    objc = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.restype, objc.objc_getClass.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    objc.sel_registerName.restype, objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    return objc


@functools.cache
def _title_bar() -> int:
    """The height in points of the title bar AppKit draws on a titled window (``NSWindow frameRectForContentRect``)."""
    objc = _objc()
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


def _hit(x: int, y: int) -> int:
    """The number of the window a click at the point ``x, y`` lands on, 0 for none: the window server's own hit test.

    It is how the pointer's input is routed, so it passes through a window's transparent parts -- the Dock's window
    spans the whole screen above every program's and takes only what lands on the Dock -- where a window's bounds
    cannot. AppKit's points run up from the main display's bottom edge, Quartz's down from its top. AppKit is asked
    as an application (``NSApplication sharedApplication``; a process running Tk already is one).
    """
    objc = _objc()
    shared = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    )
    shared(objc.objc_getClass(b'NSApplication'), objc.sel_registerName(b'sharedApplication'))
    at = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, _Point, ctypes.c_long)(
        ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    )
    main = _cg().CGDisplayBounds(_cg().CGMainDisplayID())
    point = _Point(x + 0.5, main.size.height - (y + 0.5))  # the middle of the pixel, in AppKit's points
    selector = objc.sel_registerName(b'windowNumberAtPoint:belowWindowWithWindowNumber:')
    return at(objc.objc_getClass(b'NSWindow'), selector, point, 0)


def _nobody(x: int, y: int) -> str:
    """Why no window was found under the point ``x, y``: what the window server and Accessibility each say is there."""
    ax, cf = _ax(), _cf()
    system, element = ax.AXUIElementCreateSystemWide(), ctypes.c_void_p()
    try:
        error = ax.AXUIElementCopyElementAtPosition(system, x, y, ctypes.byref(element))
    finally:
        cf.CFRelease(system)
    pid = ctypes.c_int()
    if not error and element.value:
        ax.AXUIElementGetPid(element.value, ctypes.byref(pid))
        cf.CFRelease(element.value)
    found = f'pid {pid.value}' if pid.value else f'nothing (AXError {error})'
    return f'the window server names window {_hit(x, y)} at {x},{y}; Accessibility finds {found} there'


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


def _no_picture(what: str) -> str:
    """Why macOS gave no picture of ``what``: Screen Recording not granted, or ``what`` not drawn (yet) or gone.

    Granted, the permission is no cause, and naming it would send the user to a setting that changes nothing.
    """
    if not _cg().CGPreflightScreenCaptureAccess():
        return f'macOS gave no picture of {what}: {_SCREEN_RECORDING}'
    return (
        f'macOS gave no picture of {what}, though Screen Recording is granted: the window server holds none for it '
        '(not drawn yet, or closed)'
    )


def client(window: Window) -> Region:
    """``window``'s content area in points: what the program draws and macOS shows untouched.

    Its frame less the title bar, less the edge the window server draws over the program's pixels: the :data:`_RING`
    along its sides and, at its bottom, the rows its rounded corners take (:func:`_bottom`).
    """
    frame, bar = _info(window).bounds, _title_bar()
    height = max(frame.height - bar, 0)
    bottom = _bottom(window.handle, frame) if height and frame.width else 0
    return Region(frame.left + _RING, frame.top + bar, max(frame.width - 2 * _RING, 0), max(height - bottom, 0))


@functools.lru_cache(maxsize=64)
def _bottom(handle: int, frame: Region) -> int:
    """The rows at the bottom of window ``handle``, at ``frame``, the window server draws over (:func:`_corner_rows`).

    Read from the window's own picture, once for each place and size it has.
    """
    rows = min(_PROBE, frame.height)
    rect = _Rect(_Point(frame.left, frame.top + frame.height - rows), _Size(frame.width, rows))
    image = _cg().CGWindowListCreateImage(rect, _INCLUDING_WINDOW, handle, _NOMINAL_RESOLUTION)
    if not image:
        raise OSError(_no_picture(f'window {handle}'))
    try:
        return _corner_rows(_pixels(image, frame.width, rows, _srgb(), alpha=True)[..., 3])
    finally:
        _cg().CGImageRelease(image)


def _corner_rows(alpha: np.ndarray) -> int:
    """The rows at the bottom of a window's picture whose ``alpha`` its rounded corners cut, and at least the ring.

    A rounded corner leaves its outermost column short of opaque up from the bottom edge for as many rows as it is
    tall; the border the window server draws along its curve lies within them (measured on macOS 26: corners 24 rows,
    their border 20). A window whose sides are short of opaque all the way up gives every row read.
    """
    cut = (alpha[:, 0] < 255) | (alpha[:, -1] < 255)  # noqa: PLR2004 -- opaque
    return max(len(cut) if cut.all() else int(np.argmin(cut[::-1])), _RING)


def _shown() -> list[_Info]:
    """The on-screen windows of programs (normal or kept on top), front to back, named or not."""
    return [i for i in _infos(_ON_SCREEN_ONLY | _EXCLUDE_DESKTOP) if 0 <= i.layer <= _KEPT_ON_TOP_LAYER and i.alpha > 0]


def _sized(listed: list[_Info]) -> list[Window]:
    found = [Window(i.number, i.title) for i in listed]
    return [w for w in found if client(w).width and client(w).height]


def windows() -> list[Window]:
    """The on-screen, titled windows of programs (normal or kept on top), front to back."""
    return _sized([i for i in _shown() if i.title])


def owned(pid: int) -> list[Window]:
    """The on-screen windows process ``pid`` owns, front to back, titled or not.

    macOS names another program's windows only to a process granted Screen Recording; without it they are untitled.
    """
    return _sized([i for i in _shown() if i.pid == pid])


def census() -> str:
    """What CoreGraphics tells this process, for a report when a window cannot be found.

    Its two permissions, each display (its points, its pixels, its color space), then every on-screen window: number,
    owner and pid, layer, alpha, bounds, and its name or that macOS withheld it.
    """
    cg = _cg()
    granted, posts = bool(cg.CGPreflightScreenCaptureAccess()), _may_post()
    lines = [f'Screen Recording granted: {granted}; input may be posted: {posts}; this process: pid {os.getpid()}']
    ids, count = (ctypes.c_uint32 * 32)(), ctypes.c_uint32()
    cg.CGGetActiveDisplayList(32, ids, ctypes.byref(count))
    for display in ids[: count.value]:
        r, space = cg.CGDisplayBounds(display), cg.CGDisplayCopyColorSpace(display)
        try:
            lines.append(
                f'  display {display}: {r.size.width:g}x{r.size.height:g} points at {r.origin.x:g},{r.origin.y:g}, '
                f'{cg.CGDisplayPixelsWide(display)} pixels wide, {_described(space)}'
            )
        finally:
            if space:
                _cf().CFRelease(space)
    for i in _infos(_ON_SCREEN_ONLY | _EXCLUDE_DESKTOP):
        b = i.bounds
        name = repr(i.title) if i.title else 'no name (withheld)'
        lines.append(
            f'  window {i.number}: {i.owner!r} pid {i.pid}, layer {i.layer}, alpha {i.alpha:g}, '
            f'{b.width}x{b.height} at {b.left},{b.top}, {name}'
        )
    return '\n'.join(lines)


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


@functools.cache
def _srgb() -> int:
    """The sRGB color space, made once and kept."""
    return _cg().CGColorSpaceCreateWithName(_key('kCGColorSpaceSRGB'))


def _pixels(image: int, width: int, height: int, space: int, *, alpha: bool = False) -> np.ndarray:
    """A CGImage as ``(h, w, 3)`` 8-bit RGB in color space ``space``; ``(h, w, 4)`` premultiplied RGBA with ``alpha``.

    It is drawn one pixel a pixel into a bitmap of that space, so CoreGraphics color-matches it and reads its pixel
    format, whatever that is.
    """
    cg = _cg()
    got = cg.CGImageGetWidth(image), cg.CGImageGetHeight(image)
    if got != (width, height):
        msg = f'macOS gave a {got[0]}x{got[1]} picture of a {width}x{height} area'
        raise OSError(msg)
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    info = _RGBA if alpha else _RGBX
    context = cg.CGBitmapContextCreate(pixels.ctypes.data, width, height, 8, 4 * width, space, info)
    if not context:
        msg = f'CoreGraphics made no {width}x{height} bitmap in {_described(space)}'
        raise OSError(msg)
    try:
        cg.CGContextSetBlendMode(context, _COPY)
        cg.CGContextSetInterpolationQuality(context, _NO_INTERPOLATION)
        cg.CGContextDrawImage(context, _Rect(_Point(0, 0), _Size(width, height)), image)
    finally:
        cg.CGContextRelease(context)
    return pixels if alpha else np.ascontiguousarray(pixels[..., :3])


def _image(target: Window | Region) -> tuple[int, Region]:
    """A CGImage of ``target`` (the caller releases it) and the rectangle it shows."""
    shown = client(target) if isinstance(target, Window) else target
    rect = _Rect(_Point(shown.left, shown.top), _Size(shown.width, shown.height))
    if isinstance(target, Window):
        image = _cg().CGWindowListCreateImage(rect, _INCLUDING_WINDOW, target.handle, _NOMINAL_RESOLUTION)
    else:
        image = _cg().CGWindowListCreateImage(rect, _ON_SCREEN_ONLY, 0, _NOMINAL_RESOLUTION)
    if not image:
        raise OSError(_no_picture(str(target)))
    return image, shown


def grab(target: Window | Region) -> np.ndarray:
    """The pixels of ``target`` in sRGB: a window's content area even when covered, a region as the screen shows it."""
    image, shown = _image(target)
    try:
        return _pixels(image, shown.width, shown.height, _srgb())
    finally:
        _cg().CGImageRelease(image)


def _tagged(target: Window | Region) -> tuple[np.ndarray, str]:
    """For a diagnosis: ``target``'s pixels as macOS gave them, RGBA, and the color space it tagged them with.

    What :func:`grab` color-matches to sRGB, read without a conversion; the alpha is the window's shape.
    """
    image, shown = _image(target)
    try:
        space = _cg().CGImageGetColorSpace(image)
        return _pixels(image, shown.width, shown.height, space, alpha=True), _described(space)
    finally:
        _cg().CGImageRelease(image)


def _under(x: int, y: int) -> _Info | None:
    """The window a click or the wheel at the point ``x, y`` lands on (:func:`_hit`), None for none."""
    number = _hit(x, y)
    found = [i for i in _infos(_INCLUDING_WINDOW, number) if i.number == number] if number else []
    return found[0] if found else None


def _require_input() -> None:
    if not _may_post():
        raise OSError(_ACCESSIBILITY)


def _require_on_top(target: Window | Region, x: int, y: int) -> None:
    """Refuse a wheel that would land on another window: ``target`` was grabbed even if covered, the wheel is not."""
    if isinstance(target, Window):
        under = _under(x, y)
        if under is None or under.number != target.handle:
            name = 'nothing' if under is None else f'{under.title or under.owner!r} (window {under.number})'
            msg = f'{name} covers the middle of {target.title!r}: bring it to the front (or method message)'
            if under is None:
                msg += f' ({_nobody(x, y)})'
            raise OSError(msg)


def _cursor() -> _Point:
    event = _cg().CGEventCreate(None)
    try:
        return _cg().CGEventGetLocation(event)
    finally:
        _cf().CFRelease(event)


def _wheel(notches: int, x: int, y: int, post: Callable[[int], None]) -> None:
    """``notches`` wheel notches (down when > 0) at ``x, y``, each handed to ``post`` as its own one-line event.

    One event a notch, as a wheel sends them: Tk 9 on macOS scrolls one notch for an event of any number of lines, so a
    capture pacing its steps by notches would double them until the event's delta overflows (2**16 lines: nothing).
    """
    cg, cf = _cg(), _cf()
    for _ in range(abs(notches)):
        event = cg.CGEventCreateScrollWheelEvent2(None, _LINE, 1, -1 if notches > 0 else 1, 0, 0)
        cg.CGEventSetLocation(event, _Point(x, y))
        post(event)
        cf.CFRelease(event)
        time.sleep(_NOTCH_GAP)


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
    """Scroll the view under ``target``'s center ``notches`` wheel notches (or arrow presses): down, up if < 0."""
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
        _wheel(notches, x, y, lambda event: cg.CGEventPost(_HID_TAP, event))
        time.sleep(0.05)  # the wheel goes to the window under the pointer when it is processed
        cg.CGWarpMouseCursorPosition(was)
    elif method == 'message':
        under = _under(x, y)
        if under is None:
            msg = f'no window is under the middle of {target} ({_nobody(x, y)})'
            raise OSError(msg)
        _wheel(notches, x, y, lambda event: cg.CGEventPostToPid(under.pid, event))
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
