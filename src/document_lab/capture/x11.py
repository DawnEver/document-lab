"""The Linux desktop under X11 as a capture source (`document_lab.capture.screen.System`), through ``python-xlib``.

``python-xlib`` speaks the X protocol itself -- no C library to install. The window manager's EWMH lists say which
windows there are (``_NET_CLIENT_LIST_STACKING``), which one has the keyboard (``_NET_ACTIVE_WINDOW``) and how to
bring one there; XTEST sends the wheel and the keys; the pixels are read from the root window as the screen shows
them. X11 keeps no picture of a covered window, so a covered window is refused rather than grabbed half hidden.

A Wayland session lets no program read other windows or send them input. Its X server for older programs
(XWayland) is told by its own ``XWAYLAND`` extension and refused, naming the X11 session to log in to: a capture
there would read black where every Wayland window is. (The desktop portal's screencast and remote-desktop route is
not taken.)
"""

from __future__ import annotations

import functools
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from document_lab.capture.screen import SCROLL_METHODS, Region, Window

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['client', 'grab', 'monitors', 'open_file', 'owned', 'problem', 'scroll', 'window', 'windows']

FRONT_TRIES: Final = 5  # tries to bring a window to the front before it is refused
_WHEEL_DOWN, _WHEEL_UP = 5, 4  # the pointer buttons X11 sends for a wheel notch
_ALL_DESKTOPS: Final = 0xFFFFFFFF
_PAGER: Final = 2  # the source of an _NET_ACTIVE_WINDOW request: a pager, which a window manager obeys
_WAYLAND: Final = (
    'this is a Wayland session (its X server is XWayland): Wayland lets no program read other windows or send them '
    "input. Log out and choose the X11 session on the login screen (GNOME: the gear -> 'GNOME on Xorg'; KDE: "
    "'Plasma (X11)'), then capture again"
)


def _require_linux() -> None:
    if sys.platform != 'linux':
        msg = f'document_lab.capture.x11 drives an X11 desktop on Linux; this is {sys.platform}'
        raise OSError(msg)


@functools.cache
def _display() -> Any:  # noqa: ANN401 -- Xlib.display.Display, imported only here
    """The X display ``DISPLAY`` names, opened on first use; refused by name when there is none."""
    _require_linux()
    from Xlib import display, error  # noqa: PLC0415 -- python-xlib is installed on Linux alone

    try:
        return display.Display()
    except (error.DisplayError, error.DisplayConnectionError) as missing:
        msg = (
            f'no X display to capture ({missing}): run it in an X11 desktop session, where DISPLAY names the display; '
            'a Wayland session has none for other programs'
        )
        raise OSError(msg) from missing


def _root() -> Any:  # noqa: ANN401
    return _display().screen().root


def _atom(name: str) -> int:
    return _display().intern_atom(name)


def _cardinals(owner: Any, name: str) -> list[int]:  # noqa: ANN401
    """The 32-bit values of property ``name`` of window ``owner`` (empty when it has none)."""
    from Xlib import X  # noqa: PLC0415

    prop = owner.get_full_property(_atom(name), X.AnyPropertyType)
    return [] if prop is None else list(prop.value)


def _resource(handle: int) -> Any:  # noqa: ANN401
    return _display().create_resource_object('window', handle)


def _title(handle: int) -> str:
    """The window's ``_NET_WM_NAME`` (UTF-8), else its ``WM_NAME``."""
    from Xlib import X, error  # noqa: PLC0415

    owner = _resource(handle)
    try:
        named = owner.get_full_property(_atom('_NET_WM_NAME'), _atom('UTF8_STRING'))
        if named is not None and named.value:
            value = named.value
            return value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value)
        plain = owner.get_full_property(_atom('WM_NAME'), X.AnyPropertyType)
    except error.BadWindow:
        return ''
    if plain is None:
        return ''
    return plain.value.decode('latin-1') if isinstance(plain.value, bytes) else str(plain.value)


def problem(*, sends_input: bool) -> str | None:
    """Why a capture cannot run here: no X display, a Wayland session, no EWMH window manager, no XTEST.

    The pixels must be 24-bit true color (8 bits a component), the depth of every current desktop.
    """
    try:
        shown = _display()
    except OSError as missing:
        return str(missing)
    if shown.has_extension('XWAYLAND'):
        return _WAYLAND
    if not _cardinals(_root(), '_NET_SUPPORTING_WM_CHECK'):
        return (
            'no EWMH window manager runs on this X display: the windows and which one is in front are its to say. '
            'Run a desktop session, or a window manager (openbox, xfwm4) on the display'
        )
    if sends_input and not shown.has_extension('XTEST'):
        return 'the X server has no XTEST extension, so no scroll can be sent: scroll yourself (leave out --auto)'
    return _format_problem()


def _format_problem() -> str | None:
    shown = _display()
    screen = shown.screen()
    depth = screen.root_depth
    pixmap = next((f for f in shown.display.info.pixmap_formats if f.depth == depth), None)
    visual = next(
        (v for d in screen.allowed_depths if d.depth == depth for v in d.visuals if v.visual_id == screen.root_visual),
        None,
    )
    masks = None if visual is None else (visual.red_mask, visual.green_mask, visual.blue_mask)
    if depth not in {24, 32} or pixmap is None or pixmap.bits_per_pixel != 32 or masks != (0xFF0000, 0xFF00, 0xFF):  # noqa: PLR2004
        return f'the X screen is {depth}-bit, not the 24-bit true color (8 bits a component) this reads'
    return None


def _stacking() -> list[int]:
    """The managed top-level windows, bottom to top (the window manager's ``_NET_CLIENT_LIST_STACKING``)."""
    return _cardinals(_root(), '_NET_CLIENT_LIST_STACKING')


def _shown(handle: int) -> bool:
    """Mapped, not minimized, and on the desktop in view (or on every desktop)."""
    from Xlib import X, error  # noqa: PLC0415

    owner = _resource(handle)
    try:
        if owner.get_attributes().map_state != X.IsViewable:
            return False
        states = _cardinals(owner, '_NET_WM_STATE')
        desktop = _cardinals(owner, '_NET_WM_DESKTOP')
    except error.BadWindow:
        return False
    current = _cardinals(_root(), '_NET_CURRENT_DESKTOP')
    on_view = not desktop or not current or desktop[0] in {_ALL_DESKTOPS, current[0]}
    return _atom('_NET_WM_STATE_HIDDEN') not in states and on_view


def client(window: Window) -> Region:
    """``window``'s own area (the window manager's frame aside) on the screen."""
    from Xlib import error  # noqa: PLC0415

    owner = _resource(window.handle)
    try:
        geometry = owner.get_geometry()
        origin = _root().translate_coords(owner, 0, 0)
    except error.BadWindow as gone:
        msg = f'window {window.handle:#x} ({window.title!r}) is gone'
        raise OSError(msg) from gone
    return Region(origin.x, origin.y, geometry.width, geometry.height)


def _frame(handle: int) -> Region:
    """The window with its frame (``_NET_FRAME_EXTENTS``): what it covers of the screen."""
    inner = client(Window(handle, ''))
    left, right, top, bottom = _cardinals(_resource(handle), '_NET_FRAME_EXTENTS') or [0, 0, 0, 0]
    return Region(inner.left - left, inner.top - top, inner.width + left + right, inner.height + top + bottom)


def windows() -> list[Window]:
    """The shown, titled, managed windows, front to back."""
    found = [Window(handle, title) for handle in reversed(_stacking()) if _shown(handle) and (title := _title(handle))]
    return [w for w in found if client(w).width and client(w).height]


def _pid(handle: int) -> int | None:
    """The process that says it owns the window (``_NET_WM_PID``); None when it says none."""
    from Xlib import error  # noqa: PLC0415

    try:
        said = _cardinals(_resource(handle), '_NET_WM_PID')
    except error.BadWindow:
        return None
    return said[0] if said else None


def owned(pid: int) -> list[Window]:
    """The shown, managed windows process ``pid`` says it owns (``_NET_WM_PID``), front to back, titled or not."""
    stacked = reversed(_stacking())
    found = [Window(handle, _title(handle)) for handle in stacked if _shown(handle) and _pid(handle) == pid]
    return [w for w in found if client(w).width and client(w).height]


def window(handle: int) -> Window:
    """The window with id ``handle``."""
    client(Window(handle, ''))  # refuses an id that names no window
    return Window(handle, _title(handle))


def monitors() -> list[Region]:
    """Each monitor's rectangle (RandR), the primary first, then left to right; a screen without RandR is one."""
    shown, root = _display(), _root()
    if shown.has_extension('RANDR'):
        listed = root.xrandr_get_monitors(is_active=True).monitors
        if listed:
            found = [(not m.primary, Region(m.x, m.y, m.width_in_pixels, m.height_in_pixels)) for m in listed]
            return [region for _, region in sorted(found, key=lambda f: (f[0], f[1].left, f[1].top))]
    geometry = root.get_geometry()
    return [Region(0, 0, geometry.width, geometry.height)]


def _cover(target: Window) -> Window | None:
    """The front-most shown window above ``target`` that covers part of it, None when none does."""
    order = _stacking()
    if target.handle not in order:
        return None
    inner = client(target)
    for handle in reversed(order[order.index(target.handle) + 1 :]):
        if not _shown(handle):
            continue
        over = _frame(handle)
        apart = (
            over.left >= inner.left + inner.width
            or inner.left >= over.left + over.width
            or over.top >= inner.top + inner.height
            or inner.top >= over.top + over.height
        )
        if not apart:
            return Window(handle, _title(handle))
    return None


def grab(target: Window | Region) -> np.ndarray:
    """The pixels of ``target`` as the screen shows them; a window another covers is refused."""
    from Xlib import X, error  # noqa: PLC0415

    if isinstance(target, Window) and (cover := _cover(target)) is not None:
        msg = (
            f'{cover.title!r} covers {target.title!r}; X11 keeps no picture of a covered window: bring it to the front'
        )
        raise OSError(msg)
    if why := _format_problem():
        raise OSError(why)
    shown = client(target) if isinstance(target, Window) else target
    try:
        raw = _root().get_image(shown.left, shown.top, shown.width, shown.height, X.ZPixmap, 0xFFFFFFFF)
    except error.BadMatch as outside:
        msg = f'{shown} runs off the X screen: mark an area on it'
        raise OSError(msg) from outside
    data = raw.data if isinstance(raw.data, bytes) else raw.data.encode('latin-1')
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(shown.height, shown.width, 4)
    order = [2, 1, 0] if _display().display.info.image_byte_order == X.LSBFirst else [1, 2, 3]
    return np.ascontiguousarray(pixels[..., order])


def _under(x: int, y: int) -> int | None:
    """The front-most shown managed window whose frame holds the point ``x, y``."""
    for handle in reversed(_stacking()):
        if _shown(handle):
            f = _frame(handle)
            if f.left <= x < f.left + f.width and f.top <= y < f.top + f.height:
                return handle
    return None


def _deepest(x: int, y: int) -> Any:  # noqa: ANN401
    """The innermost window at the point ``x, y`` (what a click there would go to) and the point in it."""
    root = _root()
    owner, inner = root, root.translate_coords(root, x, y)
    while inner.child:
        owner = inner.child
        inner = owner.translate_coords(root, x, y)
    return owner, inner.x, inner.y


def _require_xtest() -> Any:  # noqa: ANN401
    from Xlib.ext import xtest  # noqa: PLC0415

    if not _display().has_extension('XTEST'):
        msg = 'the X server has no XTEST extension, so no scroll can be sent: scroll yourself (leave out --auto)'
        raise OSError(msg)
    return xtest


def _active() -> int | None:
    active = _cardinals(_root(), '_NET_ACTIVE_WINDOW')
    return active[0] if active and active[0] else None


def _in_front(handle: int) -> bool:
    """Whether ``handle``, or a dialog of its own (``WM_TRANSIENT_FOR``), holds the keyboard."""
    active = _active()
    if active is None:
        return False
    return active == handle or handle in _cardinals(_resource(active), 'WM_TRANSIENT_FOR')


def _bring_to_front(handle: int) -> None:
    """Ask the window manager to activate ``handle``, or refuse: keys sent elsewhere would end the capture short."""
    from Xlib import X  # noqa: PLC0415
    from Xlib.protocol import event  # noqa: PLC0415

    shown, root = _display(), _root()
    for _ in range(FRONT_TRIES):
        if _in_front(handle):
            return
        request = event.ClientMessage(
            window=_resource(handle),
            client_type=_atom('_NET_ACTIVE_WINDOW'),
            data=(32, [_PAGER, X.CurrentTime, 0, 0, 0]),
        )
        root.send_event(request, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        shown.sync()
        time.sleep(0.2)
    if not _in_front(handle):
        msg = f'{_title(handle)!r} could not be brought to the front for method keys: click it first (or method wheel)'
        raise OSError(msg)


def scroll(target: Window | Region, notches: int, method: str) -> None:
    """Scroll the view under ``target``'s center ``notches`` wheel notches (or arrow presses): down, up if < 0."""
    from Xlib import XK, X  # noqa: PLC0415
    from Xlib.protocol import event  # noqa: PLC0415

    if method not in SCROLL_METHODS:
        msg = f'scroll method {method!r} is not one of {SCROLL_METHODS}'
        raise ValueError(msg)
    shown, root = _display(), _root()
    x, y = (client(target) if isinstance(target, Window) else target).center
    button = _WHEEL_DOWN if notches > 0 else _WHEEL_UP
    if method == 'wheel':
        xtest = _require_xtest()
        under = _under(x, y)
        if isinstance(target, Window) and under != target.handle:
            name = 'nothing' if under is None else repr(_title(under))
            msg = f'{name} covers the middle of {target.title!r}: bring it to the front (or method message)'
            raise OSError(msg)
        was = root.query_pointer()
        xtest.fake_input(shown, X.MotionNotify, x=x, y=y)
        for _ in range(abs(notches)):
            xtest.fake_input(shown, X.ButtonPress, button)
            xtest.fake_input(shown, X.ButtonRelease, button)
        shown.sync()
        time.sleep(0.05)  # the wheel goes to the window under the pointer when the server processes it
        xtest.fake_input(shown, X.MotionNotify, x=was.root_x, y=was.root_y)
        shown.sync()
    elif method == 'message':
        owner, ex, ey = _deepest(x, y)
        for _ in range(abs(notches)):
            for kind, mask in ((event.ButtonPress, X.ButtonPressMask), (event.ButtonRelease, X.ButtonReleaseMask)):
                sent = kind(
                    time=X.CurrentTime, root=root, window=owner, child=X.NONE, same_screen=1,
                    root_x=x, root_y=y, event_x=ex, event_y=ey, state=0, detail=button,
                )  # fmt: skip
                owner.send_event(sent, event_mask=mask, propagate=True)
        shown.sync()
    else:
        xtest = _require_xtest()
        handle = target.handle if isinstance(target, Window) else _under(x, y)
        if handle is None:
            msg = f'no window is under the middle of {target}'
            raise OSError(msg)
        _bring_to_front(handle)
        code = shown.keysym_to_keycode(XK.string_to_keysym('Down' if notches > 0 else 'Up'))
        for _ in range(abs(notches)):
            xtest.fake_input(shown, X.KeyPress, code)
            xtest.fake_input(shown, X.KeyRelease, code)
        shown.sync()


def open_file(path: Path) -> None:
    """Open ``path`` in the application the desktop associates with it (``xdg-open``)."""
    _require_linux()
    subprocess.Popen(['xdg-open', str(path)])  # noqa: S607 -- the desktop's own opener, found as the desktop finds it
