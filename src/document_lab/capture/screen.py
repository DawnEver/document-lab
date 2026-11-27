"""What a capture needs of a desktop, whatever the OS: its rectangles and windows, and the one interface each OS fills.

:class:`System` is the seam. `document_lab.capture.desktop` picks the module that fills it for this OS
(`win32`, `quartz` or `x11`) and everything above it -- the selection overlay, the panel, the CLI -- asks only it;
`session` and `stitch` see two callables and no OS at all. Each module fills it with plain functions, so the module
itself is the :class:`System`.

Coordinates are the OS's own screen space: physical pixels on Windows (the process is made per-monitor DPI aware)
and X11, Quartz points on macOS -- the space Tk places its windows in. A grab of a rectangle is that rectangle's
size in pixels on every OS (macOS grabs one pixel a point), so the stitcher and the PDF never see a scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

__all__ = ['SCROLL_METHODS', 'Region', 'System', 'Window', 'bounds']

#: ``wheel`` puts the pointer over the target, sends real wheel input and puts the pointer back (the target must be
#: in front there); ``message`` delivers the wheel to the window under the target without moving the pointer (some
#: programs ignore it); ``keys`` brings the window to the front and presses the Down arrow (for views that ignore
#: the wheel).
SCROLL_METHODS: Final = ('wheel', 'message', 'keys')


@dataclass(frozen=True)
class Region:
    """A rectangle of the screen, in the OS's screen space."""

    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """The middle pixel."""
        return self.left + self.width // 2, self.top + self.height // 2


def bounds(regions: list[Region]) -> Region:
    """The rectangle covering every one of ``regions`` (every monitor: the virtual screen)."""
    left, top = min(r.left for r in regions), min(r.top for r in regions)
    right, bottom = max(r.left + r.width for r in regions), max(r.top + r.height for r in regions)
    return Region(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class Window:
    """A top-level window: the OS's handle for it (an HWND, a CGWindowID, an X11 window id) and its title."""

    handle: int
    title: str


class System(Protocol):
    """One OS's desktop as a capture source: what is on it, its pixels, and its scrolling."""

    def problem(self, *, sends_input: bool) -> str | None:
        """Why a capture cannot run here, naming the remedy; None when it can.

        ``sends_input``: the capture will scroll for the user (an OS may need a permission for that alone).
        """
        ...

    def windows(self) -> list[Window]:
        """The visible, titled top-level windows a capture may take, front to back."""
        ...

    def window(self, handle: int) -> Window:
        """The window with ``handle``; refused when none has it."""
        ...

    def client(self, window: Window) -> Region:
        """``window``'s content area (no frame, no title bar) on the screen, read fresh; refused when it is gone."""
        ...

    def monitors(self) -> list[Region]:
        """Each monitor's rectangle, the primary first, then left to right."""
        ...

    def grab(self, target: Window | Region) -> np.ndarray:
        """The pixels of ``target`` -- a window's content area, or a rectangle as the screen shows it -- as RGB."""
        ...

    def scroll(self, target: Window | Region, notches: int, method: str) -> None:
        """Scroll the view under ``target``'s center ``notches`` notches (or Down presses), up when < 0.

        ``method`` is one of :data:`SCROLL_METHODS`; input the OS drops is refused (``OSError``) naming why.
        """
        ...

    def open_file(self, path: Path) -> None:
        """Open ``path`` in the program the OS associates with it."""
        ...
