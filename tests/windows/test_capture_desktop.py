"""The ``windows`` part: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Its own part, never tests/base: it needs a Windows desktop (and briefly moves the cursor); GitHub's hosted
``windows-latest`` runner has one, so CI produces it. The body is every OS part's (`tests.capture_part`); here too
is what only ``capture-files`` drives, on Windows alone: the mouse parked and whether the user is at the machine.
Where no Windows desktop exists it FAILS naming what is missing; it is never skipped.
"""

import pytest

from document_lab.capture import win32
from document_lab.capture.screen import Region
from tests.capture_part import *  # noqa: F403 -- the body every OS part runs
from tests.capture_part import require

pytestmark = pytest.mark.windows


@pytest.fixture(autouse=True)
def on_windows():
    require('win32')


def test_the_mouse_is_parked_at_a_bottom_corner_never_the_close_button_one():
    """A screen's top right corner is a close button: the left monitor is covered, so the parking is bottom right."""
    screen = Region(-1600, 0, 5440, 2560)
    left = Region(-1600, 0, 1600, 2560)
    x, y = win32.parking_spot(screen, left)
    assert y > screen.height - 100  # a bottom corner: never the top right corner, the close button of the window there
    assert screen.left + 39 < x < screen.left + screen.width - 39  # inset from the hot edge
    assert not left.left <= x < left.left + left.width


def test_a_window_covering_the_whole_virtual_screen_leaves_the_mouse_where_it_is():
    screen = Region(0, 0, 3840, 2160)
    assert win32.parking_spot(screen, screen) is None


def test_the_desktop_says_how_long_it_has_been_idle_and_whether_a_window_holds_the_keyboard(viewer):
    window, _ = viewer
    assert 0.0 <= win32.idle_seconds() < 86_400.0
    assert isinstance(win32.in_front(window), bool)
