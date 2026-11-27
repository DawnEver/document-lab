"""The ``macos`` part: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Its own part, never tests/base: it needs a macOS desktop session whose program running Python is granted Screen
Recording and Accessibility; GitHub's hosted ``macos-latest`` runner grants both to its agent, so CI produces it.
The body is every OS part's (`tests.capture_part`). Where no such desktop exists it FAILS naming what is missing; it
is never skipped.
"""

import pytest

from tests.capture_part import *  # noqa: F403 -- the body every OS part runs
from tests.capture_part import require

pytestmark = pytest.mark.macos


@pytest.fixture(autouse=True)
def on_macos():
    require('darwin')
