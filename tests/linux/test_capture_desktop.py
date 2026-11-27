"""The ``linux`` part: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Its own part, never tests/base: it needs an X11 display with an EWMH window manager. GitHub's hosted
``ubuntu-latest`` runner has neither until the part's setup starts Xvfb and openbox (``[tool.dev_lab.parts]
hosted``), so CI produces it. The body is every OS part's (`tests.capture_part`). Where no such desktop exists it
FAILS naming what is missing; it is never skipped.
"""

import pytest

from tests.capture_part import *  # noqa: F403 -- the body every OS part runs
from tests.capture_part import require

pytestmark = pytest.mark.linux


@pytest.fixture(autouse=True)
def on_linux():
    require('linux')
