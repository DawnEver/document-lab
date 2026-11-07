"""The ``windows`` part: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Its own part, never tests/base: it needs a Windows desktop (and briefly moves the cursor); GitHub's hosted
``windows-latest`` runner has one, so CI produces it. Where none exists it FAILS naming what is missing; it is never
skipped.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from document_lab.__main__ import main
from document_lab.capture import desktop
from document_lab.capture.session import Limits, capture
from tests.windows.viewer import HEIGHT, VIEW, WIDTH, picture

pytestmark = pytest.mark.windows

VIEWER = Path(__file__).with_name('viewer.py')


def launched(title: str, seed: int) -> subprocess.Popen:
    assert sys.platform == 'win32', f'the windows part needs a Windows desktop, not {sys.platform}'
    return subprocess.Popen([sys.executable, str(VIEWER), title, str(seed)])


def shown(title: str) -> desktop.Window:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            window = desktop.find_window(title)
            if window.client().height == VIEW:
                time.sleep(0.5)
                return window
        except LookupError:
            pass
        time.sleep(0.1)
    pytest.fail(f'the viewer window {title!r} never appeared: is there an interactive desktop?')


@pytest.fixture
def viewer():
    title, seed = f'document-lab capture fixture {os.getpid()}', 7
    process = launched(title, seed)
    try:
        yield shown(title), picture(seed)
    finally:
        process.kill()
        process.wait()


@pytest.mark.parametrize('method', ['wheel', 'message', 'keys'])
def test_a_scrolled_window_is_captured_whole(viewer, method):
    window, image = viewer
    source = desktop.Source(window, notches=3, method=method)
    assert np.array_equal(capture(source.grab, source.scroll), image)


def test_the_first_frame_is_the_view(viewer):
    window, image = viewer
    assert np.array_equal(desktop.grab(window), image[:VIEW])


def test_a_region_is_the_screen_under_it(viewer):
    window, image = viewer
    area = window.client()
    inner = desktop.Region(area.left + 10, area.top + 20, 100, 50)
    assert np.array_equal(desktop.grab(inner), image[20:70, 10:110])


def test_the_cli_writes_the_pdf(viewer, tmp_path):
    window, image = viewer
    out = tmp_path / 'capture.pdf'
    argv = ['capture', str(out), '--hwnd', hex(window.hwnd), '--delay', '0', '--page-height', '0']
    assert main(argv) == 0
    with pymupdf.open(out) as doc:
        assert doc.page_count == 1
        ((xref, *_),) = doc[0].get_images()
        pix = pymupdf.Pixmap(doc, xref)
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3), image)


def test_max_height_stops_early(viewer):
    window, image = viewer
    source = desktop.Source(window)
    got = capture(source.grab, source.scroll, limits=Limits(max_height=700))
    assert 700 <= len(got) < HEIGHT
    assert np.array_equal(got, image[: len(got)])


def test_a_wheel_onto_a_covered_window_refuses(viewer):
    window, _ = viewer
    title = f'document-lab capture cover {os.getpid()}'
    cover = launched(title, 8)  # opens at the same place, above the first
    try:
        shown(title)
        with pytest.raises(OSError, match='covers'):
            desktop.Source(window).scroll()
    finally:
        cover.kill()
        cover.wait()
