"""The ``windows`` part: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Its own part, never tests/base: it needs a Windows desktop (and briefly moves the cursor); GitHub's hosted
``windows-latest`` runner has one, so CI produces it. Where none exists it FAILS naming what is missing; it is never
skipped.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from document_lab.__main__ import main
from document_lab.capture import desktop
from document_lab.capture.session import Limits, Rows, capture
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
    rows = Rows()
    capture(source.grab, source.scroll, rows)
    assert np.array_equal(rows.image(), image)


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
    rows = Rows()
    capture(source.grab, source.scroll, rows, limits=Limits(max_height=700))
    got = rows.image()
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


def _gesture(*steps: tuple) -> threading.Thread:
    """Real input, played once the overlay is up: ('drag', (x, y), (x, y)), ('click', (x, y)) or ('key', vk)."""
    user32 = ctypes.WinDLL('user32')

    def play() -> None:
        deadline = time.monotonic() + 15
        while not user32.IsWindowVisible(user32.FindWindowW(None, desktop.OVERLAY_TITLE)):
            assert time.monotonic() < deadline, 'the selection overlay never appeared'
            time.sleep(0.05)
        time.sleep(0.3)  # mapped is not yet painted and taking input
        for kind, *where in steps:
            if kind == 'key':
                user32.keybd_event(where[0], 0, 0, 0)
                user32.keybd_event(where[0], 0, 2, 0)
            else:
                start, end = where[0], where[-1]
                user32.SetCursorPos(*start)
                time.sleep(0.1)
                user32.mouse_event(0x2, 0, 0, 0, 0)
                for i in range(1, 11):
                    user32.SetCursorPos(
                        start[0] + (end[0] - start[0]) * i // 10, start[1] + (end[1] - start[1]) * i // 10
                    )
                    time.sleep(0.03)
                user32.mouse_event(0x4, 0, 0, 0, 0)
            time.sleep(0.4)

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    return thread


ENTER, ESC = 0x0D, 0x1B


def test_a_drag_confirmed_by_enter_is_the_region(viewer):
    window, _ = viewer
    area = window.client()
    start, end = (area.left + 10, area.top + 20), (area.left + 110, area.top + 70)
    _gesture(('drag', start, end), ('key', ENTER))
    assert desktop.select() == desktop.Region(area.left + 10, area.top + 20, 100, 50)


def test_a_click_marks_the_window_and_start_confirms_it(viewer):
    window, _ = viewer
    area = window.client()
    right, bottom = area.left + area.width, area.top + area.height
    _gesture(
        ('drag', (area.left + 10, area.top + 20), (area.left + 110, area.top + 70)),  # replaced by the click
        ('click', area.centre),
        ('click', (right - 80, bottom + 28)),  # the start button, under the mark's right end
    )
    assert desktop.select() == window


def test_nothing_is_taken_before_it_is_confirmed(viewer):
    window, _ = viewer
    _gesture(('click', window.client().centre), ('key', ESC))
    with pytest.raises(KeyboardInterrupt, match='cancelled'):
        desktop.select()


def test_confirming_by_the_start_button_does_not_free_the_canvas_under_its_own_event():
    """Regression: Start destroyed the window inside its click handler; the canvas's own release binding then ran on
    freed memory and Tcl aborted the process ("alloc: invalid block"). Run in a child so an abort is a failure."""
    script = """
import types
from document_lab.capture import desktop
screen = desktop.virtual_screen()
o = desktop._Overlay(desktop.windows(), screen, desktop.grab(screen))
o.root.update()
ev = types.SimpleNamespace
o.press(ev(x_root=screen.left + 100, y_root=screen.top + 100))
o.release(ev(x_root=screen.left + 500, y_root=screen.top + 400))
o.root.update()
x0, y0, x1, y1 = o.canvas.bbox('start')
def click():
    for kind in ('<Motion>', '<ButtonPress-1>', '<ButtonRelease-1>'):
        o.canvas.event_generate(kind, x=(x0 + x1) // 2, y=(y0 + y1) // 2)
o.root.after(300, click)
print(o.run())
"""
    done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith('Region('), done.stdout
