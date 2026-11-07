"""The ``windows`` part: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Its own part, never tests/base: it needs a Windows desktop (and briefly moves the cursor); GitHub's hosted
``windows-latest`` runner has one, so CI produces it. Where none exists it FAILS naming what is missing; it is never
skipped.
"""

import contextlib
import json
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


LOCK = Path(__file__).resolve().parents[2] / 'output' / 'locks' / 'desktop.lock'


@contextlib.contextmanager
def the_desktop(wait: float = 900):
    """Hold the one desktop: pytest workers run in parallel, but two windows on one screen cover each other and take
    each other's input, so a test that shows one waits for this lock across processes."""
    import msvcrt  # noqa: PLC0415 -- Windows only, like the part

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait
    with LOCK.open('a+b') as handle:
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                assert time.monotonic() < deadline, f'the desktop stayed held for {wait} s'
                time.sleep(0.2)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@pytest.fixture
def viewer():
    title, seed = f'document-lab capture fixture {os.getpid()}', 7
    with the_desktop():
        process = launched(title, seed)
        try:
            yield shown(title), picture(seed)
        finally:
            process.kill()
            process.wait()


@pytest.mark.parametrize('method', ['wheel', 'keys'])
def test_a_scrolled_window_is_captured_whole(viewer, method):
    window, image = viewer
    source = desktop.Source(window, notches=3, method=method)
    rows = Rows()
    capture(source.grab, source.scroll, rows)
    assert np.array_equal(rows.image(), image)


def test_a_posted_wheel_captures_whole_or_says_it_did_not_scroll(viewer):
    """``message`` posts the wheel without moving the pointer, and Tk routes a wheel to the window under the pointer:
    where the pointer is elsewhere the viewer ignores it. What is promised is never a short image passed off whole."""
    window, image = viewer
    source = desktop.Source(window, notches=3, method='message')
    rows = Rows()
    got = capture(source.grab, source.scroll, rows)
    if got.stuck:
        assert np.array_equal(rows.image(), image[:VIEW])
    else:
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
    argv = [
        'capture',
        str(out),
        '--hwnd',
        hex(window.hwnd),
        '--auto',
        '--notches',
        '3',
        '--delay',
        '0',
        '--page-height',
        '0',
    ]
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


# The selector is driven through Tk's own event dispatch (event_generate), not OS input -- a click Windows did not
# deliver would leave select() waiting forever -- and in a child with a timeout, so even a hang is a failure.
SELECT = """
import json, sys
from document_lab.capture import desktop
steps = json.loads(sys.argv[1])
screen = desktop.virtual_screen()
o = desktop._Overlay(desktop.windows(), screen, desktop.grab(screen))
o.canvas.focus_force()
def at(x, y):
    return {'x': x - screen.left, 'y': y - screen.top}
def play():
    for kind, *where in steps:
        if kind == 'key':
            o.canvas.event_generate(where[0])
        elif kind == 'start':
            x0, y0, x1, y1 = o.canvas.bbox('start')
            for event in ('<Motion>', '<ButtonPress-1>', '<ButtonRelease-1>'):
                o.canvas.event_generate(event, x=(x0 + x1) // 2, y=(y0 + y1) // 2)
        else:
            (a, b), (c, d) = where[0], where[-1]
            o.canvas.event_generate('<ButtonPress-1>', **at(a, b))
            o.canvas.event_generate('<B1-Motion>', **at(c, d))
            o.canvas.event_generate('<ButtonRelease-1>', **at(c, d))
        o.root.update()
o.root.after(300, play)
print(repr(o.run()))
"""


def selected(*steps: tuple) -> str:
    """What the overlay returns after ``steps``: ('drag', a, b), ('click', a), ('start',) or ('key', sequence)."""
    done = subprocess.run(
        [sys.executable, '-c', SELECT, json.dumps(steps)], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_a_drag_confirmed_by_enter_is_the_region(viewer):
    window, _ = viewer
    area = window.client()
    start, end = (area.left + 10, area.top + 20), (area.left + 110, area.top + 70)
    assert selected(('drag', start, end), ('key', '<Return>')) == repr(
        desktop.Region(area.left + 10, area.top + 20, 100, 50)
    )


def test_a_click_marks_the_window_and_start_confirms_it(viewer):
    window, _ = viewer
    area = window.client()
    drag = ('drag', (area.left + 10, area.top + 20), (area.left + 110, area.top + 70))  # replaced by the click
    assert selected(drag, ('click', area.centre), ('start',)) == repr(window)


def test_nothing_is_taken_before_it_is_confirmed(viewer):
    window, _ = viewer
    assert selected(('click', window.client().centre), ('key', '<Escape>')) == 'None'


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
    with the_desktop():
        done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith('Region('), done.stdout
