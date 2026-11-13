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


def test_a_posted_wheel_places_every_row_it_captures(viewer):
    """``message`` posts the wheel without moving the pointer, and Tk acts on a posted wheel only some of the time: the
    capture may stop early, where a dropped scroll looks like the end. What it must never do is misplace a row."""
    window, image = viewer
    source = desktop.Source(window, notches=3, method='message')
    rows = Rows()
    capture(source.grab, source.scroll, rows)
    got = rows.image()
    assert VIEW <= len(got) <= len(image)
    assert np.array_equal(got, image[: len(got)])


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
o = desktop._Overlay(desktop.windows(), screen)
o.canvas.focus_force()
def at(x, y):
    return {'x': x - screen.left, 'y': y - screen.top}
def play():
    for kind, *where in steps:
        if kind == 'key':
            o.canvas.event_generate(where[0])
        elif kind == 'confirm':
            o.confirm_button.invoke()  # what a click runs; a generated click would go where the real pointer is
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
    """What the overlay returns after ``steps``: ('drag', a, b), ('click', a), ('confirm',) or ('key', sequence)."""
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
    assert selected(drag, ('click', area.centre), ('confirm',)) == repr(window)


def test_nothing_is_taken_before_it_is_confirmed(viewer):
    window, _ = viewer
    assert selected(('click', window.client().centre), ('key', '<Escape>')) == 'None'


def test_confirming_by_the_confirm_button_does_not_free_the_canvas_under_its_own_event():
    """Regression: confirming destroyed the window inside its click handler; the canvas's own release binding then ran
    on freed memory and Tcl aborted the process ("alloc: invalid block"). Run in a child so an abort is a failure."""
    script = """
import types
from document_lab.capture import desktop
screen = desktop.virtual_screen()
o = desktop._Overlay(desktop.windows(), screen)
o.root.update()
ev = types.SimpleNamespace
o.press(ev(x_root=screen.left + 100, y_root=screen.top + 100))
o.release(ev(x_root=screen.left + 500, y_root=screen.top + 400))
o.root.update()
o.root.after(300, o.confirm_button.invoke)  # inside the loop, as a click: the handler must not destroy what runs it
print(o.run())
"""
    with the_desktop():
        done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith('Region('), done.stdout


def test_the_panel_waits_for_a_start_adjust_or_cancel_and_begin_turns_it_into_finish_and_pause():
    """After the area is confirmed the HUD waits, nothing over the desktop, until its own buttons answer."""
    script = """
from document_lab.capture import desktop
for press, want in (('button', 'start'), ('auto', 'auto'), ('adjust', 'adjust'), ('cancel', 'cancel')):
    hud = desktop.Hud(desktop.Region(200, 200, 300, 200))
    hud.root.after(300, getattr(hud, press).invoke)
    assert hud.ready() == want
    if want in {'start', 'auto'}:
        hud.begin(want)
        assert hud.button.cget('text') == 'Finish and save'
        hud.pause.invoke()
        assert hud.paused and hud.pause.cget('text') == 'Resume'
        hud.pause.invoke()
        assert not hud.paused and hud.pause.cget('text') == 'Pause'
        hud.button.invoke()
        assert hud.finished
    hud.close()
print('ok')
"""
    with the_desktop():
        done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'ok', done.stdout


def test_adjusting_opens_the_overlay_with_the_area_set_already_marked():
    """Adjust reopens the selection on the area set: confirmed untouched, it comes back as it was."""
    script = """
from document_lab.capture import desktop
area = desktop.Region(150, 120, 400, 250)
o = desktop._Overlay(desktop.windows(), desktop.virtual_screen(), area)
o.root.after(300, o.confirm_button.invoke)
print(repr(o.run()))
"""
    with the_desktop():
        done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == repr(desktop.Region(150, 120, 400, 250)), done.stdout
