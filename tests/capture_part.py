"""The OS parts' one body: a real window, really scrolled, captured to its end -- pixel for pixel the picture it shows.

Each OS part (``tests/windows``, ``tests/macos``, ``tests/linux``) imports these tests into its own module, marked
with its OS, so one body proves `document_lab.capture.screen.System` on each desktop. A part run on another OS, or
where no desktop exists, FAILS naming what is missing; it is never skipped.
"""

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pymupdf
import pytest

from document_lab.capture import desktop
from document_lab.capture.screen import Region, Window
from document_lab.capture.session import Limits, Rows, capture
from tests.capture_viewer import HEIGHT, VIEW, picture

VIEWER = Path(__file__).with_name('capture_viewer.py')
LOCK = Path(__file__).resolve().parents[1] / 'output' / 'locks' / 'desktop.lock'


def require(platform: str) -> None:
    """Fail -- never skip -- when this is not ``platform``'s desktop."""
    assert sys.platform == platform, f'this part needs a {platform} desktop, not {sys.platform}'


def launched(title: str, seed: int) -> subprocess.Popen:
    """The viewer, saying its own pid first on its stdout: on Windows a venv's python.exe is a launcher that runs the
    interpreter as its child, so the Popen's pid owns no window."""
    return subprocess.Popen([sys.executable, str(VIEWER), title, str(seed)], stdout=subprocess.PIPE, text=True)


def _census() -> str:
    """macOS's own account of its permissions and its windows; empty elsewhere, where no window name is withheld."""
    if sys.platform != 'darwin':
        return ''
    from document_lab.capture import quartz  # noqa: PLC0415 -- macOS alone withholds names

    return '\n' + quartz.census()


_SHIFT: Final = (40, 8)  # the rows and columns the report searches for a shift of the whole picture
_LEVELS: Final = 8  # the bands of 0..255 the report gives the mean picture value of


def differences(got: np.ndarray, want: np.ndarray, target: Window | Region | None = None) -> str:
    """How ``got`` differs from ``want``, for an assertion's message.

    Shapes; then, over the part both cover, the share of differing pixels, the largest and mean difference per
    channel, where they lie and a few of them, the shift of the whole picture that matches best, and what ``got``
    holds for each band of ``want``'s values (a color conversion shows as a curve). On macOS, the same of ``target``
    as macOS tagged it, before ``quartz.grab`` color-matched it; and the census.
    """
    lines = _compared(got, want)
    if sys.platform == 'darwin' and target is not None:
        from document_lab.capture import quartz  # noqa: PLC0415 -- macOS alone color-matches a grab

        tagged, space = quartz._tagged(target)
        lines += [f'as macOS tagged it, in {space}:', *_compared(tagged, want)]
    return '\n'.join(lines) + _census()


def _compared(got: np.ndarray, want: np.ndarray) -> list[str]:
    h, w = min(len(got), len(want)), min(got.shape[1], want.shape[1])
    lines = [f'got {got.shape}, want {want.shape}; over the {h}x{w} both cover:']
    a, b = got[:h, :w].astype(int), want[:h, :w].astype(int)
    diff = a - b
    off = np.argwhere((diff != 0).any(axis=2))
    lines.append(
        f'  {len(off)} of {h * w} pixels differ ({len(off) / max(h * w, 1):.2%}); '
        f'max |d| per channel {np.abs(diff).max(axis=(0, 1)).tolist()}, mean |d| '
        f'{np.round(np.abs(diff).mean(axis=(0, 1)), 2).tolist()}, mean d {np.round(diff.mean(axis=(0, 1)), 2).tolist()}'
    )
    if len(off):
        (y0, x0), (y1, x1) = off.min(axis=0), off.max(axis=0)
        rows = len(np.unique(off[:, 0]))
        lines.append(f'  differing pixels lie in rows {y0}..{y1} ({rows} rows), columns {x0}..{x1}')
        picks = off[np.linspace(0, len(off) - 1, min(len(off), 6)).astype(int)]
        lines.append('  e.g. ' + '; '.join(f'({y},{x}) {a[y, x].tolist()} vs {b[y, x].tolist()}' for y, x in picks))
        lines.append('  ' + _best_shift(got, want))
        lines.append('  got (R/G/B) by want band: ' + _curve(a, b))
    return lines


def _curve(got: np.ndarray, want: np.ndarray) -> str:
    """The mean of ``got`` over each band of ``want``'s values, channel by channel: a color conversion's curve."""
    step = 256 // _LEVELS
    bands = want // step
    out = []
    for k in range(_LEVELS):
        means = [got[..., c][bands[..., c] == k] for c in range(3)]
        out.append(f'{k * step + step // 2}->' + '/'.join(f'{m.mean():.0f}' if m.size else '-' for m in means))
    return ', '.join(out)


def _best_shift(got: np.ndarray, want: np.ndarray) -> str:
    """The shift (rows, columns) of ``got`` against ``want`` under which a patch of ``want`` matches best."""
    h, w = min(len(got), len(want)), min(got.shape[1], want.shape[1])
    dy, dx = min(_SHIFT[0], h // 4), min(_SHIFT[1], w // 4)
    ph, pw = min(100, h - 2 * dy), min(100, w - 2 * dx)
    patch = want[dy : dy + ph, dx : dx + pw].astype(int)
    scores = {
        (sy, sx): np.abs(got[dy + sy : dy + sy + ph, dx + sx : dx + sx + pw].astype(int) - patch).mean()
        for sy in range(-dy, dy + 1)
        for sx in range(-dx, dx + 1)
    }
    best = min(scores, key=scores.get)
    return f'best shift (rows, columns) {best}: mean |d| {scores[best]:.2f} there, {scores[0, 0]:.2f} unshifted'


HANG: Final = 45  # seconds a child may run, by default, before it dumps where it is and exits


def child(script: str, *args: str, hang: int = HANG) -> subprocess.CompletedProcess:
    """Run ``script`` in a child Python: a hang dumps every thread's stack and ends it, so it fails saying where."""
    guarded = f'import faulthandler\nfaulthandler.dump_traceback_later({hang}, exit=True)\n{script}'
    return subprocess.run(
        [sys.executable, '-c', guarded, *args], capture_output=True, text=True, timeout=hang + 30, check=False
    )


_NOISE: Final = 40  # the spread (standard deviation) of grey levels above which a grab shows the noise picture


def shown(title: str, process: subprocess.Popen) -> Window:
    """The viewer's window once it shows its whole view and its picture: the one its process owns, else the one titled
    ``title``.

    By owner first: macOS names another program's windows only to a process granted Screen Recording. Its picture is
    noise: a window still blank, all one color, is waited for (and named when it stays so).
    """
    pid = int(process.stdout.readline() or 0)
    system = desktop.system()
    deadline = time.monotonic() + 15
    whole: list[Window] = []
    while time.monotonic() < deadline and len(whole) != 1:
        with contextlib.suppress(OSError):  # a window closing while it is read
            found = system.owned(pid) or [w for w in system.windows() if w.title == title]
            whole = [w for w in found if desktop.area(w).height == VIEW]
        time.sleep(0.1)
    if len(whole) != 1:
        code = process.poll()
        state = 'running' if code is None else f'exited with {code}'
        why = system.problem(sends_input=True) or 'the desktop names no missing permission'
        pytest.fail(f'the viewer window {title!r} (pid {pid}, {state}) never appeared: {why}{_census()}')
    while time.monotonic() < deadline:
        grabbed = system.grab(whole[0])
        if grabbed.std() > _NOISE:
            time.sleep(0.5)  # past the viewer's own redraw (``capture_viewer._redrawn``)
            return whole[0]
        time.sleep(0.2)
    pytest.fail(
        f'the viewer window {title!r} appeared but never showed its picture: {_looks(grabbed)}; the screen there: '
        f'{_looks(system.grab(desktop.area(whole[0])))}{_census()}'
    )


def _looks(image: np.ndarray) -> str:
    """What a picture is like, at a glance: its shape, how many colors, the commonest, its spread."""
    colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    return (
        f'{image.shape}, {len(colors)} colors, the commonest {colors[counts.argmax()].tolist()} '
        f'({counts.max() / counts.sum():.0%}), spread {image.std():.1f}'
    )


def test_the_desktop_lets_a_capture_run():
    """This desktop lets a capture grab and scroll -- on macOS, Screen Recording and Accessibility granted to the
    program running Python -- else every window test below fails for that one reason, which this names."""
    why = desktop.system().problem(sends_input=True)
    assert why is None, f'{why}{_census()}'


@contextlib.contextmanager
def _locked(handle: object) -> object:
    """Hold a lock on the first byte of the open file ``handle``, waiting for it; free it after."""
    if sys.platform == 'win32':
        import msvcrt  # noqa: PLC0415 -- the lock of each OS

        take = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # noqa: E731
        free = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # noqa: E731
    else:
        import fcntl  # noqa: PLC0415

        take = lambda: fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1)  # noqa: E731
        free = lambda: fcntl.lockf(handle.fileno(), fcntl.LOCK_UN, 1)  # noqa: E731
    deadline = time.monotonic() + 900
    while True:
        try:
            take()
            break
        except OSError:
            assert time.monotonic() < deadline, 'the desktop stayed held for 900 s'
            time.sleep(0.2)
    try:
        yield
    finally:
        handle.seek(0)
        free()


@contextlib.contextmanager
def the_desktop():
    """Hold the one desktop: pytest workers run in parallel, but two windows on one screen cover each other and take
    each other's input, so a test that shows one waits for this lock across processes."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+b') as handle, _locked(handle):
        yield


@pytest.fixture
def viewer():
    title, seed = f'document-lab capture fixture {os.getpid()}', 7
    with the_desktop():
        process = launched(title, seed)
        try:
            yield shown(title, process), picture(seed)
        finally:
            process.kill()
            process.wait()


@pytest.mark.parametrize('method', ['wheel', 'keys'])
def test_a_scrolled_window_is_captured_whole(viewer, method):
    window, image = viewer
    source = desktop.Source(window, notches=3, method=method)
    rows = Rows()
    done = capture(source.grab, source.scroll, rows)
    assert np.array_equal(rows.image(), image), f'{done}\n{differences(rows.image(), image)}'


def test_a_posted_wheel_places_every_row_it_captures(viewer):
    """``message`` delivers the wheel without moving the pointer, and a program acts on it only some of the time: the
    capture may stop early, where a dropped scroll looks like the end. What it must never do is misplace a row."""
    window, image = viewer
    source = desktop.Source(window, notches=3, method='message')
    rows = Rows()
    done = capture(source.grab, source.scroll, rows)
    got = rows.image()
    assert VIEW <= len(got) <= len(image), done
    assert np.array_equal(got, image[: len(got)]), f'{done}\n{differences(got, image[: len(got)])}'


def test_the_first_frame_is_the_view(viewer):
    window, image = viewer
    got = desktop.system().grab(window)
    assert np.array_equal(got, image[:VIEW]), differences(got, image[:VIEW], window)


def test_a_region_is_the_screen_under_it(viewer):
    window, image = viewer
    area = desktop.area(window)
    inner = Region(area.left + 10, area.top + 20, 100, 50)
    got = desktop.system().grab(inner)
    assert np.array_equal(got, image[20:70, 10:110]), differences(got, image[20:70, 10:110], inner)


_MAIN = """
import runpy, sys
sys.argv = sys.argv[1:]
runpy.run_module('document_lab', run_name='__main__', alter_sys=True)
"""  # ``python -m document_lab <argv>``, as a `child`: argv[0] the module


def test_the_cli_writes_the_pdf(viewer, tmp_path):
    """Run as a script runs it: ``--now`` starts without the panel's Start, in a `child`, so a wait for a click nobody
    gives is a failure saying where it waits, never a hang."""
    window, image = viewer
    out = tmp_path / 'capture.pdf'
    argv = [
        'document_lab',
        'capture',
        str(out),
        '--handle',
        hex(window.handle),
        '--now',
        '--auto',
        '--notches',
        '3',
        '--delay',
        '0',
        '--page-height',
        '0',
    ]
    done = child(_MAIN, *argv, hang=90)
    assert done.returncode == 0, done.stdout + done.stderr
    with pymupdf.open(out) as doc:
        assert doc.page_count == 1
        ((xref, *_),) = doc[0].get_images()
        pix = pymupdf.Pixmap(doc, xref)
    got = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    assert np.array_equal(got, image), differences(got, image)


def test_max_height_stops_early(viewer):
    window, image = viewer
    source = desktop.Source(window)
    rows = Rows()
    done = capture(source.grab, source.scroll, rows, limits=Limits(max_height=700))
    got = rows.image()
    assert 700 <= len(got) < HEIGHT, done
    assert np.array_equal(got, image[: len(got)]), f'{done}\n{differences(got, image[: len(got)])}'


def test_a_wheel_onto_a_covered_window_refuses(viewer):
    window, _ = viewer
    title = f'document-lab capture cover {os.getpid()}'
    cover = launched(title, 8)  # opens at the same place, above the first
    try:
        shown(title, cover)
        with pytest.raises(OSError, match='covers'):
            desktop.Source(window).scroll()
    finally:
        cover.kill()
        cover.wait()


# The selector is driven through Tk's own event dispatch (event_generate), not OS input -- a click the OS did not
# deliver would leave select() waiting forever -- and in a child (`child`), so even a hang is a failure saying where. A
# point is given on the canvas from where the overlay truly is: an OS may keep a window off its menu bar. Each step
# and what it left marked is said on stderr, for a failure to show how far it got.
SELECT = """
import json, sys
from document_lab.capture import desktop
steps = json.loads(sys.argv[1])
o = desktop._Overlay(desktop.system().windows(), desktop.virtual_screen())
o.canvas.focus_force()
def at(x, y):
    return {'x': x - o.canvas.winfo_rootx(), 'y': y - o.canvas.winfo_rooty()}
def play():
    print(f'windows {o.found}; canvas at {o.canvas.winfo_rootx()},{o.canvas.winfo_rooty()}', file=sys.stderr)
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
        print(f'{kind} {where}: marked {o.marked}, confirmed {o.confirmed}', file=sys.stderr, flush=True)
        o.root.update()
    print('played', file=sys.stderr, flush=True)
o.root.after(300, play)
chosen = o.run()
print('left the loop', file=sys.stderr, flush=True)
print(repr(chosen))
"""


def selected(*steps: tuple) -> str:
    """What the overlay returns after ``steps``: ('drag', a, b), ('click', a), ('confirm',) or ('key', sequence)."""
    done = child(SELECT, json.dumps(steps))
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()


def test_a_drag_confirmed_by_enter_is_the_region(viewer):
    window, _ = viewer
    area = desktop.area(window)
    start, end = (area.left + 10, area.top + 20), (area.left + 110, area.top + 70)
    assert selected(('drag', start, end), ('key', '<Return>')) == repr(Region(area.left + 10, area.top + 20, 100, 50))


def test_a_click_marks_the_window_and_start_confirms_it(viewer):
    window, _ = viewer
    area = desktop.area(window)
    drag = ('drag', (area.left + 10, area.top + 20), (area.left + 110, area.top + 70))  # replaced by the click
    assert selected(drag, ('click', area.center), ('confirm',)) == repr(window)


def test_nothing_is_taken_before_it_is_confirmed(viewer):
    window, _ = viewer
    assert selected(('click', desktop.area(window).center), ('key', '<Escape>')) == 'None'


def test_confirming_by_the_confirm_button_does_not_free_the_canvas_under_its_own_event():
    """Regression: confirming destroyed the window inside its click handler; the canvas's own release binding then ran
    on freed memory and Tcl aborted the process ("alloc: invalid block"). Run in a child so an abort is a failure."""
    script = """
import types
from document_lab.capture import desktop
screen = desktop.virtual_screen()
o = desktop._Overlay(desktop.system().windows(), screen)
o.root.update()
ev = types.SimpleNamespace
o.press(ev(x_root=screen.left + 100, y_root=screen.top + 100))
o.release(ev(x_root=screen.left + 500, y_root=screen.top + 400))
o.root.update()
o.root.after(300, o.confirm_button.invoke)  # inside the loop, as a click: the handler must not destroy what runs it
print(o.run())
"""
    with the_desktop():
        done = child(script)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith('Region('), done.stdout


def test_the_panel_waits_for_a_start_adjust_or_cancel_and_begin_turns_it_into_finish_and_pause():
    """After the area is confirmed the HUD waits, nothing over the desktop, until its own buttons answer."""
    script = """
from document_lab.capture import desktop
from document_lab.capture.screen import Region
for press, want in (('button', 'start'), ('auto', 'auto'), ('adjust', 'adjust'), ('cancel', 'cancel')):
    hud = desktop.Hud(Region(200, 200, 300, 200))
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
        done = child(script)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'ok', done.stdout


def test_adjusting_opens_the_overlay_with_the_area_set_already_marked():
    """Adjust reopens the selection on the area set: confirmed untouched, it comes back as it was."""
    script = """
from document_lab.capture import desktop
from document_lab.capture.screen import Region
area = Region(150, 120, 400, 250)
o = desktop._Overlay(desktop.system().windows(), desktop.virtual_screen(), area)
o.root.after(300, o.confirm_button.invoke)
print(repr(o.run()))
"""
    with the_desktop():
        done = child(script)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == repr(Region(150, 120, 400, 250)), done.stdout


#: Every test of the body, for a part module to import (``from tests.capture_part import *`` names them all).
__all__ = [
    'test_a_click_marks_the_window_and_start_confirms_it',
    'test_a_drag_confirmed_by_enter_is_the_region',
    'test_a_posted_wheel_places_every_row_it_captures',
    'test_a_region_is_the_screen_under_it',
    'test_a_scrolled_window_is_captured_whole',
    'test_a_wheel_onto_a_covered_window_refuses',
    'test_adjusting_opens_the_overlay_with_the_area_set_already_marked',
    'test_confirming_by_the_confirm_button_does_not_free_the_canvas_under_its_own_event',
    'test_max_height_stops_early',
    'test_nothing_is_taken_before_it_is_confirmed',
    'test_the_cli_writes_the_pdf',
    'test_the_desktop_lets_a_capture_run',
    'test_the_first_frame_is_the_view',
    'test_the_panel_waits_for_a_start_adjust_or_cancel_and_begin_turns_it_into_finish_and_pause',
    'viewer',
]
