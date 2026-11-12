"""The capture command end to end over a fake source: rows stream into the PDF and the PNG as they are stitched, a
capture that stops short keeps what it has and says so, and nothing is half-written under the real names."""

from typing import ClassVar

import numpy as np
import pymupdf
import pytest

from document_lab.__main__ import main
from document_lab.capture import desktop
from tests.unit.test_capture_session import Hand, Viewport
from tests.unit.test_capture_stitch import tall

WIDTH = 64  # the fake view's width: tests.unit.test_capture_stitch.tall's


class FakeSource:
    """``desktop.Source`` over a :class:`Viewport` of ``IMAGE``, ``STEP`` rows a scroll."""

    IMAGE = tall(2000)
    STEP = 170  # what --auto's one notch scrolls

    def __init__(self, target, notches, method):
        self.view = Viewport(self.IMAGE, target.height, self.STEP)

    def grab(self):
        return self.view.grab()

    def scroll(self, notches=1):
        self.view.scroll(notches)


class FakeHud:
    """``desktop.Hud`` without a screen: what was shown, in order."""

    shown: ClassVar[list] = []
    rows = 0
    done = staticmethod(lambda: False)  # what the Finish button says, per test
    answers: ClassVar[list] = []  # what the panel answers before the capture, in turn; then Start (I scroll)

    def __init__(self, area):
        FakeHud.shown, FakeHud.rows = [], 0

    def status(self, text, *, ok):
        FakeHud.shown.append((ok, text))

    def add(self, rows):
        FakeHud.rows += len(rows)

    def pump(self):
        return None

    def ready(self):
        FakeHud.shown.append('ready')
        return FakeHud.answers.pop(0) if FakeHud.answers else 'start'

    @property
    def finished(self):
        return FakeHud.done()

    def close(self):
        FakeHud.shown.append('closed')


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(desktop, 'Source', FakeSource)
    monkeypatch.setattr(desktop, 'Hud', FakeHud)
    monkeypatch.setattr(FakeHud, 'done', staticmethod(lambda: False))
    return FakeSource


def run(tmp_path, *more):
    argv = [
        'capture',
        str(tmp_path / 'c.pdf'),
        '--rect',
        f'0,0,{WIDTH},300',
        '--delay',
        '0',
        '--settle',
        '0',
        '--every',
        '0',
    ]
    return main([*argv, '--end-wait', '0', *more])


def auto(tmp_path, *more):
    return run(tmp_path, '--auto', *more)


def pdf_pixels(path):
    with pymupdf.open(path) as doc:
        pages = []
        for page in doc:
            ((xref, *_),) = page.get_images()
            pix = pymupdf.Pixmap(doc, xref)
            pages.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
        return np.concatenate(pages), doc.page_count


def png_pixels(path):
    pix = pymupdf.Pixmap(str(path))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def test_the_whole_view_streams_into_the_pdf_and_the_png(tmp_path, fake):
    assert auto(tmp_path, '--page-height', '300', '--png', str(tmp_path / 'c.png')) == 0
    image, pages = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE)
    assert pages == 7
    assert np.array_equal(png_pixels(tmp_path / 'c.png'), fake.IMAGE)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['c.pdf', 'c.png']


def test_a_capture_that_cannot_be_stitched_keeps_what_it_has_and_exits_1(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(fake, 'STEP', 290)  # past what a 300-row view can overlap
    assert auto(tmp_path, '--page-height', '100') == 1
    image, _ = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE[:300])


def test_a_page_no_reader_accepts_refuses_before_capturing(tmp_path, fake):
    with pytest.raises(ValueError, match='14400'):
        auto(tmp_path, '--page-height', '100000')
    assert not list(tmp_path.iterdir())


def test_by_default_it_follows_the_users_scrolling_until_esc(tmp_path, fake, monkeypatch):
    hand = Hand(fake.IMAGE, 300, [0, 150, 0, 0, 200, 220, 0, 100])
    monkeypatch.setattr(fake, 'grab', lambda self: hand.grab())
    monkeypatch.setattr(FakeHud, 'done', staticmethod(hand.esc))
    assert run(tmp_path, '--page-height', '0') == 0
    image, _ = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE[: 300 + 150 + 200 + 220 + 100])


def test_a_view_the_scroll_did_not_move_is_reported_not_taken_for_the_end(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setattr(fake, 'STEP', 0)
    assert auto(tmp_path) == 1
    assert 'leave out --auto' in capsys.readouterr().out


def test_the_screen_shows_the_capture_growing_and_which_way_back_when_lost(tmp_path, fake, monkeypatch):
    hand = Hand(fake.IMAGE, 300, [150, 600, -600, 150])  # on; too far down; back; on
    monkeypatch.setattr(fake, 'grab', lambda self: hand.grab())
    monkeypatch.setattr(FakeHud, 'done', staticmethod(hand.esc))
    assert run(tmp_path, '--page-height', '0') == 0
    assert FakeHud.shown[0] == 'ready'  # nothing is shown captured before Start
    oks = [ok for ok, _ in FakeHud.shown[1:-1]]
    assert oks == [True, False, True]  # growing; lost (red); growing again
    assert 'UP' in FakeHud.shown[2][1]
    assert FakeHud.shown[-1] == 'closed'
    assert FakeHud.rows == 300 + 150 + 150  # the miniature saw every row captured


def test_cancel_on_the_panel_before_start_grabs_and_writes_nothing(tmp_path, fake, monkeypatch):
    grabbed = []
    monkeypatch.setattr(fake, 'grab', lambda self: grabbed.append(1))
    monkeypatch.setattr(FakeHud, 'answers', ['cancel'])
    assert run(tmp_path) == 1
    assert not grabbed
    assert not list(tmp_path.iterdir())
    assert FakeHud.shown == ['ready', 'closed']


def test_adjust_on_the_panel_reopens_the_selection_on_the_area_set_then_waits_again(tmp_path, fake, monkeypatch):
    asked, waited = [], []
    monkeypatch.setattr(desktop, 'select', lambda marked: asked.append(marked) or marked)
    answer = FakeHud.ready
    monkeypatch.setattr(FakeHud, 'ready', lambda self: waited.append(1) or answer(self))
    monkeypatch.setattr(FakeHud, 'answers', ['adjust'])
    assert run(tmp_path, '--no-scroll') == 0
    assert asked == [desktop.Region(0, 0, WIDTH, 300)]  # the area set, shown marked to adjust
    assert len(waited) == 2  # the new area waits for Start too


def test_auto_scroll_on_the_panel_scrolls_for_the_user_to_the_end(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(FakeHud, 'answers', ['auto'])
    assert run(tmp_path, '--page-height', '0') == 0  # no --auto: the panel chose it
    image, _ = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE)


def test_an_existing_output_is_refused_before_anything_unless_overwrite(tmp_path, fake, monkeypatch):
    for name in ('c.pdf', 'c.png'):
        (tmp_path / name).write_bytes(b'the document on screen')
    asked = []
    monkeypatch.setattr(FakeHud, 'ready', lambda self: asked.append(1) or 'start')
    with pytest.raises(FileExistsError, match='--overwrite'):
        run(tmp_path, '--no-scroll', '--png', str(tmp_path / 'c.png'))
    (tmp_path / 'c.pdf').unlink()
    with pytest.raises(FileExistsError, match=r'c\.png'):
        run(tmp_path, '--no-scroll', '--png', str(tmp_path / 'c.png'))
    assert not asked  # refused before the area was even waited on
    assert (tmp_path / 'c.png').read_bytes() == b'the document on screen'
    assert run(tmp_path, '--no-scroll', '--png', str(tmp_path / 'c.png'), '--overwrite') == 0
    assert (tmp_path / 'c.png').read_bytes() != b'the document on screen'


def test_a_scroll_windows_refuses_ends_the_capture_with_what_it_has_and_why(tmp_path, fake, monkeypatch, capsys):
    scrolled = []

    def refused(self, notches=1):
        if len(scrolled) == 3:
            msg = 'SendInput was blocked (error 5): is a higher-integrity window in front?'
            raise OSError(msg)  # what desktop._send raises
        scrolled.append(notches)
        self.view.scroll(notches)

    monkeypatch.setattr(fake, 'scroll', refused)
    assert auto(tmp_path, '--page-height', '0') == 1  # no traceback: an exit status
    out = capsys.readouterr().out
    assert 'SendInput was blocked' in out
    assert 'UAC' in out
    assert 'lock screen' in out
    assert '1 page(s)' in out
    image, _ = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE[: 300 + 3 * fake.STEP])  # every frame placed before it is written


def test_frames_that_share_nothing_say_page_flip_mode_not_a_larger_tolerance(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setattr(fake, 'STEP', 300)  # a whole view a scroll: a reader turning pages
    assert auto(tmp_path, '--page-height', '0') == 1
    out = capsys.readouterr().out
    assert 'page-flip' in out
    assert 'continuous' in out
    assert 'larger --tolerance will not help' in out


def test_frames_that_nearly_match_still_suggest_a_larger_tolerance(tmp_path, fake, monkeypatch, capsys):
    rng = np.random.default_rng(1)
    grab = FakeSource.grab

    def noisy(self):  # re-rendered each grab: the true offset errs about 10 grey levels, above 8 but near it
        frame = grab(self).astype(np.int16) + rng.integers(-17, 18, (300, WIDTH, 3))
        return np.clip(frame, 0, 255).astype(np.uint8)

    monkeypatch.setattr(fake, 'grab', noisy)
    assert auto(tmp_path, '--page-height', '0') == 1
    out = capsys.readouterr().out
    assert 'try a larger --tolerance' in out
    assert 'page-flip' not in out


def test_page_height_auto_is_accepted_and_reported(tmp_path, fake, capsys):
    assert auto(tmp_path, '--page-height', 'auto') == 0
    assert 'no page separators found' in capsys.readouterr().out  # noise has none: A4-shaped, and it says so


def test_python_3_13_0_in_a_venv_without_tcl_is_told_before_tkinter_starts(tmp_path):
    venv, base = tmp_path / 'venv', tmp_path / 'base'
    venv.mkdir()
    problem = desktop.tk_problem((3, 13, 0), str(venv), str(base), {})
    assert problem is not None
    assert '3.13.1' in problem
    assert str(base / 'tcl') in problem
    assert desktop.tk_problem((3, 13, 1), str(venv), str(base), {}) is None  # the controls: fixed in 3.13.1,
    assert desktop.tk_problem((3, 13, 0), str(base), str(base), {}) is None  # not in a venv,
    assert desktop.tk_problem((3, 13, 0), str(venv), str(base), {'TCL_LIBRARY': 'x'}) is None  # Tcl named,
    (venv / 'tcl').mkdir()
    assert desktop.tk_problem((3, 13, 0), str(venv), str(base), {}) is None  # or the tcl folder copied in


def test_a_tkinter_that_cannot_start_ends_the_run_with_the_remedy(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setattr(desktop, 'tk_problem', lambda: 'Use Python 3.13.1 or later')
    shown = []
    monkeypatch.setattr(FakeHud, 'ready', lambda self: shown.append(1) or 'start')
    assert run(tmp_path) == 1
    assert 'Use Python 3.13.1' in capsys.readouterr().out
    assert not shown
    assert not list(tmp_path.iterdir())
