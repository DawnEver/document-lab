"""The capture loop against a fake viewport over a known tall image: it scrolls to the end, stops there, honours
its caps, and waits for a frame that is still changing to settle."""

import numpy as np

from document_lab.capture.session import Limits, Rows, capture, follow
from tests.unit.test_capture_stitch import tall


class Viewport:
    """A view of ``image``, ``height`` rows tall, that scrolls by ``step`` and stops at the end."""

    def __init__(self, image: np.ndarray, height: int, step: int, lag: int = 0):
        self.image, self.height, self.step, self.lag = image, height, step, lag
        self.top = self.pending = self.scrolls = self.moved = 0

    def grab(self) -> np.ndarray:
        top = self.top - self.moved * self.pending // (self.lag + 1)  # mid-animation: short by a shrinking share
        self.pending = max(self.pending - 1, 0)
        return self.image[top : top + self.height].copy()

    def scroll(self) -> None:
        self.scrolls += 1
        before = self.top
        self.top = min(self.top + self.step, len(self.image) - self.height)
        self.moved, self.pending = self.top - before, self.lag


def run(view: Viewport, **knobs) -> np.ndarray:
    rows = Rows()
    capture(view.grab, view.scroll, rows, sleep=lambda _: None, **knobs)
    return rows.image()


def test_scrolls_to_the_end_and_stops_there():
    image = tall(1234)
    view = Viewport(image, 300, 170)
    assert np.array_equal(run(view), image)
    assert view.scrolls == 7  # 6 moves, then one that did not move


def test_waits_for_the_view_to_settle():
    image = tall(900)
    assert np.array_equal(run(Viewport(image, 300, 160, lag=2)), image)


class LateViewport(Viewport):
    """A view whose scroll shows only after ``late`` grabs: equal frames first, as a queued message gives."""

    def __init__(self, image: np.ndarray, height: int, step: int, late: int):
        super().__init__(image, height, step)
        self.late, self.waiting, self.shown = late, 0, 0

    def grab(self) -> np.ndarray:
        if self.waiting:
            self.waiting -= 1
        else:
            self.shown = self.top
        return self.image[self.shown : self.shown + self.height].copy()

    def scroll(self) -> None:
        super().scroll()
        self.waiting = self.late


def test_a_scroll_handled_late_is_not_the_end():
    image = tall(1000)
    assert np.array_equal(run(LateViewport(image, 300, 150, late=3)), image)


def test_without_scroll_it_is_one_frame():
    image = tall(900)
    view = Viewport(image, 300, 160)
    rows = Rows()
    assert capture(view.grab, None, rows).stopped == 'one frame'
    assert np.array_equal(rows.image(), image[:300])
    assert view.scrolls == 0


def test_max_frames_caps_it():
    view = Viewport(tall(2000), 300, 100)
    assert len(run(view, limits=Limits(max_frames=3))) == 500


def test_max_height_caps_it():
    view = Viewport(tall(2000), 300, 100)
    assert len(run(view, limits=Limits(max_height=550))) == 600


def test_abort_stops_with_what_it_has():
    view = Viewport(tall(2000), 300, 100)
    calls = iter([False, False, True])
    assert len(run(view, aborted=lambda: next(calls))) == 500


def test_a_scroll_too_large_to_stitch_keeps_the_first_frame_and_says_why():
    view, rows = Viewport(tall(2000), 300, 290), Rows()
    got = capture(view.grab, view.scroll, rows, sleep=lambda _: None)
    assert got.stopped.startswith('unstitchable: ')
    assert not got.complete
    assert np.array_equal(rows.image(), view.image[:300])


def test_it_says_why_it_stopped():
    image = tall(1234)
    done = capture((v := Viewport(image, 300, 170)).grab, v.scroll, Rows(), sleep=lambda _: None)
    assert (done.stopped, done.complete, done.frames, done.height) == ('end', True, 7, 1234)
    capped = capture(
        (v := Viewport(image, 300, 170)).grab, v.scroll, Rows(), limits=Limits(max_frames=3), sleep=lambda _: None
    )
    assert capped.stopped == 'max_frames'


def test_progress_sees_every_placed_frame():
    view, heights = Viewport(tall(1234), 300, 170), []
    capture(view.grab, view.scroll, Rows(), sleep=lambda _: None, progress=lambda s: heights.append(s.height))
    assert heights == [470, 640, 810, 980, 1150, 1234]


def test_an_endless_view_streams_and_holds_one_view():
    image = tall(30000)
    view, rows, held = Viewport(image, 300, 170), Rows(), []
    got = capture(view.grab, view.scroll, rows, sleep=lambda _: None, progress=lambda s: held.append(s.held))
    assert got.complete
    assert max(held) <= 3 * 300
    assert np.array_equal(rows.image(), image)


class Hand:
    """A view the user scrolls: ``moves[i]`` rows before the i-th grab (negative: up), then Esc."""

    def __init__(self, image: np.ndarray, height: int, moves: list[int]):
        self.image, self.height, self.moves, self.top, self.grabs = image, height, moves, 0, 0

    def grab(self) -> np.ndarray:
        if self.grabs and self.grabs <= len(self.moves):
            self.top = min(max(self.top + self.moves[self.grabs - 1], 0), len(self.image) - self.height)
        self.grabs += 1
        return self.image[self.top : self.top + self.height].copy()

    def esc(self) -> bool:
        return self.grabs > len(self.moves)


def test_following_the_user_waits_through_pauses_and_ends_on_esc():
    image = tall(1500)
    hand, rows = Hand(image, 300, [0, 0, 120, 0, 0, 0, 200, 150, 0, 200, 0, 0]), Rows()
    got = follow(hand.grab, rows, aborted=hand.esc, sleep=lambda _: None)
    assert got.stopped == 'aborted'
    assert np.array_equal(rows.image(), image[: 300 + 120 + 200 + 150 + 200])


def test_a_scroll_too_fast_or_back_up_is_skipped_until_the_view_overlaps_again():
    image = tall(2000)
    lost = []
    hand = Hand(image, 300, [100, 400, -380, 100, -150, 150, 90])  # too far; back; on; up above it; back; on
    rows = Rows()
    follow(hand.grab, rows, aborted=hand.esc, lost=lost.append, sleep=lambda _: None)
    assert len(lost) == 2  # the jump past the view, and the view scrolled back above the last frame
    assert np.array_equal(rows.image(), image[: hand.top + 300])


def test_a_pause_over_a_blank_view_is_not_lost():
    image = np.full((1200, 64, 3), 255, dtype=np.uint8)
    image[:400] = tall(400)
    hand, rows, lost = Hand(image, 300, [0, 0, 0, 0]), Rows(), []
    follow(hand.grab, rows, aborted=hand.esc, lost=lost.append, sleep=lambda _: None)
    assert lost == []
