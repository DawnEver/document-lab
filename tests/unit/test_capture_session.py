"""The capture loop against a fake viewport over a known tall image: it scrolls to the end, stops there, honours
its caps, and waits for a frame that is still changing to settle."""

import numpy as np
import pytest

from document_lab.capture.session import Limits, capture
from document_lab.capture.stitch import NoOverlap
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
    return capture(view.grab, view.scroll, sleep=lambda _: None, **knobs)


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
    assert np.array_equal(capture(view.grab, None), image[:300])
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


def test_a_scroll_too_large_to_stitch_raises():
    with pytest.raises(NoOverlap):
        run(Viewport(tall(2000), 300, 290))
