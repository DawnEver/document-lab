"""The capture loop against a fake viewport over a known tall image: it scrolls to the end, stops there, honours
its caps, and waits for a frame that is still changing to settle."""

import numpy as np

from document_lab.capture.session import Limits, Pace, Rows, Settling, capture, follow
from document_lab.capture.stitch import Match
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

    def scroll(self, notches: int = 1) -> None:
        self.scrolls += 1
        before = self.top
        self.top = min(max(self.top + self.step * notches, 0), len(self.image) - self.height)
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

    def scroll(self, notches: int = 1) -> None:
        super().scroll(notches)
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
    assert len(run(view, limits=Limits(max_frames=3), pace=Pace(stride=0))) == 500  # 100 rows a step


def test_max_height_caps_it():
    view = Viewport(tall(2000), 300, 100)
    assert len(run(view, limits=Limits(max_height=550))) == 600


def test_abort_stops_with_what_it_has():
    view = Viewport(tall(2000), 300, 100)
    calls = iter([False, False, True])
    assert len(run(view, aborted=lambda: next(calls), pace=Pace(stride=0))) == 500


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


def test_an_endless_view_streams_and_holds_a_bounded_history():
    image = tall(30000)
    view, rows, held = Viewport(image, 300, 170), Rows(), []
    got = capture(view.grab, view.scroll, rows, sleep=lambda _: None, progress=lambda s: held.append(s.held))
    assert got.complete
    assert max(held) <= 4 * 300  # three views of history, kept to recover in, and the view
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


def test_back_up_within_what_was_captured_picks_up_there_and_too_far_is_skipped():
    image = tall(2000)
    lost = []
    hand = Hand(image, 300, [100, 400, -380, 100, -150, 150, 90])  # too far; back; on; up into the past; on
    rows = Rows()
    follow(hand.grab, rows, aborted=hand.esc, lost=lost.append, sleep=lambda _: None)
    assert [x.back for x in lost] == ['up']  # only the jump past the view is lost; back up, it picked up again
    assert np.array_equal(rows.image(), image[: hand.top + 300])


def test_after_a_flick_too_far_any_part_already_captured_resumes_the_capture():
    image = tall(3000)
    lost = []
    hand = Hand(image, 300, [200, 200, 900, -700, 200, 200])  # a flick past the view, then back into what was held
    rows = Rows()
    follow(hand.grab, rows, aborted=hand.esc, lost=lost.append, sleep=lambda _: None)
    assert len(lost) == 1
    assert np.array_equal(rows.image(), image[: hand.top + 300])


def test_a_pause_over_a_blank_view_is_not_lost():
    image = np.full((1200, 64, 3), 255, dtype=np.uint8)
    image[:400] = tall(400)
    hand, rows, lost = Hand(image, 300, [0, 0, 0, 0]), Rows(), []
    follow(hand.grab, rows, aborted=hand.esc, lost=lost.append, sleep=lambda _: None)
    assert lost == []


def test_a_small_notch_is_measured_and_the_steps_grow_to_half_the_view():
    image = tall(3000)
    paced, fixed = Viewport(image, 300, 30), Viewport(image, 300, 30)  # a notch scrolls only 30 rows
    assert np.array_equal(run(paced), image)
    assert np.array_equal(run(fixed, pace=Pace(stride=0)), image)
    assert paced.scrolls < fixed.scrolls // 4  # 1, 2, 4, then 5 notches (150 rows) a step


def test_a_large_notch_is_cut_down_to_half_the_view():
    assert Pace().next(4, 280, 300) == 2  # 70 rows a notch: two make 140, about half of 300
    assert Pace().next(3, 0, 300) == 3  # nothing moved: nothing measured
    assert Pace(stride=0).next(3, 30, 300) == 3


class Glitchy(Viewport):
    """A view whose ``bad``-th scroll goes ``jump`` rows further at once, or shows ``noise`` rendering for one grab."""

    def __init__(self, image, height, step, bad, *, jump=0, noise=False):
        super().__init__(image, height, step)
        self.bad, self.jump, self.noise, self.ups = bad, jump, noise, 0

    def scroll(self, notches=1):
        self.ups += notches < 0
        super().scroll(notches)
        if self.scrolls == self.bad and notches > 0:
            self.top = min(self.top + self.jump, len(self.image) - self.height)
            self.noisy = self.noise

    def grab(self):
        frame = super().grab()
        if getattr(self, 'noisy', False):
            self.noisy = False
            frame = np.random.default_rng(0).integers(0, 256, frame.shape, dtype=np.uint8)
        return frame


def test_a_frame_caught_mid_render_is_looked_at_again_not_the_end():
    image = tall(1500)
    view = Glitchy(image, 300, 150, bad=3, noise=True)
    assert np.array_equal(run(view, pace=Pace(stride=0)), image)
    assert not view.ups  # looking again was enough


def test_a_step_that_outran_the_overlap_scrolls_back_and_goes_on():
    image = tall(2000)
    view = Glitchy(image, 300, 150, bad=3, jump=300)  # 450 rows at once: nothing shared with a 300-row view
    lost = []
    assert np.array_equal(run(view, pace=Pace(stride=0), lost=lost.append), image)
    assert view.ups and not lost


def test_a_view_that_cannot_be_recovered_is_told_lost_and_ends_the_capture():
    image = tall(2000)
    view = Glitchy(image, 300, 150, bad=3, jump=300)
    lost = []
    rows = Rows()
    done = capture(
        view.grab, lambda n: view.scroll(max(n, 1)), rows, pace=Pace(stride=0), lost=lost.append, sleep=lambda _: None
    )  # a scroll that cannot go back up
    assert done.stopped.startswith('unstitchable: ')
    assert len(lost) == 1
    assert np.array_equal(rows.image(), image[: done.height])


class SlowRender(Viewport):
    """A reader that renders ``render`` seconds after each scroll: until then the view shows the new position with
    everything below its top third still blank -- the same unfinished frame on every grab."""

    def __init__(self, image, height, step, render):
        super().__init__(image, height, step)
        self.render, self.clock, self.since = render, 0.0, -render  # rendered before the first scroll

    def sleep(self, seconds):
        self.clock += seconds

    def scroll(self, notches=1):
        super().scroll(notches)
        self.since = self.clock

    def grab(self):
        frame = super().grab()
        if self.clock - self.since < self.render:
            frame[self.height // 3 :] = 255
        return frame


def slow_capture(view, settling):
    rows = Rows()
    done = capture(view.grab, view.scroll, rows, pace=Pace(stride=0), settling=settling, sleep=view.sleep)
    return done, rows.image()


def test_a_slow_renderer_grows_the_settle_gap_until_it_is_seen_rendered():
    image = tall(3000)
    done, rows = slow_capture(SlowRender(image, 300, 150, render=0.5), Settling(interval=0.12, max_interval=1.0))
    assert done.complete
    assert np.array_equal(rows, image)


def test_a_fixed_settle_gap_shorter_than_the_render_bounces_and_gives_up():
    image = tall(3000)  # the control: the same reader with the gap held at --settle
    done, _ = slow_capture(SlowRender(image, 300, 150, render=0.5), Settling(interval=0.12, max_interval=0.12))
    assert done.stopped.startswith('unstitchable: ')


def test_a_scroll_the_os_refuses_stops_the_capture_with_what_it_has():
    image = tall(1500)
    view = Viewport(image, 300, 150)
    scrolled = []

    def scroll(notches):
        if len(scrolled) == 2:
            msg = 'SendInput was blocked (error 5)'
            raise OSError(msg)
        scrolled.append(notches)
        view.scroll(notches)

    rows = Rows()
    done = capture(view.grab, scroll, rows, pace=Pace(stride=0), sleep=lambda _: None)
    assert done.stopped == 'blocked: SendInput was blocked (error 5)'
    assert np.array_equal(rows.image(), image[: 300 + 2 * 150])


def test_a_frame_that_shares_nothing_reports_how_far_off_its_best_match_was():
    image = tall(1500)
    view = Viewport(image, 300, 300)  # a whole view a scroll: nothing shared
    done = capture(view.grab, view.scroll, Rows(), pace=Pace(stride=0), sleep=lambda _: None)
    assert done.stopped.startswith('unstitchable: ')
    assert done.error is not None
    assert done.error > 3 * Match().tolerance  # random rows: far above the tolerance
