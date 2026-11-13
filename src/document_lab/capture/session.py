"""One capture: grab a frame, scroll, wait for the view to settle, stitch on -- until the view stops moving.

The source is two callables, ``grab() -> frame`` and ``scroll()``, so the loop knows nothing of windows or screens
(`document_lab.capture.desktop` supplies them on Windows). The end is the first scroll that moves nothing; the caps
in :class:`Limits`, the ``aborted`` callback (the CLI binds Esc) and a frame that cannot be
stitched stop it earlier. There is no cap by default: a view as long as it scrolls is captured.

The stitched rows go to ``sink`` as soon as no later frame can change them, so a capture of any length holds about
one view in memory; :class:`Rows` is a sink that keeps them, for a capture known to be short. Everything runs in the
caller's thread, in order -- grab, scroll, stitch, sink -- so nothing is shared and nothing races. Whatever stops
it, every stitched row reaches ``sink`` and the :class:`Capture` says why it stopped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab.capture.stitch import Match, NoOverlap, Stitcher, Unstitchable, offset

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ['Capture', 'Limits', 'Lost', 'Pace', 'Rows', 'Settling', 'capture', 'follow']


@dataclass(frozen=True)
class Limits:
    """When to stop before the end: frames grabbed, stitched height in pixels (``None``: no cap)."""

    max_frames: int | None = None
    max_height: int | None = None


@dataclass(frozen=True)
class Settling:
    """After a scroll, grab every ``interval`` seconds until two frames agree, at most ``tries`` times.

    ``interval`` is the least gap between the two: a reader that renders slowly shows the same unfinished frame twice
    in a short gap, so a frame taken for settled that a second look then places (it had changed after all) doubles
    the gap for every later step, up to ``max_interval`` -- the gap grows to the renderer's pace instead of the
    capture bouncing between two pages.

    A view that has not moved is looked at once more after ``end_wait`` seconds before it is taken for the end: a
    scroll the application handles late (a posted message, a busy view) shows two equal frames first, and a page that
    loads more as it nears its end (an infinite feed) needs that long to grow.
    """

    interval: float = 0.12
    tries: int = 15
    end_wait: float = 0.8
    max_interval: float = 1.0


@dataclass(frozen=True)
class Pace:
    """How far each automatic scroll goes: ``first`` notches, then as many as move ``stride`` of the view.

    A notch scrolls however far the application makes it -- a few rows at one zoom, a screenful at another -- so it is
    measured, not assumed: each step's rows per notch set the next step's notches, at most doubling, so a step never
    outruns what consecutive frames must share. ``stride`` 0 keeps ``first`` notches every step.

    It adapts, as a congestion window does: every ``streak`` steps placed in a row, the share aimed for grows by
    :data:`_GROW`, up to ``most`` of the view; a step that loses the view drops it back to ``stride`` (and halves the
    notches, as before). ``most`` at or below ``stride`` holds it at ``stride``. ``most`` stays under the 5/6 of the
    view two frames may scroll and still share the sixth that places them.
    """

    first: int = 1
    stride: float = 0.5
    most: float = 0.75
    streak: int = 3

    def next(self, notches: int, moved: int, rows: int, ceiling: int | None = None, reach: float | None = None) -> int:
        """The notches for the step after one of ``notches`` that moved ``moved`` of a ``rows``-row view.

        Aimed at ``reach`` of the view (default ``stride``); never past ``ceiling``: half a step that once lost it.
        """
        reach = self.stride if reach is None else reach
        if not (self.stride and moved):
            return notches
        grown = max(1, min(2 * notches, round(reach * rows * notches / moved)))
        return grown if ceiling is None else min(grown, ceiling)

    def grown(self, reach: float) -> float:
        """The share aimed for after ``streak`` steps placed in a row at ``reach``."""
        return max(reach, min(self.most, reach * _GROW))


@dataclass(frozen=True)
class Capture:
    """One capture: the ``frames`` it took, the ``height`` it stitched, and why it ``stopped``.

    ``stopped`` is ``end``, ``aborted``, ``max_frames``, ``max_height``, ``one frame``, ``unstitchable: <reason>``
    or ``blocked: <reason>`` (the scroll was refused: the input went nowhere); ``stuck`` is true when the first scroll
    already moved nothing -- the source did not scroll at all. ``error``: for a frame that overlapped nothing, the
    mean grey error of its best offset.
    """

    frames: int
    height: int
    stopped: str
    stuck: bool = False
    error: float | None = None

    @property
    def complete(self) -> bool:
        """Did it reach the end of the view?"""
        return self.stopped == 'end'


class Rows:
    """A sink that keeps every row it is given: the whole image, for a capture known to be short."""

    def __init__(self) -> None:
        """Empty."""
        self._parts: list[np.ndarray] = []

    def __call__(self, rows: np.ndarray) -> None:
        """Keep ``rows`` below the ones before."""
        if len(rows):
            self._parts.append(rows)

    def image(self) -> np.ndarray:
        """Every row kept, as one image."""
        return np.concatenate(self._parts)


def _settled(
    grab: Callable[[], np.ndarray], settling: Settling, sleep: Callable[[float], None], gap: float | None = None
) -> np.ndarray:
    frame = grab()
    for _ in range(settling.tries):
        sleep(settling.interval if gap is None else gap)
        again = grab()
        if np.array_equal(again, frame):
            return again
        frame = again
    return frame  # never still (a blinking caret, an animation): Match.tolerance is the answer to that


def _never() -> bool:
    return False


def _ignore(_: Stitcher) -> None:
    return None


_RECOVERIES: Final = 3  # tries to place a frame again before an automatic capture gives up


def _recover(
    stitcher: Stitcher,
    grab: Callable[[], np.ndarray],
    scroll: Callable[[int], None],
    *,
    notches: int,
    settling: Settling,
    sleep: Callable[[float], None],
    gap: float,
) -> np.ndarray | bool:
    """Place the view again after a frame that could not be: True, False or the last view seen.

    True: looking again placed it (the frame was taken mid-render); False: scrolling back did. Each try first waits
    and looks again -- a page still rendering settles into what it shows -- then scrolls back up half the last step,
    each time further, and finds the view anywhere in the rows held.
    """
    back = 0
    for _ in range(_RECOVERIES):
        sleep(settling.end_wait)
        frame = _settled(grab, settling, sleep, gap)
        for place in (stitcher.add, stitcher.relocate):
            try:
                place(frame)
            except Unstitchable:
                continue
            return not back
        step = max(1, notches // 2)
        scroll(-step)
        back += step
    scroll(back)  # where it was: what follows the lost view is still below
    return frame


def _moved(
    stitcher: Stitcher, grab: Callable[[], np.ndarray], settling: Settling, sleep: Callable[[float], None], gap: float
) -> int:
    """The rows the settled view moved, looked at once more after ``end_wait`` when none: 0 is the end."""
    if moved := stitcher.add(_settled(grab, settling, sleep, gap)):
        return moved
    sleep(settling.end_wait)
    return stitcher.add(_settled(grab, settling, sleep, gap))


class _Blocked(Exception):
    """The source refused a scroll (the OS dropped the input): the capture ends there, like any other stop."""


def _refusing(scroll: Callable[[int], None]) -> Callable[[int], None]:
    """``scroll``, its ``OSError`` (the input refused) raised as :class:`_Blocked`."""

    def step(notches: int) -> None:
        try:
            scroll(notches)
        except OSError as refused:
            raise _Blocked(str(refused)) from refused

    return step


def _cap(limits: Limits, frames: int, height: int, aborted: Callable[[], bool]) -> str | None:
    """Why to stop before the next scroll, if anything says so."""
    if limits.max_frames is not None and frames >= limits.max_frames:
        return 'max_frames'
    if limits.max_height is not None and height >= limits.max_height:
        return 'max_height'
    return 'aborted' if aborted() else None


_GROW: Final = 1.25  # the share a step aims for grows by this after a streak of steps placed
_LEAST_GAP: Final = 0.05  # seconds a settle gap grows from when --settle is 0


@dataclass(frozen=True)
class Lost:
    """A view :func:`follow` could not place below the last frame: which way back, both frames, and why.

    ``back`` is ``down`` when the view is above the last frame (scrolled back up past it), else ``up`` (scrolled down
    too far at once, past what one view can overlap).
    """

    back: str
    last: np.ndarray
    frame: np.ndarray
    reason: str


def _where(last: np.ndarray, frame: np.ndarray, match: Match) -> str:
    try:
        return 'down' if offset(frame, last, match) > 0 else 'up'
    except Unstitchable:
        return 'up'


def _report(_: Lost) -> None:
    return None


_HISTORY: Final = 3  # views of captured rows kept to find a lost view in


def capture(
    grab: Callable[[], np.ndarray],
    scroll: Callable[[int], None] | None,
    sink: Callable[[np.ndarray], None],
    *,
    limits: Limits = Limits(),  # noqa: B008 -- frozen
    match: Match = Match(),  # noqa: B008 -- frozen
    settling: Settling = Settling(),  # noqa: B008 -- frozen
    pace: Pace = Pace(),  # noqa: B008 -- frozen
    drop_overlays: bool = False,
    aborted: Callable[[], bool] = _never,
    progress: Callable[[Stitcher], None] = _ignore,
    lost: Callable[[Lost], None] = _report,
    sleep: Callable[[float], None] = time.sleep,
) -> Capture:
    """What the source shows from its current position to its end, to ``sink``; one frame when ``scroll`` is None.

    ``scroll(n)`` scrolls ``n`` notches, down as many as ``pace`` says, up when ``n`` is negative. A frame that cannot
    be placed -- caught mid-render, or a step that outran the overlap -- is recovered (:func:`_recover`), the steps
    capped at half the one that lost it; only one that cannot be recovered ends the capture, told to ``lost`` first.
    ``progress`` sees the stitcher after every frame it placed. A ``scroll`` that raises ``OSError`` (Windows refused
    the input) stops it as ``blocked: <reason>``, what was stitched kept.
    """
    first = grab()
    stitcher = Stitcher(first, match, drop_overlays=drop_overlays, history=_HISTORY * len(first))
    notches, ceiling, lapses = pace.first, None, 0  # lapses: recoveries in a row, no step placed between
    reach, placed = pace.stride, 0  # the share of the view a step aims for; steps placed in a row since it last grew
    gap = settling.interval  # grows when a frame taken for settled was still rendering; shrinks back on a streak
    frames, stopped, error = 1, 'one frame' if scroll is None else 'end', None
    step = None if scroll is None else _refusing(scroll)
    try:
        while step is not None:
            if capped := _cap(limits, frames, stitcher.height, aborted):
                stopped = capped
                break
            step(notches)
            last = stitcher.last
            try:
                if not (moved := _moved(stitcher, grab, settling, sleep, gap)):
                    break
                placed += 1
                if pace.streak and placed >= pace.streak:  # a streak placed: aim further, wait less, lift the cap
                    reach, placed = pace.grown(reach), 0
                    gap = max(settling.interval, gap / 2)
                    ceiling = None if ceiling is None else 2 * ceiling
                notches, lapses = pace.next(notches, moved, len(first), ceiling, reach), 0
            except Unstitchable as failure:
                seen = _recover(stitcher, grab, step, notches=notches, settling=settling, sleep=sleep, gap=gap)
                lapses += 1
                if seen is True:  # looking again placed it: it was still rendering, so wait longer from now on
                    gap = min(max(2 * gap, _LEAST_GAP), max(settling.max_interval, settling.interval))
                if not isinstance(seen, bool) or lapses > _RECOVERIES:  # lost, or back where it was, no smaller step
                    seen = stitcher.last if isinstance(seen, bool) else seen
                    lost(Lost(_where(last, seen, match), last, seen, str(failure)))
                    raise
                ceiling = notches = max(1, notches // 2)
                reach, placed = pace.stride, 0
            frames += 1
            sink(stitcher.drain())
            progress(stitcher)
    except Unstitchable as failure:
        stopped = f'unstitchable: {failure}'
        error = failure.error if isinstance(failure, NoOverlap) else None
    except _Blocked as refused:
        stopped = f'blocked: {refused}'
    finally:
        sink(stitcher.drain(final=True))
    stuck = scroll is not None and stopped == 'end' and frames == 1
    return Capture(frames, stitcher.height, stopped, stuck=stuck, error=error)


def follow(
    grab: Callable[[], np.ndarray],
    sink: Callable[[np.ndarray], None],
    *,
    aborted: Callable[[], bool],
    interval: float = 0.1,
    limits: Limits = Limits(),  # noqa: B008 -- frozen
    match: Match = Match(),  # noqa: B008 -- frozen
    drop_overlays: bool = False,
    progress: Callable[[Stitcher], None] = _ignore,
    lost: Callable[[Lost], None] = _report,
    sleep: Callable[[float], None] = time.sleep,
) -> Capture:
    """What the source shows while the user scrolls it, to ``sink``, until ``aborted`` (the CLI binds Esc).

    A frame is grabbed every ``interval`` seconds and placed if the view moved; standing still ends nothing. A frame
    that cannot be placed below the last is looked for in the last :data:`_HISTORY` views captured: found, the capture
    goes on from there; not found -- scrolled too far at once, or far back -- it is skipped and told to ``lost`` with
    the way back (:class:`Lost`), until the view overlaps anything held again.
    ``aborted`` is asked once per grab, so a caller can keep its own window responsive there.
    """
    first = grab()
    stitcher = Stitcher(first, match, drop_overlays=drop_overlays, history=_HISTORY * len(first))
    frames, stopped = 1, 'aborted'
    try:
        while not aborted():
            if limits.max_frames is not None and frames >= limits.max_frames:
                stopped = 'max_frames'
                break
            if limits.max_height is not None and stitcher.height >= limits.max_height:
                stopped = 'max_height'
                break
            sleep(interval)
            frame = grab()
            if np.array_equal(frame, stitcher.last):
                continue  # a pause: nothing to place, and a blank view must not read as lost
            try:
                moved = stitcher.add(frame)
            except Unstitchable as error:
                try:  # not below the last frame: anywhere in the history held, it picks up from there
                    moved = stitcher.relocate(frame)
                except Unstitchable:
                    lost(Lost(_where(stitcher.last, frame, match), stitcher.last, frame, str(error)))
                    continue
            if moved:
                frames += 1
                sink(stitcher.drain())
                progress(stitcher)
    finally:
        sink(stitcher.drain(final=True))
    return Capture(frames, stitcher.height, stopped)
