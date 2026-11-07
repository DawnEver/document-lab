"""One capture: grab a frame, scroll, wait for the view to settle, stitch on -- until the view stops moving.

The source is two callables, ``grab() -> frame`` and ``scroll()``, so the loop knows nothing of windows or screens
(`document_lab.capture.desktop` supplies them on Windows). The end is the first scroll that moves nothing; the caps
in :class:`Limits`, the ``aborted`` callback (the CLI binds Esc and a moved pointer) and a frame that cannot be
stitched stop it earlier. There is no cap by default: a view as long as it scrolls is captured.

The stitched rows go to ``sink`` as soon as no later frame can change them, so a capture of any length holds about
one view in memory; :class:`Rows` is a sink that keeps them, for a capture known to be short. Everything runs in the
caller's thread, in order -- grab, scroll, stitch, sink -- so nothing is shared and nothing races. Whatever stops
it, every stitched row reaches ``sink`` and the :class:`Capture` says why it stopped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from document_lab.capture.stitch import Match, Stitcher, Unstitchable

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ['Capture', 'Limits', 'Rows', 'Settling', 'capture']


@dataclass(frozen=True)
class Limits:
    """When to stop before the end: frames grabbed, stitched height in pixels (``None``: no cap)."""

    max_frames: int | None = None
    max_height: int | None = None


@dataclass(frozen=True)
class Settling:
    """After a scroll, grab every ``interval`` seconds until two frames agree, at most ``tries`` times.

    A view that has not moved is looked at once more after ``end_wait`` seconds before it is taken for the end: a
    scroll the application handles late (a posted message, a busy view) shows two equal frames first, and a page that
    loads more as it nears its end (an infinite feed) needs that long to grow.
    """

    interval: float = 0.12
    tries: int = 15
    end_wait: float = 0.8


@dataclass(frozen=True)
class Capture:
    """One capture: the ``frames`` it took, the ``height`` it stitched, and why it ``stopped``.

    ``stopped`` is ``end``, ``aborted``, ``max_frames``, ``max_height``, ``one frame`` or ``unstitchable: <reason>``.
    """

    frames: int
    height: int
    stopped: str

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


def _settled(grab: Callable[[], np.ndarray], settling: Settling, sleep: Callable[[float], None]) -> np.ndarray:
    frame = grab()
    for _ in range(settling.tries):
        sleep(settling.interval)
        again = grab()
        if np.array_equal(again, frame):
            return again
        frame = again
    return frame  # never still (a blinking caret, an animation): Match.tolerance is the answer to that


def _never() -> bool:
    return False


def _ignore(_: Stitcher) -> None:
    return None


def capture(
    grab: Callable[[], np.ndarray],
    scroll: Callable[[], None] | None,
    sink: Callable[[np.ndarray], None],
    *,
    limits: Limits = Limits(),  # noqa: B008 -- frozen
    match: Match = Match(),  # noqa: B008 -- frozen
    settling: Settling = Settling(),  # noqa: B008 -- frozen
    drop_overlays: bool = False,
    aborted: Callable[[], bool] = _never,
    progress: Callable[[Stitcher], None] = _ignore,
    sleep: Callable[[float], None] = time.sleep,
) -> Capture:
    """What the source shows from its current position to its end, to ``sink``; one frame when ``scroll`` is None.

    ``progress`` sees the stitcher after every frame it placed.
    """
    stitcher = Stitcher(grab(), match, drop_overlays=drop_overlays)
    frames, stopped = 1, 'one frame' if scroll is None else 'end'
    try:
        while scroll is not None:
            if limits.max_frames is not None and frames >= limits.max_frames:
                stopped = 'max_frames'
                break
            if limits.max_height is not None and stitcher.height >= limits.max_height:
                stopped = 'max_height'
                break
            if aborted():
                stopped = 'aborted'
                break
            scroll()
            if not stitcher.add(_settled(grab, settling, sleep)):
                sleep(settling.end_wait)
                if not stitcher.add(_settled(grab, settling, sleep)):
                    break
            frames += 1
            sink(stitcher.drain())
            progress(stitcher)
    except Unstitchable as error:
        stopped = f'unstitchable: {error}'
    finally:
        sink(stitcher.drain(final=True))
    return Capture(frames, stitcher.height, stopped)
