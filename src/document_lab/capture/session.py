"""One capture: grab a frame, scroll, wait for the view to settle, stitch on -- until the view stops moving.

The source is two callables, ``grab() -> frame`` and ``scroll()``, so the loop knows nothing of windows or screens
(`document_lab.capture.desktop` supplies them on Windows). The end is the first scroll that moves nothing; the caps
in :class:`Limits` and the ``aborted`` callback (the CLI binds Esc) stop it earlier with what it has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from document_lab.capture.stitch import Match, Stitcher

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ['Limits', 'Settling', 'capture']


@dataclass(frozen=True)
class Limits:
    """When to stop before the end: frames grabbed, stitched height in pixels (``None``: no cap)."""

    max_frames: int = 500
    max_height: int | None = None


@dataclass(frozen=True)
class Settling:
    """After a scroll, grab every ``interval`` seconds until two frames agree, at most ``tries`` times.

    A view that has not moved is looked at once more after ``end_wait`` seconds before it is taken for the end: a
    scroll the application handles late (a posted message, a busy view) shows two equal frames first.
    """

    interval: float = 0.12
    tries: int = 15
    end_wait: float = 0.8


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


def capture(
    grab: Callable[[], np.ndarray],
    scroll: Callable[[], None] | None,
    *,
    limits: Limits = Limits(),  # noqa: B008 -- frozen
    match: Match = Match(),  # noqa: B008 -- frozen
    settling: Settling = Settling(),  # noqa: B008 -- frozen
    aborted: Callable[[], bool] = _never,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """The tall image the source shows from its current position to its end; one frame when ``scroll`` is None."""
    stitcher = Stitcher(grab(), match)
    frames = 1
    while scroll is not None and frames < limits.max_frames and not aborted():
        if limits.max_height is not None and stitcher.height >= limits.max_height:
            break
        scroll()
        if not stitcher.add(_settled(grab, settling, sleep)):
            sleep(settling.end_wait)
            if not stitcher.add(_settled(grab, settling, sleep)):
                break
        frames += 1
    return stitcher.image()
