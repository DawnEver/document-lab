"""Two captures of one document as one: the second goes on where it overlaps the first's end, the shared rows once.

A capture stopped part way -- Finish, a lost view, a crash -- is continued by capturing again from anywhere
overlapping its end, then joining the two. Only the seam is redrawn -- the first's last pages and the second's first,
cut in the middle of the rows both show, so neither's edge (a window border caught at the top or bottom of a view)
shows -- every other page of both is copied as it is (its pixels, its OCR layer). The second's head is found in the
first's tail by `document_lab.capture.stitch.drift`: a page that moved sideways between the two (a reader reopened,
its side bar changed) is found with how far, and the second's pages are redrawn moved back in line. Two that share
nothing, or fit more than one place, are refused -- never butted together -- and two of different widths before
anything is read.

A capture is read and written by `document_lab._pdf`: each page holds one image of 8-bit gray or RGB samples, stored
losslessly; any other page is refused, naming what it holds.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from document_lab import _pdf
from document_lab.capture.export import check_page, image_page
from document_lab.capture.stitch import AmbiguousOverlap, Match, NoOverlap, drift, shifted

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['HEAD_ROWS', 'JOIN_TOLERANCE', 'Joined', 'join', 'tail_rows', 'width']

JOIN_TOLERANCE: Final = Match.tolerance  # once aligned sideways, a reopened page matches as closely (0.06, measured)
HEAD_ROWS: Final = 1000  # rows of the second capture's start looked for in the first: enough to be told apart
_REACH: Final = 3  # the first's tail searched: this many times the head, so the second may start well before its end
_CURVE_SAMPLES: Final = 1  # a capture's page draws no curve worth reading: its paths are not looked at


@dataclass(frozen=True)
class Joined:
    """The joined capture at ``path``: ``shared`` rows both showed, kept once; ``redrawn`` first pages redrawn.

    ``shift``: columns the second's content sat further right than the first's, taken out (0: none).
    """

    path: Path
    shared: int
    redrawn: int
    shift: int = 0


@dataclass(frozen=True)
class _Image:
    """A capture page's image: its object, its size in samples, and the page's width in points."""

    object: int
    width: int
    height: int
    page_width: float


class _Capture:
    """A capture PDF, read: each page's one image, found when first asked for, and its rows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reader = _pdf.open(path.read_bytes(), b'')
        self.count = len(self.reader.pages)
        self._images: dict[int, _Image] = {}

    def image(self, index: int) -> _Image:
        """Page ``index``'s (0-based) image; refused unless the page holds exactly one."""
        if index not in self._images:
            number, page_width, _ = self.reader.pages[index]
            _, _, placed, _ = self.reader.page(number, _CURVE_SAMPLES)
            if len(placed) != 1 or placed[0][3] is None:
                msg = f'page {number} of {self.path.name} holds {len(placed)} image(s), not one: it is no capture'
                raise ValueError(msg)
            _, width, height, obj = placed[0]
            self._images[index] = _Image(obj, width, height, page_width)
        return self._images[index]

    @property
    def width(self) -> int:
        """The pixel width: the first page's image's."""
        return self.image(0).width

    def heights(self) -> list[int]:
        """Each page's rows."""
        return [self.image(i).height for i in range(self.count)]

    def rows(self, index: int) -> np.ndarray:
        """Page ``index``'s pixels, ``(h, w, 3)`` uint8."""
        width, height, space, components, bits, data, codec = self.reader.image(self.image(index).object)
        if codec is not None or bits != 8 or components not in {1, 3}:  # noqa: PLR2004 -- 8-bit samples
            msg = (
                f'page {index + 1} of {self.path.name} is a {space} image of {components} x {bits}-bit samples'
                f'{f" in {codec}" if codec else ""}: a capture is 8-bit gray or RGB, stored losslessly'
            )
            raise ValueError(msg)
        pixels = np.frombuffer(data, dtype=np.uint8)[: width * height * components].reshape(height, width, components)
        return np.repeat(pixels, 3, axis=2) if components == 1 else pixels

    def dpi(self, index: int) -> float:
        """The pixels to the inch page ``index`` is laid out at."""
        image = self.image(index)
        return image.width * 72 / image.page_width


def width(path: Path) -> int:
    """The pixel width of the capture at ``path``: its first page's image's."""
    return _Capture(path).width


def _tail(doc: _Capture, least: int) -> tuple[int, np.ndarray]:
    """The last pages of ``doc`` holding at least ``least`` rows (all, if fewer): how many, and their rows."""
    heights, count, rows = doc.heights(), 0, 0
    while count < len(heights) and rows < least:
        count += 1
        rows += heights[-count]
    return count, np.concatenate([doc.rows(i) for i in range(len(heights) - count, len(heights))])


def tail_rows(path: Path, least: int) -> np.ndarray:
    """The rows of the last whole pages of the capture at ``path`` that hold at least ``least`` rows."""
    return _tail(_Capture(path), least)[1]


def _head(doc: _Capture, rows: int) -> np.ndarray:
    parts, held = [], 0
    for index in range(doc.count):
        if held >= rows:
            break
        parts.append(doc.rows(index))
        held += len(parts[-1])
    return np.concatenate(parts)[:rows]


def _found(doc: _Capture, frame: np.ndarray, match: Match) -> tuple[int, np.ndarray, int, int]:
    """Where ``frame`` starts in the end of ``doc``: the tail pages searched, their rows, the row, the sideways shift.

    The search starts :data:`_REACH` frames back and doubles until it holds the whole document, so a second capture
    begun far before the first ended is found too; what is still not found, or fits more than one place, is refused.
    """
    least = _REACH * len(frame)
    while True:
        count, tail = _tail(doc, least)
        try:
            return count, tail, *drift(tail, frame, match)
        except NoOverlap:
            if count == doc.count:
                raise
            least *= 2


def _placed(a: _Capture, b: _Capture, head: int, match: Match) -> tuple[int, np.ndarray, int, int]:
    """Where ``b`` starts in the end of ``a`` (as :func:`_found`), its head doubled while it fits more than one place.

    A short head of a standard -- a recurring page header, white space -- fits many places; a longer one fits one.
    """
    whole = sum(b.heights())
    while True:
        try:
            return _found(a, _head(b, head), match)
        except AmbiguousOverlap:
            if head >= whole:
                raise
            head *= 2


def _seam(out: _pdf.Writer, rows: np.ndarray, page_height: int, dpi: float) -> None:
    """``rows`` onto new pages of ``out``, ``page_height`` rows each, the last shorter."""
    for top in range(0, len(rows), page_height):
        image_page(out, rows[top : top + page_height], dpi)


def _second(joined: _pdf.Writer, b: _Capture, cut: int, dx: int, dpi: float) -> None:
    """``b`` from its row ``cut`` onto ``joined``.

    The page holding the cut is redrawn from there and the rest copied -- or, ``dx`` not 0, every page is redrawn,
    moved ``dx`` columns back in line with the first capture.
    """
    start = 0
    for index, height in enumerate(b.heights()):
        if start + height <= cut:
            start += height
            continue
        if dx or start < cut:
            _seam(joined, shifted(b.rows(index)[max(cut - start, 0) :], dx), height, dpi)
        else:
            joined.copy_page(b.reader, index + 1)
        start += height


def _copy(joined: _pdf.Writer, doc: _Capture, pages: int) -> None:
    for number in range(1, pages + 1):
        joined.copy_page(doc.reader, number)


def join(
    first: Path,
    second: Path,
    out: Path,
    match: Match = Match(tolerance=JOIN_TOLERANCE),  # noqa: B008 -- frozen
    *,
    head: int = HEAD_ROWS,
) -> Joined:
    """``first`` then ``second`` from where its first ``head`` rows are found in ``first``'s end, written to ``out``.

    ``out`` may be ``first`` itself: it is written beside as ``<out>.part`` and moved into place only when whole.
    """
    a, b = _Capture(first), _Capture(second)
    if a.width != b.width:
        msg = f'the captures are {a.width} and {b.width} px wide: only captures of one area join'
        raise ValueError(msg)
    count, tail, top, dx = _placed(a, b, head, match)
    whole = sum(b.heights())
    part = out.with_name(out.name + '.part')
    joined = _pdf.create(part)
    try:
        if top + whole <= len(tail):  # the second shows nothing the first does not
            _copy(joined, a, a.count)
            shared, count = whole, 0
        else:
            keep = a.count - count
            _copy(joined, a, keep)
            dpi = a.dpi(keep)
            page_height = a.image(0).height
            check_page(a.width, page_height, dpi)
            shared = len(tail) - top
            cut = shared // 2  # mid-overlap: each capture's edge rows (a window border caught) left out
            _seam(joined, tail[: top + cut], page_height, dpi)
            _second(joined, b, cut, dx, dpi)
        joined.close()
    except BaseException:
        with contextlib.suppress(ValueError):  # closed already: the failure came after
            joined.close()
        part.unlink(missing_ok=True)
        raise
    part.replace(out)
    return Joined(out, shared, count, dx)
