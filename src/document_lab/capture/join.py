"""Two captures of one document as one: the second goes on where it overlaps the first's end, the shared rows once.

A capture stopped part way -- Finish, a lost view, a crash -- is continued by capturing again from anywhere
overlapping its end, then joining the two. Only the seam is redrawn: the first's last pages, down to where the second
begins; every other page of both is copied as it is (its pixels, its OCR layer). The second's head is found in the
first's tail by `document_lab.capture.stitch.locate`, at :data:`JOIN_TOLERANCE` -- looser than within one capture: a
document reopened renders the same text a little differently. Two that share nothing, or fit more than one place,
are refused -- never butted together -- and two of different widths before anything is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pymupdf

from document_lab.capture.export import check_page
from document_lab.capture.stitch import AmbiguousOverlap, Match, NoOverlap, locate

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['HEAD_ROWS', 'JOIN_TOLERANCE', 'Joined', 'join', 'tail_rows', 'width']

JOIN_TOLERANCE: Final = 20.0  # grey levels: the same text rendered again after reopening differs ~15 (measured)
HEAD_ROWS: Final = 1000  # rows of the second capture's start looked for in the first: enough to be told apart
_REACH: Final = 3  # the first's tail searched: this many times the head, so the second may start well before its end


@dataclass(frozen=True)
class Joined:
    """The joined capture at ``path``: ``shared`` rows both showed, kept once; ``redrawn`` first pages redrawn."""

    path: Path
    shared: int
    redrawn: int


def _image(page: pymupdf.Page) -> pymupdf.Pixmap:
    ((xref, *_), *_) = page.get_images()
    pix = pymupdf.Pixmap(page.parent, xref)
    return pymupdf.Pixmap(pix, 0) if pix.alpha else pix


def _rows(page: pymupdf.Page) -> np.ndarray:
    pix = _image(page)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def _heights(doc: pymupdf.Document) -> list[int]:
    return [page.get_images()[0][3] for page in doc]


def width(doc: pymupdf.Document) -> int:
    """The pixel width of a capture: its first page's image's."""
    return doc[0].get_images()[0][2]


def _tail(doc: pymupdf.Document, least: int) -> tuple[int, np.ndarray]:
    """The last pages of ``doc`` holding at least ``least`` rows (all, if fewer): how many, and their rows."""
    heights, count, rows = _heights(doc), 0, 0
    while count < len(heights) and rows < least:
        count += 1
        rows += heights[-count]
    return count, np.concatenate([_rows(doc[i]) for i in range(len(heights) - count, len(heights))])


def tail_rows(path: Path, least: int) -> np.ndarray:
    """The rows of the last whole pages of the capture at ``path`` that hold at least ``least`` rows."""
    with pymupdf.open(path) as doc:
        return _tail(doc, least)[1]


def _head(doc: pymupdf.Document, rows: int) -> np.ndarray:
    parts, held = [], 0
    for page in doc:
        if held >= rows:
            break
        parts.append(_rows(page))
        held += len(parts[-1])
    return np.concatenate(parts)[:rows]


def _found(doc: pymupdf.Document, frame: np.ndarray, match: Match) -> tuple[int, np.ndarray, int]:
    """Where ``frame`` starts in the end of ``doc``: the tail pages searched, their rows, and the row in them.

    The search starts :data:`_REACH` frames back and doubles until it holds the whole document, so a second capture
    begun far before the first ended is found too; what is still not found, or fits more than one place, is refused.
    """
    least = _REACH * len(frame)
    while True:
        count, tail = _tail(doc, least)
        try:
            return count, tail, locate(tail, frame, match)
        except NoOverlap:
            if count == doc.page_count:
                raise
            least *= 2


def _placed(a: pymupdf.Document, b: pymupdf.Document, head: int, match: Match) -> tuple[int, np.ndarray, int]:
    """Where ``b`` starts in the end of ``a`` (as :func:`_found`), its head doubled while it fits more than one place.

    A short head of a standard -- a recurring page header, white space -- fits many places; a longer one fits one.
    """
    whole = sum(_heights(b))
    while True:
        try:
            return _found(a, _head(b, head), match)
        except AmbiguousOverlap:
            if head >= whole:
                raise
            head *= 2


def _seam(out: pymupdf.Document, rows: np.ndarray, page_height: int, dpi: float) -> None:
    """``rows`` onto new pages of ``out``, ``page_height`` rows each, the last shorter."""
    for top in range(0, len(rows), page_height):
        part = np.ascontiguousarray(rows[top : top + page_height])
        pix = pymupdf.Pixmap(pymupdf.csRGB, part.shape[1], len(part), part.tobytes(), 0)
        rect = pymupdf.Rect(0, 0, pix.width * 72 / dpi, pix.height * 72 / dpi)
        out.new_page(width=rect.width, height=rect.height).insert_image(rect, pixmap=pix)


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
    with pymupdf.open(first) as a, pymupdf.open(second) as b:
        if width(a) != width(b):
            msg = f'the captures are {width(a)} and {width(b)} px wide: only captures of one area join'
            raise ValueError(msg)
        count, tail, top = _placed(a, b, head, match)
        whole = sum(_heights(b))
        joined = pymupdf.open()
        if top + whole <= len(tail):  # the second shows nothing the first does not
            joined.insert_pdf(a)
            shared, count = whole, 0
        else:
            keep = a.page_count - count
            if keep:
                joined.insert_pdf(a, to_page=keep - 1)
            seam = a[keep]
            dpi = _image(seam).width * 72 / seam.rect.width
            page_height = _heights(a)[0]
            check_page(width(a), page_height, dpi)
            _seam(joined, tail[:top], page_height, dpi)
            joined.insert_pdf(b)
            shared = len(tail) - top
        part = out.with_name(out.name + '.part')
        joined.save(part, deflate=True)
        joined.close()
    part.replace(out)
    return Joined(out, shared, count)
