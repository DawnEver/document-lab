"""The document IR: frozen values, each able to cite where in its source it came from.

Coordinates are PDF points in page space: origin at the top-left of the page as displayed, y grows DOWN; pages
are 1-based.
A value from a format without geometry (an HTML table) cites a ``locator`` instead of a box.

A value read off a page is drawn by its content, or by an annotation's appearance (``annotation``: its subtype, when
the reader is asked for annotations). Optional content the file's default view hides (a layer turned off) is read
too and flagged ``hidden`` for the caller to keep or drop; ``hidden`` is False by default because content no hidden
layer draws is seen.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy as np

__all__ = [
    'BBox',
    'Cell',
    'Document',
    'Figure',
    'Page',
    'Provenance',
    'Raster',
    'Table',
    'TextSpan',
    'VectorPath',
    'union',
]


class BBox(NamedTuple):
    """An axis-aligned box ``(x0, y0, x1, y1)`` in PDF points, y down."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """``x1 - x0``."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """``y1 - y0``."""
        return self.y1 - self.y0

    def contains(self, x: float, y: float, tol: float = 0.0) -> bool:
        """Whether the point lies inside the box grown by ``tol`` points on every side."""
        return self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol


def union(boxes: Iterable[BBox]) -> BBox:
    """The smallest box covering every box given (at least one)."""
    xs0, ys0, xs1, ys1 = zip(*boxes, strict=True)
    return BBox(min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a value came from: the source's content hash and URI, then a page and box, or a locator."""

    sha256: str
    uri: str
    page: int | None = None
    bbox: BBox | None = None
    locator: str | None = None

    def __post_init__(self) -> None:
        """Refuse a 0-based page and an inverted box; store the box as a :class:`BBox`."""
        if self.page is not None and self.page < 1:
            msg = f'page {self.page}: pages are 1-based'
            raise ValueError(msg)
        if self.bbox is not None:
            box = BBox(*map(float, self.bbox))
            if box.x0 > box.x1 or box.y0 > box.y1:
                msg = f'bbox {tuple(box)}: needs x0 <= x1 and y0 <= y1'
                raise ValueError(msg)
            object.__setattr__(self, 'bbox', box)

    def at(self, *, bbox: BBox | None = None, locator: str | None = None) -> Provenance:
        """The same source and page, narrowed to a box or a sub-locator."""
        return dataclasses.replace(self, bbox=bbox or self.bbox, locator=locator or self.locator)


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A run of text in one font and size."""

    text: str
    size: float
    font: str
    provenance: Provenance
    bold: bool = False
    #: Drawn by optional content the default view hides (see the module).
    hidden: bool = False
    #: The subtype of the annotation whose appearance draws it; None for the page's content.
    annotation: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        """The center of the span's box."""
        box = self.provenance.bbox
        if box is None:
            msg = 'a span without a box has no center'
            raise ValueError(msg)
        return ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)


@dataclass(frozen=True, slots=True)
class VectorPath:
    """One subpath of a page's vector drawing, curves flattened to points (``read_pdf``'s ``curve_samples`` each).

    A color is RGB in 0..1 when its color space defines one, else its own components (a separation's tint); a
    pattern's has none but an uncolored tiling's tint. A shading painted over the clip (``sh``) is the closed
    rectangle of the box it may cover, filled with a ``'shading'`` pattern.
    """

    points: tuple[tuple[float, float], ...]
    #: The stroking color; None when the path is not stroked.
    stroke: tuple[float, ...] | None
    #: The stroke's line width (points); 0 when not stroked.
    width: float
    #: The stroke's dash array (points: on, off, ...); empty for a solid stroke.
    dashes: tuple[float, ...]
    #: Whether the subpath is closed (`h`, a rectangle).
    closed: bool
    provenance: Provenance
    #: Where in the dash array the stroke starts (points).
    dash_phase: float = 0.0
    #: The filling color; None when the path is not filled.
    fill: tuple[float, ...] | None = None
    #: Filled by the even-odd rule rather than the nonzero winding rule.
    even_odd: bool = False
    #: The pattern the fill paints instead of one color: ``'shading'`` (a gradient) or ``'tiling'`` (a repeated cell,
    #: not read); None for one color or no fill.
    pattern: str | None = None
    #: The painted path it is a subpath of, numbered on its page: one drawing's subpaths share it.
    group: int = 0
    #: Drawn by optional content the default view hides (see the module).
    hidden: bool = False
    #: The subtype of the annotation whose appearance draws it; None for the page's content.
    annotation: str | None = None


@dataclass(frozen=True, slots=True)
class Cell:
    """One table cell's text and where it stands."""

    text: str
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class Table:
    """A grid of cell strings, and ``cell_bboxes`` of the same shape when the source has geometry.

    A cell with no box of its own holds ``None`` there and cites the table's box.
    """

    rows: tuple[tuple[str, ...], ...]
    cell_bboxes: tuple[tuple[BBox | None, ...], ...] | None
    provenance: Provenance

    def __post_init__(self) -> None:
        """Refuse a box grid whose shape differs from the text grid."""
        if self.cell_bboxes is not None and [len(r) for r in self.cell_bboxes] != [len(r) for r in self.rows]:
            msg = 'cell_bboxes must have the shape of rows'
            raise ValueError(msg)

    def cell(self, row: int, col: int) -> Cell:
        """The cell at ``(row, col)``, citing its own box when it has one, else the table's."""
        box = self.cell_bboxes[row][col] if self.cell_bboxes is not None else None
        locator = f'{self.provenance.locator}/row[{row}]/col[{col}]' if self.provenance.locator else None
        return Cell(self.rows[row][col], self.provenance.at(bbox=box, locator=locator))


@dataclass(frozen=True, slots=True)
class Figure:
    """An image placed on a page: the box it covers, and its size in samples (its native resolution)."""

    provenance: Provenance
    #: Its width in samples.
    width: int
    #: Its height in samples.
    height: int
    #: The image's object number in the file; None for an inline image.
    image: int | None = None
    #: Drawn by optional content the default view hides (see the module).
    hidden: bool = False
    #: The subtype of the annotation whose appearance draws it; None for the page's content.
    annotation: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Raster:
    """A page region as pixels: RGB ``uint8`` rows top-down, spanning ``provenance.bbox`` edge to edge."""

    pixels: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        """Refuse pixels that are not rows x columns x RGB, and a raster that cites no box."""
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:  # noqa: PLR2004 -- RGB
            msg = f'pixels of shape {self.pixels.shape}: need rows x columns x 3'
            raise ValueError(msg)
        if self.provenance.bbox is None:
            msg = 'a raster must cite the box it covers'
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Page:
    """Everything read off one page."""

    number: int
    width: float
    height: float
    spans: tuple[TextSpan, ...] = ()
    paths: tuple[VectorPath, ...] = ()
    figures: tuple[Figure, ...] = ()
    #: Content the reader skipped and why (an operation it could not read), in words.
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Document:
    """A source document by content hash and URI, and its pages."""

    sha256: str
    uri: str
    pages: tuple[Page, ...] = field(default=())
    #: What the reader repaired to read the file (a rebuilt cross-reference table, ...), in words.
    repairs: tuple[str, ...] = ()

    def page(self, number: int) -> Page:
        """The page with 1-based ``number``."""
        return self.pages[number - 1]

    def spans(self) -> Iterator[TextSpan]:
        """Every text span, page by page."""
        for page in self.pages:
            yield from page.spans
