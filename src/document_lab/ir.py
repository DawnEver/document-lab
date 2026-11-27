"""The document IR: frozen values, each able to cite where in its source it came from.

Coordinates are PDF points in page space: origin at the top-left of the page as displayed, y grows DOWN; pages
are 1-based.
A value from a format without geometry (an HTML table) cites a ``locator`` instead of a box.

A value read off a page is drawn by its content, or by an annotation's appearance (``annotation``: its subtype, when
the reader is asked for annotations). Optional content the file's default view hides (a layer turned off) is read
too and flagged ``hidden`` for the caller to keep or drop (:meth:`Page.shown` drops it); ``hidden`` is False by
default because content no hidden layer draws is seen. A module judging a page's content leaves hidden content
out unless its ``include_hidden`` says otherwise.

A tagged PDF states its logical structure (ISO 32000-2 14.7): :attr:`Document.structure` is that tree of
:class:`Element` s in the order the document states, each covering the values it names -- the spans, paths and
figures drawn in its marked content (each value cites the :class:`Mark` it is drawn under), annotations and
images it refers to. It is None for an untagged file: a structure is never made up.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy as np

__all__ = [
    'Attribute',
    'BBox',
    'Cell',
    'Document',
    'Element',
    'Figure',
    'Mark',
    'MarkedContent',
    'ObjectReference',
    'Page',
    'Provenance',
    'Raster',
    'Structure',
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


class Mark(NamedTuple):
    """The marked-content sequence a value is drawn under, as a tagged PDF's structure refers to it.

    ``mcid`` is the sequence's marked-content identifier; ``stream`` the object number of the Form XObject whose
    content holds it, None for the page's own content.
    """

    mcid: int
    stream: int | None = None


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
    #: The number it shares, on its page, with the spans it continues or that continue it, no word gap between: a
    #: change of font, size or baseline inside a word (the ``2`` of ``H2O``). None: it continues none and none
    #: continues it. Its source decides where a word gap is; a reader setting spans into lines spaces them by it.
    unspaced: int | None = None
    #: Drawn by optional content the default view hides (see the module).
    hidden: bool = False
    #: The subtype of the annotation whose appearance draws it; None for the page's content.
    annotation: str | None = None
    #: The text its glyphs draw, when an ``/ActualText`` stated ``text`` in its place (a ligature spelled out, a
    #: line-end hyphen taken out); None when ``text`` is what they draw.
    drawn: str | None = None
    #: The marked content it is drawn under; None outside any with an identifier.
    mark: Mark | None = None
    #: An ``/Artifact`` holds it: pagination, layout or decoration the document states is not its content.
    artifact: bool = False

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
    #: The marked content it is drawn under; None outside any with an identifier.
    mark: Mark | None = None
    #: An ``/Artifact`` holds it (see :class:`TextSpan`).
    artifact: bool = False


@dataclass(frozen=True, slots=True)
class Cell:
    """One table cell's text and where it stands."""

    text: str
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class Table:
    """A grid of cell strings, and ``cell_bboxes`` of the same shape when the source has geometry.

    A cell with no box of its own holds ``None`` there and cites the table's box. A cell spanning several slots
    of the grid is named in ``merged``; its text and box stand in its first slot, the others empty.
    """

    rows: tuple[tuple[str, ...], ...]
    cell_bboxes: tuple[tuple[BBox | None, ...], ...] | None
    provenance: Provenance
    #: Each cell spanning more than one slot: ``(row, column, rows, columns)``, its first slot and its extent.
    merged: tuple[tuple[int, int, int, int], ...] = ()
    #: The slots ``(row, column)`` of the cells the source states are headers.
    heads: tuple[tuple[int, int], ...] = ()

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
    #: The marked content it is drawn under; None outside any with an identifier.
    mark: Mark | None = None
    #: An ``/Artifact`` holds it (see :class:`TextSpan`).
    artifact: bool = False


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

    def shown(self, *, include_hidden: bool = False) -> Page:
        """The page as its default view shows it: what hidden layers draw left out, unless ``include_hidden``."""
        if include_hidden:
            return self
        return dataclasses.replace(
            self,
            spans=tuple(s for s in self.spans if not s.hidden),
            paths=tuple(p for p in self.paths if not p.hidden),
            figures=tuple(f for f in self.figures if not f.hidden),
        )


#: An attribute's value: a name or text, a number, a boolean, an array of values, or a dictionary as its pairs.
type AttributeValue = bool | int | float | str | tuple[AttributeValue, ...] | tuple[tuple[str, AttributeValue], ...]


class Attribute(NamedTuple):
    """A structure element's attribute: its ``owner`` (``Layout``, ``Table``, ``List``, ...), name and value."""

    owner: str
    name: str
    value: AttributeValue


@dataclass(frozen=True, slots=True, eq=False)
class MarkedContent:
    """A marked-content sequence a structure element covers: its page and mark, and what it draws, in drawing order."""

    page: int
    mark: Mark
    spans: tuple[TextSpan, ...] = ()
    paths: tuple[VectorPath, ...] = ()
    figures: tuple[Figure, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class ObjectReference:
    """An object a structure element covers: an annotation by its rectangle, an image by its placements."""

    #: Its object number.
    object: int
    #: The page it is on, when the structure names one.
    page: int | None = None
    #: An annotation's rectangle in page space; None for any other object.
    bbox: BBox | None = None
    #: The figures that place it, when it is an image.
    figures: tuple[Figure, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class Element:
    """A structure element: its roles, what it says of itself, and its kids in the order the document states.

    Its role as stated is kept beside the standard role it maps to. Compared by identity: one element of a
    document is not another however alike.
    """

    #: Its role as stated (``/S``): a standard one, or the document's own.
    role: str
    #: The standard role it is or maps to through the document's role map (``H1``, ``P``, ``Table``); None when it
    #: maps to none.
    standard: str | None
    kids: tuple[Element | MarkedContent | ObjectReference, ...] = ()
    #: Its title (``/T``).
    title: str | None = None
    #: Its language (``/Lang``).
    lang: str | None = None
    #: Its alternate description (``/Alt``): what a figure shows, in words.
    alt: str | None = None
    #: The text stated in place of what it covers (``/ActualText``); the spans it covers state it too, keeping the
    #: text they draw (:attr:`TextSpan.drawn`).
    actual_text: str | None = None
    #: The expansion of the abbreviation it is (``/E``).
    expansion: str | None = None
    #: Its attributes: its own, then its classes'.
    attributes: tuple[Attribute, ...] = ()
    #: The page it names, else the nearest element's above it; None when none names one.
    page: int | None = None

    def elements(self) -> Iterator[Element]:
        """Itself and every element below it, in tree order."""
        yield self
        for kid in self.kids:
            if isinstance(kid, Element):
                yield from kid.elements()

    def content(self) -> Iterator[MarkedContent | ObjectReference]:
        """What it and the elements below it cover, in order."""
        for kid in self.kids:
            if isinstance(kid, Element):
                yield from kid.content()
            else:
                yield kid

    def spans(self) -> Iterator[TextSpan]:
        """The spans it covers, in order."""
        for item in self.content():
            if isinstance(item, MarkedContent):
                yield from item.spans

    def figures(self) -> Iterator[Figure]:
        """The figures it covers, in order."""
        for item in self.content():
            yield from item.figures

    def paths(self) -> Iterator[VectorPath]:
        """The paths it covers, in order."""
        for item in self.content():
            if isinstance(item, MarkedContent):
                yield from item.paths

    def attribute(self, name: str, owner: str | None = None) -> AttributeValue | None:
        """The value of its attribute ``name`` (of ``owner``, when given); None when it states none."""
        return next((a.value for a in self.attributes if a.name == name and owner in {None, a.owner}), None)

    @property
    def text(self) -> str:
        """The text it states: its ``/ActualText``, else that of what it covers, a space between two pieces."""
        if self.actual_text is not None:
            return self.actual_text
        pieces = [
            kid.text if isinstance(kid, Element) else ' '.join(s.text for s in kid.spans)
            for kid in self.kids
            if not isinstance(kid, ObjectReference)
        ]
        return ' '.join(' '.join(pieces).split())

    def bbox(self, page: int) -> BBox | None:
        """The box round what it covers on ``page``; None when it covers nothing there."""
        boxes = []
        for item in self.content():
            if item.page != page:
                continue
            values = (*item.spans, *item.paths, *item.figures) if isinstance(item, MarkedContent) else item.figures
            boxes += [v.provenance.bbox for v in values if v.provenance.bbox is not None]
            if isinstance(item, ObjectReference) and item.bbox is not None:
                boxes.append(item.bbox)
        return union(boxes) if boxes else None


@dataclass(frozen=True, slots=True, eq=False)
class Structure:
    """A tagged PDF's logical structure: the elements under its root, and what its reader left out, in words."""

    roots: tuple[Element, ...]
    #: Cycles, broken references and disagreements with the parent tree the reader found, in words.
    warnings: tuple[str, ...] = ()

    def elements(self) -> Iterator[Element]:
        """Every element, in tree order."""
        for root in self.roots:
            yield from root.elements()


@dataclass(frozen=True, slots=True)
class Document:
    """A source document by content hash and URI, and its pages."""

    sha256: str
    uri: str
    pages: tuple[Page, ...] = field(default=())
    #: What the reader repaired to read the file (a rebuilt cross-reference table, ...), in words.
    repairs: tuple[str, ...] = ()
    #: The logical structure a tagged PDF states; None for an untagged file.
    structure: Structure | None = None

    def page(self, number: int) -> Page:
        """The page with 1-based ``number``."""
        return self.pages[number - 1]

    def spans(self) -> Iterator[TextSpan]:
        """Every text span, page by page."""
        for page in self.pages:
            yield from page.spans

    def shown(self, *, include_hidden: bool = False) -> Document:
        """The document as its default view shows it, page by page (:meth:`Page.shown`)."""
        if include_hidden:
            return self
        return dataclasses.replace(self, pages=tuple(p.shown() for p in self.pages))
