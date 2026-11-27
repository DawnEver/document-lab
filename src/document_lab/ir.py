"""The document IR: frozen values, each able to cite where in its source it came from.

Coordinates are PDF points in page space: origin at the top-left of the page as displayed, y grows DOWN; pages
are 1-based.
A value from a format without geometry (an HTML table) cites a ``locator`` instead of a box.

A value read off a page is drawn by its content, or by an annotation's appearance (``annotation``: its subtype, when
the reader is asked for annotations). Optional content the file's default view hides (a layer turned off) is read
too and flagged ``hidden`` for the caller to keep or drop (:meth:`Page.shown` drops it); ``hidden`` is False by
default because content no hidden layer draws is seen. A module judging a page's content leaves hidden content
out unless its ``include_hidden`` says otherwise.

Beside its pages a document states data no page draws: its outline (bookmarks), its pages' labels, its metadata,
its form fields' values, the files it embeds, and each page's links -- each as stated, a value it does not state
absent (None, or an empty tuple), never made up. An outline item and a link lead to a :class:`Destination`, or do
an :class:`Action`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy as np

__all__ = [
    'Action',
    'Attachment',
    'BBox',
    'Cell',
    'Destination',
    'Document',
    'Field',
    'Figure',
    'Info',
    'Link',
    'Metadata',
    'OutlineItem',
    'Page',
    'Provenance',
    'Raster',
    'Table',
    'TextSpan',
    'VectorPath',
    'Widget',
    'Xmp',
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
    #: The number it shares, on its page, with the spans it continues or that continue it, no word gap between: a
    #: change of font, size or baseline inside a word (the ``2`` of ``H2O``). None: it continues none and none
    #: continues it. Its source decides where a word gap is; a reader setting spans into lines spaces them by it.
    unspaced: int | None = None
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
class Destination:
    """A place in the document as a viewer shows it (ISO 32000-2 12.3.2): a page, and where on it, in page space.

    ``x`` and ``y`` are the page point the window's top-left corner shows; a coordinate the destination leaves to
    the viewer is None (``Fit`` states neither, ``FitH`` only ``y`` on an upright page).
    """

    page: int
    #: How the page is fitted to the window, as stated: ``XYZ``, ``Fit``, ``FitH``, ``FitV``, ``FitR``, ``FitB``,
    #: ``FitBH``, ``FitBV``.
    fit: str
    x: float | None = None
    y: float | None = None
    #: The zoom an ``XYZ`` states; None when it leaves it to the viewer (stated null or 0).
    zoom: float | None = None
    #: The box a ``FitR`` fits to the window.
    box: BBox | None = None


@dataclass(frozen=True, slots=True)
class Action:
    """What an outline item or a link does other than go to a place in this document (ISO 32000-2 12.6.4).

    A kind not read further is its type alone. For a ``GoToR`` (a place in another file), ``destination`` names the
    other file's page and fit, the position there not read, or ``name`` its named destination.
    """

    #: The action type as stated: ``URI``, ``GoToR``, ``Named``, ``Launch``, ``JavaScript``, ...
    kind: str
    destination: Destination | None = None
    #: A ``URI`` action's address, as stated.
    uri: str | None = None
    #: The file a ``GoToR``, ``GoToE`` or ``Launch`` names.
    file: str | None = None
    #: A ``Named`` action's name (``NextPage``, ...), or the named destination a ``GoToR`` leads to.
    name: str | None = None


@dataclass(frozen=True, slots=True)
class OutlineItem:
    """One outline item (a bookmark, ISO 32000-2 12.3.3): its title, where it leads, and its children."""

    #: Its title as stated; None when it states none.
    title: str | None
    #: Where it leads in this document.
    destination: Destination | None = None
    #: What it does instead of leading to a place in this document.
    action: Action | None = None
    children: tuple[OutlineItem, ...] = ()


@dataclass(frozen=True, slots=True)
class Link:
    """A link on a page (ISO 32000-2 12.5.6.5): the area that activates it (``provenance.bbox``) and its target.

    Read as data whether or not annotations' appearances are drawn.
    """

    provenance: Provenance
    #: Where it leads in this document.
    destination: Destination | None = None
    #: What it does instead (a ``URI``, ...).
    action: Action | None = None

    @property
    def uri(self) -> str | None:
        """The address a ``URI`` action opens; None for any other target."""
        return self.action.uri if self.action is not None else None


@dataclass(frozen=True, slots=True)
class Info:
    """The document information dictionary (``/Info``, ISO 32000-2 14.3.3), each entry as stated."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    #: ``/CreationDate`` as stated (a PDF date, ``D:YYYYMMDDHHmmSSOHH'mm``).
    created: str | None = None
    #: ``/ModDate`` as stated.
    modified: str | None = None
    #: Every other entry (``Trapped``, a writer's own keys), (key, value) in file order.
    other: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Xmp:
    """The XMP metadata packet (ISO 32000-2 14.3.2): what it states of the document, each value as stated."""

    #: The whole packet: everything else it states.
    packet: str
    #: ``dc:title`` (its ``x-default`` alternative, else its first).
    title: str | None = None
    #: ``dc:creator``, in order.
    creators: tuple[str, ...] = ()
    #: ``dc:description`` (its ``x-default`` alternative, else its first).
    description: str | None = None
    #: ``xmp:CreateDate`` (ISO 8601).
    created: str | None = None
    #: ``xmp:ModifyDate``.
    modified: str | None = None
    #: ``xmp:MetadataDate``.
    metadata_date: str | None = None
    creator_tool: str | None = None
    #: ``pdf:Producer``.
    producer: str | None = None
    #: ``pdf:Keywords``.
    keywords: str | None = None


@dataclass(frozen=True, slots=True)
class Metadata:
    """Both metadata records a PDF may state, neither preferred, and which states the later date.

    ``newer`` compares the latest date each record states (``/Info``: creation and modification; XMP: create,
    modify and metadata dates): ``'info'``, ``'xmp'``, ``'same'``, or None when either states no date or the two
    cannot be ordered (a date without its offset from UT, within a day of the other).
    """

    info: Info | None = None
    xmp: Xmp | None = None
    newer: Literal['info', 'xmp', 'same'] | None = None


@dataclass(frozen=True, slots=True)
class Widget:
    """Where a form field is shown: its page and box (``provenance``; both None when the file places it on none)."""

    provenance: Provenance
    #: Its appearance state (``/AS``): a check box's or radio button's on state, or ``Off``.
    state: str | None = None
    #: The export value a check box's or radio button's widget stands for (the field's ``/Opt`` entry for it, else
    #: its on appearance state).
    export: str | None = None


@dataclass(frozen=True, slots=True)
class Field:
    """An interactive form field (ISO 32000-2 12.7): its value as stated -- data, not its appearance."""

    #: Its fully qualified name: its ancestors' partial names and its own, ``.``-joined.
    name: str | None
    #: Its type as stated: ``Btn``, ``Tx``, ``Ch``, ``Sig``.
    kind: str | None
    #: Its flags (``/Ff``) as stated: which kind of button, a multi-line text, a combo box, ...
    flags: int | None = None
    #: Its value (``/V``): a text, a button's state or a choice; several for a multiple choice. None when unset, and
    #: for a signature.
    value: str | tuple[str, ...] | None = None
    #: Its default value (``/DV``), read as ``value`` is.
    default: str | tuple[str, ...] | None = None
    #: A choice field's options: (export value, text shown).
    options: tuple[tuple[str, str], ...] = ()
    widgets: tuple[Widget, ...] = ()


@dataclass(frozen=True, slots=True)
class Attachment:
    """A file the document embeds (ISO 32000-2 7.11.4); its bytes: :func:`document_lab.pdf.read_attachment`.

    ``provenance`` cites the page whose file attachment annotation carries it, no page for the document's own.
    """

    provenance: Provenance
    #: The embedded file stream's object number in the file.
    object: int
    #: Its key in the document's embedded files; None for a page's attachment.
    key: str | None = None
    #: Its file name as stated.
    name: str | None = None
    description: str | None = None
    #: Its MIME type as stated (``application/pdf``, ...).
    mime: str | None = None
    #: Its size in bytes as stated.
    size: int | None = None
    #: When it was made, as stated (a PDF date).
    created: str | None = None
    #: When it was last changed, as stated.
    modified: str | None = None


@dataclass(frozen=True, slots=True)
class Page:
    """Everything read off one page."""

    number: int
    width: float
    height: float
    spans: tuple[TextSpan, ...] = ()
    paths: tuple[VectorPath, ...] = ()
    figures: tuple[Figure, ...] = ()
    #: Its links, in the file's order.
    links: tuple[Link, ...] = ()
    #: Content the reader skipped and why (an operation it could not read, a link whose target does not resolve).
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


@dataclass(frozen=True, slots=True)
class Document:
    """A source document by content hash and URI, its pages, and what it states beside them."""

    sha256: str
    uri: str
    pages: tuple[Page, ...] = field(default=())
    #: What the reader repaired to read the file (a rebuilt cross-reference table, ...), in words.
    repairs: tuple[str, ...] = ()
    #: Its outline's top-level items; empty when it has none.
    outline: tuple[OutlineItem, ...] = ()
    #: Each page's label (``iv``, ``A-3``), in page order, None for a page no range labels; empty when it states
    #: no labels.
    page_labels: tuple[str | None, ...] = ()
    metadata: Metadata = field(default_factory=Metadata)
    #: Its interactive form's terminal fields, in the form's order.
    fields: tuple[Field, ...] = ()
    #: The files it embeds: its own, then its pages'.
    attachments: tuple[Attachment, ...] = ()
    #: What of the document's own data the reader skipped and why (an outline that loops, a destination that names
    #: nothing), in words.
    warnings: tuple[str, ...] = ()

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
