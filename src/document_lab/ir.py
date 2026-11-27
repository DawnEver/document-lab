"""The document IR: frozen values, each able to cite where in its source it came from.

Coordinates are PDF points in page space: origin at the top-left of the page as displayed, y grows DOWN; pages
are 1-based.
A value from a format without geometry (an HTML table) cites a ``locator`` instead of a box.

A value read off a page is drawn by its content, or by an annotation's appearance (``annotation``: its subtype, when
the reader is asked for annotations). What the page draws but its default view does not show is read too, flagged
for the caller to keep or drop (:meth:`Page.shown` drops both): ``hidden``, drawn by optional content the default
view hides (a layer turned off), and ``inkless``, painted in the colorant ``None``, which marks nothing in any view
(a path, an image or a shading; text is text whatever its color, as an invisible OCR layer's is). A span is
flagged a third way, ``in_background_color``: filled in the color painted under it (white on the white page, black
on a black box), it shows nothing in the view that draws it -- a fact of its paint and what lies under it, not of
its layer or its colorant, so neither of the other two. All are False by default because what none flags is seen.
A module judging a page's content leaves them out unless its ``include_hidden``, ``include_inkless`` or
``include_in_background_color`` says otherwise.

Beside its pages a document states data no page draws: its outline (bookmarks), its pages' labels, its metadata,
its form fields' values, the files it embeds, and each page's links -- each as stated, a value it does not state
absent (None, or an empty tuple), never made up. An outline item and a link lead to a :class:`Destination`, or do
an :class:`Action`.

A tagged PDF states its logical structure (ISO 32000-2 14.7): :attr:`Document.structure` is that tree of
:class:`Element` s in the order the document states, each covering the values it names -- the spans, paths and
figures drawn in its marked content (each value cites the :class:`Mark` it is drawn under), annotations and
images it refers to. It is None for an untagged file: a structure is never made up.

A span's text is what its document states, a replacement (``/ActualText``) too: a no-break or zero-width space
stays one. A module that joins, splits or matches text reads it through :func:`spaced`.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, NamedTuple
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    import numpy as np

__all__ = [
    'Action',
    'Attachment',
    'Attribute',
    'BBox',
    'Cell',
    'Destination',
    'Document',
    'Element',
    'Field',
    'Figure',
    'Info',
    'Link',
    'Mark',
    'MarkedContent',
    'Metadata',
    'ObjectReference',
    'OutlineItem',
    'Page',
    'Provenance',
    'Raster',
    'Structure',
    'Table',
    'TextSpan',
    'VectorPath',
    'Widget',
    'Xmp',
    'file_uri',
    'spaced',
    'union',
]

#: Every space separator (Unicode category Zs: the no-break, en, em, thin, ideographic ... spaces) is a space, and so
#: is the zero-width space (U+200B), a word break unseen; a word joiner (U+2060, and U+FEFF read as one) forbids a
#: break and is dropped.
_SPACES: Final = {
    **{c: ' ' for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == 'Zs'},
    0x200B: ' ',
    0x2060: None,
    0xFEFF: None,
}
_RUN: Final = re.compile(' {2,}')


def spaced(text: str) -> str:
    """``text`` with every space a plain one and a run of them one, word joiners dropped (see :data:`_SPACES`).

    Tabs and line breaks are no space separators: they stay.
    """
    return _RUN.sub(' ', text.translate(_SPACES))


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


def file_uri(path: Path) -> str:
    """The ``file:`` URI of the absolute ``path``, a :class:`Provenance`'s ``uri`` for a local file: ``path.as_uri()``.

    Made here because ``as_uri`` imports ``urllib.request`` (``http.client``, ``email``, ``tempfile``) the first
    time a process calls it, about 0.1 s (Python 3.15) -- more than reading a short PDF takes. The steps are
    ``urllib.request.pathname2url``'s, which ``tests/unit/test_ir.py`` holds this to.

    Raises:
        ValueError: ``path`` is relative.

    """
    if not path.is_absolute():
        msg = f"{path}: relative paths can't be expressed as file URIs"
        raise ValueError(msg)
    encoding, errors = sys.getfilesystemencoding(), sys.getfilesystemencodeerrors()
    text = str(path).replace('\\', '/') if os.name == 'nt' else str(path)
    drive, root, tail = os.path.splitroot(text)
    if drive:
        if drive[:4] == '//?/':  # a verbatim path: its drive, or its share
            drive = drive[4:]
            if drive[:4].upper() == 'UNC/':
                drive = '//' + drive[4:]
        if drive[1:] == ':':  # a drive letter: an empty authority, then the path
            drive = '///' + drive
        drive = quote(drive, encoding=encoding, errors=errors, safe='/:')
    elif root:
        root = '//' + root  # an empty authority, so a path starting // is no authority
    return 'file:' + drive + root + quote(tail, encoding=encoding, errors=errors)


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
    """A run of text in one font, size and fill: along a baseline, or down a column when written vertically.

    Its fill is a color as in :class:`VectorPath`.
    """

    text: str
    size: float
    font: str
    provenance: Provenance
    bold: bool = False
    #: The number it shares, on its page, with the spans it continues or that continue it, no word gap between: a
    #: change of font, size or baseline inside a word (the ``2`` of ``H2O``). None: it continues none and none
    #: continues it. Its source decides where a word gap is; a reader setting spans into lines spaces them by it.
    unspaced: int | None = None
    #: How its glyphs are painted, the PDF text render mode: 0 fill, 1 stroke, 2 both, 3 invisible (an OCR layer),
    #: 4-7 those adding the glyphs to the clip. None: its source has no render modes.
    render_mode: int | None = None
    #: Written vertically (its font's writing mode, WMode 1): its glyphs stand upright and advance down a column, its
    #: text reads top to bottom, its box is the column's. Text turned sideways is not this: it is written
    #: horizontally along a turned baseline.
    vertical: bool = False
    #: The unit vector its baseline runs along, in page space (y down): ``(1, 0)`` for text set left to right,
    #: ``(0, -1)`` for text set bottom to top (a margin stamp turned sideways).
    direction: tuple[float, float] = (1.0, 0.0)
    #: Drawn by optional content the default view hides (see the module).
    hidden: bool = False
    #: The subtype of the annotation whose appearance draws it; None for the page's content.
    annotation: str | None = None
    #: The color its glyphs are filled with; None when they are not filled (stroked or invisible text) or the source
    #: states no fill.
    fill: tuple[float, ...] | None = None
    #: The text its glyphs draw, when an ``/ActualText`` stated ``text`` in its place (a ligature spelled out, a
    #: line-end hyphen taken out); None when ``text`` is what they draw.
    drawn: str | None = None
    #: The marked content it is drawn under; None outside any with an identifier.
    mark: Mark | None = None
    #: An ``/Artifact`` holds it: pagination, layout or decoration the document states is not its content.
    artifact: bool = False
    #: It continues the run of the span before it on its page: its source set them as one run, which it split where
    #: the fill changes (a scan's glyphs each a grey of their own, a colored digit). False: it starts a run, or its
    #: source states no runs.
    continues_run: bool = False
    #: Filled (and only filled) in the color its source paints under its box -- the page's white, or the paint of what
    #: was drawn under it -- so it shows nothing (see the module). Its source compares the colors exactly.
    in_background_color: bool = False

    @property
    def center(self) -> tuple[float, float]:
        """The center of the span's box."""
        box = self.provenance.bbox
        if box is None:
            msg = 'a span without a box has no center'
            raise ValueError(msg)
        return ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)


@dataclass(frozen=True, slots=True, eq=False)
class VectorPath:
    """One subpath of a page's vector drawing, curves flattened to points (as ``read_pdf`` says).

    A color is sRGB in 0..1 -- an ICC-based color through its profile, a separation's or DeviceN's through its
    tint transform, a calibrated or Lab color by the CIE formulas, a device color by the standard's own (the reader's
    ``pdf::color`` says how each space converts) -- else, when the reader cannot convert it, its own components; a
    pattern's has none, but an uncolored tiling pattern's is its tint. A path painted in the colorant ``None`` is
    ``inkless``, its color its tint's components; one whose fill marks and whose stroke does not (or the reverse) is
    two paintings, one ``inkless``. A shading painted over the clip (``sh``) is the closed rectangle of the box it
    may cover, filled with a ``'shading'`` pattern.

    A path is compared and hashed by identity (``eq=False``): it is one mark drawn on its page -- two drawn alike
    are still two -- and its points, an array, have no single truth value to compare by.
    """

    #: Its points in drawing order: an ``(n, 2)`` float64 array of ``(x, y)`` rows, n >= 2, read-only (the
    #: reader's are views of one array per page).
    points: np.ndarray
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
    #: Painted in the colorant ``None``: it marks nothing (see the module).
    inkless: bool = False
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
    #: A stencil mask painted, or samples, in the colorant ``None``: it marks nothing (see the module).
    inkless: bool = False
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

    def shown(
        self, *, include_hidden: bool = False, include_inkless: bool = False, include_in_background_color: bool = False
    ) -> Page:
        """The page as its default view shows it.

        What hidden layers draw is left out unless ``include_hidden``, what the colorant ``None`` paints unless
        ``include_inkless``, text filled in the color under it unless ``include_in_background_color``.
        """
        if include_hidden and include_inkless and include_in_background_color:
            return self
        return dataclasses.replace(
            self,
            spans=tuple(
                s
                for s in self.spans
                if (include_hidden or not s.hidden) and (include_in_background_color or not s.in_background_color)
            ),
            paths=tuple(
                p for p in self.paths if (include_hidden or not p.hidden) and (include_inkless or not p.inkless)
            ),
            figures=tuple(
                f for f in self.figures if (include_hidden or not f.hidden) and (include_inkless or not f.inkless)
            ),
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
    """A source document by content hash and URI, its pages, and what it states beside them.

    ``pages`` is any sequence of pages: a tuple, or ``read_pdf``'s, which reads each page when it is first asked for.
    """

    sha256: str
    uri: str
    pages: Sequence[Page] = field(default=())
    #: What the reader repaired to read the file (a rebuilt cross-reference table, ...), in words: a reader reading
    #: pages when asked for adds what it repairs to read one as it reads it.
    repairs: Sequence[str] = ()
    #: The logical structure a tagged PDF states; None for an untagged file.
    structure: Structure | None = None
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

    def shown(
        self, *, include_hidden: bool = False, include_inkless: bool = False, include_in_background_color: bool = False
    ) -> Document:
        """The document as its default view shows it, page by page (:meth:`Page.shown`)."""
        if include_hidden and include_inkless and include_in_background_color:
            return self
        pages = tuple(
            p.shown(
                include_hidden=include_hidden,
                include_inkless=include_inkless,
                include_in_background_color=include_in_background_color,
            )
            for p in self.pages
        )
        return dataclasses.replace(self, pages=pages)
