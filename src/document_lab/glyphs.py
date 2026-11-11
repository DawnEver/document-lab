"""Numbers read from glyph SHAPES -- outlines drawn as paths, or ink in an image -- where a page has no text for them.

A glyph is a bitmap. A digit is the reference digit (``REFERENCE_FONT``, rendered by PyMuPDF) whose shape it is
nearest: both are cropped to their ink, sampled on a ``GRID`` of ink fractions, and compared by Euclidean distance
plus ``ASPECT_WEIGHT`` times the difference of their width-to-height ratios. A glyph is read only when its nearest
digit is within ``ACCEPT`` AND the second nearest is ``MARGIN`` farther; otherwise it raises
:class:`UnreadableGlyph` -- a near tie is a guess, and a guessed tick label silently rescales a whole axis.

A word is glyphs on one row. Its sign and decimal point are read from geometry relative to its digits: a glyph no
taller than ``MARK_HEIGHT`` of the tallest digit is a minus when it is wider than tall and sits at mid height, a
point when it sits on the baseline. Only a minus leading the word and at most one point are numbers.

A word that is TEXT (a unit such as ``mT``, a grade such as ``N96``) is read by :func:`read_word` over ``ALPHABET``,
where shape alone cannot tell an ``o`` from an ``O``: each glyph is also compared by where its ink sits against the
word's baseline (the median glyph bottom), in ems taken from the word's tallest top assumed to be ``ASCENT`` -- a
word with no capital, ascender or digit is mis-scaled and is then refused, never guessed. The same ``ACCEPT`` and
``MARGIN`` refuse a near tie (``l``, ``I`` and ``1`` are one bar in Helvetica, so a word holding one raises). A
small raised mark taller than wide is a prime (``mu'``, ``mu''`` read as ``µ'``, ``µ''``); a word of one letter is
read by shape alone (:func:`_word_chars`).

The references are a :class:`Font`. ``HELVETICA`` renders them; ``STROKE`` strokes the polylines of
:mod:`document_lab.strokefont` (the stroke-font lettering of TDK SIFERRIT sheets) with :func:`stroke`, which is also
how a stroked glyph's own ink is made. Every reading takes the font it compares against; a character whose
reference lies within ``TWIN`` of another's in that font is refused, never read.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pymupdf

from document_lab.strokefont import TEMPLATES

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    'ALPHABET',
    'HELVETICA',
    'STROKE',
    'Font',
    'Glyph',
    'UnreadableGlyph',
    'has_letter',
    'rasterise',
    'read_glyph',
    'read_number',
    'read_word',
    'stroke',
]

#: The font the reference digits are rendered in (PyMuPDF's built-in Helvetica).
REFERENCE_FONT: Final = 'helv'
#: The characters :func:`read_word` reads.
ALPHABET: Final = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzµ'
#: The height above the baseline (ems) of a ``REFERENCE_FONT`` capital, ascender or digit (0.73 measured).
ASCENT: Final = 0.73
#: Weight of a glyph's top and bottom (ems from the baseline) in a word reading's distance.
PLACE_WEIGHT: Final = 8.0
#: Two characters whose references (nearest phases, with place) are this close are one shape to the reader: at 32
#: rows ``I``/``l`` are 1.65 apart, anti-aliasing noise, and the nearest distinct pair ``O``/``Q`` 1.91 (2026-11-10).
TWIN: Final = 1.8
#: A minus' centre sits at least this far (ems) above the baseline; a point (as wide as tall) sits on it.
MINUS_FLOOR: Final = 0.12
#: A mark taller than wide whose bottom sits at least this far (ems) above the baseline is a prime (``mu'``).
PRIME_FLOOR: Final = 0.3
#: Rows and columns of the ink-fraction grid a glyph is compared on.
GRID: Final = (12, 8)
#: Weight of the width-to-height ratio difference in the distance (a ``1`` is narrow, a ``0`` is not).
ASPECT_WEIGHT: Final = 2.0
#: The largest distance at which a glyph still reads as its nearest digit.
ACCEPT: Final = 2.5
#: How much farther (distance) the second nearest digit must be than the nearest.
MARGIN: Final = 0.75
#: A glyph at most this fraction of the word's tallest glyph is a sign or a point, not a digit.
MARK_HEIGHT: Final = 0.45
#: Rows a vector outline is rasterised to.
RASTER_ROWS: Final = 32
#: The darkness (0 white .. 1 black) above which a pixel counts as ink when a glyph is cropped.
INK: Final = 0.25
#: The sub-pixel offsets (pixels, each of x and y) a reference digit is rendered at.
PHASES: Final = (0.0, 1 / 3, 2 / 3)
#: The fewest pixel rows a glyph needs before its shape is compared at all: below it a 6 and an 8 of the reference
#: font itself come within ``MARGIN`` (measured 2026-11-08 on Helvetica 9..13 px high).
MIN_ROWS: Final = 14
#: The thinnest stroke (pixels) :func:`stroke` draws: a hairline still inks the pixels it crosses.
MIN_STROKE_PX: Final = 1.0

#: Shapes at one pixel height: per character, its variants (features, aspect) and its (top, bottom) in ems.
Shapes = dict[str, tuple[list[tuple[np.ndarray, float]], tuple[float, float]]]


class UnreadableGlyph(ValueError):
    """A glyph matches no reference digit clearly enough to read."""


@dataclass(frozen=True, slots=True, eq=False)
class Glyph:
    """One glyph: its ink (rows top-down; darkness 0..1 or bool) and its box (x0, y0, x1, y1), y growing down."""

    ink: np.ndarray
    box: tuple[float, float, float, float]


def rasterise(polygons: Sequence[Sequence[tuple[float, float]]], box: tuple[float, float, float, float]) -> np.ndarray:
    """The even-odd fill of ``polygons`` (a glyph's subpaths, holes too) over ``box``, ``RASTER_ROWS`` high."""
    x0, y0, x1, y1 = box
    scale = RASTER_ROWS / max(y1 - y0, 1e-9)
    cols = max(1, round((x1 - x0) * scale))
    x, y = np.meshgrid(x0 + (np.arange(cols) + 0.5) / scale, y0 + (np.arange(RASTER_ROWS) + 0.5) / scale)
    ink = np.zeros(x.shape, dtype=bool)
    for polygon in polygons:
        p = np.asarray(polygon, dtype=float)
        for (ax, ay), (bx, by) in zip(p, np.roll(p, -1, axis=0), strict=True):
            if ay == by:
                continue
            crosses = (ay > y) != (by > y)
            ink ^= crosses & (x < (bx - ax) * (y - ay) / (by - ay) + ax)
    return ink


def stroke(
    polylines: Sequence[Sequence[tuple[float, float]]],
    width: float,
    rows: int = RASTER_ROWS,
    phase: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """The ink of ``polylines`` stroked ``width`` wide, ``rows`` high, shifted by ``phase`` pixels.

    The image is the lines' box grown by half the width on every side, so a glyph is never clipped at its own ink
    box; the stroke is at least ``MIN_STROKE_PX`` wide, so a hairline still inks the pixels it crosses.
    """
    lines = [np.asarray(line, dtype=float).reshape(-1, 2) for line in polylines]
    every = np.vstack(lines)
    (x0, y0), (x1, y1) = every.min(axis=0) - width / 2, every.max(axis=0) + width / 2
    scale = rows / max(y1 - y0, 1e-9)
    half = max(width, MIN_STROKE_PX / scale) / 2
    cols = max(1, round((x1 - x0) * scale))
    x, y = np.meshgrid(x0 + (np.arange(cols) + 0.5 - phase[0]) / scale, y0 + (np.arange(rows) + 0.5 - phase[1]) / scale)
    ink = np.zeros(x.shape, dtype=bool)
    for p in lines:
        for (ax, ay), (bx, by) in zip(p, p[1:] if len(p) > 1 else p, strict=False):
            dx, dy = bx - ax, by - ay
            length = dx * dx + dy * dy
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / length, 0.0, 1.0) if length > 0 else 0.0
            ink |= np.hypot(x - ax - t * dx, y - ay - t * dy) <= half
    return ink


class Font:
    """The references a glyph is read against: shapes per pixel height, and vertical metrics in ems."""

    name: str

    def digits(self, rows: int) -> dict[str, list[tuple[np.ndarray, float]]]:
        """Each digit's shape variants (features, aspect) ``rows`` pixels high."""
        raise NotImplementedError

    def letters(self, rows: int) -> Shapes:
        """Each character's shape variants ``rows`` pixels high and its (top, bottom) in ems above the baseline."""
        raise NotImplementedError

    def ascent(self) -> float:
        """The height above the baseline (ems) of a capital, ascender or digit."""
        raise NotImplementedError

    def x_height(self) -> float:
        """The height above the baseline (ems) of a lowercase ``x``."""
        raise NotImplementedError


class _Helvetica(Font):
    """``REFERENCE_FONT`` rendered by PyMuPDF: every ``ALPHABET`` character."""

    name = REFERENCE_FONT

    def digits(self, rows: int) -> dict[str, list[tuple[np.ndarray, float]]]:
        return _references(rows)

    def letters(self, rows: int) -> Shapes:
        return _letter_references(rows)

    def ascent(self) -> float:
        return ASCENT

    def x_height(self) -> float:
        return _x_height()


class _Stroke(Font):
    """The polylines of :data:`document_lab.strokefont.TEMPLATES`, stroked; an em is a digit's height."""

    name = 'stroke'

    def digits(self, rows: int) -> dict[str, list[tuple[np.ndarray, float]]]:
        return {c: variants for c, (variants, _) in _stroke_references(rows).items() if c.isdigit()}

    def letters(self, rows: int) -> Shapes:
        return _stroke_references(rows)

    def ascent(self) -> float:
        return 1.0

    def x_height(self) -> float:
        return TEMPLATES['z'][0][0]


#: The reference font text and filled outlines are read against.
HELVETICA: Final[Font] = _Helvetica()
#: The stroke font TDK SIFERRIT sheets letter their plots in (:mod:`document_lab.strokefont`).
STROKE: Final[Font] = _Stroke()


def read_glyph(ink: np.ndarray, font: Font = HELVETICA) -> str:
    """The digit ``ink`` (darkness 0..1, or bool) shows; raises :class:`UnreadableGlyph` when none is clearly best.

    A glyph nearer a letter of ``font`` than any digit is that letter (a ``C`` is no ``0``), unless the font draws
    the two as twins.
    """
    ink = _crop(ink)
    if ink.shape[0] < MIN_ROWS:
        msg = f'a glyph {ink.shape[0]} px high is below the {MIN_ROWS} px a digit needs to be told apart'
        raise UnreadableGlyph(msg)
    features, aspect = _features(ink)
    ranked = sorted(
        (min(float(np.linalg.norm(features - f) + ASPECT_WEIGHT * abs(aspect - a)) for f, a in variants), digit)
        for digit, variants in font.digits(ink.shape[0]).items()
    )
    (best, digit), (second, runner) = ranked[0], ranked[1]
    if best > ACCEPT or second - best < MARGIN:
        msg = f'a glyph reads as {digit!r} at {best:.2f} and {runner!r} at {second:.2f}: no clear digit'
        raise UnreadableGlyph(msg)
    near, letter = min(
        (min(float(np.linalg.norm(features - f) + ASPECT_WEIGHT * abs(aspect - a)) for f, a in variants), char)
        for char, (variants, _) in font.letters(ink.shape[0]).items()
    )
    if near < best and not letter.isdigit() and digit not in _twins(ink.shape[0], font, placed=False)[letter]:
        msg = f'a glyph reads as the letter {letter!r} at {near:.2f}, nearer than the digit {digit!r} at {best:.2f}'
        raise UnreadableGlyph(msg)
    return digit


def read_word(glyphs: Sequence[Glyph], font: Font = HELVETICA) -> str:
    """The text a row of glyphs spells (left to right) over ``ALPHABET`` plus ``-`` and ``.`` marks.

    Raises :class:`UnreadableGlyph` when a glyph matches no character clearly.
    """
    text = ''
    for char in _word_chars(glyphs, font):
        if isinstance(char, UnreadableGlyph):
            raise char
        text += char
    return text


def has_letter(glyphs: Sequence[Glyph], font: Font = HELVETICA) -> bool:
    """Whether a glyph of the row reads clearly as a letter, however the rest read: the row is a word, not a number."""
    return any(isinstance(char, str) and char.isalpha() for char in _word_chars(glyphs, font))


def _word_chars(glyphs: Sequence[Glyph], font: Font) -> list[str | UnreadableGlyph]:
    """Per glyph left to right, the character it reads as in its word, else the reader's refusal.

    The baseline is the median bottom of the word's letters (its marks aside). The word's top is the ascent (a
    capital, ascender or digit) or the x-height (``max.``, ``mm``): the word is read both ways and the way with fewer
    refusals wins; when both read every letter but spell different text (``ox`` or ``OX``) each letter is refused. A
    word of ONE letter has no baseline to place it by -- a lone ``u`` scaled as if its top were the ascent is a ``U``
    -- so that letter is read by shape alone, and refused when the reference font draws another character of that
    shape (``o`` and ``O``).
    """
    ordered = sorted(glyphs, key=lambda g: g.box[0])
    tall = max(g.box[3] - g.box[1] for g in ordered)
    letters = [g for g in ordered if g.box[3] - g.box[1] > MARK_HEIGHT * tall]
    baseline = float(np.median([g.box[3] for g in letters]))
    top = min(g.box[1] for g in ordered)
    readings = []
    for height in (font.ascent(), font.x_height()) if len(letters) > 1 else (font.ascent(),):
        em = (baseline - top) / height
        readings.append(
            _marked(
                [_letter(g, baseline, em, font, placed=len(letters) > 1) for g in letters], ordered, baseline, em, font
            )
        )
    best = min(readings, key=lambda r: sum(isinstance(c, UnreadableGlyph) for c in r))
    clean = [r for r in readings if not any(isinstance(c, UnreadableGlyph) for c in r)]
    if len(clean) > 1 and clean[0] != clean[1]:
        msg = f'the word reads as {"".join(clean[0])!r} at the ascent and {"".join(clean[1])!r} at the x-height'
        return [UnreadableGlyph(msg) for _ in ordered]
    return best


def _letter(glyph: Glyph, baseline: float, em: float, font: Font, *, placed: bool) -> str | UnreadableGlyph:
    _, y0, _, y1 = glyph.box
    try:
        return _read_char(glyph.ink, ((baseline - y0) / em, (baseline - y1) / em) if placed else None, font)
    except UnreadableGlyph as refused:
        return refused


def _marked(
    read: list[str | UnreadableGlyph], ordered: Sequence[Glyph], baseline: float, em: float, font: Font
) -> list[str | UnreadableGlyph]:
    """The word's characters in order: its letters as ``read``, each mark a minus, a point or a prime."""
    tall = max(g.box[3] - g.box[1] for g in ordered)
    letters = iter(read)
    out: list[str | UnreadableGlyph] = []
    for glyph in ordered:
        x0, y0, x1, y1 = glyph.box
        if y1 - y0 > MARK_HEIGHT * tall:
            out.append(next(letters))
        elif x1 - x0 > y1 - y0 and baseline - font.ascent() * em < (y0 + y1) / 2 < baseline - MINUS_FLOOR * em:
            out.append('-')
        elif y1 >= baseline - 0.1 * em:
            out.append('.')
        elif y1 - y0 > x1 - x0 and y1 < baseline - PRIME_FLOOR * em:
            out.append("'")
        else:
            msg = f'a mark at {tuple(round(v, 1) for v in glyph.box)} is neither a minus, a point nor a prime'
            out.append(UnreadableGlyph(msg))
    return out


def _read_char(ink: np.ndarray, place: tuple[float, float] | None, font: Font) -> str:
    """The ``font`` character ``ink`` shows with its (top, bottom) ems from the baseline at ``place``.

    With no ``place`` the character is read by shape alone, and a twin is then judged by shape alone too.
    """
    ink = _crop(ink)
    if ink.shape[0] < MIN_ROWS:
        msg = f'a glyph {ink.shape[0]} px high is below the {MIN_ROWS} px a character needs to be told apart'
        raise UnreadableGlyph(msg)
    features, aspect = _features(ink)
    ranked = sorted(
        (
            min(float(np.linalg.norm(features - f) + ASPECT_WEIGHT * abs(aspect - a)) for f, a in variants)
            + (0.0 if place is None else PLACE_WEIGHT * (abs(place[0] - top) + abs(place[1] - bottom))),
            char,
        )
        for char, (variants, (top, bottom)) in font.letters(ink.shape[0]).items()
    )
    (best, char), (second, runner) = ranked[0], ranked[1]
    if best > ACCEPT or second - best < MARGIN:
        msg = f'a glyph reads as {char!r} at {best:.2f} and {runner!r} at {second:.2f}: no clear character'
        raise UnreadableGlyph(msg)
    twins = _twins(ink.shape[0], font, placed=place is not None)[char]
    if twins:  # the reference itself cannot be told apart: matching it clearly says nothing
        msg = f'a glyph reads as {char!r}, which the reference font draws within {TWIN} of {"".join(twins)!r}'
        raise UnreadableGlyph(msg)
    return char


@functools.cache
def _twins(rows: int, font: Font, *, placed: bool = True) -> dict[str, tuple[str, ...]]:
    """Per ``font`` character, the others whose reference (shape, and place when ``placed``) is within ``TWIN``."""
    refs = font.letters(rows)

    def distance(a: str, b: str) -> float:
        (ta, ba), (tb, bb) = refs[a][1], refs[b][1]
        shape = min(
            float(np.linalg.norm(fa - fb) + ASPECT_WEIGHT * abs(aa - ab))
            for fa, aa in refs[a][0]
            for fb, ab in refs[b][0]
        )
        return shape + (PLACE_WEIGHT * (abs(ta - tb) + abs(ba - bb)) if placed else 0.0)

    return {a: tuple(b for b in refs if b != a and distance(a, b) <= TWIN) for a in refs}


def read_number(glyphs: Sequence[Glyph], font: Font = HELVETICA) -> float:
    """The number a row of glyphs spells (left to right); raises :class:`UnreadableGlyph` when it spells none."""
    ordered = sorted(glyphs, key=lambda g: g.box[0])
    tall = max(g.box[3] - g.box[1] for g in ordered)
    top = min(g.box[1] for g in ordered)
    bottom = max(g.box[3] for g in ordered)
    text = ''
    for glyph in ordered:
        x0, y0, x1, y1 = glyph.box
        if y1 - y0 > MARK_HEIGHT * tall:
            text += read_glyph(glyph.ink, font)
        elif x1 - x0 > y1 - y0 and top < (y0 + y1) / 2 < bottom - (bottom - top) / 4:
            text += '-'
        elif y1 >= bottom - (bottom - top) / 4:
            text += '.'
        else:
            msg = f'a mark at {tuple(round(v, 1) for v in glyph.box)} is neither a minus nor a point'
            raise UnreadableGlyph(msg)
    if '-' in text[1:] or text.count('.') > 1 or not any(c.isdigit() for c in text):
        msg = f'the glyphs spell {text!r}, not a number'
        raise UnreadableGlyph(msg)
    return float(text)


def _crop(ink: np.ndarray) -> np.ndarray:
    dark = ink > INK
    rows, cols = np.flatnonzero(dark.any(axis=1)), np.flatnonzero(dark.any(axis=0))
    if rows.size == 0:
        return ink[:0, :0]
    return ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _features(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """Ink fractions on the ``GRID`` cells (each cell at least one pixel) and the width-to-height ratio."""
    h, w = ink.shape
    gh, gw = GRID
    cells = np.empty(GRID)
    for i in range(gh):
        r0 = i * h // gh
        r1 = max(r0 + 1, (i + 1) * h // gh)
        for j in range(gw):
            c0 = j * w // gw
            c1 = max(c0 + 1, (j + 1) * w // gw)
            cells[i, j] = float(ink[r0:r1, c0:c1].mean())
    return cells.ravel(), w / h


@functools.cache
def _references(rows: int) -> dict[str, list[tuple[np.ndarray, float]]]:
    """Each digit of ``REFERENCE_FONT`` rendered ``rows`` pixels high at each ``PHASES`` sub-pixel offset, as features.

    A glyph a dozen pixels high is mostly edge: compared with a large reference its anti-aliased rim reads as a
    different shape, so the reference is sampled the way the glyph was.
    """
    size = rows / _digit_height()
    out: dict[str, list[tuple[np.ndarray, float]]] = {}
    for digit in '0123456789':
        out[digit] = [_features(_crop(_render(digit, size, (dx, dy)))) for dx in PHASES for dy in PHASES]
    return out


@functools.cache
def _letter_references(rows: int) -> dict[str, tuple[list[tuple[np.ndarray, float]], tuple[float, float]]]:
    """Each ``ALPHABET`` character's features ``rows`` pixels high (as :func:`_references`) and its place in ems."""
    out = {}
    for char in ALPHABET:
        top, bottom = _place(char)
        size = rows / (top - bottom)
        variants = [_features(_crop(_render(char, size, (dx, dy)))) for dx in PHASES for dy in PHASES]
        out[char] = (variants, (top, bottom))
    return out


@functools.cache
def _stroke_references(rows: int) -> Shapes:
    """Each stroke-font character stroked ``rows`` pixels high at each ``PHASES`` offset, as features, and its place."""
    out: Shapes = {}
    for char, (place, width, drawn) in TEMPLATES.items():
        lines = [[tuple(float(v) / 100 for v in p.split(',')) for p in line.split()] for line in drawn.split(';')]
        variants = [_features(_crop(stroke(lines, width, rows, (dx, dy)))) for dx in PHASES for dy in PHASES]
        out[char] = (variants, place)
    return out


@functools.cache
def _place(char: str) -> tuple[float, float]:
    """The top and bottom of ``char``'s ink, in ems above ``REFERENCE_FONT``'s baseline."""
    size = 100.0
    rows = np.flatnonzero((_render(char, size, (0.0, 0.0)) > INK).any(axis=1))
    flat = np.flatnonzero((_render('H', size, (0.0, 0.0)) > INK).any(axis=1))[-1] + 1  # a flat bottom is the baseline
    return (flat - rows[0]) / size, (flat - rows[-1] - 1) / size


@functools.cache
def _x_height() -> float:
    """The height above the baseline (ems) of a ``REFERENCE_FONT`` lowercase ``x``."""
    return _place('x')[0]


@functools.cache
def _digit_height() -> float:
    """A ``REFERENCE_FONT`` digit's height per point of font size."""
    return _crop(_render('0', 100.0, (0.0, 0.0))).shape[0] / 100.0


def _render(text: str, size: float, offset: tuple[float, float]) -> np.ndarray:
    """``text`` at ``size`` (one pixel per point), shifted by ``offset`` pixels, as darkness."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=size * 2 + 10, height=size * 2 + 10)
        page.insert_text((5 + offset[0], size * 1.5 + offset[1]), text, fontname=REFERENCE_FONT, fontsize=size)
        pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
    return 1 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w) / 255
