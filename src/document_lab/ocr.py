"""A text layer over a PDF of page images, by Tesseract: its words readable, its pixels exactly as they were.

Each page that is one image and has no text is recognized by the ``tesseract`` executable in the languages asked
for (its ``-l``: ``eng``, ``chi_sim+eng``, ``jpn_vert``), run on the image at the resolution it carries and as the
page shows it -- though its samples be placed turned or mirrored by quarter turns (a scan on a ``/Rotate`` page); a
placement at another angle is refused. Lossless samples are handed over as a PGM/PPM,
turned and, under a ``/Decode`` of ``[1 0]``, inverted exactly; any other ``/Decode`` is refused. CCITT fax data goes,
as stored, inside a TIFF (:func:`_fax_tiff`) whose tags say its coding, its black and how it is shown; JPEG and JPEG
2000 go as their encoded data when shown as stored, and are decoded (`document_lab.pdf.read_raster`) only to be
measured, or turned, inverted or deskewed; any other codec is refused by name.

What the page shows is not always upright. Tesseract's orientation detection (OSD) says how far to turn it -- a
table printed sideways -- and a turn it is sure enough of (:data:`ORIENTATION_CONFIDENCE`) is read beside the page as
placed, line by line the surer kept, so an upright running head over a sideways table reads too; one it is unsure of
is reported and the page read as placed, never guessed. A page skewed more than :data:`DESKEW_ABOVE` degrees
(`document_lab.ink.skew`) is handed over turned back by its skew, resampled for Tesseract's eyes only: the PDF's
pixels stay as they were. The words Tesseract reports (its PAGE XML at word level, each with its pixel box on the
image it read and the direction it reads in) are mapped back through every turn and laid invisible over where they
are seen, along the baseline they are seen on, by `document_lab._pdf`'s writer (`Writer.copy_page_with_text`) -- a
word read top to bottom (CJK set in columns, which the ``_vert`` languages read) written vertically, so a reader reads
it as vertical writing; every page is otherwise copied as it is, so a layer is never laid twice. How each page was
read is returned (:class:`Layered`).

Tesseract is the user's to install, never bundled: :func:`check` finds it -- on PATH, else in the
:class:`TesseractPlaces` an entry reads from the environment (beside ``TESSDATA_PREFIX``, where its Windows
installer puts it) -- and refuses, before any work is spent, a missing one, one older than 5.3 (the first to write
PAGE XML), and a language it has no data for, saying what it has, where it looked and how to install it.
"""

from __future__ import annotations

import itertools
import math
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import numpy as np
from lab_commons.temp import TEMP_PREFIX

from document_lab import _pdf, ink
from document_lab.ir import BBox
from document_lab.pdf import read_raster

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ['DESKEW_ABOVE', 'ORIENTATION_CONFIDENCE', 'Layered', 'Reading', 'TesseractPlaces', 'check', 'text_layer']

#: OSD's confidence at or above which a page is read turned as it says. Measured over 795 pages of 25 scanned IEC
#: standards (96 dpi captures): OSD proposed a turn on 36, by eye every one of 27 sideways tables right (confidence
#: 4.76 to 18.83) and 9 near-blank pages or a back cover wrong (0.59 to 3.65); the default stands between them.
ORIENTATION_CONFIDENCE: Final = 4.2
#: Degrees of skew above which Tesseract is handed the page turned back by it. Measured on synthetic scans (prose
#: and a ruled table, 186 words, at 300 and 150 dpi): as placed it reads 172-173 and 167 words at 2 and 2.5 degrees
#: and almost none from 3 (0-62 of 186 at 3 to 8 degrees), deskewed 176-186 at every skew; at 1.5 deskewing gains 2
#: and 13 words, below 1 degree it changes a few either way -- there the picture is handed exactly as it is.
DESKEW_ABOVE: Final = 1.0
_BAND: Final = 256  # rows deskewed at a time
_TOO_FEW: Final = 'Too few characters'  # Tesseract OSD's failure on a page with too little text to judge by
#: The first Tesseract to write PAGE XML, whose words say the direction they read in.
_PAGE_XML_SINCE: Final = (5, 3)

_INSTALL: Final = (
    'install it -- Windows: the UB Mannheim installer (https://github.com/UB-Mannheim/tesseract/wiki), then add its '
    'folder to PATH or set TESSDATA_PREFIX to its tessdata; macOS: `brew install tesseract tesseract-lang`; '
    'Debian/Ubuntu: `apt install tesseract-ocr tesseract-ocr-<lang>`'
)
#: Codecs Tesseract reads from a file of their own, by the suffix that names the file's format.
_CODEC_FILES: Final = {'DCTDecode': '.jpg', 'JPXDecode': '.jp2'}
_J2K_CODESTREAM: Final = b'\xff\x4f\xff\x51'  # a JPEG 2000 codestream without the JP2 box around it
#: A PAGE XML word's reading direction (Tesseract writes these three), as whether it is written vertically.
_VERTICAL: Final = {'left-to-right': False, 'right-to-left': False, 'top-to-bottom': True}
#: Tesseract's options for PAGE XML of each word in a box, and its config for PAGE XML.
_PAGE_XML: Final = ('-c', 'page_xml_level=1', '-c', 'page_xml_polygon=0')
_CURVE_SAMPLES: Final = 1  # a page's paths are not looked at
#: An image's placement (the reader's: turned, rows reversed, columns reversed) when it shows its samples as stored.
_UPRIGHT: Final = (False, False, False)
_TIFF_SHORT: Final = 3
_TIFF_LONG: Final = 4
_TIFF_HEADER: Final = 8  # bytes before the one image directory
_TIFF_G4: Final = 4  # Compression: CCITT T.6
_TIFF_G3: Final = 3  # Compression: CCITT T.4, each row behind an end-of-line code
_TIFF_RLE: Final = 2  # Compression: CCITT modified Huffman, rows byte-aligned, no end-of-line codes
_T4_2D: Final = 1  # T4Options: rows coded two-dimensionally too
_T4_FILL: Final = 4  # T4Options: fill bits end each end-of-line code on a byte boundary

type Box = tuple[float, float, float, float]
type Placement = tuple[bool, bool, bool]
type Words = list[tuple[str, Box]]


@dataclass(frozen=True)
class TesseractPlaces:
    """The folders ``tesseract`` is looked for in after PATH: facts of the environment, read once at an entry."""

    folders: tuple[Path, ...]

    @classmethod
    def of(cls, variables: Mapping[str, str]) -> Self:
        """The folders the environment ``variables`` name.

        ``TESSDATA_PREFIX``'s and the one above it (the Windows installer puts tessdata beside the executable), then
        the Windows installer's own: for every user, under ``ProgramFiles``; for one (as winget installs it), under
        ``LOCALAPPDATA``'s ``Programs``.
        """
        folders = []
        if prefix := variables.get('TESSDATA_PREFIX'):
            folders += [Path(prefix), Path(prefix).parent]
        if program_files := variables.get('ProgramFiles'):
            folders.append(Path(program_files) / 'Tesseract-OCR')
        if local := variables.get('LOCALAPPDATA'):
            folders.append(Path(local) / 'Programs' / 'Tesseract-OCR')
        return cls(tuple(folders))


def _run(command: list[str]) -> str:
    """``command``'s standard output; refused, with its error output, when it fails."""
    done = subprocess.run(command, capture_output=True, check=False)
    if done.returncode:
        msg = f'{" ".join(command)} failed ({done.returncode}): {done.stderr.decode(errors="replace").strip()}'
        raise OSError(msg)
    return done.stdout.decode('utf-8')


def _executable(places: TesseractPlaces) -> Path:
    """The ``tesseract`` executable: on PATH, else in the first of ``places`` that holds it."""
    if found := shutil.which('tesseract'):
        return Path(found)
    for folder in places.folders:
        if found := shutil.which('tesseract', path=str(folder)):
            return Path(found)
    looked = ', '.join(['PATH', *(str(f) for f in places.folders)])
    msg = f'OCR needs the tesseract executable, found in none of: {looked}; {_INSTALL}'
    raise OSError(msg)


def check(languages: str, places: TesseractPlaces) -> Path:
    """The ``tesseract`` executable, found on PATH or in ``places``, able to read in ``languages``.

    It is 5.3 or later and has data for every ``+``-joined language in ``languages``. Refused, naming what is
    missing, where it looked and how to install it, when any is missing: for a language, the languages it has and
    the folder their data is in, where a language's ``<lang>.traineddata`` goes.
    """
    executable = _executable(places)
    said = _run([str(executable), '--version'])
    version = re.match(r'\s*tesseract v?(\d+)\.(\d+)', said)
    if version is None or (int(version[1]), int(version[2])) < _PAGE_XML_SINCE:
        first = said.strip().splitlines()[0] if said.strip() else 'silent'
        msg = f'{executable} says {first!r}: OCR needs Tesseract 5.3 or later, which writes PAGE XML; {_INSTALL}'
        raise OSError(msg)
    header, *listed = _run([str(executable), '--list-langs']).splitlines()
    available = sorted(name.strip() for name in listed if name.strip())
    if missing := [lang for lang in languages.split('+') if lang not in available]:
        folder = re.search(r'"(.+)"', header)
        where = folder[1] if folder else header.strip()
        msg = (
            f'{executable} has no language data for {missing}: it has {available}, in {where}; a language is its '
            f'<lang>.traineddata in that folder (https://github.com/tesseract-ocr/tessdata_best), or {_INSTALL}'
        )
        raise OSError(msg)
    return executable


def _tiff_orientation(placement: Placement) -> int:
    """The TIFF Orientation (tag 274) of samples shown as ``placement`` says: 1 upright, 6 a quarter clockwise, ..."""
    turned, rows_reversed, columns_reversed = placement
    flips = {(False, False): 1, (False, True): 2, (True, True): 3, (True, False): 4}
    return flips[rows_reversed, columns_reversed] + (4 if turned else 0)


def _upright(rows: np.ndarray, placement: Placement) -> np.ndarray:
    """``rows`` (height, width, components) as stored, as the page shows them under ``placement``."""
    turned, rows_reversed, columns_reversed = placement
    if turned:
        rows = rows.transpose(1, 0, 2)
    return rows[:: -1 if rows_reversed else 1, :: -1 if columns_reversed else 1]


def _inverted(decode: list[float] | None, components: int, name: str) -> bool:
    """Whether the ``/Decode`` array ``decode`` inverts each of ``components`` (``[1 0]``).

    Refused, by name, when it maps them otherwise than as stored or inverted.
    """
    if decode is None or decode == [0.0, 1.0] * components:
        return False
    if decode == [1.0, 0.0] * components:
        return True
    msg = f'{name} maps its samples by /Decode {decode}: Tesseract is handed them as stored or inverted only'
    raise ValueError(msg)


def _fax_tiff(
    width: int, height: int, data: bytes, fax: tuple, *, inverted: bool, orientation: int, name: str
) -> bytes:
    """CCITT fax ``data``, as stored, in a TIFF of one strip: packaging, never decoding.

    Its tags say the coding ``fax`` (``Reader.image``) states -- Compression 4 for Group 4; 3 for Group 3 behind
    end-of-line codes; 2 for one-dimensional rows byte-aligned without them -- which of its bits are black (BlackIs1,
    turned over by an ``inverted`` ``/Decode``) as its PhotometricInterpretation, and how the page places it as its
    Orientation. A coding no TIFF holds is refused by name.
    """
    k, columns, end_of_line, byte_align, black_is_1 = fax
    if columns != width:
        msg = f'{name} is CCITT fax of {columns} columns on an image {width} samples wide'
        raise ValueError(msg)
    options = {}
    if k < 0 and not (end_of_line or byte_align):
        compression = _TIFF_G4
    elif k >= 0 and end_of_line:
        compression = _TIFF_G3
        options[292] = (_TIFF_LONG, (_T4_2D if k > 0 else 0) | (_T4_FILL if byte_align else 0))  # T4Options
    elif k == 0 and byte_align:
        compression = _TIFF_RLE
    else:
        coding = f'K {k}, EndOfLine {end_of_line}, EncodedByteAlign {byte_align}'
        msg = f'{name} is CCITT fax coded as no TIFF holds ({coding}): Tesseract is handed fax data in a TIFF only'
        raise ValueError(msg)
    # Decoded, a black run is 1 bits: WhiteIsZero (0) shows them black, as BlackIs1 false does; BlackIsZero (1) white.
    photometric = int(black_is_1 != inverted)
    tags = {
        256: (_TIFF_LONG, width),  # ImageWidth
        257: (_TIFF_LONG, height),  # ImageLength
        258: (_TIFF_SHORT, 1),  # BitsPerSample
        259: (_TIFF_SHORT, compression),
        262: (_TIFF_SHORT, photometric),
        273: (_TIFF_LONG, 0),  # StripOffsets: the data's, once the directory's size is known
        274: (_TIFF_SHORT, orientation),
        277: (_TIFF_SHORT, 1),  # SamplesPerPixel
        278: (_TIFF_LONG, height),  # RowsPerStrip: one strip
        279: (_TIFF_LONG, len(data)),  # StripByteCounts
        **options,
    }
    tags[273] = (_TIFF_LONG, _TIFF_HEADER + 2 + 12 * len(tags) + 4)
    entries = b''.join(
        struct.pack('<HHI', tag, kind, 1)
        + (struct.pack('<HH', value, 0) if kind == _TIFF_SHORT else struct.pack('<I', value))
        for tag, (kind, value) in sorted(tags.items())
    )
    directory = struct.pack('<H', len(tags)) + entries + struct.pack('<I', 0)  # no next directory
    return struct.pack('<2sHI', b'II', 42, _TIFF_HEADER) + directory + data


def _lossless(samples: tuple, placement: Placement, name: str) -> np.ndarray | None:
    """The image ``samples`` (``Reader.image``) as the page shows them, 8-bit rows by columns by components, exactly.

    Lossless samples only, inverted as their ``/Decode`` says; None for a codec's. Refused, by name, when Tesseract
    could not be handed them.
    """
    width, height, space, components, bits, data, codec, decode, _masked, _fax = samples
    if codec is not None:
        return None
    if components not in {1, 3} or space == 'Indexed':
        msg = f'{name} is {space} ({components} components): Tesseract is handed gray or RGB samples only'
        raise ValueError(msg)
    row = (width * components * bits + 7) // 8
    stored = np.frombuffer(data, dtype=np.uint8)[: row * height].reshape(height, row)
    if bits == 16:  # noqa: PLR2004 -- the high byte of each big-endian sample
        values = stored[:, ::2]
    elif bits == 8:  # noqa: PLR2004
        values = stored
    else:  # 1, 2, 4: each sample spread over 0-255
        unpacked = np.unpackbits(stored, axis=1)[:, : width * components * bits]
        weights = 1 << np.arange(bits - 1, -1, -1)
        levels = unpacked.reshape(height, width * components, bits) @ weights
        values = (levels * 255 // (2**bits - 1)).astype(np.uint8)
    if _inverted(decode, components, name):  # exact: each depth's levels spread over 0-255 evenly, so 255 less
        values = 255 - values
    return _upright(values[:, : width * components].reshape(height, width, components), placement)


def _pnm(rows: np.ndarray, folder: Path) -> Path:
    """``rows`` (rows by columns by 1 or 3 components) as an 8-bit PGM/PPM in ``folder``."""
    kind, suffix = ('P5', '.pgm') if rows.shape[2] == 1 else ('P6', '.ppm')
    path = folder / f'page{suffix}'
    path.write_bytes(f'{kind}\n{rows.shape[1]} {rows.shape[0]}\n255\n'.encode() + np.ascontiguousarray(rows).tobytes())
    return path


def _turned_placement(placement: Placement, turn: int) -> Placement:
    """The placement that shows samples placed by ``placement`` turned ``turn`` degrees further, clockwise."""
    probe = np.arange(6).reshape(2, 3, 1)
    wanted = np.rot90(_upright(probe, placement), -turn // 90)
    return next(p for p in itertools.product((False, True), repeat=3) if np.array_equal(_upright(probe, p), wanted))


def _stored_file(samples: tuple, placement: Placement, turn: int, folder: Path, name: str) -> Path | None:
    """A codec's data, as stored, in a file Tesseract reads as ``placement`` shows it, turned ``turn`` degrees on.

    CCITT fax data inside a TIFF whose Orientation says how it is shown; JPEG and JPEG 2000 as their own files when
    shown as stored, else None (turning or inverting them needs their pixels decoded; a TIFF's Orientation over a
    color JPEG is not honored alike by every Tesseract build). Refused, by name, when no TIFF holds its fax coding.
    """
    width, height, _space, components, _bits, data, codec, decode, _masked, fax = samples
    inverted = _inverted(decode, components, name)
    if codec == 'CCITTFaxDecode':
        path = folder / 'page.tif'
        orientation = _tiff_orientation(_turned_placement(placement, turn))
        path.write_bytes(_fax_tiff(width, height, data, fax, inverted=inverted, orientation=orientation, name=name))
        return path
    if placement != _UPRIGHT or turn or inverted:
        return None
    path = folder / f'page{".j2k" if data.startswith(_J2K_CODESTREAM) else _CODEC_FILES[codec]}'
    path.write_bytes(data)
    return path


def _orientation(tesseract: str, path: Path) -> tuple[int, float] | None:
    """Tesseract's OSD of the image at ``path``: the degrees to turn it clockwise to read it upright, and how sure.

    None when OSD tells nothing: too few characters to judge by (a blank page, a picture).
    """
    try:
        found = _run([tesseract, str(path), 'stdout', '--psm', '0'])
    except OSError as failed:
        if _TOO_FEW in str(failed):
            return None
        raise
    turn, confidence = (re.search(rf'{field}: ([-\d.]+)', found) for field in ('Rotate', 'Orientation confidence'))
    if turn is None or confidence is None:
        msg = f'Tesseract OSD of {path.name} said no turn and confidence: {found!r}'
        raise OSError(msg)
    return int(turn[1]) % 360, float(confidence[1])


def _deskewed(rows: np.ndarray, degrees: float) -> np.ndarray:
    """``rows`` turned back by ``degrees`` (their skew, clockwise as seen) about their center, on the same canvas.

    Resampled bilinearly, white where the turn brings in nothing: for Tesseract's eyes only, never written to a PDF.
    """
    height, width = rows.shape[:2]
    cx, cy = width / 2, height / 2
    sin, cos = math.sin(math.radians(degrees)), math.cos(math.radians(degrees))
    padded = np.pad(rows, ((1, 1), (1, 1), (0, 0)), constant_values=255)
    out = np.empty_like(rows)
    x = np.arange(width, dtype=np.float32) + 0.5 - cx
    for start in range(0, height, _BAND):  # a band of rows at a time: a page's grids would take a gigabyte at once
        y = np.arange(start, min(start + _BAND, height), dtype=np.float32)[:, None] + 0.5 - cy
        # Each pixel of the picture turned back shows the point the skew turned it to (pixel centers at the halves).
        sx, sy = cx + x * cos - y * sin - 0.5, cy + x * sin + y * cos - 0.5
        left, top = np.floor(sx), np.floor(sy)
        fx, fy = (sx - left)[..., None], (sy - top)[..., None]
        x0, y0 = np.clip(left + 1, 0, width + 1).astype(np.intp), np.clip(top + 1, 0, height + 1).astype(np.intp)
        x1, y1 = np.clip(left + 2, 0, width + 1).astype(np.intp), np.clip(top + 2, 0, height + 1).astype(np.intp)
        above = padded[y0, x0] * (1 - fx) + padded[y0, x1] * fx
        below = padded[y1, x0] * (1 - fx) + padded[y1, x1] * fx
        out[start : start + y.shape[0]] = np.clip(above * (1 - fy) + below * fy + 0.5, 0, 255)
    return out


def _frame(box: Box, size: tuple[int, int], skew: float, turn: int) -> np.ndarray:
    """The affine map (3 x 3) from a pixel of the image Tesseract read to page space.

    That image is the picture as the page shows it (``size``: columns, rows), filling ``box``, turned back by
    ``skew`` degrees about its center, then turned ``turn`` degrees clockwise.
    """
    width, height = size
    x0, y0, x1, y1 = box
    onto = np.array([[(x1 - x0) / width, 0, x0], [0, (y1 - y0) / height, y0], [0, 0, 1]])
    sin, cos = math.sin(math.radians(skew)), math.cos(math.radians(skew))
    cx, cy = width / 2, height / 2
    back = np.array([[cos, -sin, cx - cx * cos + cy * sin], [sin, cos, cy - cx * sin - cy * cos], [0, 0, 1]])
    unturn = {  # a point of the image turned clockwise, on the image before the turn
        0: [[1, 0, 0], [0, 1, 0]],
        90: [[0, 1, 0], [-1, 0, height]],
        180: [[-1, 0, width], [0, -1, height]],
        270: [[0, -1, width], [1, 0, 0]],
    }[turn]
    return onto @ back @ np.array([*unturn, [0, 0, 1]], dtype=np.float64)


type _Line = tuple[int, int]  # a word's line: its reading's index, then the line's place in Tesseract's PAGE XML


@dataclass(frozen=True)
class _Word:
    """A recognized word, in page space.

    Its ``text``, its ``box`` turned back level, its baseline's ``angle`` (degrees clockwise as seen), its ``line``,
    Tesseract's ``confidence`` in it (0 to 1), and whether it is written ``vertical`` (it reads top to bottom).
    """

    text: str
    box: Box
    angle: float
    line: _Line
    confidence: float
    vertical: bool

    def covers(self) -> Box:
        """The box it covers on the page, turned as it is."""
        x0, y0, x1, y1 = self.box
        cx, cy, half_w, half_h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        sin, cos = abs(math.sin(math.radians(self.angle))), abs(math.cos(math.radians(self.angle)))
        across, down = half_w * cos + half_h * sin, half_w * sin + half_h * cos
        return cx - across, cy - down, cx + across, cy + down


def _local(element: ET.Element) -> str:
    """``element``'s name without its namespace."""
    return element.tag.rpartition('}')[2]


def _words(page: str, frame: np.ndarray, reading: int, name: str) -> list[_Word]:
    """Each word of Tesseract's PAGE XML ``page`` (word level, boxes), mapped by ``frame`` (:func:`_frame`).

    Into page space, with its line, Tesseract's confidence in it and whether it reads top to bottom. A word read in
    a direction Tesseract does not write is refused by name.
    """
    linear = frame[:2, :2]
    along, across = linear @ (1.0, 0.0), linear @ (0.0, 1.0)
    angle = round(math.degrees(math.atan2(along[1], along[0])), 6)
    out = []
    # Tesseract's own output, of the image we handed it: no entity or reference beyond the document in it
    lines = [e for e in ET.fromstring(page).iter() if _local(e) == 'TextLine']  # noqa: S314
    for place, line in enumerate(lines):
        for word in (e for e in line if _local(e) == 'Word'):
            children = {_local(c): c for c in word}
            equiv = {_local(c): c for c in children['TextEquiv']} if 'TextEquiv' in children else {}
            text = ''.join(equiv['Unicode'].itertext()).strip() if 'Unicode' in equiv else ''
            coords = children['Coords'].get('points', '') if 'Coords' in children else ''
            points = [tuple(map(int, p.split(','))) for p in coords.split()]
            xs, ys = [x for x, _ in points], [y for _, y in points]
            if not text or not points or max(xs) <= min(xs) or max(ys) <= min(ys):
                continue
            direction = word.get('readingDirection')
            if direction not in _VERTICAL:
                msg = f'Tesseract read {text!r} on {name} {direction or "in no direction"}: a word is laid as read so'
                raise ValueError(msg)
            left, top, wide, high = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
            cx, cy, _ = frame @ (left + wide / 2, top + high / 2, 1.0)
            half_w, half_h = wide * math.hypot(*along) / 2, high * math.hypot(*across) / 2
            box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            confidence = float(children['TextEquiv'].get('conf', '0'))
            out.append(_Word(text, box, angle, (reading, place), confidence, _VERTICAL[direction]))
    return out


def _surer(readings: list[list[_Word]]) -> list[_Word]:
    """The words of every reading but those whose line another reading reads surer where it lies.

    A page read as placed and turned -- a table printed sideways under an upright running head -- is misread by each
    reading where the other reads right. Line by line, a line is kept unless a line of another reading overlaps it
    with a higher mean confidence.
    """
    lines: dict[_Line, list[_Word]] = defaultdict(list)
    for word in itertools.chain.from_iterable(readings):
        lines[word.line].append(word)
    measured: dict[_Line, tuple[Box, float]] = {}
    for key, words in lines.items():
        boxes = [w.covers() for w in words]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        measured[key] = (box, sum(w.confidence for w in words) / len(words))

    def beaten(key: _Line) -> bool:
        (x0, y0, x1, y1), mine = measured[key]
        return any(
            other[0] != key[0] and theirs > mine and ox0 < x1 and x0 < ox1 and oy0 < y1 and y0 < oy1
            for other, ((ox0, oy0, ox1, oy1), theirs) in measured.items()
        )

    kept = {key for key in lines if not beaten(key)}
    return [w for w in itertools.chain.from_iterable(readings) if w.line in kept]


@dataclass(frozen=True)
class Reading:
    """How page ``page`` was recognized.

    ``osd``: Tesseract's orientation detection -- the degrees to turn the page clockwise to read it upright, and how
    sure it is -- or None when it told nothing (too few characters) or was not asked; ``turn``: the degrees the page
    was read turned besides as placed (0: as placed only); ``unsure``: OSD told nothing, or proposed a turn it was
    not sure enough of, so the page was read as placed. ``skew``: the degrees its lines stand turned, clockwise as
    seen (None: no lines tell, or it was not measured); ``deskewed``: whether Tesseract was handed it turned back.
    """

    page: int
    osd: tuple[int, float] | None
    turn: int
    unsure: bool
    skew: float | None
    deskewed: bool

    def note(self) -> str | None:
        """What a reader of the layer should know of how this page was read; None when read as placed, unskewed."""
        parts = []
        if self.turn and self.osd is not None:
            parts.append(f'read turned {self.turn} degrees as well as placed (OSD confidence {self.osd[1]:.2f})')
        elif self.unsure and self.osd is None:
            parts.append('orientation untold (OSD: too few characters), read as placed')
        elif self.unsure and self.osd is not None:
            turn, confidence = self.osd
            parts.append(
                f'OSD proposed turning {turn} degrees at confidence {confidence:.2f}, too unsure: read as placed'
            )
        if self.deskewed:
            parts.append(f'read deskewed by {self.skew:.2f} degrees')
        return f'page {self.page}: {"; ".join(parts)}' if parts else None


@dataclass(frozen=True)
class Layered:
    """What :func:`text_layer` did: a :class:`Reading` of each page it recognized, in page order (none: a copy)."""

    readings: tuple[Reading, ...]


@dataclass(frozen=True)
class _Asked:
    """How pages are to be read.

    The OSD confidence a turn is read at (None: never turned, OSD not asked) and the skew above which a page is
    deskewed (None: never, skew not measured).
    """

    orientation_confidence: float | None
    deskew_above: float | None


def _image(reader: _pdf.Reader, number: int) -> tuple | None:
    """Page ``number``'s one image row (``Reader.page``) or None: text, or not one image."""
    spans, _, images, _ = reader.page(number, annotations=False, curve_samples=_CURVE_SAMPLES)
    if len(images) != 1 or images[0][3] is None or any(span[0].strip() for span in spans):
        return None
    return images[0]


def _recognize(
    pdf: Path, reader: _pdf.Reader, number: int, image: tuple, *, tesseract: list[str], asked: _Asked
) -> tuple[list[_Word], Reading]:
    """The words ``tesseract`` (its command, but for the image) reads on page ``number``'s ``image``, in page space.

    The picture as the page shows it is measured -- its skew (`document_lab.ink.skew`) and, by OSD, its orientation
    -- and Tesseract reads it turned back by its skew when above ``asked``'s and, when OSD is sure enough of a turn,
    turned so as well as placed, line by line the surer kept (:func:`_surer`). It is handed over exact where it can
    be -- lossless samples turned exactly, fax data and an upright JPEG as stored -- and decoded
    (`document_lab.pdf.read_raster`) only to be measured, or when in a codec to be turned or deskewed.
    """
    name = f'page {number} of {pdf.name}'
    box, _width, _height, obj, matrix, placement, _ = image
    if placement is None:
        msg = f'{name} is placed at an angle no quarter turn makes ({matrix}): Tesseract reads it upright only'
        raise ValueError(msg)
    samples = reader.image(obj)
    if samples[6] is not None and samples[6] not in {*_CODEC_FILES, 'CCITTFaxDecode'}:
        msg = f'{name} is in {samples[6]}, which Tesseract does not read'
        raise ValueError(msg)
    shown = (samples[1], samples[0]) if placement[0] else (samples[0], samples[1])  # columns, rows as shown
    exact = _lossless(samples, placement, name)
    measure = asked.deskew_above is not None
    pixels = exact
    if pixels is None and measure:  # a codec's pixels, at their own resolution, as the page shows them
        decoded = read_raster(pdf, number, BBox(*box), dpi=None).pixels
        pixels = decoded[..., :1] if samples[3] == 1 else decoded[..., :3]
    skew = ink.skew(pixels) if measure and pixels is not None else None
    deskew = skew is not None and asked.deskew_above is not None and abs(skew) > asked.deskew_above
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as made:
        folder = Path(made)

        def handed(turn: int) -> tuple[Path, tuple[int, int]]:
            """The file Tesseract reads turned ``turn`` degrees clockwise (deskewed when it is to be), and its size.

            The size (columns, rows) is the picture's before that turn.
            """
            nonlocal pixels
            if not deskew and exact is None and (stored := _stored_file(samples, placement, turn, folder, name)):
                return stored, shown
            if pixels is None:  # a codec turned or inverted, never measured
                decoded = read_raster(pdf, number, BBox(*box), dpi=None).pixels
                pixels = decoded[..., :1] if samples[3] == 1 else decoded[..., :3]
            rows = _deskewed(pixels, skew) if deskew and skew is not None else pixels
            return _pnm(np.rot90(rows, -turn // 90), folder), (pixels.shape[1], pixels.shape[0])

        osd = None if asked.orientation_confidence is None else _orientation(tesseract[0], handed(0)[0])
        sure = osd is not None and asked.orientation_confidence is not None and osd[1] >= asked.orientation_confidence
        turn = osd[0] if sure and osd is not None else 0
        readings = []
        for reading, quarter in enumerate([0, turn] if turn else [0]):
            path, size = handed(quarter)
            dpi = str(round(size[0] * 72 / (box[2] - box[0])))
            page = _run([tesseract[0], str(path), *tesseract[1:], '--dpi', dpi, 'page'])
            frame = _frame(box, size, skew if deskew and skew is not None else 0.0, quarter)
            readings.append(_words(page, frame, reading, name))
    unsure = asked.orientation_confidence is not None and (osd is None or (osd[0] != 0 and not sure))
    words = _surer(readings) if len(readings) > 1 else readings[0]
    return words, Reading(number, osd, turn, unsure, skew, deskew)


def text_layer(
    pdf: Path,
    out: Path,
    languages: str = 'eng',
    *,
    places: TesseractPlaces,
    orientation_confidence: float | None = ORIENTATION_CONFIDENCE,
    deskew_above: float | None = DESKEW_ABOVE,
) -> Layered:
    """``pdf`` with a text layer in Tesseract's ``languages`` over every page of one image and no text, to ``out``.

    ``languages`` are Tesseract's, ``+``-joined (``chi_sim+eng``; ``jpn_vert`` for text set in columns).

    A page OSD says to turn with ``orientation_confidence`` or more is read turned as well as placed (None: OSD is
    not asked, and Tesseract's ``osd`` data not needed); a page skewed more than ``deskew_above`` degrees is read
    turned back (None: skew is not measured). Tesseract is found on PATH or in ``places``, and only once there is a
    page to recognize: a document already read needs none. ``out`` may be ``pdf`` itself: it is written beside as
    ``<out>.part`` and moved into place when whole; nothing is written when a page cannot be recognized.
    """
    reader = _pdf.open(pdf, b'')
    asked = _Asked(orientation_confidence, deskew_above)
    layers: dict[int, list[_Word]] = {}
    readings = []
    tesseract: list[str] = []
    for number, _, _ in reader.pages:
        if (image := _image(reader, number)) is None:
            continue
        needed = languages if orientation_confidence is None else f'{languages}+osd'
        tesseract = tesseract or [str(check(needed, places)), 'stdout', '-l', languages, *_PAGE_XML]
        layers[number], reading = _recognize(pdf, reader, number, image, tesseract=tesseract, asked=asked)
        readings.append(reading)
    part = out.with_name(out.name + '.part')
    writer = _pdf.create(part)
    try:
        for number, _, _ in reader.pages:
            if number in layers:
                words = [(w.text, w.box, w.angle, w.vertical) for w in layers[number]]
                writer.copy_page_with_text(reader, number, words)
            else:
                writer.copy_page(reader, number)
        writer.close()
    except BaseException:
        del writer  # its file closed, so the part can be removed
        part.unlink(missing_ok=True)
        raise
    part.replace(out)
    return Layered(tuple(readings))
