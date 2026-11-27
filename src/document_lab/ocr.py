"""A text layer over a PDF of page images, by Tesseract: its words readable, its pixels exactly as they were.

Each page that is one image and has no text is recognized by the ``tesseract`` executable, run on the image at the
resolution it carries and as the page shows it -- upright, though its samples be placed turned or mirrored by quarter
turns (a scan on a ``/Rotate`` page); a placement at another angle is refused. Lossless samples are handed over as a
PGM/PPM, turned upright and, under a ``/Decode`` of ``[1 0]``, inverted exactly; any other ``/Decode`` is refused. An
image still in a codec is never decoded here: JPEG and JPEG 2000 go as their encoded data, untouched, and are refused
placed turned or inverted; CCITT fax data goes, as stored, inside a TIFF (:func:`_fax_tiff`) whose tags say its coding,
its black and its placement; any other codec is refused by name. The words Tesseract reports (TSV, each with its
pixel box on the upright image) are laid invisible over where they are seen, by `document_lab._pdf`'s writer
(`Writer.copy_page_with_text`); every page is otherwise copied as it is, so a layer is never laid twice.

Tesseract is the user's to install, never bundled: :func:`check` finds it -- on PATH, else in the
:class:`TesseractPlaces` an entry reads from the environment (beside ``TESSDATA_PREFIX``, where its Windows
installer puts it) -- and refuses a missing one or a missing language, saying where it looked and how to install
it, before any work is spent.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import numpy as np
from lab_commons.temp import TEMP_PREFIX

from document_lab import _pdf

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ['TesseractPlaces', 'check', 'text_layer']

_INSTALL: Final = (
    'install it -- Windows: the UB Mannheim installer (https://github.com/UB-Mannheim/tesseract/wiki), then add its '
    'folder to PATH or set TESSDATA_PREFIX to its tessdata; macOS: `brew install tesseract tesseract-lang`; '
    'Debian/Ubuntu: `apt install tesseract-ocr tesseract-ocr-<lang>`'
)
#: Codecs Tesseract reads from a file of their own, by the suffix that names the file's format.
_CODEC_FILES: Final = {'DCTDecode': '.jpg', 'JPXDecode': '.jp2'}
_J2K_CODESTREAM: Final = b'\xff\x4f\xff\x51'  # a JPEG 2000 codestream without the JP2 box around it
_WORD_LEVEL: Final = '5'  # a TSV row's level for one word
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
    """The ``tesseract`` executable, found on PATH or in ``places``, with every ``+``-joined language in ``languages``.

    Refused, naming what is missing, where it looked and how to install it, when either is missing.
    """
    executable = _executable(places)
    header, *listed = _run([str(executable), '--list-langs']).splitlines()
    available = {name.strip() for name in listed}
    if missing := [lang for lang in languages.split('+') if lang not in available]:
        msg = f'{executable} has no language data for {missing} ({header.strip()}); {_INSTALL}'
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


def _image_file(samples: tuple, placement: Placement, folder: Path, name: str) -> Path:
    """The image ``samples`` (``Reader.image``), shown as ``placement`` says, as a file Tesseract reads upright.

    Written in ``folder``: lossless samples as an 8-bit PGM/PPM, turned upright and inverted as their ``/Decode``
    says; a codec's data as stored, CCITT fax's inside a TIFF. Refused, by name, when Tesseract could not read it,
    or not as the page shows it.
    """
    width, height, space, components, bits, data, codec, decode, _masked, fax = samples
    if codec is None and (components not in {1, 3} or space == 'Indexed'):
        msg = f'{name} is {space} ({components} components): Tesseract is handed gray or RGB samples only'
        raise ValueError(msg)
    inverted = _inverted(decode, components, name)
    if codec == 'CCITTFaxDecode':
        path = folder / 'page.tif'
        orientation = _tiff_orientation(placement)
        path.write_bytes(_fax_tiff(width, height, data, fax, inverted=inverted, orientation=orientation, name=name))
        return path
    if codec is not None:
        if codec not in _CODEC_FILES:
            msg = f'{name} is in {codec}, which Tesseract does not read'
            raise ValueError(msg)
        for wrong, how in ((placement != _UPRIGHT, 'placed turned or mirrored'), (inverted, 'inverted by its /Decode')):
            if wrong:
                msg = f'{name} is in {codec}, {how}: showing it so needs its pixels decoded, which is not done here'
                raise ValueError(msg)
        path = folder / f'page{".j2k" if data.startswith(_J2K_CODESTREAM) else _CODEC_FILES[codec]}'
        path.write_bytes(data)
        return path
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
    if inverted:  # exact: each depth's levels spread over 0-255 evenly, so a level's inverse is 255 less it
        values = 255 - values
    rows = _upright(values[:, : width * components].reshape(height, width, components), placement)
    kind, suffix = ('P5', '.pgm') if components == 1 else ('P6', '.ppm')
    path = folder / f'page{suffix}'
    header = f'{kind}\n{rows.shape[1]} {rows.shape[0]}\n255\n'.encode()
    path.write_bytes(header + np.ascontiguousarray(rows).tobytes())
    return path


def _words(tsv: str, box: Box, width: int, height: int) -> Words:
    """Each word of Tesseract's ``tsv`` and its box in page space: its pixels mapped onto the image's ``box``."""
    x0, y0, x1, y1 = box
    sx, sy = (x1 - x0) / width, (y1 - y0) / height
    out = []
    for line in tsv.splitlines()[1:]:
        level, *_, left, top, across, down, _conf, text = line.split('\t')
        left, top, across, down = int(left), int(top), int(across), int(down)
        if level == _WORD_LEVEL and text.strip() and across > 0 and down > 0:
            seen = (x0 + left * sx, y0 + top * sy, x0 + (left + across) * sx, y0 + (top + down) * sy)
            out.append((text.strip(), seen))
    return out


def _image(reader: _pdf.Reader, number: int) -> tuple | None:
    """Page ``number``'s one image row (``Reader.page``) or None: text, or not one image."""
    spans, _, images, _ = reader.page(number, curve_samples=_CURVE_SAMPLES)
    if len(images) != 1 or images[0][3] is None or any(span[0].strip() for span in spans):
        return None
    return images[0]


def _recognize(reader: _pdf.Reader, image: tuple, tesseract: list[str], name: str) -> Words:
    """The words ``tesseract`` (its command, but for the image) reads on ``image`` upright, in page space.

    The upright image fills its box, so each word's pixel box maps onto the box.
    """
    box, width, height, obj, matrix, placement = image
    if placement is None:
        msg = f'{name} is placed at an angle no quarter turn makes ({matrix}): Tesseract reads it upright only'
        raise ValueError(msg)
    across, down = (height, width) if placement[0] else (width, height)
    dpi = round(across * 72 / (box[2] - box[0]))
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as folder:
        path = _image_file(reader.image(obj), placement, Path(folder), name)
        tsv = _run([tesseract[0], str(path), *tesseract[1:], '--dpi', str(dpi), 'tsv'])
    return _words(tsv, box, across, down)


def text_layer(pdf: Path, out: Path, languages: str = 'eng', *, places: TesseractPlaces) -> int:
    """``pdf`` with a text layer in ``languages`` over every page that is one image and has no text, to ``out``.

    Tesseract is found on PATH or in ``places``, and only once there is a page to recognize: a document already
    read needs none. ``out`` may be ``pdf`` itself: it is written beside as ``<out>.part`` and moved into place when
    whole; nothing is written when a page cannot be recognized. How many pages were recognized (0: ``out`` is a
    copy).
    """
    reader = _pdf.open(pdf.read_bytes(), b'')
    layers: dict[int, Words] = {}
    tesseract: list[str] = []
    for number, _, _ in reader.pages:
        if (image := _image(reader, number)) is None:
            continue
        tesseract = tesseract or [str(check(languages, places)), 'stdout', '-l', languages]
        layers[number] = _recognize(reader, image, tesseract, f'page {number} of {pdf.name}')
    part = out.with_name(out.name + '.part')
    writer = _pdf.create(part)
    try:
        for number, _, _ in reader.pages:
            if number in layers:
                writer.copy_page_with_text(reader, number, layers[number])
            else:
                writer.copy_page(reader, number)
        writer.close()
    except BaseException:
        del writer  # its file closed, so the part can be removed
        part.unlink(missing_ok=True)
        raise
    part.replace(out)
    return len(layers)
