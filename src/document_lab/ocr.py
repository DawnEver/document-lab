"""A text layer over a PDF of page images, by Tesseract: its words readable, its pixels exactly as they were.

Each page that is one image and has no text is recognized by the ``tesseract`` executable, run on the image at the
resolution it carries: lossless samples are handed over as a PGM/PPM, an image still in a codec Tesseract reads
(JPEG, JPEG 2000) as its encoded data, untouched -- never decoded here; any other is refused by name. The words it
reports (TSV, each with its pixel box) are laid invisible over where they are seen, by `document_lab._pdf`'s writer
(`Writer.copy_page_with_text`); every page is otherwise copied as it is, so a layer is never laid twice. The image is
taken as drawn upright over its box, as a capture's and a scan's is.

Tesseract is the user's to install, never bundled: :func:`check` finds it -- on PATH, else in the
:class:`TesseractPlaces` an entry reads from the environment (beside ``TESSDATA_PREFIX``, where its Windows
installer puts it) -- and refuses a missing one or a missing language, saying where it looked and how to install
it, before any work is spent.
"""

from __future__ import annotations

import shutil
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

type Box = tuple[float, float, float, float]
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


def _image_file(samples: tuple, folder: Path, name: str) -> Path:
    """The image ``samples`` (``Reader.image``) as a file Tesseract reads, in ``folder``.

    Its codec's data as stored, or its samples as an 8-bit PGM/PPM; refused, by name, when Tesseract could not read
    it.
    """
    width, height, space, components, bits, data, codec = samples
    if codec is not None:
        if codec not in _CODEC_FILES:
            msg = f'{name} is in {codec}, which Tesseract does not read'
            raise ValueError(msg)
        path = folder / f'page{".j2k" if data.startswith(_J2K_CODESTREAM) else _CODEC_FILES[codec]}'
        path.write_bytes(data)
        return path
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
    kind, suffix = ('P5', '.pgm') if components == 1 else ('P6', '.ppm')
    path = folder / f'page{suffix}'
    path.write_bytes(f'{kind}\n{width} {height}\n255\n'.encode() + np.ascontiguousarray(values).tobytes())
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


def _image(reader: _pdf.Reader, number: int) -> tuple[Box, int, int, int] | None:
    """Page ``number``'s one image -- box, samples across and down, object -- or None: text, or not one image."""
    spans, _, images, _ = reader.page(number, _CURVE_SAMPLES)
    if len(images) != 1 or images[0][3] is None or any(span[0].strip() for span in spans):
        return None
    return images[0]


def _recognize(reader: _pdf.Reader, image: tuple[Box, int, int, int], tesseract: list[str], name: str) -> Words:
    """The words ``tesseract`` (its command, but for the image) reads on ``image``, in page space."""
    box, width, height, obj = image
    dpi = round(width * 72 / (box[2] - box[0]))
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as folder:
        path = _image_file(reader.image(obj), Path(folder), name)
        tsv = _run([tesseract[0], str(path), *tesseract[1:], '--dpi', str(dpi), 'tsv'])
    return _words(tsv, box, width, height)


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
