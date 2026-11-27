"""The text layer, with a fake Tesseract: what it refuses before any work, the image it hands Tesseract -- as
placed, turned as OSD says, turned back by its skew -- where the words it gets back are laid, and the pages it leaves
alone.

That real Tesseract reads a real capture is the ``ocr`` part's (tests/ocr)."""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from document_lab import _pdf, ocr
from document_lab.capture.export import to_pdf
from document_lab.ink import skew
from document_lab.ocr import TesseractPlaces, check, text_layer
from document_lab.pdf import read_pdf
from tests.synthetic import fax_g4, image_pdf, scan_image, scan_pdf, turned

PAGE_NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'


def region(*words: tuple, confidence: float = 0.965) -> str:
    """Tesseract's PAGE XML text region (word level, boxes) for ``words`` -- (text, left, top, width, height[, line[,
    direction]]) in pixels, each line in its own text line, a word read left to right unless its direction says --
    each word ``confidence`` sure (0 to 1)."""
    lines: dict[int, list[str]] = {}
    for i, (text, x, y, w, h, *rest) in enumerate(words):
        line, reads = (rest + [1, 'left-to-right'][len(rest) :])[:2]
        points = f'{x},{y} {x + w},{y} {x + w},{y + h} {x},{y + h}'
        lines.setdefault(line, []).append(
            f'<Word id="w{i}" readingDirection="{reads}"><Coords points="{points}"/>'
            f'<TextEquiv index="1" conf="{confidence}"><Unicode>{text}</Unicode></TextEquiv></Word>'
        )
    laid = ''.join(f'<TextLine id="l{n}">{"".join(ws)}</TextLine>' for n, ws in sorted(lines.items()))
    return f'<TextRegion id="r0">{laid}</TextRegion>'


def document(*regions: str) -> str:
    """Tesseract's PAGE XML of a page holding ``regions``."""
    return f'<?xml version="1.0" encoding="UTF-8"?><PcGts xmlns="{PAGE_NS}"><Page>{"".join(regions)}</Page></PcGts>'


def page(*words: tuple, confidence: float = 0.965) -> str:
    """Tesseract's PAGE XML of a page of one region of ``words`` (:func:`region`)."""
    return document(region(*words, confidence=confidence))


#: How Tesseract names itself and its Leptonica (``--version``).
VERSION = 'tesseract v5.4.0.20240606\n leptonica-1.84.1\n'
#: Its options for PAGE XML of words in boxes, before the page config.
PAGE = ['-c', 'page_xml_level=1', '-c', 'page_xml_polygon=0']


def osd(turn: int, confidence: float) -> str:
    """Tesseract's OSD answer: turn the page ``turn`` degrees clockwise, ``confidence`` sure."""
    seen = (360 - turn) % 360
    return (
        f'Page number: 0\nOrientation in degrees: {seen}\nRotate: {turn}\nOrientation confidence: {confidence}\n'
        'Script: Latin\nScript confidence: 2.00\n'
    )


TOO_FEW = None  # OSD's answer on a page of too few characters: it fails


class FakeTesseract:
    """Answers each recognition with the next canned PAGE XML and each OSD with ``orientation`` (:data:`TOO_FEW`: it
    fails as on a blank page), keeping the commands and the image files recognized."""

    def __init__(self, *answers: str, orientation: str | None = TOO_FEW) -> None:
        self.answers = list(answers)
        self.orientation = orientation
        self.version = VERSION
        self.images: list[tuple[str, bytes]] = []
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        if command[1] == '--version':
            return self.version
        if command[1] == '--list-langs':
            return 'List of available languages in "/fake/tessdata/" (4):\neng\nchi_sim\nchi_sim_vert\nosd\n'
        if '--psm' in command:
            if self.orientation is None:
                msg = f'{" ".join(command)} failed (1): Too few characters. Skipping this page'
                raise OSError(msg)
            return self.orientation
        image = Path(command[1])
        self.images.append((image.suffix, image.read_bytes()))
        return self.answers.pop(0)


PLACES = TesseractPlaces(())  # the fake is found wherever it is looked for


@pytest.fixture
def tesseract(monkeypatch: pytest.MonkeyPatch):
    def install(*answers: str, orientation: str | None = TOO_FEW) -> FakeTesseract:
        fake = FakeTesseract(*answers, orientation=orientation)
        monkeypatch.setattr(ocr, '_executable', lambda _places: Path('tesseract'))
        monkeypatch.setattr(ocr, '_run', fake)
        return fake

    return install


@pytest.fixture
def no_tesseract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TesseractPlaces:
    """No tesseract anywhere: PATH empty, and the places an environment names -- ``TESSDATA_PREFIX`` an empty
    folder, ``ProgramFiles`` one without Tesseract -- holding none."""
    prefix = tmp_path / 'tessdata'
    prefix.mkdir()
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    return TesseractPlaces.of({'TESSDATA_PREFIX': str(prefix), 'ProgramFiles': str(tmp_path / 'programs')})


def test_the_places_are_beside_tessdata_then_the_windows_installer_s(tmp_path):
    prefix = tmp_path / 'Tesseract-OCR' / 'tessdata'
    variables = {'TESSDATA_PREFIX': str(prefix), 'ProgramFiles': str(tmp_path / 'all'), 'LOCALAPPDATA': str(tmp_path)}
    assert TesseractPlaces.of(variables).folders == (
        prefix,
        prefix.parent,
        tmp_path / 'all' / 'Tesseract-OCR',  # installed for every user
        tmp_path / 'Programs' / 'Tesseract-OCR',  # installed for one (winget)
    )
    assert TesseractPlaces.of({}).folders == ()


def words(pdf: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Each span our reader reads from ``pdf``'s first page: its text and box."""
    return [(s.text, tuple(s.provenance.bbox)) for s in read_pdf(pdf).pages[0].spans]


def near(a, b, tolerance: float = 0.01) -> bool:
    return all(abs(x - y) < tolerance for x, y in zip(a, b, strict=True))


def capture(tmp_path: Path, rows: int = 300, cols: int = 400) -> tuple[Path, np.ndarray]:
    image = np.random.default_rng(0).integers(0, 256, (rows, cols, 3), dtype=np.uint8)
    return to_pdf(image, tmp_path / 'capture.pdf', page_height=0, dpi=96), image


def one_image(path: Path, *, size: tuple[int, int], space: str, bits: int, codec: str | None, data: bytes) -> Path:
    """A one-page PDF of one image, as many points across and down as it has samples (``size``)."""
    writer = _pdf.create(path)
    writer.image_page(*size, space, bits, codec, data, *size)
    writer.close()
    return path


def test_no_tesseract_is_refused_naming_where_it_looked_and_how_to_install_it(no_tesseract):
    with pytest.raises(OSError, match='tesseract') as refused:
        check('eng', no_tesseract)
    message = str(refused.value)
    assert 'PATH' in message
    for folder in no_tesseract.folders:
        assert str(folder) in message
    for step in ('UB Mannheim', 'brew install tesseract tesseract-lang', 'apt install tesseract-ocr'):
        assert step in message


def test_tesseract_is_found_beside_its_tessdata(no_tesseract, monkeypatch):
    executable = no_tesseract.folders[1] / ('tesseract.exe' if os.name == 'nt' else 'tesseract')
    executable.write_bytes(b'')
    executable.chmod(0o755)
    monkeypatch.setattr(ocr, '_run', FakeTesseract())
    assert check('eng', no_tesseract) == executable


def test_a_missing_language_is_refused_naming_those_installed_and_where_language_data_goes(tesseract):
    tesseract()
    check('eng', PLACES)
    check('chi_sim_vert+chi_sim+eng', places=PLACES)
    for asked in ('jpn+eng', 'eng+', 'jpn_vert'):
        with pytest.raises(OSError, match='no language data') as refused:
            check(asked, PLACES)
        message = str(refused.value)
        for said in (
            "['chi_sim', 'chi_sim_vert', 'eng', 'osd']",
            '/fake/tessdata/',
            '<lang>.traineddata',
            'tessdata_best',
        ):
            assert said in message
        assert 'tesseract-ocr-<lang>' in message


@pytest.mark.parametrize('said', ['tesseract 4.1.1\n leptonica-1.82.0\n', 'tesseract v5.2.0\n', ''])
def test_a_tesseract_older_than_5_3_is_refused_by_its_version(tesseract, said):
    fake = tesseract()
    fake.version = said
    with pytest.raises(OSError, match=r'5\.3 or later') as refused:
        check('eng', PLACES)
    assert (said.splitlines()[0] if said else 'silent') in str(refused.value)
    assert 'UB Mannheim' in str(refused.value)


def test_the_words_are_laid_invisible_where_they_are_seen_over_the_same_pixels(tmp_path, tesseract):
    pdf, image = capture(tmp_path)
    fake = tesseract(page(('stator', 40, 30, 120, 36), ('winding', 200, 150, 160, 40)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', 'eng', places=PLACES).readings) == 1
    ((suffix, handed),) = fake.images
    assert suffix == '.ppm'
    assert handed == b'P6\n400 300\n255\n' + image.tobytes()  # the samples as stored, losslessly
    assert fake.commands[-1][2:] == ['stdout', '-l', 'eng', *PAGE, '--dpi', '96', 'page']
    assert fake.commands[-2][2:] == ['stdout', '--psm', '0']  # its orientation asked first, of the same file
    scale = 72 / 96
    expected = [('stator', (40, 30, 160, 66)), ('winding', (200, 150, 360, 190))]
    read = words(tmp_path / 'ocr.pdf')
    assert [t for t, _ in read] == [t for t, _ in expected]
    for (_, box), (_, pixels) in zip(read, expected, strict=True):
        assert near(box, [v * scale for v in pixels]), box
    with pymupdf.open(tmp_path / 'ocr.pdf') as doc:  # an independent reader finds them where they are seen
        found = [(w[4], w[:4]) for w in doc[0].get_text('words')]
        assert [t for t, _ in found] == ['stator', 'winding']
        for (_, box), (_, pixels) in zip(found, expected, strict=True):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            assert pixels[0] * scale < cx < pixels[2] * scale
            assert pixels[1] * scale < cy < pixels[3] * scale
        ((xref, *_),) = doc[0].get_images()
        pix = pymupdf.Pixmap(doc, xref)
        assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(image.shape), image)


def test_the_layer_s_streams_are_deflated_and_an_independent_reader_reads_them_as_stored_ones(tmp_path):
    pdf, _ = capture(tmp_path)
    reader = _pdf.open(pdf.read_bytes(), b'')
    layer = [
        ('stator', (30.0, 20.0, 120.0, 47.0), 0.0, False),
        (chr(0x5B9A) + chr(0x5B50), (150.0, 112.0, 270.0, 142.0), 0.0, False),
    ]
    read = {}
    for level in (0, None):  # stored as deflate's stored blocks, and at the measured default
        out = tmp_path / f'level-{level}.pdf'
        writer = _pdf.create(out) if level is None else _pdf.create(out, level)
        writer.copy_page_with_text(reader, 1, layer)
        writer.close()
        with pymupdf.open(out) as doc:
            streams = [x for x in range(1, doc.xref_length()) if doc.xref_is_stream(x) and not doc.xref_is_image(x)]
            deflated = ('name', '/FlateDecode')  # spelling-ignore-line: the PDF name
            assert [doc.xref_get_key(x, 'Filter') for x in streams] == [deflated] * len(streams)
            read[level] = (
                [doc.xref_stream(x) for x in streams],
                [w[4] for w in doc[0].get_text('words')],
                out.stat().st_size,
            )
    assert read[0][:2] == read[None][:2]
    assert read[0][1] == [text for text, *_ in layer]
    assert read[None][2] < read[0][2]
    with pytest.raises(ValueError, match='0 to 10, not 11'):
        _pdf.create(tmp_path / 'never.pdf', 11)


def test_cjk_text_is_laid_and_read_back(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    recognized = [''.join(map(chr, (0x5B9A, 0x5B50, 0x7ED5, 0x7EC4))), ''.join(map(chr, (0x6E29, 0x5347)))]
    tesseract(page((recognized[0], 40, 30, 160, 40), (recognized[1], 40, 100, 80, 40)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', 'chi_sim+eng', places=PLACES).readings) == 1
    assert [t for t, _ in words(tmp_path / 'ocr.pdf')] == recognized
    with pymupdf.open(tmp_path / 'ocr.pdf') as doc:
        assert [w[4] for w in doc[0].get_text('words')] == recognized


def test_words_read_top_to_bottom_are_laid_as_vertical_writing(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    column = ''.join(map(chr, (0x5B9A, 0x5B50, 0x7ED5, 0x7EC4)))  # stator winding, down a column
    tesseract(page((column, 340, 20, 40, 160, 1, 'top-to-bottom'), ('stator', 40, 30, 120, 36, 2)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', 'chi_sim_vert+eng', places=PLACES).readings) == 1
    spans = read_pdf(tmp_path / 'ocr.pdf').pages[0].spans
    assert [(s.text, s.vertical) for s in spans] == [(column, True), ('stator', False)]
    assert near(tuple(spans[0].provenance.bbox), [v * 72 / 96 for v in (340, 20, 380, 180)])


def test_a_word_read_in_a_direction_no_layer_writes_is_refused_by_name(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    tesseract(page(('upward', 40, 30, 36, 120, 1, 'bottom-to-top')))
    with pytest.raises(ValueError, match=r"'upward'.*bottom-to-top"):
        text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)
    assert not (tmp_path / 'ocr.pdf').exists()


def test_a_page_with_text_is_copied_untouched_and_needs_no_tesseract(tmp_path, no_tesseract):
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'already searchable')
        doc.save(tmp_path / 'text.pdf')
    assert text_layer(tmp_path / 'text.pdf', tmp_path / 'out.pdf', places=no_tesseract).readings == ()
    assert [t for t, _ in words(tmp_path / 'out.pdf')] == ['already searchable']


def test_only_the_pages_of_one_image_without_text_are_recognized(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    with pymupdf.open(pdf) as doc, pymupdf.open(pdf) as again:
        doc.insert_pdf(again)  # page 2: the same image ...
        doc[1].insert_text((20, 20), 'searchable')  # ... with text: left alone
        doc.save(tmp_path / 'two.pdf')
    fake = tesseract(page(('stator', 40, 30, 120, 36)))
    assert len(text_layer(tmp_path / 'two.pdf', tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    assert len(fake.images) == 1
    pages = read_pdf(tmp_path / 'ocr.pdf').pages
    assert [[s.text for s in p.spans] for p in pages] == [['stator'], ['searchable']]


def test_a_jpeg_is_handed_to_tesseract_as_it_is_stored(tmp_path, tesseract):
    jpeg = pymupdf.Pixmap(pymupdf.csRGB, 30, 20, bytes([200]) * 30 * 20 * 3, False).tobytes('jpg')
    pdf = one_image(tmp_path / 'scan.pdf', size=(30, 20), space='DeviceRGB', bits=8, codec='DCTDecode', data=jpeg)
    fake = tesseract(page(('scan', 2, 3, 20, 10)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    assert fake.images == [('.jpg', jpeg)]  # never decoded by us
    assert words(tmp_path / 'ocr.pdf')[0][0] == 'scan'


def test_one_bit_samples_are_handed_as_eight_bit_gray(tmp_path, tesseract):
    pdf = one_image(
        tmp_path / 'bits.pdf', size=(3, 2), space='DeviceGray', bits=1, codec=None, data=bytes([0b01000000, 0b10100000])
    )
    fake = tesseract(page())
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    assert fake.images == [('.pgm', b'P5\n3 2\n255\n' + bytes([0, 255, 0, 255, 0, 255]))]
    assert words(tmp_path / 'ocr.pdf') == []


@pytest.mark.parametrize(
    'stored',
    [('DeviceGray', 1, 'JBIG2Decode'), ('DeviceCMYK', 8, None)],
)
def test_an_image_tesseract_cannot_be_handed_is_refused_by_name(tmp_path, tesseract, stored):
    space, bits, codec = stored
    pdf = one_image(tmp_path / 'scan.pdf', size=(2, 2), space=space, bits=bits, codec=codec, data=bytes(16))
    tesseract()
    with pytest.raises(ValueError, match=codec or space):
        text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['scan.pdf']


def test_a_capture_without_tesseract_is_refused_and_left_untouched(tmp_path, no_tesseract):
    pdf, _ = capture(tmp_path)
    before = pdf.read_bytes()
    with pytest.raises(OSError, match='tesseract'):
        text_layer(pdf, pdf, places=no_tesseract)
    assert pdf.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['capture.pdf', 'tessdata']


def test_the_layer_replaces_the_pdf_it_was_laid_on(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    tesseract(page(('stator', 40, 30, 120, 36)))
    assert len(text_layer(pdf, pdf, places=PLACES).readings) == 1
    assert [t for t, _ in words(pdf)] == ['stator']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['capture.pdf']


def tiff_tags(tiff: bytes) -> dict[int, int]:
    """Each tag of the one-strip TIFF ``tiff`` (little-endian, one value a tag) and its value."""
    assert tiff[:4] == b'II*\x00'
    (directory,) = struct.unpack_from('<I', tiff, 4)
    (count,) = struct.unpack_from('<H', tiff, directory)
    tags = {}
    for i in range(count):
        tag, kind, _, value = struct.unpack_from('<HHII', tiff, directory + 2 + 12 * i)
        tags[tag] = value & 0xFFFF if kind == 3 else value
    return tags


PARAMS = 'DecodeParms'  # spelling-ignore-line: the PDF name of a filter's parameters

#: A fax page's black and white: a bar with a hole, off center, so a turn or a mirror shows.
FAX = np.ones((40, 64), dtype=np.uint8)
FAX[10:30, 8:50] = 0
FAX[15:20, 20:30] = 1
FAX[2, 1:4] = 0


def fax_pdf(path: Path, data: bytes, params: str, *, extra: str = '', size=(64, 40), rotate: int = 0) -> Path:
    """A page of one 64 x 40 CCITT fax image coded ``data`` with ``params`` (its fax parameters' entries)."""
    image = f'/Width 64 /Height 40 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode {extra}'
    return image_pdf(path, data, f'{image} /{PARAMS} << {params} >>', size=size, deflate=False, rotate=rotate)


@pytest.mark.parametrize('black_is_1', ['false', 'true'])
@pytest.mark.parametrize('decode', ['', '/Decode [1 0]'])
def test_a_fax_image_is_handed_as_stored_in_a_tiff_that_shows_what_the_page_shows(
    tmp_path, tesseract, black_is_1, decode
):
    data = fax_g4(FAX)
    pdf = fax_pdf(tmp_path / 'fax.pdf', data, f'/K -1 /Columns 64 /BlackIs1 {black_is_1}', extra=decode)
    fake = tesseract(page(('fax', 2, 3, 20, 10)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    ((suffix, handed),) = fake.images
    assert suffix == '.tif'
    assert handed.endswith(data)  # packaged, never decoded
    tags = tiff_tags(handed)
    assert (tags[256], tags[257], tags[259], tags[274]) == (64, 40, 4, 1)  # Group 4, upright
    with pymupdf.open(pdf) as doc:  # an independent decoder shows the TIFF as it shows the page's image
        ((xref, *_),) = doc[0].get_images()
        assert pymupdf.Pixmap(handed).samples == pymupdf.Pixmap(doc, xref).samples
    ((text, box),) = words(tmp_path / 'ocr.pdf')
    assert text == 'fax'
    assert near(box, (2, 3, 22, 13))


def test_a_fax_image_placed_turned_is_handed_with_its_placement_as_the_tiff_s_orientation(tmp_path, tesseract):
    data = fax_g4(FAX)
    # 64 x 40 samples on a 32 x 20 point page turned a quarter clockwise: shown 20 x 32, 40 x 64 pixels upright
    pdf = fax_pdf(tmp_path / 'fax.pdf', data, '/K -1 /Columns 64', size=(32, 20), rotate=90)
    fake = tesseract(page(('turned', 4, 6, 16, 4)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    ((_, handed),) = fake.images
    assert handed.endswith(data)
    assert tiff_tags(handed)[274] == 6
    assert fake.commands[-1][-2:] == ['144', 'page']  # 40 pixels across 20 points
    ((text, box),) = words(tmp_path / 'ocr.pdf')
    assert text == 'turned'
    assert near(box, (2, 3, 10, 5))


@pytest.mark.parametrize(
    ('params', 'compression', 't4'),
    [
        ('/K 0 /EndOfLine true', 3, 0),
        ('/K 4 /EndOfLine true /EncodedByteAlign true', 3, 5),
        ('/K 0 /EncodedByteAlign true', 2, None),
    ],
)
def test_a_group_3_fax_image_s_coding_is_stated_in_its_tiff(tmp_path, tesseract, params, compression, t4):
    pdf = fax_pdf(tmp_path / 'fax.pdf', b'coded rows', f'{params} /Columns 64 /BlackIs1 true')
    fake = tesseract(page())
    text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)
    ((_, handed),) = fake.images
    tags = tiff_tags(handed)
    assert (tags[259], tags.get(292), tags[262]) == (compression, t4, 1)  # BlackIs1: BlackIsZero
    assert handed.endswith(b'coded rows')


@pytest.mark.parametrize(
    ('params', 'refusal'),
    [
        ('/K 0 /Columns 64', 'coded as no TIFF holds'),  # rows neither behind end-of-line codes nor byte-aligned
        ('/K -1 /Columns 64 /EncodedByteAlign true', 'coded as no TIFF holds'),
        ('/K -1 /Columns 32', '32 columns on an image 64 samples wide'),
    ],
)
def test_a_fax_coding_no_tiff_holds_is_refused_by_name(tmp_path, tesseract, params, refusal):
    pdf = fax_pdf(tmp_path / 'fax.pdf', b'coded rows', params)
    tesseract()
    with pytest.raises(ValueError, match=refusal):
        text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)


@pytest.mark.parametrize(
    ('image', 'stored', 'handed'),
    [
        ('/ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0]', [0, 10, 200, 255], ('P5', [255, 245, 55, 0])),
        ('/ColorSpace /DeviceGray /BitsPerComponent 2 /Decode [1 0]', [0b00011011, 0], ('P5', [255, 170, 255, 255])),
        (
            '/ColorSpace /DeviceRGB /BitsPerComponent 8 /Decode [1 0 1 0 1 0]',
            [0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255],
            ('P6', [255, 254, 253, 252, 251, 250, 5, 4, 3, 2, 1, 0]),
        ),
    ],
)
def test_lossless_samples_a_decode_inverts_are_handed_inverted_exactly(tmp_path, tesseract, image, stored, handed):
    pdf = image_pdf(tmp_path / 'inverted.pdf', bytes(stored), f'/Width 2 /Height 2 {image}', size=(2, 2))
    fake = tesseract(page())
    text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)
    kind, samples = handed
    assert fake.images[0][1] == f'{kind}\n2 2\n255\n'.encode() + bytes(samples)


def test_a_decode_that_neither_keeps_nor_inverts_the_samples_is_refused_by_name(tmp_path, tesseract):
    image = '/Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [0 0.5]'
    pdf = image_pdf(tmp_path / 'halved.pdf', bytes([0, 255]), image, size=(2, 1))
    tesseract()
    with pytest.raises(ValueError, match=r'/Decode \[0\.0, 0\.5\]'):
        text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)


@pytest.mark.parametrize(('decode', 'rotate'), [(' /Decode [1 0 1 0 1 0]', 0), ('', 90)])
def test_a_jpeg_shown_inverted_or_turned_is_handed_decoded_as_the_page_shows_it(tmp_path, tesseract, decode, rotate):
    left = np.zeros((20, 30, 3), dtype=np.uint8)
    left[:, 15:] = 230  # dark on the left, light on the right: a turn or an inversion shows
    jpeg = pymupdf.Pixmap(pymupdf.csRGB, 30, 20, left.tobytes(), False).tobytes('jpg')
    image = f'/Width 30 /Height 20 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode{decode}'
    pdf = image_pdf(tmp_path / 'scan.pdf', jpeg, image, size=(30, 20), deflate=False, rotate=rotate)
    fake = tesseract(page(('scan', 2, 3, 10, 6)))
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    ((suffix, handed),) = fake.images
    header, pixels = handed.split(b'\n', 3)[:3], np.frombuffer(handed.split(b'\n', 3)[3], dtype=np.uint8)
    assert suffix == '.ppm'
    columns, rows = map(int, header[1].split())
    shown = pixels.reshape(rows, columns, 3).mean(axis=2)
    if rotate:  # a quarter turn clockwise: the dark left half is shown on top
        assert (columns, rows) == (20, 30)
        assert shown[:12].mean() < 60 and shown[18:].mean() > 170
    else:  # inverted: the dark half shown light
        assert (columns, rows) == (30, 20)
        assert shown[:, :12].mean() > 170 and shown[:, 18:].mean() < 60
    assert words(tmp_path / 'ocr.pdf')[0][0] == 'scan'


#: 8 x 6 gray samples, each its own value, so any turn or mirror shows.
SAMPLES = np.arange(48, dtype=np.uint8).reshape(6, 8) * 5


@pytest.mark.parametrize(
    ('placement', 'shown'),
    [  # placement: the page's size (points), the cm drawing the image (None: over the page, upright), its /Rotate
        (((4, 3), None, 90), np.rot90(SAMPLES, -1)),  # a page turned a quarter clockwise
        (((4, 3), None, 180), np.rot90(SAMPLES, 2)),
        (((4, 3), None, 270), np.rot90(SAMPLES, 1)),
        (((3, 4), (0, 4, -3, 0, 3, 0), 0), np.rot90(SAMPLES, 1)),  # drawn turned a quarter counterclockwise
        (((4, 3), (-4, 0, 0, 3, 4, 0), 0), np.fliplr(SAMPLES)),  # drawn mirrored left to right
        (((4, 3), (4, 0, 0, -3, 0, 3), 0), np.flipud(SAMPLES)),  # drawn upside down
    ],
)
def test_samples_placed_turned_or_mirrored_are_handed_upright_and_their_words_laid_back(
    tmp_path, tesseract, placement, shown
):
    size, matrix, rotate = placement
    image = '/Width 8 /Height 6 /ColorSpace /DeviceGray /BitsPerComponent 8'
    pdf = image_pdf(tmp_path / 'placed.pdf', SAMPLES.tobytes(), image, size=size, matrix=matrix, rotate=rotate)
    fake = tesseract(page(('placed', 1, 2, 4, 3)))  # pixels of the upright image, two to the point
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings) == 1
    rows, columns = shown.shape
    assert fake.images == [('.pgm', f'P5\n{columns} {rows}\n255\n'.encode() + shown.tobytes())]
    assert fake.commands[-1][-2:] == ['144', 'page']
    ((text, box),) = words(tmp_path / 'ocr.pdf')
    assert text == 'placed'
    assert near(box, (0.5, 1, 2.5, 2.5))


def test_an_image_placed_at_an_angle_no_quarter_turn_makes_is_refused_by_name(tmp_path, tesseract):
    image = '/Width 8 /Height 6 /ColorSpace /DeviceGray /BitsPerComponent 8'
    turn = np.radians(30)
    matrix = (4 * np.cos(turn), 4 * np.sin(turn), -3 * np.sin(turn), 3 * np.cos(turn), 3, 1)
    pdf = image_pdf(tmp_path / 'skewed.pdf', SAMPLES.tobytes(), image, size=(8, 8), matrix=matrix)
    tesseract()
    with pytest.raises(ValueError, match='no quarter turn'):
        text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)


# Orientation: OSD's turn read beside the page as placed, the surer line kept; an unsure one said, never guessed.


def test_a_turn_osd_is_sure_of_is_read_as_well_as_placed_and_the_surer_line_kept(tmp_path, tesseract):
    pdf, image = capture(tmp_path)  # 400 x 300 pixels, 96 dpi
    as_placed = document(region(('ssiaw', 95, 150, 40, 100, 1), confidence=0.2), region(('header', 200, 20, 150, 20)))
    turned = page(('sideways', 40, 100, 120, 30), confidence=0.9)  # on the image turned a quarter clockwise
    fake = tesseract(as_placed, turned, orientation=osd(90, 15.0))
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings
    assert (reading.osd, reading.turn, reading.unsure) == ((90, 15.0), 90, False)
    assert reading.note() == 'page 1: read turned 90 degrees as well as placed (OSD confidence 15.00)'
    assert [handed for _, handed in fake.images] == [
        b'P6\n400 300\n255\n' + image.tobytes(),
        b'P6\n300 400\n255\n' + np.ascontiguousarray(np.rot90(image, -1)).tobytes(),  # turned exactly
    ]
    spans = {s.text: s for s in read_pdf(tmp_path / 'ocr.pdf').pages[0].spans}
    assert sorted(spans) == ['header', 'sideways']  # 'ssiaw', read less sure where 'sideways' lies, is dropped
    scale = 72 / 96  # 'sideways', 120 x 30 on the turned image at (40, 100), is 30 x 120 on the page, bottom to top
    assert near(spans['sideways'].provenance.bbox, [100 * scale, 140 * scale, 130 * scale, 260 * scale], 0.2)
    assert near(spans['sideways'].direction, (0, -1), 1e-6)
    assert near(spans['header'].direction, (1, 0), 1e-6)


@pytest.mark.parametrize(
    ('orientation', 'note'),
    [
        (osd(90, 3.0), 'page 1: OSD proposed turning 90 degrees at confidence 3.00, too unsure: read as placed'),
        (TOO_FEW, 'page 1: orientation untold (OSD: too few characters), read as placed'),
        (osd(0, 3.0), None),  # upright, however unsure: as placed, nothing guessed
    ],
)
def test_an_unsure_or_untold_orientation_is_read_as_placed_and_said(tmp_path, tesseract, orientation, note):
    pdf, _ = capture(tmp_path)
    fake = tesseract(page(('stator', 40, 30, 120, 36)), orientation=orientation)
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings
    assert (reading.turn, reading.unsure, reading.note()) == (0, note is not None, note)
    assert len(fake.images) == 1


def test_the_orientation_confidence_is_an_option_and_off_asks_no_osd(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    fake = tesseract(page(), page(), page(), orientation=osd(90, 3.0))
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES, orientation_confidence=2.5).readings
    assert reading.turn == 90
    del fake.commands[:]
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES, orientation_confidence=None).readings
    assert (reading.osd, reading.turn, reading.unsure) == (None, 0, False)
    assert not any('--psm' in command for command in fake.commands)


def test_missing_osd_data_is_refused_by_name_unless_orientation_is_off(tmp_path, tesseract, monkeypatch):
    pdf, _ = capture(tmp_path)
    fake = tesseract(page())
    listed = fake.__call__

    def without_osd(command):
        return listed(command).replace('\nosd', '') if command[1] == '--list-langs' else listed(command)

    monkeypatch.setattr(ocr, '_run', without_osd)
    with pytest.raises(OSError, match=r"no language data for \['osd'\]"):
        text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES)
    assert len(text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES, orientation_confidence=None).readings) == 1


# Skew: a page skewed past DESKEW_ABOVE is handed turned back, and its words laid back along the slant it shows.


def _skewed(tmp_path, degrees):
    def prose(page):
        for i in range(30):
            page.insert_text((72, 90 + 20 * i), 'The rated output of a machine is the value in its rating', fontsize=10)

    grey = scan_image(prose, dpi=72, angle=degrees)
    return scan_pdf(tmp_path / 'skewed.pdf', grey), grey


def test_a_skewed_page_is_handed_turned_back_and_its_words_laid_along_the_slant_it_shows(tmp_path, tesseract):
    pdf, _ = _skewed(tmp_path, 5.0)  # 612 x 792 pixels, a pixel a point
    fake = tesseract(page(('centered', 256, 386, 100, 20), ('aside', 100, 100, 60, 20)), orientation=osd(0, 20.0))
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings
    assert reading.skew == pytest.approx(5.0, abs=0.1)
    assert reading.deskewed
    assert reading.note() == f'page 1: read deskewed by {reading.skew:.2f} degrees'
    ((suffix, handed),) = fake.images
    assert (suffix, handed[:15]) == ('.pgm', b'P5\n612 792\n255\n')
    assert skew(np.frombuffer(handed[15:], dtype=np.uint8).reshape(792, 612)) == pytest.approx(0, abs=0.1)
    spans = {s.text: s for s in read_pdf(tmp_path / 'ocr.pdf').pages[0].spans}
    angle = math.radians(reading.skew)
    for text, (cx, cy) in (('centered', (306, 396)), ('aside', (130, 110))):
        box = spans[text].provenance.bbox
        x, y = turned(cx, cy, reading.skew)  # where the skewed page shows the point read turned back
        assert near(((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2), (x, y), 0.3), text
        assert near(spans[text].direction, (math.cos(angle), math.sin(angle)), 1e-3)
    assert not near(spans['aside'].provenance.bbox[:2], (100, 100), 1)  # a planted wrong truth: not where read


def test_a_page_skewed_no_more_than_the_option_is_handed_exactly_as_it_is(tmp_path, tesseract):
    pdf, grey = _skewed(tmp_path, 0.6)
    fake = tesseract(page(), page(), page(), orientation=osd(0, 20.0))
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES).readings
    assert reading.skew == pytest.approx(0.6, abs=0.1)
    assert not reading.deskewed
    assert fake.images[0][1] == b'P5\n612 792\n255\n' + grey.tobytes()  # its own samples, exactly
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES, deskew_above=0.5).readings
    assert reading.deskewed
    (reading,) = text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES, deskew_above=None).readings
    assert (reading.skew, reading.deskewed) == (None, False)
