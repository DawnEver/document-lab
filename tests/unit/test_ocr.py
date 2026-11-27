"""The text layer, with a fake Tesseract: what it refuses before any work, the image it hands Tesseract, where the
words it gets back are laid, and the pages it leaves alone.

That real Tesseract reads a real capture is the ``ocr`` part's (tests/ocr)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from document_lab import _pdf, ocr
from document_lab.capture.export import to_pdf
from document_lab.ocr import TesseractPlaces, check, text_layer
from document_lab.pdf import read_pdf

TSV_HEADER = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext'


def tsv(*words: tuple[str, int, int, int, int]) -> str:
    """Tesseract's TSV for ``words`` -- (text, left, top, width, height) in pixels -- behind a page row."""
    rows = [TSV_HEADER, '1\t1\t0\t0\t0\t0\t0\t0\t400\t300\t-1\t']
    rows += [f'5\t1\t1\t1\t1\t{i}\t{x}\t{y}\t{w}\t{h}\t96.5\t{t}' for i, (t, x, y, w, h) in enumerate(words, 1)]
    return '\n'.join(rows) + '\n'


class FakeTesseract:
    """Answers each recognition with the next canned TSV, keeping the command and the image file it was handed."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.images: list[tuple[str, bytes]] = []
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        if command[1] == '--list-langs':
            return 'List of available languages in "/fake/tessdata/" (2):\neng\nchi_sim\n'
        image = Path(command[1])
        self.images.append((image.suffix, image.read_bytes()))
        return self.answers.pop(0)


PLACES = TesseractPlaces(())  # the fake is found wherever it is looked for


@pytest.fixture
def tesseract(monkeypatch: pytest.MonkeyPatch):
    def install(*answers: str) -> FakeTesseract:
        fake = FakeTesseract(*answers)
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


def test_a_missing_language_is_refused_naming_it_where_it_looked_and_how_to_add_it(tesseract):
    tesseract()
    check('eng', PLACES)
    check('chi_sim+eng', places=PLACES)
    with pytest.raises(OSError, match='jpn') as refused:
        check('jpn+eng', PLACES)
    assert '/fake/tessdata/' in str(refused.value)
    assert 'tesseract-ocr-<lang>' in str(refused.value)


def test_the_words_are_laid_invisible_where_they_are_seen_over_the_same_pixels(tmp_path, tesseract):
    pdf, image = capture(tmp_path)
    fake = tesseract(tsv(('stator', 40, 30, 120, 36), ('winding', 200, 150, 160, 40)))
    assert text_layer(pdf, tmp_path / 'ocr.pdf', 'eng', places=PLACES) == 1
    ((suffix, handed),) = fake.images
    assert suffix == '.ppm'
    assert handed == b'P6\n400 300\n255\n' + image.tobytes()  # the samples as stored, losslessly
    assert fake.commands[-1][2:] == ['stdout', '-l', 'eng', '--dpi', '96', 'tsv']
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


def test_cjk_text_is_laid_and_read_back(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    recognized = [''.join(map(chr, (0x5B9A, 0x5B50, 0x7ED5, 0x7EC4))), ''.join(map(chr, (0x6E29, 0x5347)))]
    tesseract(tsv((recognized[0], 40, 30, 160, 40), (recognized[1], 40, 100, 80, 40)))
    assert text_layer(pdf, tmp_path / 'ocr.pdf', 'chi_sim+eng', places=PLACES) == 1
    assert [t for t, _ in words(tmp_path / 'ocr.pdf')] == recognized
    with pymupdf.open(tmp_path / 'ocr.pdf') as doc:
        assert [w[4] for w in doc[0].get_text('words')] == recognized


def test_a_page_with_text_is_copied_untouched_and_needs_no_tesseract(tmp_path, no_tesseract):
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), 'already searchable')
        doc.save(tmp_path / 'text.pdf')
    assert text_layer(tmp_path / 'text.pdf', tmp_path / 'out.pdf', places=no_tesseract) == 0
    assert [t for t, _ in words(tmp_path / 'out.pdf')] == ['already searchable']


def test_only_the_pages_of_one_image_without_text_are_recognized(tmp_path, tesseract):
    pdf, _ = capture(tmp_path)
    with pymupdf.open(pdf) as doc, pymupdf.open(pdf) as again:
        doc.insert_pdf(again)  # page 2: the same image ...
        doc[1].insert_text((20, 20), 'searchable')  # ... with text: left alone
        doc.save(tmp_path / 'two.pdf')
    fake = tesseract(tsv(('stator', 40, 30, 120, 36)))
    assert text_layer(tmp_path / 'two.pdf', tmp_path / 'ocr.pdf', places=PLACES) == 1
    assert len(fake.images) == 1
    pages = read_pdf(tmp_path / 'ocr.pdf').pages
    assert [[s.text for s in p.spans] for p in pages] == [['stator'], ['searchable']]


def test_a_jpeg_is_handed_to_tesseract_as_it_is_stored(tmp_path, tesseract):
    jpeg = pymupdf.Pixmap(pymupdf.csRGB, 30, 20, bytes([200]) * 30 * 20 * 3, False).tobytes('jpg')
    pdf = one_image(tmp_path / 'scan.pdf', size=(30, 20), space='DeviceRGB', bits=8, codec='DCTDecode', data=jpeg)
    fake = tesseract(tsv(('scan', 2, 3, 20, 10)))
    assert text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES) == 1
    assert fake.images == [('.jpg', jpeg)]  # never decoded by us
    assert words(tmp_path / 'ocr.pdf')[0][0] == 'scan'


def test_one_bit_samples_are_handed_as_eight_bit_gray(tmp_path, tesseract):
    pdf = one_image(
        tmp_path / 'bits.pdf', size=(3, 2), space='DeviceGray', bits=1, codec=None, data=bytes([0b01000000, 0b10100000])
    )
    fake = tesseract(tsv())
    assert text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES) == 1
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
    tesseract(tsv(('stator', 40, 30, 120, 36)))
    assert text_layer(pdf, pdf, places=PLACES) == 1
    assert [t for t, _ in words(pdf)] == ['stator']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['capture.pdf']
