"""The ``ocr`` part: a captured image of known words, given a Tesseract text layer, reads back as those words, its
pixels and page size unchanged; so does a fax scan of them, and a JPEG scan, each handed over in a TIFF, upright on a
page turned or not; so do Chinese and Japanese, set in lines or in columns, the columns as vertical writing.
Its own part, never tests/base: it needs Tesseract 5.3 or later with English, simplified Chinese and Japanese data
(``eng``, ``chi_sim``, ``chi_sim_vert``, ``jpn``, ``jpn_vert``), and FAILS naming what is missing where there is
none; it is never skipped."""

import difflib
import os
import subprocess

import numpy as np
import pymupdf
import pytest

from document_lab import ocr
from document_lab.capture.export import to_pdf
from document_lab.ocr import TesseractPlaces, check, text_layer
from document_lab.pdf import read_pdf
from tests.synthetic import fax_g4, image_pdf

pytestmark = pytest.mark.ocr

PLACES = TesseractPlaces.of(os.environ)  # where this machine's Tesseract may be, beyond PATH
WORDS = ('stator', 'winding', 'capture')


def rendered(words: tuple[str, ...]) -> np.ndarray:
    """``words``, one per line in large type, as the RGB pixels a screen capture would give."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=60 * len(words) + 40)
        for i, word in enumerate(words):
            page.insert_text((30, 60 + 60 * i), word, fontsize=36)
        pix = page.get_pixmap(dpi=96)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()


def test_the_text_layer_reads_back_over_the_same_pixels(tmp_path):
    check('eng', PLACES)
    image = rendered(WORDS)
    capture = to_pdf(image, tmp_path / 'capture.pdf', page_height=0, dpi=96)
    assert text_layer(capture, tmp_path / 'ocr.pdf', places=PLACES) == 1
    with pymupdf.open(capture) as before, pymupdf.open(tmp_path / 'ocr.pdf') as after:
        assert after.page_count == 1
        assert after[0].rect == before[0].rect
        ((xref, *_),) = after[0].get_images()
        pix = pymupdf.Pixmap(after, xref)
        assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(image.shape), image)
        assert all(word in after[0].get_text().casefold() for word in WORDS)  # an independent reader reads them
    drawn_where_they_are(tmp_path / 'ocr.pdf')
    laid_again = text_layer(tmp_path / 'ocr.pdf', tmp_path / 'again.pdf', places=PLACES)
    assert laid_again == 0  # a page with text is never laid twice


def drawn_where_they_are(pdf) -> None:
    """Each of :data:`WORDS` read back from ``pdf`` over where :func:`rendered` draws it."""
    spans = {span.text.casefold(): span.provenance.bbox for span in read_pdf(pdf).pages[0].spans}
    for i, word in enumerate(WORDS):  # from x 30, on the baseline at y 60 + 60 i
        x0, y0, x1, y1 = spans[word]
        assert 25 < x0 < 35 < x1
        assert y0 < 60 + 60 * i - 10 < y1


@pytest.mark.parametrize('rotate', [0, 90])
def test_a_fax_scan_reads_back_upright_on_a_page_turned_or_not(tmp_path, rotate):
    check('eng', PLACES)
    white = rendered(WORDS)[..., 0] >= 128  # the scanner's threshold
    stored = np.rot90(white, rotate // 90)  # turned back as far as the page turns it: shown upright
    rows, columns = stored.shape
    image = f'/Width {columns} /Height {rows} /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode'
    params = f'/DecodeParms << /K -1 /Columns {columns} >>'  # spelling-ignore-line: the PDF name
    size = (columns * 72 / 96, rows * 72 / 96)
    pdf = image_pdf(tmp_path / 'fax.pdf', fax_g4(stored), f'{image} {params}', size=size, deflate=False, rotate=rotate)
    assert text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES) == 1
    with pymupdf.open(tmp_path / 'ocr.pdf') as after:
        assert all(word in after[0].get_text().casefold() for word in WORDS)  # an independent reader reads them
    drawn_where_they_are(tmp_path / 'ocr.pdf')


def scanned(words: tuple[str, ...], rotate: int, components: int) -> tuple[bytes, tuple[int, int]]:
    """:func:`rendered` ``words`` as a JPEG of ``components`` (1: gray, 3: RGB), stored turned back as far as a page
    turned ``rotate`` turns it, so the page shows it upright; and its samples across and down."""
    shown = rendered(words)[..., :components]
    stored = np.ascontiguousarray(np.rot90(shown, rotate // 90))
    rows, columns = stored.shape[:2]
    space = pymupdf.csGRAY if components == 1 else pymupdf.csRGB
    return pymupdf.Pixmap(space, columns, rows, stored.tobytes(), False).tobytes('jpg', jpg_quality=90), (columns, rows)


@pytest.mark.parametrize(('rotate', 'components'), [(90, 3), (180, 3), (270, 1), (90, 1)])
def test_a_jpeg_scan_reads_back_upright_on_a_page_turned(tmp_path, rotate, components):
    check('eng', PLACES)
    data, (columns, rows) = scanned(WORDS, rotate, components)
    space = 'DeviceGray' if components == 1 else 'DeviceRGB'
    image = f'/Width {columns} /Height {rows} /ColorSpace /{space} /BitsPerComponent 8 /Filter /DCTDecode'
    size = (columns * 72 / 96, rows * 72 / 96)
    pdf = image_pdf(tmp_path / 'scan.pdf', data, image, size=size, deflate=False, rotate=rotate)
    assert text_layer(pdf, tmp_path / 'ocr.pdf', places=PLACES) == 1
    with pymupdf.open(tmp_path / 'ocr.pdf') as after:
        assert all(word in after[0].get_text().casefold() for word in WORDS)  # an independent reader reads them
    drawn_where_they_are(tmp_path / 'ocr.pdf')


def test_leptonica_still_shows_a_color_jpeg_at_tiff_orientation_8_turned_half_round(tmp_path):
    """The evidence behind ``ocr``'s one refused JPEG placement: should Leptonica show it upright, this fails, and
    the refusal (``_COLOR_JPEG_SHOWN_WRONG``) is to go."""
    executable = check('eng', PLACES)
    data, (columns, rows) = scanned(WORDS, 270, 3)  # stored so Orientation 8 shows it upright
    tags = {
        258: (3, (8, 8, 8)),
        259: (3, (7,)),
        262: (3, (6,)),
        274: (3, (ocr._COLOR_JPEG_SHOWN_WRONG,)),
        277: (3, (3,)),
        284: (3, (1,)),
        530: (3, (1, 1)),
    }
    tiff = tmp_path / 'scan.tif'
    tiff.write_bytes(ocr._tiff(columns, rows, data, tags))
    done = subprocess.run([executable, str(tiff), 'stdout', '--dpi', '96'], capture_output=True, check=True)
    read = done.stdout.decode('utf-8').casefold()
    assert not any(word in read for word in WORDS), read  # shown half round: none reads


#: Chinese (simplified): the stator winding of a hydro generator heats and vibrates in service, so its insulation is
#: inspected regularly; the cooling system matters greatly. Four columns or lines.
ZH = (
    '\u6c34\u7535\u673a\u7ec4\u7684\u5b9a\u5b50\u7ed5\u7ec4\u5728\u8fd0\u884c\u65f6',
    '\u4ea7\u751f\u70ed\u91cf\u548c\u632f\u52a8\u56e0\u6b64\u9700\u8981\u5b9a',
    '\u671f\u68c0\u67e5\u7edd\u7f18\u72b6\u6001\u4ee5\u4fdd\u8bc1\u5b89\u5168',
    '\u53d1\u7535\u673a\u7684\u51b7\u5374\u7cfb\u7edf\u975e\u5e38\u91cd\u8981',
)
#: Japanese: the generator's stator winding heats and vibrates in service, so its insulation is inspected regularly.
JA = (
    '\u767a\u96fb\u6a5f\u306e\u56fa\u5b9a\u5b50\u5dfb\u7dda\u306f\u904b\u8ee2\u4e2d',
    '\u306b\u71b1\u3068\u632f\u52d5\u3092\u767a\u751f\u3059\u308b\u305f\u3081\u7d76',
    '\u7e01\u306e\u72b6\u614b\u3092\u5b9a\u671f\u7684\u306b\u70b9\u691c\u3059\u308b',
)
CJK_SIZE = 24  # points
CJK_DPI = 200
COLUMN_PITCH = 1.6  # sizes from one column's middle to the next


def cjk_page(lines: tuple[str, ...], font: str, *, vertical: bool) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """``lines`` in PyMuPDF's built-in CJK ``font`` -- one under another, or in columns from the right, each
    character standing upright an em below the last -- as RGB pixels at :data:`CJK_DPI`; and each line's extent
    across the page (x for a column, y for a line), in points."""
    longest = max(map(len, lines))
    if vertical:
        width, height = len(lines) * CJK_SIZE * COLUMN_PITCH + 80, longest * CJK_SIZE + 80
    else:
        width, height = longest * CJK_SIZE + 80, len(lines) * CJK_SIZE * COLUMN_PITCH + 80
    extents = []
    with pymupdf.open() as doc:
        page = doc.new_page(width=width, height=height)
        for i, line in enumerate(lines):
            if vertical:
                left = width - 40 - CJK_SIZE * COLUMN_PITCH * i - CJK_SIZE
                for j, char in enumerate(line):
                    page.insert_text((left, 40 + CJK_SIZE * (j + 0.88)), char, fontname=font, fontsize=CJK_SIZE)
                extents.append((left, left + CJK_SIZE))
            else:
                top = 40 + CJK_SIZE * COLUMN_PITCH * i
                page.insert_text((40, top + CJK_SIZE * 0.88), line, fontname=font, fontsize=CJK_SIZE)
                extents.append((top, top + CJK_SIZE))
        pix = page.get_pixmap(dpi=CJK_DPI)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    return pixels, extents


@pytest.mark.parametrize(
    ('lines', 'font', 'languages', 'vertical'),
    [
        (ZH, 'china-s', 'chi_sim', False),
        (ZH, 'china-s', 'chi_sim_vert', True),
        (JA, 'japan', 'jpn', False),
        (JA, 'japan', 'jpn_vert', True),
    ],
    ids=['chinese-lines', 'chinese-columns', 'japanese-lines', 'japanese-columns'],
)
def test_cjk_reads_back_in_its_lines_or_columns_columns_as_vertical_writing(tmp_path, lines, font, languages, vertical):
    check(languages, PLACES)
    pixels, extents = cjk_page(lines, font, vertical=vertical)
    capture = to_pdf(pixels, tmp_path / 'capture.pdf', page_height=0, dpi=CJK_DPI)
    assert text_layer(capture, tmp_path / 'ocr.pdf', languages, places=PLACES) == 1
    page = read_pdf(tmp_path / 'ocr.pdf').pages[0]
    assert page.spans
    assert all(span.vertical == vertical for span in page.spans), [(s.text, s.vertical) for s in page.spans]
    across, along = (0, 1) if vertical else (1, 0)
    on = [[] for _ in lines]
    for span in page.spans:  # each on the column or line it was drawn in (Tesseract's CJK boxes run wide)
        drawn = [i for i, (start, end) in enumerate(extents) if start < span.center[across] < end]
        assert drawn, (span.text, span.provenance.bbox)
        on[drawn[0]].append(span)
    read = [''.join(s.text for s in sorted(spans, key=lambda s: s.center[along])) for spans in on]
    matched = sum(
        block.size
        for got, drawn in zip(read, lines, strict=True)
        for block in difflib.SequenceMatcher(None, got, drawn, autojunk=False).get_matching_blocks()
    )
    assert matched > 0.85 * sum(map(len, lines)), read  # each column read top to bottom, each line left to right
