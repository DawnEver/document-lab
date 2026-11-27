"""``--formulas latex``: each equation crop's LaTeX written beside it, the crop kept; the model is the ``formula``
extra's, loaded only when asked, and refused by name without it -- never a silent fallback."""

import struct
import subprocess
import sys
import zlib

import numpy as np
import pymupdf
import pytest

from document_lab import formula, png
from document_lab.__main__ import main
from document_lab.paper.ingest import ingest
from document_lab.per_page import write_pages
from tests.paper.support import LONG_BODY


class StandIn:
    """A transcriber that answers what the test planted and keeps the pixels it was shown."""

    def __init__(self, latex: str) -> None:
        self.latex = latex
        self.shown: list[np.ndarray] = []

    def __call__(self, pixels: np.ndarray) -> str:
        self.shown.append(pixels)
        return self.latex


@pytest.fixture
def stand_in(monkeypatch):
    model = StandIn(r'\[ U = R I \qquad (3) \]')
    asked = []

    def transcriber(formulas, **options):
        asked.append((formulas, options))
        return model

    monkeypatch.setattr(formula, 'transcriber', transcriber)
    model.asked = asked
    return model


def _decoded(path):
    """The pixels of a PNG `document_lab.png` wrote (8-bit gray or RGB, every row filter 0), read with zlib alone."""
    data = path.read_bytes()
    assert data.startswith(png.SIGNATURE)
    at, header, idat = len(png.SIGNATURE), b'', b''
    while at < len(data):
        (length,), kind = struct.unpack('>I', data[at : at + 4]), data[at + 4 : at + 8]
        body = data[at + 8 : at + 8 + length]
        header, idat = (body, idat) if kind == b'IHDR' else (header, idat + body if kind == b'IDAT' else idat)
        at += 12 + length
    width, height, depth, color = struct.unpack('>IIBB', header[:10])
    assert depth == 8
    channels = {0: 1, 2: 3}[color]
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + width * channels)
    assert not rows[:, 0].any()  # filter 0: each row as it is
    pixels = rows[:, 1:].reshape(height, width, channels)
    return pixels[..., 0] if channels == 1 else pixels


def _standard(path):
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=840)
        page.insert_text((50, 100), prose, fontsize=10)
        page.insert_text((230, 170), 'U = R I', fontsize=20)
        page.insert_text((520, 170), '(3)', fontsize=10)
        page.insert_text((50, 220), prose, fontsize=10)
        doc.save(path)
    return path


def test_the_latex_is_written_beside_the_equation_crop_which_is_kept(tmp_path, stand_in):
    write_pages(_standard(tmp_path / 'std.pdf'), tmp_path / 'out', formulas='latex')
    crop = tmp_path / 'out' / 'page-001-equation-3.png'
    md = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert '![(3)](page-001-equation-3.png)\n\n$$\nU = R I\n$$' in md
    assert stand_in.asked == [('latex', {'model': 'texteller'})]  # the default model, loaded once for the document
    (shown,) = stand_in.shown
    assert np.array_equal(shown, _decoded(crop))  # the model reads the very pixels the crop holds


def test_without_the_option_no_model_is_asked(tmp_path, stand_in):
    write_pages(_standard(tmp_path / 'std.pdf'), tmp_path / 'out')
    assert '$$' not in (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert stand_in.asked == []


def _paper(path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 80), 'I. INTRODUCTION', fontsize=12)
        y = 110
        for line in (LONG_BODY[i : i + 90] for i in range(0, 450, 90)):
            page.insert_text((72, y), line, fontsize=10)
            y += 14
        page.insert_text((250, y + 30), 'U = R I', fontsize=14)
        page.insert_text((500, y + 30), '(1)', fontsize=10)
        y += 70
        for line in (LONG_BODY[i : i + 90] for i in range(0, 450, 90)):
            page.insert_text((72, y), line, fontsize=10)
            y += 14
        doc.save(path)
    return path


def test_a_paper_s_equation_carries_its_latex_under_its_image(tmp_path, stand_in):
    stand_in.latex = r'\[ U = R I \qquad (1) \]'
    ingest(_paper(tmp_path / 'paper.pdf'), tmp_path / 'out', tool='ir', formulas='latex')
    body = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'out' / 'md').glob('*.md'))
    assert '-equation-1.png)\n\n$$\nU = R I\n$$' in body
    assert stand_in.shown


def test_a_paper_s_equation_is_read_from_the_very_array_its_crop_was_encoded_from(tmp_path, stand_in, monkeypatch):
    encoded = []

    def encode(image):
        encoded.append(image)
        return png_encode(image)

    png_encode = png.encode
    monkeypatch.setattr(png, 'encode', encode)
    ingest(_paper(tmp_path / 'paper.pdf'), tmp_path / 'out', tool='ir', formulas='latex')
    (shown,) = stand_in.shown
    assert any(shown is image for image in encoded)  # the array handed to the PNG writer, not a reading of it
    (crop,) = (tmp_path / 'out' / 'img').rglob('*-equation-1.png')
    assert np.array_equal(shown, _decoded(crop))  # and the file holds exactly it


def test_a_rerun_into_the_same_directory_transcribes_every_equation_again(tmp_path, stand_in):
    stand_in.latex = r'\[ U = R I \qquad (1) \]'
    pdf, out = _paper(tmp_path / 'paper.pdf'), tmp_path / 'out'
    ingest(pdf, out, tool='ir', formulas='latex')
    first = list(stand_in.shown)
    stand_in.shown.clear()
    ingest(pdf, out, tool='ir', formulas='latex')  # its crop is on disk already: never silently left unread
    assert len(stand_in.shown) == len(first) == 1
    assert np.array_equal(stand_in.shown[0], first[0])
    body = '\n'.join(p.read_text(encoding='utf-8') for p in (out / 'md').glob('*.md'))
    assert '-equation-1.png)\n\n$$\nU = R I\n$$' in body


def test_the_block_drops_the_display_delimiters_and_the_equation_s_own_number_only():
    assert formula.beside(r'\[ U = R I \qquad (3) \]', '3') == '$$\nU = R I\n$$'
    assert formula.beside(r'$$x = y \quad (2)$$', '3') == '$$\nx = y \\quad (2)\n$$'  # another number is content
    assert formula.beside(r'x = y \tag{3}', '3') == '$$\nx = y\n$$'
    assert formula.beside(r'\[B = 1 \hskip 113.811024pt(17)\]', '17') == '$$\nB = 1\n$$'
    assert formula.beside(r'\[A = J \qquad \left(23\right)\]', '23') == '$$\nA = J\n$$'
    assert formula.beside('x = y', None) == '$$\nx = y\n$$'
    two = r'\[r = 2w \qquad (2)\] \[\theta = \pi \qquad (3)\]'  # a reading of two displays: a block each
    assert formula.beside(two, '3') == '$$\nr = 2w \\qquad (2)\n$$\n\n$$\n\\theta = \\pi\n$$'


def test_the_base_package_imports_without_torch_or_transformers_and_the_option_refuses_by_name():
    probe = (
        "import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None\n"
        'import document_lab, document_lab.per_page, document_lab.paper.ingest, document_lab.formula\n'
        'try:\n    document_lab.formula.transcriber("latex")\n'
        'except document_lab.formula.FormulaUnavailable as e:\n    print(e)\n'
    )
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert "the 'formula' extra" in out.stdout


def test_a_crop_the_model_gives_no_reading_of_says_so_beside_it(tmp_path, stand_in):
    stand_in.latex = None
    write_pages(_standard(tmp_path / 'std.pdf'), tmp_path / 'out', formulas='latex')
    md = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert f'![(3)](page-001-equation-3.png)\n\n{formula.UNREAD}' in md


def test_the_command_line_names_the_model_and_refuses_one_named_without_formulas(tmp_path, stand_in):
    pdf = _standard(tmp_path / 'std.pdf')
    assert (
        main(['pages', str(pdf), str(tmp_path / 'out'), '--formulas', 'latex', '--formula-model', 'codeformula-v2'])
        == 0
    )
    assert stand_in.asked == [('latex', {'model': 'codeformula-v2'})]
    with pytest.raises(SystemExit) as refused:
        main(['pages', str(pdf), str(tmp_path / 'again'), '--formula-model', 'codeformula-v2'])
    assert refused.value.code == 2
    assert not (tmp_path / 'again').exists()


def test_a_model_not_offered_is_refused_by_name():
    with pytest.raises(ValueError, match='texteller, codeformula-v2'):
        formula.transcriber('latex', model='pix2tex')


def test_a_number_whose_digits_the_model_spaced_is_still_the_equation_s_own():
    assert formula.beside(r'u = R i \quad ( 1 2 )', '12') == '$$\nu = R i\n$$'
    assert formula.beside(r'u = R i \quad ( 1 2 )', '2') == '$$\nu = R i \\quad ( 1 2 )\n$$'  # not its number
