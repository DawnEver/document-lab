"""``--formulas latex``: each equation crop's LaTeX written beside it, the crop kept; the model is the ``formula``
extra's, loaded only when asked, and refused by name without it -- never a silent fallback."""

import subprocess
import sys

import numpy as np
import pymupdf
import pytest

from document_lab import formula
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
    assert stand_in.asked == [('latex', {})]  # loaded once for the document
    (shown,) = stand_in.shown
    pix = pymupdf.Pixmap(str(crop))
    assert shown.shape == (pix.height, pix.width, 3)  # the model reads the very pixels the crop holds


def test_without_the_option_no_model_is_asked(tmp_path, stand_in):
    write_pages(_standard(tmp_path / 'std.pdf'), tmp_path / 'out')
    assert '$$' not in (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert stand_in.asked == []


def test_a_paper_s_equation_carries_its_latex_under_its_image(tmp_path, stand_in):
    stand_in.latex = r'\[ U = R I \qquad (1) \]'
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
        doc.save(tmp_path / 'paper.pdf')
    ingest(tmp_path / 'paper.pdf', tmp_path / 'out', tool='ir', formulas='latex')
    body = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'out' / 'md').glob('*.md'))
    assert '-equation-1.png)\n\n$$\nU = R I\n$$' in body
    assert stand_in.shown


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
