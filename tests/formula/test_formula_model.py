"""The ``formula`` part: each real model (`document_lab.formula`: TexTeller, CodeFormulaV2) reads a drawn equation as
its LaTeX, on the CPU and on whatever device it picks. Its own part, never tests/base: it needs the ``formula`` extra
and the models' weights (the Hugging Face cache, or the network to fill it), and FAILS naming what is missing where
they are not -- it is never skipped."""

import functools
import re

import numpy as np
import pymupdf
import pytest

from document_lab import formula

pytestmark = pytest.mark.formula

MODELS = sorted(formula.MODELS)


def drawn() -> np.ndarray:
    """``E = mc`` with a raised ``2``, as the RGB pixels of a 2x crop (how `paper` renders one)."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=60)
        page.insert_text((40, 40), 'E = mc', fontsize=24, fontname='tiro')
        page.insert_text((118, 28), '2', fontsize=14, fontname='tiro')
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


@functools.cache  # one model a device: each holds over half a GB
def transcriber(model: str, device: str | None) -> formula.Transcribe:
    try:
        return formula.transcriber('latex', model=model, device=device)
    except formula.FormulaUnavailable as missing:
        pytest.fail(str(missing))


@pytest.mark.parametrize('device', [None, 'cpu'])
@pytest.mark.parametrize('model', MODELS)
def test_a_drawn_equation_reads_as_its_latex(model, device):
    latex = transcriber(model, device)(drawn()) or ''
    # Upright type may read as \rm, and spacing commands are the model's to set: neither is the formula.
    assert re.sub(r'\\(?:rm|mathrm|[,;:!\[\]])|[\s{}]', '', latex) == 'E=mc^2', latex


@pytest.mark.parametrize('model', MODELS)
def test_a_blank_crop_states_nothing(model):
    assert transcriber(model, 'cpu')(np.full((40, 120, 3), 255, dtype=np.uint8)) is None
