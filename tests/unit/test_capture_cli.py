"""The capture command's write step: a PDF that cannot be written leaves the capture as a PNG, never lost."""

import argparse

import numpy as np
import pymupdf
import pytest

from document_lab.capture import cli
from document_lab.capture import export as export_module


def args(tmp_path, **over):
    base = {'out': tmp_path / 'c.pdf', 'png': None, 'page_height': None, 'dpi': 96, 'ocr': None}
    return argparse.Namespace(**{**base, **over})


def test_it_writes_the_pdf(tmp_path):
    image = np.zeros((300, 40, 3), dtype=np.uint8)
    cli._write(image, args(tmp_path))
    with pymupdf.open(tmp_path / 'c.pdf') as doc:
        assert doc.page_count == 6


def test_a_failed_pdf_keeps_the_capture_as_png(tmp_path, monkeypatch):
    def refuse(*_, **__):
        msg = 'no Tesseract'
        raise OSError(msg)

    monkeypatch.setattr(export_module, 'to_pdf', refuse)
    image = np.random.default_rng(0).integers(0, 256, (50, 30, 3), dtype=np.uint8)
    with pytest.raises(OSError, match='Tesseract'):
        cli._write(image, args(tmp_path))
    pix = pymupdf.Pixmap(str(tmp_path / 'c.png'))
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(50, 30, 3), image)
