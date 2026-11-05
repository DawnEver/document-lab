from pathlib import Path

import pymupdf
import pytest

from document_lab.paper.pagemap import build_figure_page_map


def test_a_missing_pdf_raises():
    with pytest.raises(pymupdf.FileNotFoundError):
        build_figure_page_map(Path('/nonexistent/path.pdf'))
