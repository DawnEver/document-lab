import re

from document_lab.markdown import to_markdown
from document_lab.paper.convert import choose_tool, convert, detect_gpu_vram_gb
from document_lab.pdf import read_pdf
from tests.synthetic import text_pdf

_TEXT = (
    'the rated output is the mechanical power available at the shaft and shall be expressed in watts while the '
    'machine runs at its rated speed under the stated conditions of supply cooling and duty for the whole test'
)


class TestDetectGpuVram:
    def test_returns_float(self):
        result = detect_gpu_vram_gb()
        assert isinstance(result, float)

    def test_returns_non_negative(self):
        result = detect_gpu_vram_gb()
        assert result >= 0.0


class TestChooseTool:
    def test_returns_valid_tool(self):
        tool = choose_tool()
        assert tool in ('marker', 'ir')


def _journal(tmp_path):
    """Three pages under a journal's running head, each numbered at its foot; page 1 alone has a notice at its top."""
    pages = []
    for n in range(3):
        body = [(72, 120 + 14 * i, ' '.join(_TEXT.split()[(i + 7 * n) % 20 :][:12]), 'helv', 10) for i in range(30)]
        head = [(72, 40, 'Journal of Synthetic Results 12 (2026)', 'helv', 8)]
        notice = [(72, 60, 'Preprint notice for the first page', 'helv', 8)] if n == 0 else []
        pages.append([*head, *notice, *body, (300, 760, str(n + 1), 'helv', 8)])
    return text_pdf(tmp_path / 'journal.pdf', pages)


def test_the_ir_markdown_leaves_out_running_heads_and_feet(tmp_path):
    """The running head and the page numbers repeat at every page's edge: the document's furniture, not its text."""
    pdf = _journal(tmp_path)
    md, used = convert(pdf, tmp_path / 'out', tool='ir')
    assert used == 'ir'
    assert 'Journal of Synthetic Results' not in md
    assert not re.search(r'^\d+$', md, re.MULTILINE)  # no page number stands as a paragraph
    assert 'Preprint notice for the first page' in md  # one page's top line repeats nowhere: it stays
    assert 'rated output' in md
    raw = to_markdown(read_pdf(pdf))  # planted: the same check on the document with its running lines
    assert 'Journal of Synthetic Results' in raw
    assert re.search(r'^\d+$', raw, re.MULTILINE)
