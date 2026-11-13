"""A PDF written page by page: a markdown file and an image per page, an index, and the pages without text named."""

import numpy as np
import pymupdf

from document_lab.capture.export import to_pdf
from document_lab.per_page import write_pages


def test_each_page_gets_its_markdown_and_image_and_the_index_names_them(tmp_path):
    with pymupdf.open() as doc:
        first = doc.new_page(width=400, height=300)
        first.insert_text((40, 60), '4 Duty', fontsize=14, fontname='hebo')
        first.insert_text((40, 100), 'The duty shall be declared by the purchaser.', fontsize=10)
        doc.new_page(width=400, height=300)  # blank
        third = doc.new_page(width=400, height=300)
        third.insert_text((40, 60), 'Annex A continues here.', fontsize=10)
        doc.save(tmp_path / 'std.pdf')
    done = write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    assert done.pages == 3
    assert done.textless == (2,)
    names = sorted(p.name for p in (tmp_path / 'out').iterdir())
    assert names == ['INDEX.md', *(f'page-00{n}.{ext}' for n in (1, 2, 3) for ext in ('md', 'png'))]
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert page1.startswith('[std.pdf, page 1](page-001.png)')
    assert 'declared by the purchaser' in page1
    assert 'Annex A' not in page1  # a page stands alone
    assert '- [page 3](page-003.md)' in (tmp_path / 'out' / 'INDEX.md').read_text(encoding='utf-8')


def test_a_page_that_is_one_image_keeps_its_own_pixels(tmp_path):
    image = np.random.default_rng(1).integers(0, 256, (90, 70, 3), dtype=np.uint8)
    done = write_pages(to_pdf(image, tmp_path / 'capture.pdf'), tmp_path / 'out')
    assert done.textless == (1,)
    pix = pymupdf.Pixmap(str(tmp_path / 'out' / 'page-001.png'))
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(90, 70, 3), image)
