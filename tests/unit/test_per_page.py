"""A PDF written page by page: a markdown file and an image per page, an index, and the pages without text named."""

import zipfile

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
    assert names == ['INDEX.md', *(f'page-00{n}.{ext}' for n in (1, 2, 3) for ext in ('md', 'png')), 'std.pdf']
    page1 = (tmp_path / 'out' / 'page-001.md').read_text(encoding='utf-8')
    assert page1.startswith('[std.pdf, page 1](page-001.png)')
    assert 'declared by the purchaser' in page1
    assert 'Annex A' not in page1  # a page stands alone
    index = (tmp_path / 'out' / 'INDEX.md').read_text(encoding='utf-8')
    assert '- [page 3](page-003.md)\n' in index
    assert '- [page 1](page-001.md) -- 4 Duty' in index


def test_a_page_that_is_one_image_keeps_its_own_pixels(tmp_path):
    image = np.random.default_rng(1).integers(0, 256, (90, 70, 3), dtype=np.uint8)
    done = write_pages(to_pdf(image, tmp_path / 'capture.pdf'), tmp_path / 'out')
    assert done.textless == (1,)
    pix = pymupdf.Pixmap(str(tmp_path / 'out' / 'page-001.png'))
    assert np.array_equal(np.frombuffer(pix.samples, dtype=np.uint8).reshape(90, 70, 3), image)


def test_figures_tables_and_equations_are_cut_out_and_linked_where_their_text_is(tmp_path):
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=840)
        page.insert_text((50, 100), prose, fontsize=10)
        page.draw_rect(pymupdf.Rect(150, 140, 450, 320), color=(0, 0, 0))
        page.insert_text((200, 360), 'Figure 2 - Short-time duty', fontsize=10)
        page.insert_text((50, 420), prose, fontsize=10)
        page.insert_text((230, 470), 'U = R I', fontsize=20)
        page.insert_text((520, 470), '(3)', fontsize=10)
        page.insert_text((50, 520), prose, fontsize=10)
        page.insert_text((200, 600), 'Table 1 - Preferred voltage ratings', fontsize=10)
        page.insert_text((120, 630), 'Rated voltage', fontsize=9)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'out')
    out = tmp_path / 'out'
    for name in ('page-001-figure-2.png', 'page-001-equation-3.png', 'page-001-table-1.png'):
        assert (out / name).is_file(), name
    md = (out / 'page-001.md').read_text(encoding='utf-8')
    figure = md.index('![Figure 2](page-001-figure-2.png)')
    assert md.index('Figure 2 - Short-time duty') < figure < md.index(prose, md.index('Figure 2'))
    assert '![(3)](page-001-equation-3.png)' in md
    assert '![Table 1](page-001-table-1.png) (continues on the next page)' in md
    assert 'Figure 2, (3), Table 1' in (out / 'INDEX.md').read_text(encoding='utf-8')


def test_a_zip_out_holds_the_pdf_and_everything_a_directory_would(tmp_path):
    with pymupdf.open() as doc:
        doc.new_page(width=400, height=300).insert_text((40, 60), '4 Duty', fontsize=12)
        doc.save(tmp_path / 'std.pdf')
    write_pages(tmp_path / 'std.pdf', tmp_path / 'dir')
    done = write_pages(tmp_path / 'std.pdf', tmp_path / 'std.zip')
    assert done.out == tmp_path / 'std.zip'
    with zipfile.ZipFile(tmp_path / 'std.zip') as archive:
        names = sorted(archive.namelist())
        assert names == sorted(p.name for p in (tmp_path / 'dir').iterdir())
        assert 'std.pdf' in names
        assert archive.read('page-001.md') == (tmp_path / 'dir' / 'page-001.md').read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['dir', 'std.pdf', 'std.zip']  # nothing left beside it
