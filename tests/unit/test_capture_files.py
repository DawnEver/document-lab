"""capture-files without a desktop: which PDFs a run names, a page told from its viewer's backdrop, the end of paging,
and the pages written one image each."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import PageWriter
from document_lab.capture.files import listed, page_box, page_count, same_page


def _pdf(path, pages=1, password=None):
    with pymupdf.open() as doc:
        for _ in range(pages):
            doc.new_page().insert_text((72, 72), 'Rated output', fontsize=11)
        if password:
            doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
        else:
            doc.save(path)
    return path


def test_a_run_names_pdfs_directories_and_lists_each_pdf_once_in_order(tmp_path):
    folder = tmp_path / 'raw'
    folder.mkdir()
    b, a = _pdf(folder / 'b.pdf'), _pdf(folder / 'A.pdf')
    alone = _pdf(tmp_path / 'alone.pdf')
    listing = tmp_path / 'list.txt'
    listing.write_text('# the standards\nalone.pdf\n\nraw/b.pdf\n', encoding='utf-8')
    assert listed([folder, listing, alone]) == [a.resolve(), b.resolve(), alone.resolve()]


def test_a_source_naming_nothing_is_refused_not_skipped(tmp_path):
    listing = tmp_path / 'list.txt'
    listing.write_text('gone.pdf\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match=r'gone\.pdf'):
        listed([listing])
    with pytest.raises(FileNotFoundError, match='no PDF'):
        listed([tmp_path / 'missing'])


def test_the_page_count_is_known_when_the_pdf_opens_here_and_not_when_it_is_secured(tmp_path):
    assert page_count(_pdf(tmp_path / 'open.pdf', pages=3)) == 3
    secret = 'only the viewer'  # noqa: S105 -- a planted secured PDF's, not a credential
    assert page_count(_pdf(tmp_path / 'secured.pdf', pages=3, password=secret)) is None


def _view(height=300, width=400, page=(40, 260, 120, 280), backdrop=110):
    """A viewer's view: a grey backdrop round a white page holding a line of black text."""
    view = np.full((height, width, 3), backdrop, dtype=np.uint8)
    top, bottom, left, right = page
    view[top:bottom, left:right] = 255
    view[top + 30 : top + 34, left + 10 : right - 10] = 0
    return view


def test_the_page_is_the_part_of_the_view_that_differs_from_its_backdrop():
    assert page_box(_view()) == (40, 260, 120, 280)


def test_a_view_the_page_fills_wholly_is_refused_as_no_whole_page_in_view():
    with pytest.raises(LookupError, match='whole page in view'):
        page_box(np.full((300, 400, 3), 255, dtype=np.uint8))  # zoomed past the page: no backdrop round it
    with pytest.raises(LookupError, match='whole page in view'):
        page_box(_view(page=(40, 260, 190, 210)))  # a sliver, not a page


def test_a_page_running_off_the_top_or_foot_of_the_view_is_refused_as_no_whole_page():
    """A pagedown that scrolled a screenful: the page break mid-view, each page cut by the view's edge."""
    with pytest.raises(LookupError, match='runs off the view'):
        page_box(_view(page=(0, 200, 120, 280)))
    with pytest.raises(LookupError, match='runs off the view'):
        page_box(_view(page=(100, 300, 120, 280)))


def test_a_page_fitted_to_the_view_height_fills_it_top_to_foot_and_is_taken():
    """Acrobat's whole-page zoom of an A4 page: as high as the view, no backdrop above or below; a backdrop 248."""
    assert page_box(_view(page=(0, 300, 120, 280), backdrop=248)) == (0, 300, 120, 280)


def test_a_page_fitted_across_a_portrait_view_fills_its_width_with_the_backdrop_above_and_below():
    assert page_box(_view(height=500, width=300, page=(60, 440, 0, 300), backdrop=248)) == (60, 440, 0, 300)


def test_paging_past_the_last_page_changes_nothing_and_a_new_page_does():
    first = _view()
    assert same_page(first, first.copy())
    turned = first.copy()
    turned[150:200, 130:270] = 0
    assert not same_page(first, turned)
    assert not same_page(first, first[:, :-1])  # another size is another page


def test_pages_are_written_one_image_each_at_their_own_widths(tmp_path):
    out = tmp_path / 'taken.pdf'
    with PageWriter(out, dpi=72) as pdf:
        pdf.add(_view())
        pdf.add(_view(height=200, width=300, page=(10, 190, 50, 250)))
    with pymupdf.open(out) as doc:
        assert [(p.rect.width, p.rect.height) for p in doc] == [(400, 300), (300, 200)]
    assert not out.with_name(out.name + '.part').exists()


def test_a_writer_given_no_page_writes_nothing(tmp_path):
    with PageWriter(tmp_path / 'none.pdf') as pdf:
        assert pdf.pages == 0
    assert not (tmp_path / 'none.pdf').exists()
