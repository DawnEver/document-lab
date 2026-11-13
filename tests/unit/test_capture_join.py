"""Two captures of one document joined where they overlap: the rows they share kept once, every other page copied
as it is, and two that share nothing -- or differ in width -- refused, never butted together."""

import numpy as np
import pymupdf
import pytest

from document_lab.capture.export import to_pdf
from document_lab.capture.join import join, tail_rows
from document_lab.capture.stitch import Match, NoOverlap
from tests.unit.test_capture_cli import pdf_pixels
from tests.unit.test_capture_stitch import tall


def two(tmp_path, image, first_end, second_start, page_height=150):
    a = to_pdf(image[:first_end], tmp_path / 'a.pdf', page_height=page_height)
    b = to_pdf(image[second_start:], tmp_path / 'b.pdf', page_height=page_height)
    return a, b


@pytest.mark.parametrize(('first_end', 'second_start'), [(1000, 700), (1000, 980), (1000, 300), (1000, 1000 - 64)])
def test_the_rows_both_show_are_kept_once(tmp_path, first_end, second_start):
    image = tall(2000)
    a, b = two(tmp_path, image, first_end, second_start)
    joined = join(a, b, tmp_path / 'c.pdf', Match(min_overlap=16))
    rows, _ = pdf_pixels(joined.path)
    assert np.array_equal(rows, image)
    assert joined.shared == first_end - second_start


def test_pages_away_from_the_seam_are_copied_not_redrawn(tmp_path):
    image = tall(2000)
    a, b = two(tmp_path, image, 1200, 1000)
    joined = join(a, b, tmp_path / 'c.pdf', head=200)
    with pymupdf.open(a) as first, pymupdf.open(joined.path) as out:
        kept = first.page_count - joined.redrawn
        assert kept > 0
        assert all(out[i].rect == first[i].rect for i in range(kept))


def test_a_second_capture_inside_the_first_leaves_the_first(tmp_path):
    image = tall(1500)
    a = to_pdf(image, tmp_path / 'a.pdf', page_height=150)
    b = to_pdf(image[600:900], tmp_path / 'b.pdf', page_height=150)
    rows, _ = pdf_pixels(join(a, b, tmp_path / 'c.pdf').path)
    assert np.array_equal(rows, image)


def test_captures_that_share_nothing_are_refused_and_nothing_is_written(tmp_path):
    image = tall(2000)
    a, b = two(tmp_path, image, 800, 1100)
    with pytest.raises(NoOverlap):
        join(a, b, tmp_path / 'c.pdf')
    assert not (tmp_path / 'c.pdf').exists()
    assert not (tmp_path / 'c.pdf.part').exists()


def test_captures_of_different_widths_are_refused(tmp_path):
    a = to_pdf(tall(400), tmp_path / 'a.pdf')
    b = to_pdf(tall(400)[:, :32], tmp_path / 'b.pdf')
    with pytest.raises(ValueError, match='64 and 32 px wide'):
        join(a, b, tmp_path / 'c.pdf')


def test_the_out_may_be_the_first_capture_itself(tmp_path):
    image = tall(1600)
    a, b = two(tmp_path, image, 900, 700)
    join(a, b, a)
    rows, _ = pdf_pixels(a)
    assert np.array_equal(rows, image)


def test_the_tail_of_a_capture_is_its_last_pages_rows(tmp_path):
    image = tall(1000)
    a = to_pdf(image, tmp_path / 'a.pdf', page_height=150)
    assert np.array_equal(tail_rows(a, 200), image[-250:])  # whole pages: the last 100 and one of 150
