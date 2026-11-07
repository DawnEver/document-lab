"""The capture command end to end over a fake source: rows stream into the PDF and the PNG as they are stitched, a
capture that stops short keeps what it has and says so, and nothing is half-written under the real names."""

import numpy as np
import pymupdf
import pytest

from document_lab.__main__ import main
from document_lab.capture import desktop
from tests.unit.test_capture_session import Viewport
from tests.unit.test_capture_stitch import tall

WIDTH = 64  # the fake view's width: tests.unit.test_capture_stitch.tall's


class FakeSource:
    """``desktop.Source`` over a :class:`Viewport` of ``IMAGE``, ``STEP`` rows a scroll."""

    IMAGE = tall(2000)
    STEP = 170

    def __init__(self, target, notches, method):
        self.view = Viewport(self.IMAGE, target.height, self.STEP)

    def grab(self):
        return self.view.grab()

    def scroll(self):
        self.view.scroll()

    def stopped(self):
        return False


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(desktop, 'Source', FakeSource)
    return FakeSource


def run(tmp_path, *more):
    argv = ['capture', str(tmp_path / 'c.pdf'), '--rect', f'0,0,{WIDTH},300', '--delay', '0', '--settle', '0']
    return main([*argv, '--end-wait', '0', *more])


def pdf_pixels(path):
    with pymupdf.open(path) as doc:
        pages = []
        for page in doc:
            ((xref, *_),) = page.get_images()
            pix = pymupdf.Pixmap(doc, xref)
            pages.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
        return np.concatenate(pages), doc.page_count


def png_pixels(path):
    pix = pymupdf.Pixmap(str(path))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def test_the_whole_view_streams_into_the_pdf_and_the_png(tmp_path, fake):
    assert run(tmp_path, '--page-height', '300', '--png', str(tmp_path / 'c.png')) == 0
    image, pages = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE)
    assert pages == 7
    assert np.array_equal(png_pixels(tmp_path / 'c.png'), fake.IMAGE)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['c.pdf', 'c.png']


def test_a_capture_that_cannot_be_stitched_keeps_what_it_has_and_exits_1(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(fake, 'STEP', 290)  # past what a 300-row view can overlap
    assert run(tmp_path, '--page-height', '100') == 1
    image, _ = pdf_pixels(tmp_path / 'c.pdf')
    assert np.array_equal(image, fake.IMAGE[:300])


def test_a_page_no_reader_accepts_refuses_before_capturing(tmp_path, fake):
    with pytest.raises(ValueError, match='14400'):
        run(tmp_path, '--page-height', '100000')
    assert not list(tmp_path.iterdir())
