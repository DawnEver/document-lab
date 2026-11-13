"""Figure images: a page that is one picture (a screen-captured PDF) is linked whole, once, never cropped into copies;
a drawn page is still cropped per label."""

import numpy as np
import pymupdf

from document_lab.paper.render import is_full_page_image, render_figure_pages


def _picture_page(doc: pymupdf.Document, captions: list[str]) -> None:
    page = doc.new_page()
    pixels = np.random.default_rng(len(doc)).integers(0, 256, (200, 150, 3), dtype=np.uint8)
    page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, 150, 200, pixels.tobytes(), 0))
    for i, caption in enumerate(captions):  # an OCR layer: invisible text over the picture
        page.insert_text((72, 300 + 200 * i), caption, fontsize=10, render_mode=3)


def _drawn_page(doc: pymupdf.Document, caption: str) -> None:
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 100, 300, 250), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    page.insert_text((72, 270), caption, fontsize=10)


def test_a_picture_of_a_page_is_told_from_a_drawn_one(tmp_path):
    doc = pymupdf.open()
    _picture_page(doc, ['Figure 1. One'])
    _drawn_page(doc, 'Figure 2. Two')
    small = doc.new_page()  # the control: a small picture on a page is a figure, not the page
    small.insert_image(pymupdf.Rect(72, 72, 200, 200), pixmap=pymupdf.Pixmap(pymupdf.csRGB, 4, 4, bytes(48), 0))
    assert [is_full_page_image(p) for p in doc] == [True, False, False]


def test_two_figures_on_a_picture_page_share_one_whole_page_image(tmp_path):
    pdf = tmp_path / 'captured.pdf'
    doc = pymupdf.open()
    _picture_page(doc, ['Figure 1. One', 'Figure 2. Two'])
    _drawn_page(doc, 'Figure 3. Three')
    doc.save(pdf)
    out = tmp_path / 'out'
    (out / 'md').mkdir(parents=True)
    body = 'Figure 1. One\n\nFigure 2. Two\n\nFigure 3. Three\n'
    (out / 'md' / '01-shapes.md').write_text(body, encoding='utf-8')
    images, _ = render_figure_pages(pdf, out, [('## Shapes', body)])
    assert images['Figure 1'] == images['Figure 2'] == 'img/sec01/page-01.png'
    assert images['Figure 3'] == 'img/sec01/page-02-figure-3.png'  # a drawn page is still cropped per label
    assert sorted(p.name for p in (out / 'img' / 'sec01').glob('*.png')) == ['page-01.png', 'page-02-figure-3.png']
