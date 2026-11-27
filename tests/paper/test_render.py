"""Figure images: a page that is one picture (a screen-captured PDF) is linked whole, once, never cropped into copies;
a drawn page is still cropped per label."""

import numpy as np
import pymupdf
import pytest

from document_lab.paper.render import is_full_page_image, render_equation_images, render_figure_pages
from document_lab.pdf import read_pdf


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
    doc.save(tmp_path / 'pages.pdf')
    assert [is_full_page_image(p) for p in read_pdf(tmp_path / 'pages.pdf').pages] == [True, False, False]


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


def test_a_figure_on_a_picture_page_its_text_locates_is_cut_out_not_linked_whole(tmp_path):
    pdf = tmp_path / 'captured.pdf'
    prose = 'The rated output is the mechanical power available at the shaft and shall be expressed in watts.'
    doc = pymupdf.open()
    page = doc.new_page()
    pixels = np.random.default_rng(7).integers(0, 256, (842, 595, 3), dtype=np.uint8)
    page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, 595, 842, pixels.tobytes(), 0))
    for y, text in ((100, prose), (115, prose), (360, 'Figure 4 - Short-time duty'), (420, prose)):
        page.insert_text((60, y), text, fontsize=10, render_mode=3)  # an OCR layer over the picture
    doc.save(pdf)
    out = tmp_path / 'out'
    (out / 'md').mkdir(parents=True)
    body = 'Figure 4 - Short-time duty\n'
    (out / 'md' / '01-duty.md').write_text(body, encoding='utf-8')
    images, _ = render_figure_pages(pdf, out, [('## Duty', body)])
    assert images['Figure 4'] == 'img/sec01/page-01-figure-4.png'
    pix = pymupdf.Pixmap(str(out / images['Figure 4']))
    assert pix.height < pix.width  # the band between the prose and the caption, not the whole page


#: The text a page draws but its default view does not show, planted two ways, and the option that reads it.
UNSHOWN = {'hidden': 'include_hidden', 'in_background_color': 'include_in_background_color'}


def _unshown(doc: pymupdf.Document, page: pymupdf.Page, point: tuple[float, float], text: str, kind: str) -> None:
    """``text`` set at ``point`` as the default view does not show it: on a layer turned off, or white on white."""
    if kind == 'hidden':
        page.insert_text(point, text, fontsize=10, oc=doc.add_ocg('Off', on=False))
    else:
        page.insert_text(point, text, fontsize=10, color=(1, 1, 1))


@pytest.mark.parametrize('kind', UNSHOWN)
def test_a_caption_is_located_where_the_page_shows_it(tmp_path, kind):
    pdf = tmp_path / 'paper.pdf'
    doc = pymupdf.open()
    _drawn_page(doc, 'Figure 2. Two')  # the control: a caption the page shows
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 100, 300, 250), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    _unshown(doc, page, (72, 270), 'Figure 3. Three', kind)
    doc.save(pdf)
    body = 'Figure 2. Two\n\nFigure 3. Three\n'
    for asked in (False, True):
        out = tmp_path / f'out-{asked}'
        (out / 'md').mkdir(parents=True)
        (out / 'md' / '01-shapes.md').write_text(body, encoding='utf-8')
        images, _ = render_figure_pages(pdf, out, [('## Shapes', body)], **{UNSHOWN[kind]: asked})
        assert images['Figure 2'] == 'img/sec01/page-01-figure-2.png'
        assert ('Figure 3' in images) is asked


@pytest.mark.parametrize('kind', UNSHOWN)
def test_an_equation_is_located_where_the_page_shows_its_number(tmp_path, kind):
    pdf = tmp_path / 'paper.pdf'
    prose = 'The torque follows from the flux linkage and the current of each phase as stated below.'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), prose, fontsize=10)
    page.insert_text((200, 140), 'T = p (psi i)', fontsize=10)
    page.insert_text((500, 140), '(1)', fontsize=10)  # the control: a number the page shows
    page.insert_text((72, 180), prose, fontsize=10)
    page.insert_text((200, 220), 'P = T w', fontsize=10)
    _unshown(doc, page, (500, 220), '(2)', kind)
    page.insert_text((72, 260), prose, fontsize=10)
    doc.save(pdf)
    body = 'Equation (1) and Equation (2).\n'
    for asked in (False, True):
        out = tmp_path / f'out-{asked}'
        (out / 'md').mkdir(parents=True)
        (out / 'md' / '01-torque.md').write_text(body, encoding='utf-8')
        images = render_equation_images(pdf, out, [('## Torque', body)], {}, {0: 1}, **{UNSHOWN[kind]: asked})
        assert images['Equation 1'] == 'img/sec01/page-01-equation-1.png'
        assert ('Equation 2' in images) is asked
