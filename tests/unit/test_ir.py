import dataclasses
import os
from pathlib import Path

import pytest

from document_lab.ir import BBox, Document, Page, Provenance, Table, TextSpan, file_uri, union


def _prov(page=1, bbox=(0.0, 0.0, 10.0, 10.0)):
    return Provenance(sha256='ab' * 32, uri='file:///x.pdf', page=page, bbox=bbox)


def test_the_ir_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _prov().page = 2


def test_a_provenance_refuses_page_zero_because_pages_are_one_based():
    with pytest.raises(ValueError, match='1-based'):
        _prov(page=0)


def test_a_bbox_refuses_inverted_corners():
    with pytest.raises(ValueError, match='x0 <= x1'):
        _prov(bbox=(10.0, 0.0, 0.0, 10.0))


def test_a_table_cell_points_back_to_its_own_box():
    table = Table(
        rows=(('a', 'b'), ('1', '2')),
        cell_bboxes=((BBox(0, 0, 5, 5), BBox(5, 0, 10, 5)), (BBox(0, 5, 5, 10), None)),
        provenance=_prov(),
    )
    assert table.cell(1, 0).text == '1'
    assert table.cell(1, 0).provenance.bbox == (0, 5, 5, 10)
    assert table.cell(1, 0).provenance.page == 1
    # A cell with no box of its own cites the table's.
    assert table.cell(1, 1).provenance.bbox == table.provenance.bbox


def test_a_table_refuses_a_ragged_bbox_grid():
    with pytest.raises(ValueError, match='shape'):
        Table(rows=(('a', 'b'),), cell_bboxes=((None,),), provenance=_prov())


def test_an_html_cell_cites_its_locator_instead_of_a_box():
    prov = Provenance(sha256='cd' * 32, uri='page.html', locator='table[0]')
    table = Table(rows=(('k', 'v'),), cell_bboxes=None, provenance=prov)
    assert table.cell(0, 1).provenance.locator == 'table[0]/row[0]/col[1]'
    assert table.cell(0, 1).provenance.bbox is None


def test_a_document_walks_its_pages_in_order():
    span = TextSpan(text='hi', size=9.0, font='Helvetica', provenance=_prov(page=2))
    doc = Document(
        sha256='ab' * 32,
        uri='file:///x.pdf',
        pages=(Page(number=1, width=612, height=792), Page(number=2, width=612, height=792, spans=(span,))),
    )
    assert [s.text for s in doc.spans()] == ['hi']
    assert doc.page(2).spans == (span,)


def test_union_is_the_smallest_box_covering_all():
    assert union([BBox(0, 0, 1, 1), BBox(2, -1, 3, 0.5)]) == BBox(0, -1, 3, 1)


#: Where an absolute path starts past the host's own root (``tmp_path``'s), in every form the host's paths take: its
#: verbatim (``//?/``) form, a share and a verbatim share on Windows; a leading double slash (an authority's place)
#: elsewhere.
OTHER_ROOTS = ['own verbatim', '//server/share/', '//?/UNC/server/share/'] if os.name == 'nt' else ['//double/']


@pytest.mark.parametrize('root', ['own', *OTHER_ROOTS])
@pytest.mark.parametrize('name', ['plain.pdf', 'a space, a #hash, 100% and \u00fc.pdf', 'dir/\u4e2d\u6587 name.pdf'])
def test_a_file_uri_is_the_paths_uri(tmp_path, root, name):
    anchor = tmp_path.anchor.replace(os.sep, '/')
    start = {'own': anchor, 'own verbatim': '//?/' + anchor}.get(root, root)
    path = Path(start + name)
    assert file_uri(path) == path.as_uri()


def test_a_relative_path_has_no_file_uri():
    with pytest.raises(ValueError, match='relative'):
        file_uri(Path('x.pdf'))
