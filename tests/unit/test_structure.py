"""A tagged PDF's structure read into the IR: elements in the stated order, each holding the values it covers."""

from __future__ import annotations

import pytest

from document_lab.ir import Element, Mark, MarkedContent, ObjectReference
from document_lab.pdf import read_pdf
from tests.synthetic import marked, mixed_pdf, shown, tagged_pdf

ROOT = '<< /Type /StructTreeRoot /K {doc} /ParentTree {parents} /RoleMap << /Heading1 /H1 /Mine /Unknown >> >>'


def _paper(tmp_path):
    """A page: a running head (an artifact), a heading tagged ``Heading1`` (mapped to H1), a paragraph, a figure with
    an alternate description drawn as a square, a link on the paragraph's word; then a second page continuing the
    paragraph."""
    content = ' '.join(
        [
            '/Artifact BMC ' + shown(20, 20, 'Running head', size=8) + ' EMC',
            marked('H1', 0, shown(20, 50, 'Introduction', size=14, font='F2')),
            marked('P', 1, shown(20, 80, 'A tagged paragraph.')),
            marked('Figure', 2, '20 120 60 60 re f'),
        ]
    )
    return tagged_pdf(
        tmp_path / 'tagged.pdf',
        [content, marked('P', 0, shown(20, 50, 'It goes on.'))],
        {
            'root': ROOT,
            'parents': '<< /Nums [0 [{h1} {p} {fig}] 1 [{p}]] >>',
            'doc': '<< /Type /StructElem /S /Document /K [{h1} {p} {fig} {mine}] /Lang (en) >>',
            'h1': '<< /Type /StructElem /S /Heading1 /P {doc} /Pg {page1} /K 0 /T (first) >>',
            'p': '<< /Type /StructElem /S /P /P {doc} /Pg {page1} /K [1 << /Type /MCR /Pg {page2} /MCID 0 >>] >>',
            'fig': '<< /Type /StructElem /S /Figure /P {doc} /Pg {page1} /K 2 /Alt (a black square) '
            '/A << /O /Layout /BBox [20 120 80 180] /Placement /Block >> >>',
            'mine': '<< /Type /StructElem /S /Mine /P {doc} >>',
        },
    )


def test_an_untagged_file_has_no_structure(tmp_path):
    assert read_pdf(mixed_pdf(tmp_path / 'm.pdf')).structure is None


def test_elements_come_in_the_stated_order_with_both_roles_and_their_entries(tmp_path):
    structure = read_pdf(_paper(tmp_path)).structure
    assert structure is not None
    assert structure.warnings == ()
    (document,) = structure.roots
    assert [(e.role, e.standard) for e in structure.elements()] == [
        ('Document', 'Document'),
        ('Heading1', 'H1'),
        ('P', 'P'),
        ('Figure', 'Figure'),
        ('Mine', None),
    ]
    heading, _, figure, _ = (k for k in document.kids if isinstance(k, Element))
    assert (document.lang, heading.title, heading.page) == ('en', 'first', 1)
    assert figure.alt == 'a black square'
    assert figure.attribute('Placement') == 'Block'
    assert figure.attribute('BBox', owner='Layout') == (20, 120, 80, 180)
    assert figure.attribute('BBox', owner='Table') is None


def test_an_element_holds_the_very_values_drawn_under_its_marked_content(tmp_path):
    doc = read_pdf(_paper(tmp_path))
    heading, paragraph, figure, _ = (k for k in doc.structure.roots[0].kids if isinstance(k, Element))
    assert [s.text for s in heading.spans()] == ['Introduction']
    assert next(heading.spans()) is next(s for s in doc.page(1).spans if s.text == 'Introduction')
    assert paragraph.text == 'A tagged paragraph. It goes on.'
    assert [(c.page, c.mark) for c in paragraph.content()] == [(1, Mark(1)), (2, Mark(0))]
    (square,) = figure.paths()
    assert square.mark == Mark(2)
    assert figure.bbox(1) == pytest.approx((20, 120, 80, 180))
    assert figure.bbox(2) is None


def test_each_value_cites_its_mark_and_an_artifact_is_flagged(tmp_path):
    page = read_pdf(_paper(tmp_path)).page(1)
    cited = {s.text: (s.mark, s.artifact) for s in page.spans}
    assert cited == {
        'Running head': (None, True),
        'Introduction': (Mark(0), False),
        'A tagged paragraph.': (Mark(1), False),
    }


def test_an_elements_actual_text_is_stated_by_its_spans_keeping_what_they_draw(tmp_path):
    content = ' '.join(
        [
            marked('Span', 0, shown(20, 50, 'E=mc2')),
            marked('Span', 1, shown(80, 50, 'squared')),
            marked('Span', 2, shown(20, 80, 'kept')),
        ]
    )
    doc = read_pdf(
        tagged_pdf(
            tmp_path / 'a.pdf',
            [content],
            {
                'root': '<< /K [{formula} {p}] >>',
                'formula': '<< /S /Formula /Pg {page1} /K [0 1] /ActualText (E = m c^2) /Alt (energy) >>',
                'p': '<< /S /P /Pg {page1} /K 2 >>',
            },
        )
    )
    spans = {(s.text, s.drawn) for s in doc.page(1).spans}
    assert spans == {('E = m c^2', 'E=mc2'), ('', 'squared'), ('kept', None)}
    formula = doc.structure.roots[0]
    assert (formula.text, formula.alt) == ('E = m c^2', 'energy')


def test_a_marked_contents_actual_text_keeps_the_glyphs_text_too(tmp_path):
    content = 'BT /F1 10 Tf 20 250 Td (Ef) Tj /Span << /ActualText (fi) >> BDC (XY) Tj EMC (ne) Tj ET'
    doc = read_pdf(tagged_pdf(tmp_path / 'l.pdf', [content], {'root': '<< /K [] >>'}))
    ((text, drawn),) = {(s.text, s.drawn) for s in doc.page(1).spans}
    assert (text, drawn) == ('Effine', 'EfXYne')


def test_an_object_reference_covers_an_annotation_by_its_rectangle(tmp_path):
    path = tagged_pdf(
        tmp_path / 'link.pdf',
        [marked('Link', 0, shown(20, 50, 'see this'))],
        {
            'root': '<< /K {link} >>',
            'link': '<< /S /Link /Pg {page1} /K [0 << /Type /OBJR /Obj {annot} >>] >>',
            'annot': '<< /Type /Annot /Subtype /Link /Rect [20 240 80 255] /Border [0 0 0] >>',
        },
    )
    (link,) = read_pdf(path).structure.roots
    marked_content, reference = link.kids
    assert isinstance(marked_content, MarkedContent)
    assert isinstance(reference, ObjectReference)
    assert reference.page == 1
    assert reference.bbox == pytest.approx((20, 45, 80, 60))


def test_what_the_reader_leaves_out_is_named_in_the_warnings(tmp_path):
    path = tagged_pdf(
        tmp_path / 'cycle.pdf',
        [marked('P', 0, shown(20, 50, 'text'))],
        {'root': '<< /K {a} >>', 'a': '<< /S /Sect /K [{b}] >>', 'b': '<< /S /P /Pg {page1} /K [0 {a}] >>'},
    )
    structure = read_pdf(path).structure
    assert [e.role for e in structure.elements()] == ['Sect', 'P']
    (warning,) = structure.warnings
    assert 'cycle' in warning
