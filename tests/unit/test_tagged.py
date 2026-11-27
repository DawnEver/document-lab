"""A tagged PDF's tags used where they state what document-lab infers -- each an option, the inferred kept."""

from __future__ import annotations

import pytest

from document_lab import tags
from document_lab.agreement import compare
from document_lab.markdown import headings, page_markdowns, to_markdown
from document_lab.pdf import read_pdf
from document_lab.per_page import write_pages
from document_lab.regions import regions
from tests.synthetic import marked, mixed_pdf, shown, tagged_pdf

WORDS = 'words enough to read as one line of prose in a paragraph'


def _doc(tmp_path, content, tree, name='t.pdf'):
    return read_pdf(tagged_pdf(tmp_path / name, [content] if isinstance(content, str) else content, tree))


def test_markdown_reads_lines_in_the_order_the_tags_state(tmp_path):
    content = ' '.join(
        [
            marked('P', 0, shown(20, 50, 'Drawn first, read second.')),
            marked('P', 1, shown(20, 150, 'Drawn second, read first.')),
        ]
    )
    doc = _doc(
        tmp_path,
        content,
        {'root': '<< /K [{a} {b}] >>', 'a': '<< /S /P /Pg {page1} /K 1 >>', 'b': '<< /S /P /Pg {page1} /K 0 >>'},
    )
    assert to_markdown(doc) == 'Drawn second, read first.\n\nDrawn first, read second.\n'
    assert to_markdown(doc, tagged=()) == 'Drawn first, read second.\n\nDrawn second, read first.\n'


def test_a_line_heads_as_its_tag_says_and_no_other_does(tmp_path):
    content = ' '.join(
        [
            marked('Title', 0, shown(20, 30, 'A plain title')),
            marked('H1', 1, shown(20, 60, 'Background')),
            marked('P', 2, shown(20, 90, '2 Methods', font='F2')),
            marked('H2', 3, shown(20, 120, 'Detail')),
            marked('P', 4, shown(20, 150, WORDS)),
        ]
    )
    tree = {
        'root': '<< /K {doc} /RoleMap << /Title /P >> >>',
        'doc': '<< /S /Document /Pg {page1} /K [{t} {h1} {p1} {h2} {p2}] >>',
        't': '<< /S /Title /K 0 >>',
        'h1': '<< /S /H1 /K 1 >>',
        'p1': '<< /S /P /K 2 >>',
        'h2': '<< /S /H2 /K 3 >>',
        'p2': '<< /S /P /K 4 >>',
    }
    doc = _doc(tmp_path, content, tree)
    assert [h for _, h in headings(doc)] == ['# A plain title', '## Background', '### Detail']
    # Inferred alone, the bold numbered line heads and the plain heading does not.
    assert '## 2 Methods' in [h for _, h in headings(doc, tagged=())]
    assert '## Background' not in [h for _, h in headings(doc, tagged=())]


def test_an_h_heads_one_level_below_the_sections_it_is_in(tmp_path):
    content = ' '.join(marked('H', i, shown(20, 50 + 40 * i, f'Heading {i}')) for i in range(2))
    tree = {
        'root': '<< /K {s1} >>',
        's1': '<< /S /Sect /Pg {page1} /K [{h0} {s2}] >>',
        'h0': '<< /S /H /K 0 >>',
        's2': '<< /S /Sect /K {h1} >>',
        'h1': '<< /S /H /K 1 >>',
    }
    assert [h for _, h in headings(_doc(tmp_path, content, tree))] == ['## Heading 0', '### Heading 1']


def test_the_lines_of_one_tagged_block_are_one_paragraph_and_of_two_are_two(tmp_path):
    content = ' '.join(
        [
            marked(
                'P', 0, shown(20, 50, 'One paragraph whose second line') + ' ' + shown(20, 120, 'stands far below.')
            ),
            marked('P', 1, shown(20, 132, 'Another paragraph,')),
            marked('P', 2, shown(20, 144, 'set close under it.')),
        ]
    )
    tree = {
        'root': '<< /K [{a} {b} {c}] >>',
        'a': '<< /S /P /Pg {page1} /K 0 >>',
        'b': '<< /S /P /Pg {page1} /K 1 >>',
        'c': '<< /S /P /Pg {page1} /K 2 >>',
    }
    doc = _doc(tmp_path, content, tree)
    assert page_markdowns(doc) == [
        'One paragraph whose second line stands far below.\n\nAnother paragraph,\n\nset close under it.\n'
    ]
    assert page_markdowns(doc, tagged=()) == [
        'One paragraph whose second line stands far below. Another paragraph, set close under it.\n'
    ]


def test_a_line_the_tags_do_not_cover_is_read_as_untagged_text_is(tmp_path):
    content = ' '.join(
        [
            '/Artifact BMC ' + shown(20, 20, 'Running head') + ' EMC',
            marked('P', 0, shown(20, 60, 'Tagged text.')),
            shown(20, 100, '3 Untagged heading', font='F2'),
            shown(20, 112, WORDS),
        ]
    )
    doc = _doc(tmp_path, content, {'root': '<< /K {p} >>', 'p': '<< /S /P /Pg {page1} /K 0 >>'})
    assert page_markdowns(doc) == [f'Running head\n\nTagged text.\n\n## 3 Untagged heading\n\n{WORDS}\n']


def _table_doc(tmp_path):
    cells = [
        ('TH', 'Name', 20, 50),
        ('TH', 'Value', 120, 50),
        ('TH', 'Unit', 220, 50),
        ('TD', 'Spanning two', 20, 70),
        ('TD', 'm', 220, 70),
        ('TD', 'Tall', 20, 90),
        ('TD', '3', 120, 90),
        ('TD', 's', 220, 90),
        ('TD', '4', 120, 110),
        ('TD', 'kg', 220, 110),
    ]
    content = ' '.join(marked(tag, i, shown(x, y, text)) for i, (tag, text, x, y) in enumerate(cells))
    content += ' ' + marked('Caption', len(cells), shown(20, 30, 'Table 4. Quantities'))
    tree = {
        'root': '<< /K {table} >>',
        'table': '<< /S /Table /Pg {page1} /K [{cap} {head} {body}] >>',
        'cap': f'<< /S /Caption /K {len(cells)} >>',
        'head': '<< /S /THead /K {r0} >>',
        'body': '<< /S /TBody /K [{r1} {r2} {r3}] >>',
        'r0': '<< /S /TR /K [{c0} {c1} {c2}] >>',
        'r1': '<< /S /TR /K [{c3} {c4}] >>',
        'r2': '<< /S /TR /K [{c5} {c6} {c7}] >>',
        'r3': '<< /S /TR /K [{c8} {c9}] >>',
    }
    for i, (tag, *_rest) in enumerate(cells):
        spans = {3: '/A << /O /Table /ColSpan 2 >>', 5: '/A << /O /Table /RowSpan 2 >>'}.get(i, '')
        tree[f'c{i}'] = f'<< /S /{tag} /K {i} {spans} >>'
    return _doc(tmp_path, content, tree)


def test_a_tagged_table_reads_as_rows_of_cells_spanning_the_slots_they_state(tmp_path):
    doc = _table_doc(tmp_path)
    (table,) = tags.tables(doc, 1)
    assert table.rows == (
        ('Name', 'Value', 'Unit'),
        ('Spanning two', '', 'm'),
        ('Tall', '3', 's'),
        ('', '4', 'kg'),
    )
    assert table.merged == ((1, 0, 1, 2), (2, 0, 2, 1))
    assert table.heads == ((0, 0), (0, 1), (0, 2))
    assert table.cell_bboxes[1][1] is None
    assert table.cell(2, 1).provenance.bbox.x0 == pytest.approx(120)
    assert tags.tables(doc, 2) == []
    assert tags.tables(read_pdf(mixed_pdf(tmp_path / 'm.pdf')), 1) == []


def test_regions_take_a_tagged_table_and_figure_in_place_of_the_inferred(tmp_path):
    doc = _table_doc(tmp_path)
    page = doc.page(1)
    (table,) = [r for r in regions(page, 10.0, structure=doc.structure) if r.kind == 'table']
    assert table.label == 'Table 4'
    assert table.bbox.y0 < 30 - 10 < 110 < table.bbox.y1


def test_a_tagged_figure_keeps_its_alt_and_an_uncovered_page_is_inferred(tmp_path):
    figure = marked('Figure', 0, '20 100 100 80 re f') + ' ' + marked('Caption', 1, shown(20, 200, 'Fig. 7. A box.'))
    doc = _doc(
        tmp_path,
        [figure, shown(20, 100, 'Figure 2 - Untagged figure') + ' ' + shown(20, 60, WORDS)],
        {
            'root': '<< /K [{fig} {cap}] >>',
            'fig': '<< /S /Figure /Pg {page1} /K 0 /Alt (a filled box) >>',
            'cap': '<< /S /Caption /Pg {page1} /K 1 >>',
        },
    )
    (stated,) = regions(doc.page(1), 10.0, structure=doc.structure)
    assert (stated.kind, stated.label, stated.alt) == ('figure', 'Figure 7', 'a filled box')
    # The box drawn from y 100 up to 180 on a 300-point page stands from 120 down to 200, padded.
    assert stated.bbox.y0 < 120
    assert stated.bbox.y1 > 200
    (inferred,) = regions(doc.page(2), 10.0, structure=doc.structure)
    assert (inferred.label, inferred.alt) == ('Figure 2', None)


def test_the_comparison_names_where_the_tags_and_the_inference_disagree(tmp_path):
    content = ' '.join(
        [
            marked('H1', 0, shown(20, 60, 'Background')),
            marked('P', 1, shown(20, 90, '2 Methods', font='F2')),
            marked('P', 2, shown(20, 120, WORDS)),
        ]
    )
    tree = {
        'root': '<< /K [{h} {a} {b}] >>',
        'h': '<< /S /H1 /Pg {page1} /K 0 >>',
        'a': '<< /S /P /Pg {page1} /K 1 >>',
        'b': '<< /S /P /Pg {page1} /K 2 >>',
    }
    compared = compare(_doc(tmp_path, content, tree))
    assert (compared.headings.tagged, compared.headings.inferred, compared.headings.both) == (1, 1, 0)
    assert compared.headings.agreement == 0
    assert compared.order == 1.0
    assert compared.disagreements == (
        "heading tagged, not inferred: page 1 '## Background'",
        "heading inferred, not tagged: page 1 '## 2 Methods'",
    )
    assert compare(read_pdf(mixed_pdf(tmp_path / 'm.pdf'))) is None


def test_pages_use_the_tags_by_default_and_report_disagreement(tmp_path):
    content = marked('H1', 0, shown(20, 60, 'Background')) + ' ' + marked('P', 1, shown(20, 90, WORDS))
    path = tagged_pdf(
        tmp_path / 'p.pdf',
        [content],
        {'root': '<< /K [{h} {p}] >>', 'h': '<< /S /H1 /Pg {page1} /K 0 >>', 'p': '<< /S /P /Pg {page1} /K 1 >>'},
    )
    done = write_pages(path, tmp_path / 'tagged')
    assert done.disagreements == ("heading tagged, not inferred: page 1 '## Background'",)
    assert '## Background' in (tmp_path / 'tagged' / 'page-001.md').read_text(encoding='utf-8')
    write_pages(path, tmp_path / 'untagged', tagged=())
    assert '## Background' not in (tmp_path / 'untagged' / 'page-001.md').read_text(encoding='utf-8')


def test_each_use_of_the_tags_is_its_own_option(tmp_path):
    content = ' '.join(
        [
            marked('P', 0, shown(20, 50, 'Drawn first, read second.')),
            marked('H1', 1, shown(20, 150, 'Background')),
        ]
    )
    doc = _doc(
        tmp_path,
        content,
        {'root': '<< /K [{h} {p}] >>', 'h': '<< /S /H1 /Pg {page1} /K 1 >>', 'p': '<< /S /P /Pg {page1} /K 0 >>'},
    )
    assert to_markdown(doc, tagged={'headings'}) == 'Drawn first, read second.\n\n## Background\n'
    assert to_markdown(doc, tagged={'order'}) == 'Background\n\nDrawn first, read second.\n'
    with pytest.raises(ValueError, match=r'unknown use.*color'):
        to_markdown(doc, tagged={'order', 'color'})


def test_tags_stating_no_heading_leave_the_headings_and_the_title_to_inference(tmp_path):
    content = ' '.join(
        [
            marked('P', 0, shown(20, 40, 'A Large Title', size=24)),
            marked('P', 1, shown(20, 80, 'IV. RESULTS') + ' ' + shown(20, 92, WORDS)),
        ]
    )
    tree = {'root': '<< /K [{t} {p}] >>', 't': '<< /S /P /Pg {page1} /K 0 >>', 'p': '<< /S /P /Pg {page1} /K 1 >>'}
    # The heading the rules find heads no tagged block: the rest of its paragraph stands apart.
    assert page_markdowns(_doc(tmp_path, content, tree)) == [f'# A Large Title\n\n## IV. RESULTS\n\n{WORDS}\n']


def test_tags_stating_headings_but_no_title_leave_the_title_to_inference(tmp_path):
    content = ' '.join(
        [
            marked('P', 0, shown(20, 40, 'A Large Title', size=24)),
            marked('H1', 1, shown(20, 80, 'Stated heading')),
            marked('P', 2, shown(20, 110, 'IV. NOT A HEADING HERE')),
        ]
    )
    tree = {
        'root': '<< /K [{t} {h} {p}] >>',
        't': '<< /S /P /Pg {page1} /K 0 >>',
        'h': '<< /S /H1 /Pg {page1} /K 1 >>',
        'p': '<< /S /P /Pg {page1} /K 2 >>',
    }
    assert [h for _, h in headings(_doc(tmp_path, content, tree))] == ['# A Large Title', '## Stated heading']


def test_regions_take_from_the_tags_only_the_kinds_asked_and_stated(tmp_path):
    doc = _table_doc(tmp_path)
    assert [r.label for r in regions(doc.page(1), 10.0, structure=doc.structure, tagged={'figures'})] != ['Table 4']
    assert [r.label for r in regions(doc.page(1), 10.0, structure=doc.structure, tagged={'tables'})] == ['Table 4']


def test_a_tagged_heading_the_rules_find_too_takes_their_level_and_no_level_is_skipped(tmp_path):
    content = ' '.join(
        [
            marked('H1', 0, shown(20, 50, 'Background')),
            marked('H5', 1, shown(20, 80, 'REFERENCES')),
            marked('H4', 2, shown(20, 110, 'Deep heading')),
        ]
    )
    tree = {'root': '<< /K [{a} {b} {c}] >>'} | {
        name: f'<< /S /{role} /Pg {{page1}} /K {i} >>'
        for i, (name, role) in enumerate(zip('abc', ('H1', 'H5', 'H4'), strict=True))
    }
    # REFERENCES heads by the rules too: a section, not the H5 a word processor's style made it.
    assert [h for _, h in headings(_doc(tmp_path, content, tree))] == [
        '## Background',
        '## REFERENCES',
        '### Deep heading',
    ]
