import hashlib

from document_lab.html import read_html, tables_from_html

PAGE = """<html><head><style>td {color: red}</style><script>var x = '<td>no</td>';</script></head><body>
<table class="t">
  <thead><tr><th>Part</th><th>Area [mm<sup>2</sup>]</th><th colspan="2">Gap</th></tr></thead>
  <tbody>
    <tr><td rowspan="2"> A1 </td><td>12.5</td><td>0</td><td><a href="x.pdf">sheet</a></td></tr>
    <tr><td>13<br>.0</td><td>100</td></tr>
  </tbody>
</table>
<p>between</p>
<table><tr><td>outer<table><tr><td>inner</td></tr></table></td></tr></table>
</body></html>"""


def test_a_table_reads_as_its_visible_text_grid():
    first = tables_from_html(PAGE, sha256='s', uri='u')[0]
    assert first.rows == (
        ('Part', 'Area [mm2]', 'Gap', 'Gap'),
        ('A1', '12.5', '0', 'sheet'),
        ('A1', '13 .0', '100', ''),
    )
    assert first.cell_bboxes is None


def test_a_cell_cites_its_table_row_and_column():
    first = tables_from_html(PAGE, sha256='s', uri='u')[0]
    cell = first.cell(2, 1)
    assert (cell.provenance.sha256, cell.provenance.uri) == ('s', 'u')
    assert cell.provenance.locator == 'table[0]/row[2]/col[1]'
    assert cell.provenance.page is None


def test_a_nested_table_is_its_own_table_in_document_order():
    tables = tables_from_html(PAGE, sha256='s', uri='u')
    assert [t.provenance.locator for t in tables] == ['table[0]', 'table[1]', 'table[2]']
    assert tables[1].rows == (('outer',),)
    assert tables[2].rows == (('inner',),)


def test_read_html_names_the_source_by_its_bytes(tmp_path):
    path = tmp_path / 'p.html'
    path.write_text(PAGE, encoding='utf-8')
    (first, *_) = read_html(path)
    assert first.provenance.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.provenance.uri == path.resolve().as_uri()
