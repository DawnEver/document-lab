import pytest

from document_lab.extract import AmbiguousField, Basis, Field, Schema, from_columns, from_rows
from document_lab.ir import BBox, Provenance, Table

SRC = Provenance(sha256='ab' * 32, uri='file:///d.pdf', page=3, bbox=BBox(0, 0, 300, 60))

SCHEMA = Schema(
    (
        Field('length', 'number', 'effective length', labels=('Lef [mm]',)),
        Field('count', 'integer', labels=('turns',)),
        Field('grade', 'text', labels=('material',)),
    )
)


def _boxes(rows):
    return tuple(
        tuple(BBox(c * 100, r * 20, c * 100 + 100, r * 20 + 20) for c in range(len(row))) for r, row in enumerate(rows)
    )


def test_a_key_value_table_yields_typed_values_citing_their_cells():
    rows = (('Lef [mm]', '274'), ('Turns', '12'), ('Material', ' 3C90 '), ('Colour', 'grey'))
    values = from_rows(Table(rows, _boxes(rows), SRC), SCHEMA)
    assert values['length'].value == 274.0
    assert values['count'].value == 12
    assert values['grade'].value == '3C90'
    assert values['length'].basis is Basis.RULE
    assert values['length'].text == '274'
    assert values['length'].provenance.bbox == (100, 0, 200, 20)
    assert values['length'].provenance.page == 3


def test_a_value_keeps_the_text_it_was_read_from():
    rows = (('Lef [mm]', '≈274 ± 2%'),)
    value = from_rows(Table(rows, _boxes(rows), SRC), SCHEMA)['length']
    assert value.value == 274.0
    assert value.text == '≈274 ± 2%'  # the qualifier is not lost: it is in the text


def test_a_field_with_no_matching_row_is_absent_not_none():
    rows = (('Colour', 'grey'),)
    assert from_rows(Table(rows, _boxes(rows), SRC), SCHEMA) == {}


def test_an_unreadable_value_is_absent():
    rows = (('Lef [mm]', 'n/a'), ('Turns', '12.5'))
    assert from_rows(Table(rows, _boxes(rows), SRC), SCHEMA) == {}


def test_a_column_table_yields_one_record_per_row():
    rows = (
        ('Part', 'Effective length- Lef [mm] Lef [mm] *', 'Material'),
        ('A', '274', '3C90'),
        ('B', '97', '3C95'),
    )
    records = from_columns(Table(rows, None, Provenance('cd' * 32, 'page.html', locator='table[0]')), SCHEMA)
    assert [r['length'].value for r in records] == [274.0, 97.0]
    assert [r['grade'].value for r in records] == ['3C90', '3C95']
    assert records[1]['length'].provenance.locator == 'table[0]/row[2]/col[1]'


def test_two_columns_matching_one_field_refuse_rather_than_guess():
    rows = (('Lef [mm] min', 'Lef [mm] max'), ('1', '2'))
    with pytest.raises(AmbiguousField, match='length'):
        from_columns(Table(rows, None, SRC), SCHEMA)


def test_a_schema_renders_as_a_json_schema_for_a_model():
    js = SCHEMA.json_schema()
    assert js['type'] == 'object'
    assert js['required'] == ['length', 'count', 'grade']
    assert js['properties']['length']['properties']['value'] == {'type': ['number', 'null']}
    assert js['properties']['count']['properties']['value'] == {'type': ['integer', 'null']}
    assert js['properties']['length']['description'] == 'effective length'


def test_a_field_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match='kind'):
        Field('x', 'float')
