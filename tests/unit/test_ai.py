import subprocess
import sys

from ai_lab import Image, Response, Text

from document_lab.ai import ask
from document_lab.extract import Basis, Field, Schema
from document_lab.ir import BBox, Page, Provenance, Table, TextSpan

SCHEMA = Schema(
    (Field('length', 'number', 'effective length in mm'), Field('grade', 'text'), Field('count', 'integer'))
)
SRC = Provenance('ab' * 32, 'file:///d.pdf', page=2)


class StandIn:
    """A Responder that records the request and answers what the test planted."""

    name = 'stand-in'

    def __init__(self, value):
        self.value = value
        self.request = None

    def respond(self, request):
        self.request = request
        return Response(value=self.value, provider='stand-in', model='m')


def _page():
    spans = (
        TextSpan('Effective length 274 mm', 9, 'helv', SRC.at(bbox=BBox(10, 10, 120, 20))),
        TextSpan('Grade 3C90', 9, 'helv', SRC.at(bbox=BBox(10, 30, 60, 40))),
    )
    return Page(number=2, width=612, height=792, spans=spans)


def test_a_model_answer_is_a_model_value_citing_the_span_it_quoted():
    responder = StandIn(
        {
            'length': {'value': 274, 'quote': 'length 274 mm'},
            'grade': {'value': '3C90', 'quote': 'nowhere on the page'},
            'count': {'value': None, 'quote': None},
        }
    )
    values = ask(responder, SCHEMA, _page())
    assert values['length'].value == 274.0
    assert values['length'].basis is Basis.MODEL
    assert values['length'].provenance.bbox == (10, 10, 120, 20)
    # A quote the page does not contain cites the page, not a guessed box.
    assert values['grade'].provenance.bbox is None
    assert values['grade'].provenance.page == 2
    assert 'count' not in values  # null is absent


def test_the_request_carries_the_page_text_the_schema_and_an_image():
    responder = StandIn({'length': {'value': None, 'quote': None}})
    ask(responder, SCHEMA, _page(), instructions='Values at 25 C.', image=b'\x89PNG\r\n\x1a\nx')
    request = responder.request
    assert request.schema == SCHEMA.json_schema()
    assert 'Values at 25 C.' in request.instructions
    texts = [p.text for p in request.context if isinstance(p, Text)]
    assert 'Effective length 274 mm' in texts[0]
    assert any(isinstance(p, Image) for p in request.context)


def test_a_table_answer_cites_the_quoted_cell():
    rows = (('Lef', '274'), ('Grade', '3C90'))
    table = Table(rows, None, Provenance('cd' * 32, 'p.html', locator='table[0]'))
    values = ask(StandIn({'grade': {'value': '3C90', 'quote': '3C90'}}), SCHEMA, table)
    assert values['grade'].provenance.locator == 'table[0]/row[1]/col[1]'


def test_a_model_number_of_the_wrong_kind_is_absent():
    values = ask(
        StandIn({'count': {'value': 2.5, 'quote': '2.5'}, 'length': {'value': True, 'quote': 'x'}}), SCHEMA, _page()
    )
    assert values == {}


def test_the_base_package_imports_without_ai_lab():
    probe = (
        "import sys; sys.modules['ai_lab'] = None\n"
        'import document_lab, document_lab.ir, document_lab.pdf, document_lab.html, document_lab.plot\n'
        'import document_lab.extract, document_lab.numbers\n'
        'try:\n    import document_lab.ai\nexcept ImportError as e:\n    print(e)\n'
    )
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert "the 'ai' extra" in out.stdout
