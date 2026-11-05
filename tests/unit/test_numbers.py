import pytest

from document_lab.numbers import first_number, parse_number

MINUS = chr(0x2212)
THIN = chr(0x2009)


@pytest.mark.parametrize(
    ('text', 'value'),
    [
        (MINUS + '1.5', -1.5),
        ('2e3', 2000.0),
        ('.5', 0.5),
        ('10' + chr(0x207B) + chr(0x00B3), 1e-3),
        ('10' + chr(0x00B2), 100.0),
        ('1' + THIN + '000', 1000.0),
    ],
)
def test_a_whole_number_spelling_parses(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize('text', ['Fig. 1', '', '1-2', 'kHz', '100     100', '12 mm'])
def test_text_that_is_not_only_a_number_is_none(text):
    assert parse_number(text) is None


@pytest.mark.parametrize(
    ('text', 'value'),
    [
        ('7100' + chr(0xB1) + '25%', 7100.0),
        (chr(0x2248) + '2100', 2100.0),
        ('202,000', 202000.0),
        ('1,5', 1.0),  # a decimal comma is not read as one: the first number is 1
        ('4.5e2 s', 450.0),
        ('3.2 x 10-3 W', 3.2e-3),
        ('3.2' + chr(0xD7) + '10' + chr(0x207B) + chr(0xB2), 3.2e-2),
        ('Tc ' + MINUS + '40 to 125', -40.0),
    ],
)
def test_the_first_number_in_a_cell(text, value):
    assert first_number(text) == pytest.approx(value, rel=1e-12, abs=0)


def test_no_number_is_none():
    assert first_number('ungapped') is None
