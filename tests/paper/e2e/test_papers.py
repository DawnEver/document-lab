"""The three real papers, ingested on the IR path, checked STRUCTURALLY against what each PDF itself says.

The ground truth below was read off the PDFs (headings, caption numbers), not off any converter's output. Every
check can fail: test_a_wrong_truth_is_found plants one wrong fact per kind and requires a finding for each.
"""

import dataclasses

import pymupdf
import pytest

from tests.paper.structure import Paper, problems
from tests.paper.support import DATA, ingested

IEEE = Paper(
    'ieee-conference.pdf',
    'Conference Paper Title',
    (
        'I. INTRODUCTION',
        'II. EASE OF USE',
        'III. PREPARE YOUR PAPER BEFORE STYLING',
        'ACKNOWLEDGMENT',
        'REFERENCES',
        'REFERENCES',  # the template states its references heading twice
    ),
    figures=frozenset({1}),
    tables=frozenset({'I'}),
)
LIANG = Paper(
    'liang-planar-2025.pdf',
    'Equivalent Circuit-Based Analysis of Current Sharing in Planar Transformer Parallel Windings',
    (
        'I. INTRODUCTION',
        'II. FREQUENCY-RELATED CURRENT REDISTRIBUTION AND PARALLEL WINDING ANALYSIS',
        'III. CIRCUIT-BASED MODEL FOR CURRENT SHARING BETWEEN PARALLEL WINDINGS AND WINDING LOSS',
        'IV. CONCLUSION',
        'REFERENCES',
    ),
    figures=frozenset({*range(1, 9), *range(10, 17)}),
    tables=frozenset({'I'}),
    absent_figures=frozenset({9}),  # the paper's numbering skips 9
)
HAIRPIN = Paper(
    'wang-hairpin-2025.pdf',
    'Scripting-Based 3D Geometry Modelling of Hairpin Windings in EV Traction Motors',
    (
        'I. INTRODUCTION',
        'II. SCRIPTING-BASED TOOL FOR AUTOMATED DESIGN AND MODELLING OF HAIRPIN WINDINGS',
        'III. INTERFERENCE MITIGATION AND MANUFACTURABILITY CONSIDERATIONS',
        'IV. INTERFERENCE CHECK AND SELF-TUNING FRAMEWORK FOR MANUFACTURABLE HAIRPIN WINDING MODEL GENERATION',
        'V. CONCLUSION',
        'ACKNOWLEDGMENT',
        'REFERENCES',
    ),
    figures=frozenset(range(1, 14)),
    tables=frozenset({'I', 'II', 'III'}),
)
PAPERS = (IEEE, LIANG, HAIRPIN)


@pytest.fixture(scope='module')
def bound(tmp_path_factory):
    """liang with hairpin bound in after it: one PDF, the main paper plus one appended paper."""
    path = tmp_path_factory.mktemp('bound') / 'liang-then-hairpin.pdf'
    with pymupdf.open(DATA / LIANG.pdf) as doc, pymupdf.open(DATA / HAIRPIN.pdf) as tail:
        doc.insert_pdf(tail)
        doc.save(path)
    return ingested(path, 'pymupdf', tmp_path_factory)


@pytest.mark.parametrize('paper', PAPERS, ids=lambda p: p.pdf)
def test_the_ingested_paper_has_the_pdfs_structure(paper, tmp_path_factory):
    assert problems(ingested(DATA / paper.pdf, 'pymupdf', tmp_path_factory), paper) == []


def test_a_second_paper_bound_in_is_split_off(bound):
    assert problems(bound, dataclasses.replace(LIANG, appended=1)) == []
    (appended,) = sorted((bound / 'appended').iterdir())
    assert HAIRPIN.title in (appended / 'paper.md').read_text(encoding='utf-8')


WRONG = {
    'title': lambda p: dataclasses.replace(p, title='Some Other Title'),
    'a section dropped': lambda p: dataclasses.replace(p, sections=p.sections[:-1]),
    'sections reordered': lambda p: dataclasses.replace(p, sections=(p.sections[1], p.sections[0], *p.sections[2:])),
    'a heading misspelt': lambda p: dataclasses.replace(p, sections=(p.sections[0] + 'X', *p.sections[1:])),
    'a figure invented': lambda p: dataclasses.replace(p, figures=p.figures | {99}),
    'a figure missed': lambda p: dataclasses.replace(p, figures=p.figures - {max(p.figures)}),
    'a table invented': lambda p: dataclasses.replace(p, tables=p.tables | {'IX'}),
    'a table missed': lambda p: dataclasses.replace(p, tables=frozenset()),
    'an appended paper invented': lambda p: dataclasses.replace(p, appended=1),
    'a skipped number indexed': lambda p: dataclasses.replace(p, absent_figures=frozenset({min(p.figures)})),
}


@pytest.mark.parametrize('wrong', sorted(WRONG))
@pytest.mark.parametrize('paper', PAPERS, ids=lambda p: p.pdf)
def test_a_wrong_truth_is_found(paper, wrong, tmp_path_factory):
    """The planted control: the same check against a falsified truth must report it."""
    out = ingested(DATA / paper.pdf, 'pymupdf', tmp_path_factory)
    assert problems(out, WRONG[wrong](paper)), f'{wrong}: the check could not say no'


def test_the_split_check_can_say_no(bound):
    assert problems(bound, LIANG)  # the bound PDF without its appended paper declared
