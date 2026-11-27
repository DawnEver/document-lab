"""``document-lab paper`` end to end in CI: two LaTeX arXiv papers of the corpus, checked STRUCTURALLY.

The local ``papers`` part (tests/paper/e2e) checks the IEEE papers that may not be published; this part keeps the
paper layout's end-to-end coverage on papers that may (user ruling: CI keeps it without the real papers). The truth
below was read off each PDF -- its title, its top-level headings in order, the numbers its figure and table captions
carry -- never off any converter's output. Each kind is its own check (`tests.paper.structure.CHECKS`), so a
regression names the kind it breaks; every planted wrong truth (`tests.paper.structure.WRONG`) must be reported.
"""

from pathlib import Path

import pytest

from tests.paper.structure import CHECKS, WRONG, Paper
from tests.paper.support import ingested

pytestmark = pytest.mark.corpus

ISMIR = Paper(
    'latex-type3-arxiv-2204.08094v1',
    'A DATA-DRIVEN METHODOLOGY FOR CONSIDERING FEASIBILITY AND PAIRWISE LIKELIHOOD IN DEEP LEARNING BASED GUITAR '
    'TABLATURE TRANSCRIPTION SYSTEMS',
    # Its headings at section level, after the abstract (subsections 2.1-4.3 are inside them).
    (
        '1. INTRODUCTION',
        '2. PROPOSED METHOD',
        '3. EXPERIMENTAL SETUP',
        '4. RESULTS & DISCUSSION',
        '5. CONCLUSION',
        'Acknowledgments',
        '6. REFERENCES',
    ),
    figures=frozenset({1, 2, 3}),
    tables=frozenset({'1', '2'}),
)
PREPRINT = Paper(
    'cjk-ja-euc-h-arxiv-2608.20674v1',
    'Analysis of Potential Generative AI Use in Abstracts of KAKENHI-Funded Projects',
    # Its 12 pt headings after the abstract, the number and the name as the page sets them apart.
    (
        '1 Introduction',
        '2 Method for Estimating AI-Generated Text',
        '3 Data',
        '4 Methods',
        '5 Experiments',
        '6 Conclusion',
        'References',
        'Appendix',
        'A Sentence-Level Estimation',
        'B Trial Estimation Using Word Distributions',
    ),
    figures=frozenset(range(1, 8)),
    tables=frozenset(str(n) for n in range(1, 11)),
)
PAPERS = {paper.pdf: paper for paper in (ISMIR, PREPRINT)}


@pytest.mark.parametrize('kind', sorted(CHECKS))
@pytest.mark.parametrize('pdf', sorted(PAPERS), indirect=True)
def test_the_ingested_paper_has_the_pdfs_structure(pdf: Path, kind: str, tmp_path_factory, request):
    truth = PAPERS[request.node.callspec.params['pdf']]
    assert CHECKS[kind](ingested(pdf, 'ir', tmp_path_factory), truth) == []


@pytest.mark.parametrize('wrong', sorted(WRONG))
@pytest.mark.parametrize('pdf', sorted(PAPERS), indirect=True)
def test_a_wrong_truth_is_found(pdf: Path, wrong: str, tmp_path_factory, request):
    """The planted control: each kind's check against a falsified truth must report it."""
    truth = PAPERS[request.node.callspec.params['pdf']]
    kind, falsify = WRONG[wrong]
    assert CHECKS[kind](ingested(pdf, 'ir', tmp_path_factory), falsify(truth)), f'{wrong}: the check could not say no'
