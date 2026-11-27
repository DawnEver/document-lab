"""The ``gpu`` part: `document-lab paper` through marker-pdf. Its own part, never tests/base; where marker cannot
run (no CUDA GPU with >= 4 GB, no gpu extra, no model server for surya) it FAILS naming what is missing -- it is
never skipped."""

import pytest

from document_lab.paper.convert import choose_tool
from tests.paper.e2e.test_papers import IEEE
from tests.paper.structure import problems
from tests.paper.support import DATA, ingested

pytestmark = pytest.mark.gpu


def test_marker_ingests_a_paper_with_its_structure(tmp_path_factory):
    choice = choose_tool()
    assert choice.tool == 'marker', choice.why
    assert problems(ingested(DATA / IEEE.pdf, 'marker', tmp_path_factory), IEEE) == []
