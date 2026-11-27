"""The ``gpu`` part: `document-lab paper --tool marker` through marker-pdf. Its own part, never tests/base; where
marker cannot run (no gpu extra, no model server for surya) the forced choice REFUSES naming what is missing and the
test fails with it -- it is never skipped."""

import pytest

from tests.paper.e2e.test_papers import IEEE
from tests.paper.structure import problems
from tests.paper.support import DATA, ingested

pytestmark = pytest.mark.gpu


def test_marker_ingests_a_paper_with_its_structure(tmp_path_factory):
    assert problems(ingested(DATA / IEEE.pdf, 'marker', tmp_path_factory), IEEE) == []
