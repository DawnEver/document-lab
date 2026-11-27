"""The corpus part's fixture: a source fetched at SETUP, so a fetch failure is a test error, never a FAILED check."""

from pathlib import Path

import pytest

from tests.corpus.sources import fetch


@pytest.fixture
def pdf(request: pytest.FixtureRequest) -> Path:
    """The verified local copy of the source named by the test's (indirect) parameter."""
    return fetch(request.param)
