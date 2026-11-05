"""Each real paper ingested once per session per converter (marker only where a GPU can run it)."""

import pytest

from tests.paper.support import DATA, ingested, marker_available


def _out(name, tool, factory):
    if tool == 'marker' and not marker_available():
        pytest.skip('marker needs marker_single and a GPU with >= 4 GB')
    return ingested(DATA / name, tool, factory)


@pytest.fixture(scope='module')
def ieee_out_pymupdf4llm(tmp_path_factory):
    return _out('ieee-conference.pdf', 'pymupdf4llm', tmp_path_factory)


@pytest.fixture(scope='module')
def ieee_out_marker(tmp_path_factory):
    return _out('ieee-conference.pdf', 'marker', tmp_path_factory)


@pytest.fixture(scope='module')
def liang_out(tmp_path_factory):
    return _out('liang-planar-2025.pdf', 'pymupdf4llm', tmp_path_factory)


@pytest.fixture(scope='module')
def liang_out_marker(tmp_path_factory):
    return _out('liang-planar-2025.pdf', 'marker', tmp_path_factory)


@pytest.fixture(scope='module')
def hairpin_out_pymupdf4llm(tmp_path_factory):
    return _out('wang-hairpin-2025.pdf', 'pymupdf4llm', tmp_path_factory)


@pytest.fixture(scope='module')
def hairpin_out_marker(tmp_path_factory):
    return _out('wang-hairpin-2025.pdf', 'marker', tmp_path_factory)
