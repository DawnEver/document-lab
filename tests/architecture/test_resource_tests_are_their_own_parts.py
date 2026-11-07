"""`gpu` (marker-pdf on a CUDA GPU), `vendor` (a real vendor PDF) and `desktop` (an interactive Windows desktop) are
declared resources: base never selects them, so they cannot skip inside it, and all are declared unproduced on hosted
runners (none exists there)."""

import tomllib
from pathlib import Path

from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from tests.desktop import test_capture_desktop
from tests.gpu import test_marker
from tests.vendor import test_vendor_pdf

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


RESOURCES = {'gpu', 'vendor', 'desktop'}


def test_each_is_a_declared_resource_so_base_excludes_it():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert set(vocabulary.resources) == RESOURCES
    expression = selecting(vocabulary, BASE)
    for resource in RESOURCES:
        assert f'not {resource}' in expression


def test_no_default_marker_expression_hides_them_from_part_derivation():
    options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in options.get('addopts', '')


def test_each_is_declared_unproduced():
    assert unproduced(PYPROJECT.read_text(encoding='utf-8')) == frozenset(RESOURCES)


def test_each_part_marks_its_whole_module():
    assert test_marker.pytestmark.name == 'gpu'
    assert test_vendor_pdf.pytestmark.name == 'vendor'
    assert test_capture_desktop.pytestmark.name == 'desktop'
