"""`gpu` (marker-pdf on a CUDA GPU) and `vendor` (a real vendor PDF) are declared resources: base never selects
them, so they cannot skip inside it, and both are declared unproduced on hosted runners (neither exists there)."""

import tomllib
from pathlib import Path

from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from tests.gpu import test_marker
from tests.vendor import test_vendor_pdf

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


def test_both_are_declared_resources_so_base_excludes_them():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert set(vocabulary.resources) == {'gpu', 'vendor'}
    expression = selecting(vocabulary, BASE)
    assert 'not gpu' in expression
    assert 'not vendor' in expression


def test_no_default_marker_expression_hides_them_from_part_derivation():
    options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in options.get('addopts', '')


def test_both_are_declared_unproduced():
    assert unproduced(PYPROJECT.read_text(encoding='utf-8')) == frozenset({'gpu', 'vendor'})


def test_each_part_marks_its_whole_module():
    assert test_marker.pytestmark.name == 'gpu'
    assert test_vendor_pdf.pytestmark.name == 'vendor'
