"""`gpu` (marker-pdf on a CUDA GPU), `vendor` (a real vendor PDF), `ocr` (Tesseract) and `windows` (a Windows desktop)
are declared resources: base never selects them, so they cannot skip inside it. The first three are declared
unproduced on hosted runners (none exists there); `windows` is an OS, so it runs on the hosted windows runner."""

import tomllib
from pathlib import Path

from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from tests.gpu import test_marker
from tests.ocr import test_ocr
from tests.vendor import test_vendor_pdf
from tests.windows import test_capture_desktop

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


RESOURCES = {'gpu', 'vendor', 'ocr', 'windows'}
UNPRODUCED = {'gpu', 'vendor', 'ocr'}


def test_each_is_a_declared_resource_so_base_excludes_it():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert set(vocabulary.resources) == RESOURCES
    expression = selecting(vocabulary, BASE)
    for resource in RESOURCES:
        assert f'not {resource}' in expression


def test_no_default_marker_expression_hides_them_from_part_derivation():
    options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in options.get('addopts', '')


def test_all_but_windows_are_declared_unproduced():
    assert unproduced(PYPROJECT.read_text(encoding='utf-8')) == frozenset(UNPRODUCED)


def test_each_part_marks_its_whole_module():
    assert test_marker.pytestmark.name == 'gpu'
    assert test_vendor_pdf.pytestmark.name == 'vendor'
    assert test_capture_desktop.pytestmark.name == 'windows'
    assert test_ocr.pytestmark.name == 'ocr'
