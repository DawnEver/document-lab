"""`gpu` (marker-pdf on a CUDA GPU), `vendor` (a real vendor PDF), `papers` (the real papers, local), `ocr`
(Tesseract) and the OS parts `windows`, `macos` and `linux` (a desktop of each) are declared resources: base never
selects them, so they cannot skip inside it. All but the OS parts are declared unproduced on hosted runners (none
exists there); an OS part runs on its OS's hosted runner, the linux one after a setup that starts a display."""

import tomllib
from pathlib import Path

from dev_lab.dev.hostedparts import hosted_parts
from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from tests.gpu import test_marker
from tests.linux import test_capture_desktop as linux_capture
from tests.macos import test_capture_desktop as macos_capture
from tests.ocr import test_ocr
from tests.paper.e2e import test_papers
from tests.vendor import test_vendor_pdf
from tests.windows import test_capture_desktop as windows_capture

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


RESOURCES = {'gpu', 'vendor', 'papers', 'ocr', 'windows', 'macos', 'linux'}
UNPRODUCED = {'gpu', 'vendor', 'papers', 'ocr'}


def test_each_is_a_declared_resource_so_base_excludes_it():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert set(vocabulary.resources) == RESOURCES
    expression = selecting(vocabulary, BASE)
    for resource in RESOURCES:
        assert f'not {resource}' in expression


def test_no_default_marker_expression_hides_them_from_part_derivation():
    options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in options.get('addopts', '')


def test_all_but_the_os_parts_are_declared_unproduced():
    assert unproduced(PYPROJECT.read_text(encoding='utf-8')) == frozenset(UNPRODUCED)


def test_the_linux_part_starts_a_display_and_a_window_manager_before_it_runs():
    setup = hosted_parts(PYPROJECT.read_text(encoding='utf-8'))['linux'].setup
    assert 'Xvfb' in setup
    assert 'openbox' in setup
    assert 'GITHUB_ENV' in setup


def test_each_part_marks_its_whole_module():
    assert test_marker.pytestmark.name == 'gpu'
    assert test_vendor_pdf.pytestmark.name == 'vendor'
    assert test_papers.pytestmark.name == 'papers'
    assert windows_capture.pytestmark.name == 'windows'
    assert macos_capture.pytestmark.name == 'macos'
    assert linux_capture.pytestmark.name == 'linux'
    assert test_ocr.pytestmark.name == 'ocr'
