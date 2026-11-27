"""`gpu` (marker-pdf on a CUDA GPU), `vendor` (a real vendor PDF), `papers` (the real papers, local), `reference`
(a local corpus of real PDFs), `ocr` (Tesseract), the OS parts `windows`, `macos` and `linux` (a desktop of each) and
`corpus` (public PDFs fetched from their URLs) are declared resources: base never selects them, so they cannot skip
inside it. The first five are declared unproduced on hosted runners (none exists there); an OS part runs on its OS's
hosted runner, the linux and macos ones after a setup that gives the job's interpreter Tk (the overlay, the panel and
the viewer are Tk windows), the linux one starting a display as well; `corpus` needs the network a hosted runner has, so
it runs there in a job of its own whose setup fetches the sources."""

import tomllib
from pathlib import Path

from dev_lab.dev.hostedparts import hosted_parts
from dev_lab.dev.unproduced import unproduced
from lab_commons.hpc.needsets import BASE, read_vocabulary, selecting

from tests.corpus import test_corpus, test_paper_structure
from tests.gpu import test_marker
from tests.linux import test_capture_desktop as linux_capture
from tests.macos import test_capture_desktop as macos_capture
from tests.ocr import test_ocr
from tests.paper.e2e import test_papers
from tests.reference import test_reference
from tests.vendor import test_vendor_pdf
from tests.windows import test_capture_desktop as windows_capture

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


RESOURCES = {'gpu', 'vendor', 'papers', 'reference', 'ocr', 'windows', 'macos', 'linux', 'corpus'}
UNPRODUCED = {'gpu', 'vendor', 'papers', 'reference', 'ocr'}


def test_each_is_a_declared_resource_so_base_excludes_it():
    vocabulary = read_vocabulary(PYPROJECT.read_text(encoding='utf-8'))
    assert set(vocabulary.resources) == RESOURCES
    expression = selecting(vocabulary, BASE)
    for resource in RESOURCES:
        assert f'not {resource}' in expression


def test_no_default_marker_expression_hides_them_from_part_derivation():
    options = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pytest']['ini_options']
    assert '-m' not in options.get('addopts', '')


def test_all_but_the_os_parts_and_corpus_are_declared_unproduced():
    assert unproduced(PYPROJECT.read_text(encoding='utf-8')) == frozenset(UNPRODUCED)


def test_the_hosted_setups_are_tk_for_linux_and_macos_and_the_corpus_fetch():
    assert set(hosted_parts(PYPROJECT.read_text(encoding='utf-8'))) == {'linux', 'macos', 'corpus'}


def test_the_linux_part_starts_a_display_and_a_window_manager_before_it_runs():
    setup = hosted_parts(PYPROJECT.read_text(encoding='utf-8'))['linux'].setup
    assert 'Xvfb' in setup
    assert 'openbox' in setup
    assert 'GITHUB_ENV' in setup


def test_the_linux_part_gives_the_system_python_tk():
    """uv builds the job's venv on the runner's /usr/bin/python3, whose tkinter Debian ships apart (python3-tk)."""
    assert 'python3-tk' in hosted_parts(PYPROJECT.read_text(encoding='utf-8'))['linux'].setup.split()


def test_the_macos_part_gives_the_venv_interpreter_tk():
    """uv builds the job's venv on Homebrew's python@X.Y, whose _tkinter is the separate python-tk@X.Y formula."""
    setup = hosted_parts(PYPROJECT.read_text(encoding='utf-8'))['macos'].setup
    assert setup.startswith('brew install')
    assert 'python-tk@' in setup
    assert 'uv run --no-sync python' in setup  # the version is the venv's own, never a pinned one


def test_corpus_is_hosted_with_its_sources_fetched_first():
    declared = hosted_parts(PYPROJECT.read_text(encoding='utf-8'))
    assert declared['corpus'].setup.endswith('python -m tests.corpus.sources')
    assert not declared['corpus'].tasks  # no secret: it runs on every event, a pull request's included


def test_each_part_marks_its_whole_module():
    assert test_marker.pytestmark.name == 'gpu'
    assert test_vendor_pdf.pytestmark.name == 'vendor'
    assert test_papers.pytestmark.name == 'papers'
    assert test_reference.pytestmark.name == 'reference'
    assert windows_capture.pytestmark.name == 'windows'
    assert macos_capture.pytestmark.name == 'macos'
    assert linux_capture.pytestmark.name == 'linux'
    assert test_ocr.pytestmark.name == 'ocr'
    assert test_corpus.pytestmark.name == 'corpus'
    assert test_paper_structure.pytestmark.name == 'corpus'
