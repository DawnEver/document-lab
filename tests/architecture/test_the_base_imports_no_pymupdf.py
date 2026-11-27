"""The package runs without PyMuPDF (D118): it is the dev extra's, the tests' independent reference alone.

A subprocess makes ``pymupdf`` and ``fitz`` unimportable (``sys.modules[name] = None``), then imports every module of
the package -- one an optional extra's dependency keeps from importing (``ai_lab``) is set apart by that dependency's
name, never passed -- and runs the production entry point, ``document-lab pages`` and ``paper``, over a synthetic PDF
whose pages and figure crop the own renderer draws: an import made only when a page is drawn is reached too. The
planted control: modules importing PyMuPDF, put beside the package's, are reported by the same probe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.synthetic import mixed_pdf

#: The names PyMuPDF imports as.
BLOCKED = ('pymupdf', 'fitz')

PROBE = """
import importlib, json, pkgutil, sys
for name in {blocked!r}:
    sys.modules[name] = None
import document_lab
from document_lab.__main__ import main
names = [m.name for m in pkgutil.walk_packages(document_lab.__path__, 'document_lab.')] + sys.argv[3:]
offenders, apart = [], {{}}
for name in names:
    try:
        importlib.import_module(name)
    except ImportError as error:
        if error.name in {blocked!r}:
            offenders.append(name)
        else:
            apart[name] = error.name
pdf, out = sys.argv[1], sys.argv[2]
runs = {{command: main([command, pdf, f'{{out}}/{{command}}']) for command in ('pages', 'paper')}}
print(json.dumps({{'offenders': offenders, 'apart': apart, 'runs': runs}}))
"""


def _probe(tmp_path: Path, *planted: str) -> dict:
    """What the probe finds with PyMuPDF unimportable: the package's modules and ``planted`` (modules in
    ``tmp_path``) imported, ``pages`` and ``paper`` run on a synthetic PDF into ``tmp_path / 'out'``."""
    pdf = mixed_pdf(tmp_path / 'mixed.pdf')
    argv = [sys.executable, '-c', PROBE.format(blocked=BLOCKED), str(pdf), str(tmp_path / 'out'), *planted]
    env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    done = subprocess.run(argv, capture_output=True, text=True, check=True, env=env)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_package_imports_and_runs_without_pymupdf(tmp_path):
    found = _probe(tmp_path)
    assert found['offenders'] == []
    assert set(found['apart'].values()) <= {'ai_lab'}, found['apart']  # the ai extra's, and only it
    assert found['runs'] == {'pages': 0, 'paper': 0}
    out = tmp_path / 'out'
    assert {p.name for p in (out / 'pages').glob('page-*.png')} >= {'page-001.png', 'page-002.png'}
    assert (out / 'paper' / 'paper.md').is_file()


def test_a_module_importing_pymupdf_is_reported(tmp_path):
    """PLANTED CONTROL: a module importing PyMuPDF, under either name, is an offender the probe names."""
    (tmp_path / 'planted_pymupdf.py').write_text('import pymupdf\n', encoding='utf-8')
    (tmp_path / 'planted_fitz.py').write_text('from fitz import Document\n', encoding='utf-8')
    found = _probe(tmp_path, 'planted_pymupdf', 'planted_fitz')
    assert found['offenders'] == ['planted_pymupdf', 'planted_fitz']
