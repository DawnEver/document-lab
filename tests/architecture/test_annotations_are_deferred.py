"""A module that imports names only for type checking defers its annotations, so no Python evaluates them eagerly.

Python 3.15 evaluates annotations lazily, so a name imported under ``TYPE_CHECKING`` and used in a signature works
there; an older Python (CI's runners) evaluates the signature at import and raises ``NameError``. #136 landed
``paper/output.py`` that way and every Windows run of the paper tests failed at import.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'src'


def _undeferred(source: str) -> bool:
    """True when ``source`` imports under ``if TYPE_CHECKING:`` without ``from __future__ import annotations``."""
    tree = ast.parse(source)
    checks = any(
        isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING'
        for node in ast.walk(tree)
    )
    deferred = any(
        isinstance(node, ast.ImportFrom)
        and node.module == '__future__'
        and any(alias.name == 'annotations' for alias in node.names)
        for node in tree.body
    )
    return checks and not deferred


def test_every_module_importing_for_type_checking_defers_its_annotations():
    found = [str(p.relative_to(SRC)) for p in sorted(SRC.rglob('*.py')) if _undeferred(p.read_text(encoding='utf-8'))]
    assert found == []


def test_a_module_that_does_not_defer_is_reported():
    """PLANTED CONTROL: the shape #136 landed, and the same with the import that fixes it."""
    planted = 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from x import Y\ndef f(y: Y) -> None: ...\n'
    assert _undeferred(planted)
    assert not _undeferred('from __future__ import annotations\n' + planted)
