"""The Rust extension is built into the package and is the crate this repo declares."""

import tomllib
from pathlib import Path

from document_lab import _pdf

_ROOT = Path(__file__).resolve().parents[2]


def test_the_extension_is_the_workspace_crate() -> None:
    declared = tomllib.loads((_ROOT / 'Cargo.toml').read_text(encoding='utf-8'))['workspace']['package']['version']
    assert _pdf.__version__ == declared
