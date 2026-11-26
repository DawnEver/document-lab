"""document-lab's delta against each family config base -- data, computed by nothing.

The bases live in ``dev_lab.dev._famconfig_rows`` and :func:`dev_lab.dev.famconfig.render`
turns base plus delta into the file. document-lab adds only what its Rust extension needs; a fix upstream
lands here by re-rendering.
"""

from __future__ import annotations

from typing import Final

from dev_lab.dev.famconfig import Delta

__all__ = ['DELTAS', 'REPO']

REPO: Final = 'document-lab'

DELTAS: Final[dict[str, Delta]] = {
    # A PDF fixture is bytes: a cross-reference row ends in a space the format requires.
    '.gitattributes': Delta(repo=REPO, added=('*.pdf binary',), dropped={}, ceiling=1),
    # The built extension (`document_lab._pdf`) lands beside the Python sources: a build output, never source.
    '.gitignore': Delta(repo=REPO, added=('*.pyd', '*.so'), dropped={}, ceiling=2),
}
