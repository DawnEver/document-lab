"""What the reader makes of a corpus document, page by page, and the check that says where it departs from a golden.

A :class:`Page` is one page as the reader states it: its size as displayed (to 0.01 pt), the sha256 of its text
(span texts joined, whitespace collapsed to single spaces) and how many spans, vector paths and images it places.
:func:`problems` lists every way a document's pages depart from the golden ones -- an empty list is the pass.

``.venv/Scripts/python.exe -m tests.corpus.reading`` reads every source of :mod:`tests.corpus.sources` and writes
:mod:`tests.corpus.goldens` whole: a deliberate reader improvement updates the goldens in one step, and the diff
shows what it changed.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import NamedTuple

from document_lab.pdf import read_pdf

__all__ = ['Page', 'problems', 'summarize']


class Page(NamedTuple):
    """One page as the reader states it."""

    number: int
    width: float
    height: float
    text_sha256: str
    spans: int
    paths: int
    images: int


@cache
def summarize(pdf: Path, password: str = '') -> tuple[Page, ...]:
    """Every page of ``pdf`` (opened with ``password``) as a :class:`Page`."""
    return tuple(
        Page(
            page.number,
            round(page.width, 2),
            round(page.height, 2),
            hashlib.sha256(' '.join(' '.join(s.text for s in page.spans).split()).encode()).hexdigest(),
            len(page.spans),
            len(page.paths),
            len(page.figures),
        )
        for page in read_pdf(pdf, password=password).pages
    )


def problems(found: tuple[Page, ...], golden: tuple[Page, ...]) -> list[str]:
    """Every departure of ``found`` from ``golden``, in words; empty when they agree."""
    out = [f'{len(found)} pages, golden {len(golden)}'] if len(found) != len(golden) else []
    for page, want in zip(found, golden, strict=False):
        n = want.number
        if (page.width, page.height) != (want.width, want.height):  # both rounded to 0.01 pt
            out.append(f'page {n}: {page.width} x {page.height} pt, golden {want.width} x {want.height}')
        if page.text_sha256 != want.text_sha256:
            out.append(f'page {n}: text sha256 {page.text_sha256[:12]}, golden {want.text_sha256[:12]}')
        out += [
            f'page {n}: {getattr(page, kind)} {kind}, golden {getattr(want, kind)}'
            for kind in ('spans', 'paths', 'images')
            if getattr(page, kind) != getattr(want, kind)
        ]
    return out


def _written(goldens: dict[str, tuple[Page, ...]]) -> str:
    rows = []
    for name, pages in goldens.items():
        rows.append(f'    {name!r}: (')
        rows += [f'        Page{tuple(page)!r},' for page in pages]
        rows.append('    ),')
    body = '\n'.join(rows)
    return f'''"""The golden pages of each corpus source (tests.corpus.reading): written by a command, never edited.

The command: ``.venv/Scripts/python.exe -m tests.corpus.reading``. A deliberate reader change re-runs it, and the
diff is the change's evidence.
"""

from tests.corpus.reading import Page

GOLDENS: dict[str, tuple[Page, ...]] = {{
{body}
}}
'''


def main() -> None:
    """Write :mod:`tests.corpus.goldens` from the reader's reading of every source, fetched as needed."""
    from tests.corpus.sources import SOURCES, fetch  # noqa: PLC0415 -- the generator alone fetches

    goldens = {name: summarize(fetch(name)) for name in SOURCES}
    Path(__file__).with_name('goldens.py').write_text(_written(goldens), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
