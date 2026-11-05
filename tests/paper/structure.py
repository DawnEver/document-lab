"""What a paper's ingested layout must hold, read off the PDF itself, and the check that says what it misses.

A :class:`Paper` is the ground truth a reader of the PDF sees: its title, its numbered section headings in order,
the figure and table numbers its captions carry, and how many further papers are bound in after it. :func:`problems`
lists every way an ingested directory departs from it -- an empty list is the pass. Because the check returns its
findings instead of asserting, tests/paper/e2e plants wrong truths and requires a finding for each.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Paper:
    pdf: str
    title: str
    sections: tuple[str, ...]
    figures: frozenset[int]
    tables: frozenset[str]  # Roman numerals as captioned: 'I', 'II'
    appended: int = 0
    absent_figures: frozenset[int] = field(default_factory=frozenset)  # numbers the PDF skips


def _headings(out: Path) -> list[str]:
    return [
        p.read_text(encoding='utf-8').split('\n', 1)[0].lstrip('#').strip() for p in sorted((out / 'md').glob('*.md'))
    ]


def _images(out: Path, kind: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for png in sorted(out.glob('img/sec*/*.png')):
        m = re.search(rf'-{kind}-([0-9ivx]+)\.png$', png.name)
        if m:
            found.setdefault(m.group(1), []).append(png)
    return found


def problems(out: Path, paper: Paper) -> list[str]:
    """Every departure of the ingested ``out`` from ``paper``; empty when it holds."""
    found: list[str] = []
    title = (out / 'paper.md').read_text(encoding='utf-8').split('\n', 1)[0]
    if paper.title not in title:
        found.append(f'title {title!r} lacks {paper.title!r}')
    headings = _headings(out)
    if headings != list(paper.sections):
        found.append(f'sections {headings} != {list(paper.sections)}')
    found += _figure_problems(out, paper) + _table_problems(out, paper)
    appended = sorted((out / 'appended').glob('*/paper.md')) if (out / 'appended').exists() else []
    if len(appended) != paper.appended:
        found.append(f'{len(appended)} appended paper(s), expected {paper.appended}')
    return found


def _figure_problems(out: Path, paper: Paper) -> list[str]:
    figures = _images(out, 'figure')
    index = (out / 'INDEX.md').read_text(encoding='utf-8')
    found = []
    for n in sorted(paper.figures):
        if str(n) not in figures:
            found.append(f'Figure {n}: no image file')
        elif f'| Figure {n} | {figures[str(n)][0].relative_to(out).as_posix()} |' not in index:
            found.append(f'Figure {n}: INDEX.md does not point at its image')
    found += [
        f'Figure {n}: an image for a figure the paper does not caption'
        for n in sorted(set(map(int, figures)) - paper.figures)
    ]
    found += [
        f'Figure {n}: indexed, but the paper has none'
        for n in sorted(paper.absent_figures)
        if f'| Figure {n} |' in index
    ]
    return found


def _table_problems(out: Path, paper: Paper) -> list[str]:
    tables = _images(out, 'table')
    expected = {t.lower() for t in paper.tables}
    return [f'TABLE {t.upper()}: no image file' for t in sorted(expected - set(tables))] + [
        f'TABLE {t.upper()}: an image for a table the paper does not caption' for t in sorted(set(tables) - expected)
    ]
