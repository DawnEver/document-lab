"""Shared by the paper tests: a body long enough to pass the noise filters, the marker gate, the real-paper runner."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from document_lab.paper.convert import MARKER_MIN_VRAM_GB, detect_gpu_vram_gb, find_marker_single
from document_lab.paper.ingest import ingest

LONG_BODY = 'Long body text that passes all thresholds. ' * 20  # well over 400 chars
DATA = Path(__file__).resolve().parent / 'data'
FULL_PAGE = (1224, 1584)  # US letter rendered at 2x: the crop fell back to the whole page


def marker_available() -> bool:
    return detect_gpu_vram_gb() >= MARKER_MIN_VRAM_GB and find_marker_single() is not None


marker_skip = pytest.mark.skipif(not marker_available(), reason='marker needs marker_single and a GPU with >= 4 GB')

_cache: dict[tuple[str, str], Path] = {}


def ingested(pdf: Path, tool: str, factory: pytest.TempPathFactory) -> Path:
    """The paper layout of ``pdf`` by ``tool``, ingested once per session."""
    key = (str(pdf), tool)
    if key not in _cache:
        out = factory.mktemp(f'{pdf.stem}_{tool}')
        done = ingest(pdf, out, tool=tool)
        assert done.tool == tool
        _cache[key] = out
    return _cache[key]


def read_png_size(path: Path) -> tuple[int, int]:
    """PNG width and height from its IHDR chunk."""
    with path.open('rb') as f:
        if f.read(8) != b'\x89PNG\r\n\x1a\n':
            msg = f'Not a PNG: {path.name}'
            raise ValueError(msg)
        f.seek(16)
        return struct.unpack('>ii', f.read(8))


def collect_pngs(out_dir: Path) -> list[Path]:
    return sorted(out_dir.rglob('img/sec*/*.png'))


def png_matches_label(p: Path, label: str) -> bool:
    return p.name.lower().endswith(f'-{label.lower().replace(" ", "-")}.png')


def section_md(out_dir: Path, prefix: str) -> Path | None:
    return next(iter((out_dir / 'md').glob(f'{prefix}-*.md')), None)
