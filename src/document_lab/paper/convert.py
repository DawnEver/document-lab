"""A paper PDF to markdown: marker-pdf on a GPU (the ``gpu`` extra), else document_lab's own IR on CPU."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final, Literal

from document_lab.paper.markdown import to_markdown
from document_lab.pdf import read_pdf

__all__ = [
    'MARKER_MIN_VRAM_GB',
    'Tool',
    'choose_tool',
    'convert',
    'detect_gpu_vram_gb',
    'find_marker_single',
]

Tool = Literal['marker', 'pymupdf']

#: marker-pdf is chosen only with at least this much VRAM; below it the CPU path is faster.
MARKER_MIN_VRAM_GB: Final = 4.0


def detect_gpu_vram_gb() -> float:
    """VRAM of the first CUDA GPU in GB, else 0.

    macOS returns 0: surya (marker's OCR) does not run on MPS and falls back to CPU, where the IR path is faster.
    """
    if sys.platform == 'darwin':
        return 0.0
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return 0.0
    try:
        out = subprocess.check_output(
            [smi, '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True, timeout=5
        )
        return float(out.strip().splitlines()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0.0


def find_marker_single() -> str | None:
    """The ``marker_single`` executable on PATH or beside this interpreter, else ``None``."""
    found = shutil.which('marker_single')
    if found:
        return found
    name = 'marker_single.exe' if sys.platform == 'win32' else 'marker_single'
    for folder in (Path(sys.executable).parent, Path(sys.prefix) / 'bin'):
        if (folder / name).is_file():
            return str(folder / name)
    return None


def choose_tool() -> Tool:
    """Marker on a GPU with enough VRAM when it is installed, else the IR path (``pymupdf``)."""
    if detect_gpu_vram_gb() >= MARKER_MIN_VRAM_GB and find_marker_single():
        return 'marker'
    return 'pymupdf'


def convert(pdf: Path, out_dir: Path, *, tool: Tool | None = None) -> tuple[str, Tool]:
    """The markdown of ``pdf`` and the tool used (``tool``, else :func:`choose_tool`).

    marker's images land in ``out_dir/img/flat``; the IR path writes none (figures are cropped later).
    """
    used = tool or choose_tool()
    flat = out_dir / 'img' / 'flat'
    flat.mkdir(parents=True, exist_ok=True)
    return (_run_marker(pdf, out_dir) if used == 'marker' else to_markdown(read_pdf(pdf, tables=False))), used


def _run_marker(pdf: Path, out_dir: Path) -> str:
    marker = find_marker_single()
    if marker is None:
        msg = 'marker_single not found: install the gpu extra (document-lab[gpu])'
        raise FileNotFoundError(msg)
    tmp = out_dir / '_marker_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([marker, str(pdf), '--output_format', 'markdown', '--output_dir', str(tmp)], check=True)
        md_files = list(tmp.rglob('*.md'))
        if not md_files:
            msg = 'marker produced no markdown'
            raise RuntimeError(msg)
        text = md_files[0].read_text(encoding='utf-8')
        for img in (p for ext in ('*.png', '*.jpeg', '*.jpg') for p in tmp.rglob(ext)):
            shutil.copy2(img, out_dir / 'img' / 'flat' / img.name)
        return text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
