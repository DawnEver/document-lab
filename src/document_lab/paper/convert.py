"""A paper PDF to markdown: marker-pdf on a GPU (the ``gpu`` extra), pymupdf4llm on CPU (the ``paper`` extra)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final, Literal

import pymupdf

__all__ = [
    'MARKER_MIN_VRAM_GB',
    'Tool',
    'augment_markdown_with_formulas',
    'choose_tool',
    'convert',
    'detect_formula_regions',
    'detect_gpu_vram_gb',
    'find_marker_single',
]

Tool = Literal['marker', 'pymupdf4llm']

#: marker-pdf is chosen only with at least this much VRAM; below it pymupdf4llm on CPU is faster.
MARKER_MIN_VRAM_GB: Final = 4.0
_MATH_CHARS: Final = '=+-*/()[]{}^_∫∑∏√∂∞≈≠≤≥±×÷'  # noqa: RUF001 -- these ARE the math glyphs counted


def detect_gpu_vram_gb() -> float:
    """VRAM of the first CUDA GPU in GB, else 0.

    macOS returns 0: surya (marker's OCR) does not run on MPS and falls back to CPU, where pymupdf4llm is faster.
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
    """Marker on a GPU with enough VRAM when it is installed, else pymupdf4llm."""
    if detect_gpu_vram_gb() >= MARKER_MIN_VRAM_GB and find_marker_single():
        return 'marker'
    return 'pymupdf4llm'


def convert(pdf: Path, out_dir: Path, *, tool: Tool | None = None) -> tuple[str, Tool]:
    """The markdown of ``pdf`` and the tool used (``tool``, else :func:`choose_tool`).

    The converter's images land in ``out_dir/img/flat``.
    """
    used = tool or choose_tool()
    flat = out_dir / 'img' / 'flat'
    flat.mkdir(parents=True, exist_ok=True)
    return (_run_marker(pdf, out_dir) if used == 'marker' else _run_pymupdf4llm(pdf, flat)), used


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


def _run_pymupdf4llm(pdf: Path, flat: Path) -> str:
    try:
        import pymupdf4llm  # noqa: PLC0415 -- the optional `paper` extra, imported only when converting
    except ImportError as missing:
        msg = "converting a paper needs the 'paper' extra: install document-lab[paper]"
        raise ImportError(msg) from missing
    return str(pymupdf4llm.to_markdown(str(pdf), write_images=True, image_path=str(flat)))


def detect_formula_regions(pdf_path: Path) -> list[dict]:
    """Formula-like text blocks of every page: short blocks with at least two math symbols.

    Each is ``{'page': 0-based, 'bbox': (x0, y0, x1, y1), 'text': str}``.
    """
    formulas: list[dict] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc):
            for blk in page.get_text('dict', flags=pymupdf.TEXT_PRESERVE_WHITESPACE)['blocks']:
                if blk['type'] != 0:
                    continue
                text = ''.join(s.get('text', '') for line in blk.get('lines', []) for s in line.get('spans', []))
                text = text.strip()
                if text and len(text) < 200 and sum(c in _MATH_CHARS for c in text) >= 2:
                    formulas.append({'page': page_num, 'bbox': tuple(blk['bbox']), 'text': text})
    return formulas


def augment_markdown_with_formulas(md_text: str, pdf_path: Path) -> str:
    """After each formula image of the markdown, a ```math block with the formula's text from the PDF."""
    page_formulas: dict[int, list[str]] = {}
    for f in detect_formula_regions(pdf_path):
        page_formulas.setdefault(f['page'], []).append(f['text'])
    if not page_formulas:
        return md_text
    result: list[str] = []
    for line in md_text.split('\n'):
        result.append(line)
        # A formula image: short alt text, a file name carrying its 1-based page number.
        img = re.match(r'!\[([^\]]{0,20})\]\(([^)]*?(\d{4})[^)]*)\)', line.strip())
        if not img or len(img.group(1)) > 10:
            continue
        queue = page_formulas.get(int(img.group(3)) - 1)
        if queue:
            text = queue.pop(0)
            if text != img.group(1):
                result.append(f'```math\n{text}\n```')
    return '\n'.join(result)
