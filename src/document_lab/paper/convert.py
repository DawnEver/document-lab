"""A paper PDF to markdown: marker-pdf on a GPU (the ``gpu`` extra), else document_lab's own IR on CPU."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from document_lab.markdown import to_markdown
from document_lab.pdf import read_pdf

__all__ = [
    'MARKER_MIN_VRAM_GB',
    'Tool',
    'ToolChoice',
    'choose_tool',
    'convert',
    'detect_gpu_vram_gb',
    'find_marker_single',
    'missing_surya_backend',
]

Tool = Literal['marker', 'ir']

#: marker-pdf is chosen only with at least this much VRAM; below it the CPU path is faster.
MARKER_MIN_VRAM_GB: Final = 4.0

#: Seconds ``docker info`` may take to show its daemon answers.
DOCKER_PROBE_S: Final = 15.0


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """The converter a run uses and why: a fallback to the IR path always names what marker lacked."""

    tool: Tool
    why: str


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
    """The ``marker_single`` beside this interpreter (the ``gpu`` extra's), else the one on PATH, else ``None``.

    This environment's own comes first: one on PATH may be another install (a ``uv tool``) with its own marker and
    torch, which this package neither pins nor tests.
    """
    name = 'marker_single.exe' if sys.platform == 'win32' else 'marker_single'
    for folder in (Path(sys.executable).parent, Path(sys.prefix) / 'bin'):
        if (folder / name).is_file():
            return str(folder / name)
    return shutil.which('marker_single')


def missing_surya_backend() -> str | None:
    """Why surya, marker's OCR, has no model server on this host, else ``None``.

    surya 0.23 runs its model in an inference server it picks itself: on an NVIDIA GPU -- the only host marker is
    chosen on -- that is vllm, which it starts as a Docker container, so ``docker`` must be on PATH and its daemon
    must answer. surya's own settings can name another server (``SURYA_INFERENCE_BACKEND=llamacpp`` with a
    ``llama-server``, or ``SURYA_INFERENCE_URL``); they live in the environment, which this module does not read,
    so a run set up that way forces marker with ``tool='marker'``.
    """
    lead = "surya (marker's OCR) serves its model through vllm in Docker on an NVIDIA GPU"
    docker = shutil.which('docker')
    if docker is None:
        return f'{lead}, and docker is not on PATH'
    try:
        subprocess.run([docker, 'info'], capture_output=True, timeout=DOCKER_PROBE_S, check=True)
    except (OSError, subprocess.SubprocessError):
        return f'{lead}, and the docker daemon does not answer `docker info` ({docker})'
    return None


def choose_tool() -> ToolChoice:
    """Marker when a GPU with enough VRAM, ``marker_single`` and surya's model server are all here, else the IR path.

    Each fallback says which of the three was missing.
    """
    vram = detect_gpu_vram_gb()
    if vram < MARKER_MIN_VRAM_GB:
        return ToolChoice('ir', f'no CUDA GPU with >= {MARKER_MIN_VRAM_GB:g} GB VRAM (found {vram:.1f} GB)')
    marker = find_marker_single()
    if marker is None:
        return ToolChoice('ir', 'marker_single not found: install the gpu extra (document-lab[gpu])')
    missing = missing_surya_backend()
    if missing is not None:
        return ToolChoice('ir', missing)
    return ToolChoice('marker', f'a {vram:.1f} GB GPU, {marker}, and surya served by vllm in Docker')


def convert(pdf: Path, out_dir: Path, *, tool: Tool | None = None) -> tuple[str, ToolChoice]:
    """The markdown of ``pdf`` and the choice it was made by (``tool``, else :func:`choose_tool`).

    marker's images land in ``out_dir/img/flat``; the IR path writes none (figures are cropped later).
    """
    choice = ToolChoice(tool, 'forced by the caller') if tool else choose_tool()
    flat = out_dir / 'img' / 'flat'
    flat.mkdir(parents=True, exist_ok=True)
    return (_run_marker(pdf, out_dir) if choice.tool == 'marker' else to_markdown(read_pdf(pdf))), choice


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
