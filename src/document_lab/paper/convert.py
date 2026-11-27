"""A paper PDF to markdown: document_lab's own IR by default; marker-pdf (the ``gpu`` extra) only when asked.

marker is never chosen for the caller: surya 0.23, its OCR, needs a model server (vllm in Docker on an NVIDIA GPU, else
a ``llama-server``), and marker is GPL-3.0 with model weights under a non-commercial license. Its remaining value over
the IR path is equations as LaTeX and an ML layout, so it stays an option (``tool='marker'``) that refuses by name
when its server is missing (D54).

The IR path writes the paper without its running heads and feet (`document_lab.layout.without_running`): a journal's
name and a page number repeated at every page's edge are no part of its text.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Collection

from document_lab import layout
from document_lab.markdown import Headings, headings, to_markdown
from document_lab.pdf import read_pdf
from document_lab.tags import DEFAULT_USES

__all__ = [
    'MarkerUnavailable',
    'Tool',
    'ToolChoice',
    'choose_tool',
    'convert',
    'find_marker_single',
    'has_nvidia_gpu',
    'missing_surya_backend',
]

Tool = Literal['marker', 'ir']

#: Seconds ``docker info`` or ``nvidia-smi -L`` may take to answer.
PROBE_S: Final = 15.0

#: The device node an NVIDIA driver makes on Linux: surya's first sign of an NVIDIA GPU.
NVIDIA_DEVICE: Final = Path('/dev/nvidia0')


class MarkerUnavailable(RuntimeError):
    """The caller asked for marker and something it needs is missing; the message names it."""


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """The converter a run uses and why."""

    tool: Tool
    why: str


def has_nvidia_gpu() -> bool:
    """An NVIDIA GPU is on this host, by surya's own test: the device node, else ``nvidia-smi -L`` listing one."""
    if NVIDIA_DEVICE.exists():
        return True
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return False
    try:
        listed = subprocess.run([smi, '-L'], capture_output=True, text=True, timeout=PROBE_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return listed.returncode == 0 and 'GPU' in listed.stdout


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

    surya 0.23 runs its model in an inference server it picks itself: vllm on an NVIDIA GPU, which it starts as a
    Docker container (so ``docker`` on PATH whose daemon answers), else llama.cpp's ``llama-server`` on PATH. Its own
    settings can name another (``SURYA_INFERENCE_BACKEND``, ``SURYA_INFERENCE_URL``, ``LLAMA_CPP_BINARY``); they
    live in the environment, which this module does not read, so this mirrors surya's default.
    """
    if not has_nvidia_gpu():
        if shutil.which('llama-server') is None:
            return (
                "surya (marker's OCR) serves its model through llama.cpp without an NVIDIA GPU, "
                'and llama-server is not on PATH'
            )
        return None
    lead = "surya (marker's OCR) serves its model through vllm in Docker on an NVIDIA GPU"
    docker = shutil.which('docker')
    if docker is None:
        return f'{lead}, and docker is not on PATH'
    try:
        subprocess.run([docker, 'info'], capture_output=True, timeout=PROBE_S, check=True)
    except (OSError, subprocess.SubprocessError):
        return f'{lead}, and the docker daemon does not answer `docker info` ({docker})'
    return None


def choose_tool(tool: Tool | None = None) -> ToolChoice:
    """The IR path unless the caller names marker, which refuses by name when its executable or server is missing.

    Raises:
        MarkerUnavailable: ``tool='marker'`` and the gpu extra or surya's model server is missing.

    """
    if tool is None:
        return ToolChoice('ir', 'the default')
    if tool == 'ir':
        return ToolChoice('ir', 'asked by the caller')
    marker = find_marker_single()
    if marker is None:
        msg = 'marker asked for, and marker_single is not found: install the gpu extra (document-lab[gpu])'
        raise MarkerUnavailable(msg)
    missing = missing_surya_backend()
    if missing is not None:
        msg = f'marker asked for, and {missing}'
        raise MarkerUnavailable(msg)
    return ToolChoice('marker', f'asked by the caller ({marker})')


def convert(
    pdf: Path,
    out_dir: Path,
    *,
    tool: Tool | None = None,
    outline: bool = True,
    tagged: Collection[str] = DEFAULT_USES,
    include_in_background_color: bool = False,
) -> tuple[str, ToolChoice, Headings | None]:
    """The markdown of ``pdf``, its :func:`choose_tool` choice of ``tool``, and how its sections were headed.

    On the IR path the PDF's stated outline names the sections when it matches the pages and ``outline`` asks for it,
    else its tags and the rules head them (`document_lab.markdown.headings`, returned for the caller to report);
    marker heads its own (None). ``tagged``: on the IR path, what a tagged PDF's tags give its markdown
    (`document_lab.markdown`; none infers it all). marker's images land in ``out_dir/img/flat``; the IR path writes
    none (figures are cropped later). ``include_in_background_color``: the IR path writes the text filled in the color
    under it too (see `document_lab.ir`).

    Raises:
        MarkerUnavailable: ``tool='marker'`` and something marker needs is missing.

    """
    choice = choose_tool(tool)
    flat = out_dir / 'img' / 'flat'
    flat.mkdir(parents=True, exist_ok=True)
    if choice.tool == 'marker':
        return _run_marker(pdf, out_dir), choice, None
    doc = layout.without_running(read_pdf(pdf))
    sections = headings(doc, outline=outline, tagged=tagged, include_in_background_color=include_in_background_color)
    return (
        to_markdown(doc, tagged=tagged, sections=sections, include_in_background_color=include_in_background_color),
        choice,
        sections,
    )


def _run_marker(pdf: Path, out_dir: Path) -> str:
    marker = find_marker_single()
    if marker is None:
        msg = 'marker_single not found: install the gpu extra (document-lab[gpu])'
        raise MarkerUnavailable(msg)
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
