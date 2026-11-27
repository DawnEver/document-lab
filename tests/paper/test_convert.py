import sys
from pathlib import Path

import pymupdf
import pytest

from document_lab.paper.convert import choose_tool, convert, detect_gpu_vram_gb, find_marker_single

MARKER = 'marker_single.exe' if sys.platform == 'win32' else 'marker_single'


class Host:
    """A host whose PATH holds only the commands a test puts on it, and whose interpreter has no marker beside it."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bin = root / 'bin'
        self.bin.mkdir()
        (root / 'env' / 'Scripts').mkdir(parents=True)
        monkeypatch.setattr(sys, 'executable', str(root / 'env' / 'Scripts' / 'python'))
        monkeypatch.setattr(sys, 'prefix', str(root / 'env'))
        monkeypatch.setenv('PATH', str(self.bin))
        if sys.platform == 'darwin':  # detect_gpu_vram_gb reads no GPU on macOS by design
            monkeypatch.setattr(sys, 'platform', 'linux')

    def put(self, name: str, *, prints: str = '', exits: int = 0) -> Path:
        """A command ``name`` on PATH that prints ``prints`` and exits ``exits``."""
        if sys.platform == 'win32':
            made = self.bin / f'{name}.bat'
            made.write_text(f'@echo off\r\necho {prints}\r\nexit /b {exits}\r\n', encoding='ascii')
        else:
            made = self.bin / name
            made.write_text(f'#!/bin/sh\necho "{prints}"\nexit {exits}\n', encoding='ascii')
            made.chmod(0o755)
        return made

    def gpu(self, mib: int = 24467) -> None:
        self.put('nvidia-smi', prints=str(mib))


def _pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), 'A line of text', fontsize=10)
    doc.save(path)
    return path


@pytest.fixture
def host(tmp_path, monkeypatch):
    return Host(tmp_path, monkeypatch)


class TestDetectGpuVram:
    def test_returns_float(self):
        result = detect_gpu_vram_gb()
        assert isinstance(result, float)

    def test_returns_non_negative(self):
        result = detect_gpu_vram_gb()
        assert result >= 0.0

    def test_reads_nvidia_smi(self, host):
        host.gpu(8192)
        assert detect_gpu_vram_gb() == 8.0


class TestChooseTool:
    def test_returns_valid_tool(self):
        assert choose_tool().tool in ('marker', 'ir')

    def test_ir_without_a_gpu_says_so(self, host):
        choice = choose_tool()
        assert choice.tool == 'ir'
        assert 'no CUDA GPU with >= 4 GB VRAM' in choice.why

    def test_ir_on_a_small_gpu_says_so(self, host):
        host.gpu(2048)
        host.put('marker_single')
        host.put('docker')
        choice = choose_tool()
        assert choice.tool == 'ir'
        assert '2.0 GB' in choice.why

    def test_ir_without_marker_says_so(self, host):
        host.gpu()
        host.put('docker')
        choice = choose_tool()
        assert choice.tool == 'ir'
        assert 'marker_single not found' in choice.why

    def test_ir_without_docker_names_the_missing_backend(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('llama-server')  # surya picks vllm on an NVIDIA GPU: llama-server alone is not its backend
        choice = choose_tool()
        assert choice.tool == 'ir'
        assert 'docker is not on PATH' in choice.why

    def test_ir_when_the_docker_daemon_does_not_answer(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('docker', prints='Cannot connect to the Docker daemon', exits=1)
        choice = choose_tool()
        assert choice.tool == 'ir'
        assert 'daemon does not answer' in choice.why

    def test_marker_with_a_gpu_marker_and_a_docker_daemon(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('docker', prints='29.1.0')
        choice = choose_tool()
        assert choice.tool == 'marker'
        assert '23.9 GB' in choice.why


class TestConvert:
    def test_reports_why_it_chose(self, host, tmp_path):
        pdf = _pdf(tmp_path / 'paper.pdf')
        text, choice = convert(pdf, tmp_path / 'out')
        assert text
        assert choice.tool == 'ir'
        assert 'no CUDA GPU' in choice.why

    def test_a_forced_tool_says_it_was_forced(self, tmp_path):
        pdf = _pdf(tmp_path / 'paper.pdf')
        _, choice = convert(pdf, tmp_path / 'out', tool='ir')
        assert choice.tool == 'ir'
        assert choice.why == 'forced by the caller'


class TestFindMarkerSingle:
    @staticmethod
    def _executable(folder):
        folder.mkdir(parents=True, exist_ok=True)
        made = folder / MARKER
        made.write_text('')
        made.chmod(0o755)
        return made

    def test_this_interpreters_own_wins_over_one_on_path(self, host, tmp_path):
        own = self._executable(tmp_path / 'env' / 'Scripts')
        host.put('marker_single')
        assert Path(find_marker_single() or '') == own

    def test_the_one_on_path_when_this_environment_has_none(self, host):
        on_path = host.put('marker_single')
        found = find_marker_single()
        assert found is not None
        assert Path(found).samefile(on_path)
