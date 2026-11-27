import sys
from pathlib import Path

import pymupdf
import pytest

from document_lab.__main__ import main
from document_lab.paper import convert as convert_module
from document_lab.paper.convert import MarkerUnavailable, choose_tool, convert, find_marker_single, has_nvidia_gpu

MARKER = 'marker_single.exe' if sys.platform == 'win32' else 'marker_single'


class Host:
    """A host whose PATH holds only the commands a test puts on it: no NVIDIA device node, no marker beside the
    interpreter."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bin = root / 'bin'
        self.bin.mkdir()
        (root / 'env' / 'Scripts').mkdir(parents=True)
        monkeypatch.setattr(sys, 'executable', str(root / 'env' / 'Scripts' / 'python'))
        monkeypatch.setattr(sys, 'prefix', str(root / 'env'))
        monkeypatch.setattr(convert_module, 'NVIDIA_DEVICE', root / 'dev' / 'nvidia0')
        monkeypatch.setenv('PATH', str(self.bin))

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

    def gpu(self) -> None:
        self.put('nvidia-smi', prints='GPU 0: NVIDIA RTX PRO 4000 Blackwell')


def _pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), 'A line of text', fontsize=10)
    doc.save(path)
    return path


@pytest.fixture
def host(tmp_path, monkeypatch):
    return Host(tmp_path, monkeypatch)


class TestHasNvidiaGpu:
    def test_none_without_nvidia_smi_or_device(self, host):
        assert not has_nvidia_gpu()

    def test_one_nvidia_smi_lists(self, host):
        host.gpu()
        assert has_nvidia_gpu()

    def test_none_when_nvidia_smi_fails(self, host):
        host.put('nvidia-smi', prints='NVIDIA-SMI has failed', exits=9)
        assert not has_nvidia_gpu()

    def test_the_device_node_alone(self, host, tmp_path):
        (tmp_path / 'dev').mkdir()
        (tmp_path / 'dev' / 'nvidia0').write_text('')
        assert has_nvidia_gpu()


class TestChooseTool:
    def test_the_ir_path_by_default_even_where_marker_could_run(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('docker', prints='29.1.0')
        assert choose_tool() == convert_module.ToolChoice('ir', 'the default')

    def test_the_ir_path_when_asked(self, host):
        assert choose_tool('ir') == convert_module.ToolChoice('ir', 'asked by the caller')

    def test_marker_refuses_without_marker_single(self, host):
        host.gpu()
        host.put('docker', prints='29.1.0')
        with pytest.raises(MarkerUnavailable, match='marker_single is not found'):
            choose_tool('marker')

    def test_marker_on_nvidia_refuses_without_docker(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('llama-server')  # surya picks vllm on an NVIDIA GPU: llama-server alone is not its server
        with pytest.raises(MarkerUnavailable, match='docker is not on PATH'):
            choose_tool('marker')

    def test_marker_on_nvidia_refuses_when_the_docker_daemon_does_not_answer(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('docker', prints='Cannot connect to the Docker daemon', exits=1)
        with pytest.raises(MarkerUnavailable, match='daemon does not answer'):
            choose_tool('marker')

    def test_marker_on_nvidia_with_a_docker_daemon(self, host):
        host.gpu()
        host.put('marker_single')
        host.put('docker', prints='29.1.0')
        choice = choose_tool('marker')
        assert choice.tool == 'marker'
        assert choice.why.startswith('asked by the caller')

    def test_marker_without_nvidia_refuses_without_llama_server(self, host):
        host.put('marker_single')
        host.put('docker', prints='29.1.0')  # vllm is surya's server on an NVIDIA GPU only
        with pytest.raises(MarkerUnavailable, match='llama-server is not on PATH'):
            choose_tool('marker')

    def test_marker_without_nvidia_with_llama_server(self, host):
        host.put('marker_single')
        host.put('llama-server')
        assert choose_tool('marker').tool == 'marker'


class TestConvert:
    def test_reports_the_default(self, host, tmp_path):
        text, choice = convert(_pdf(tmp_path / 'paper.pdf'), tmp_path / 'out')
        assert text
        assert choice == convert_module.ToolChoice('ir', 'the default')

    def test_marker_asked_for_and_missing_refuses_before_writing_markdown(self, host, tmp_path):
        with pytest.raises(MarkerUnavailable):
            convert(_pdf(tmp_path / 'paper.pdf'), tmp_path / 'out', tool='marker')


class TestPaperCommand:
    def test_marker_asked_for_and_missing_is_a_named_refusal(self, host, tmp_path, capsys):
        assert main(['paper', str(_pdf(tmp_path / 'paper.pdf')), str(tmp_path / 'out'), '--tool', 'marker']) == 2
        assert 'refused: marker asked for, and marker_single is not found' in capsys.readouterr().out


class TestFindMarkerSingle:
    def test_this_interpreters_own_wins_over_one_on_path(self, host, tmp_path):
        own = tmp_path / 'env' / 'Scripts' / MARKER
        own.write_text('')
        host.put('marker_single')
        assert Path(find_marker_single() or '') == own

    def test_the_one_on_path_when_this_environment_has_none(self, host):
        on_path = host.put('marker_single')
        found = find_marker_single()
        assert found is not None
        assert Path(found).samefile(on_path)
