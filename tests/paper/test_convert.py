import sys
from pathlib import Path

from document_lab.paper.convert import choose_tool, detect_gpu_vram_gb, find_marker_single

MARKER = 'marker_single.exe' if sys.platform == 'win32' else 'marker_single'


class TestDetectGpuVram:
    def test_returns_float(self):
        result = detect_gpu_vram_gb()
        assert isinstance(result, float)

    def test_returns_non_negative(self):
        result = detect_gpu_vram_gb()
        assert result >= 0.0


class TestChooseTool:
    def test_returns_valid_tool(self):
        tool = choose_tool()
        assert tool in ('marker', 'ir')


class TestFindMarkerSingle:
    @staticmethod
    def _executable(folder):
        folder.mkdir(parents=True)
        made = folder / MARKER
        made.write_text('')
        made.chmod(0o755)
        return made

    def test_this_interpreters_own_wins_over_one_on_path(self, tmp_path, monkeypatch):
        own = self._executable(tmp_path / 'env' / 'Scripts')
        self._executable(tmp_path / 'tool')
        monkeypatch.setattr(sys, 'executable', str(own.parent / 'python'))
        monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'env'))
        monkeypatch.setenv('PATH', str(tmp_path / 'tool'))
        assert Path(find_marker_single() or '') == own

    def test_the_one_on_path_when_this_environment_has_none(self, tmp_path, monkeypatch):
        (tmp_path / 'env' / 'Scripts').mkdir(parents=True)
        on_path = self._executable(tmp_path / 'tool')
        monkeypatch.setattr(sys, 'executable', str(tmp_path / 'env' / 'Scripts' / 'python'))
        monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'env'))
        monkeypatch.setenv('PATH', str(tmp_path / 'tool'))
        found = find_marker_single()
        assert found is not None
        assert Path(found).samefile(on_path)
