"""Which windows macOS's window list offers as programs' windows, read from a census a runner reported."""

from document_lab.capture import quartz
from document_lab.capture.screen import Region


def _info(number: int, owner: str, layer: int, title: str) -> quartz._Info:
    return quartz._Info(number, 100 + number, layer, 1.0, title, owner, Region(0, 0, 360, 332))


# macos-latest, 2026-11-27: the test viewer (Tk, kept on top) at layer 19, under the Dock, the menu bar and its items.
_CENSUS = [
    _info(29, 'Control Center', 25, 'Item-0'),
    _info(16, 'Window Server', 24, 'Menubar'),
    _info(6, 'Dock', 20, 'Dock'),
    _info(40, 'Python', 19, 'document-lab capture fixture'),
    _info(41, 'Notes', 3, 'floating note'),
    _info(42, 'Finder', 0, 'Documents'),
]


def test_a_window_tk_keeps_on_top_is_a_program_window_and_the_dock_is_not(monkeypatch):
    monkeypatch.setattr(quartz, '_infos', lambda *_: _CENSUS)
    assert [i.title for i in quartz._shown()] == ['document-lab capture fixture', 'floating note', 'Documents']
