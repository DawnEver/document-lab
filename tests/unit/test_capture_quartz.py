"""Which windows macOS's window list offers as programs' windows, and which one a wheel lands on, read from a census a
runner reported."""

import pytest

from document_lab.capture import quartz
from document_lab.capture.screen import Region, Window

_WINDOW = Region(200, 200, 360, 332)


def _info(number: int, owner: str, layer: int, title: str, bounds: Region = _WINDOW) -> quartz._Info:
    return quartz._Info(number, 100 + number, layer, 1.0, title, owner, bounds)


# macos-latest, 2026-11-27: the test viewer (Tk, kept on top) at layer 19, under the Dock -- whose window spans the
# whole screen, transparent but where the Dock is drawn -- the menu bar and its items.
_CENSUS = [
    _info(29, 'Control Center', 25, 'Item-0'),
    _info(16, 'Window Server', 24, 'Menubar'),
    _info(6, 'Dock', 20, 'Dock', Region(0, 0, 1024, 768)),
    _info(40, 'Python', 19, 'document-lab capture fixture'),
    _info(41, 'Notes', 3, 'floating note'),
    _info(42, 'Finder', 0, 'Documents'),
]
_VIEWER = Window(40, 'document-lab capture fixture')


def test_a_window_tk_keeps_on_top_is_a_program_window_and_the_dock_is_not(monkeypatch):
    monkeypatch.setattr(quartz, '_infos', lambda *_: _CENSUS)
    assert [i.title for i in quartz._shown()] == ['document-lab capture fixture', 'floating note', 'Documents']


def test_a_wheel_goes_where_the_window_server_routes_it_not_to_the_screen_wide_dock(monkeypatch):
    """The Dock's window holds every point of the screen; the window server's hit test passes through it."""
    monkeypatch.setattr(quartz, '_infos', lambda *_: _CENSUS)
    monkeypatch.setattr(quartz, '_hit', lambda *_: 40)
    quartz._require_on_top(_VIEWER, 380, 382)


@pytest.mark.parametrize(('hit', 'named'), [(41, "'floating note'"), (0, 'nothing')])
def test_a_wheel_the_window_server_routes_elsewhere_is_refused_naming_where(monkeypatch, hit, named):
    monkeypatch.setattr(quartz, '_infos', lambda *_: _CENSUS)
    monkeypatch.setattr(quartz, '_hit', lambda *_: hit)
    with pytest.raises(OSError, match=f'^{named}.* covers the middle of'):
        quartz._require_on_top(_VIEWER, 380, 382)
