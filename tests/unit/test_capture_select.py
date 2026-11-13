"""Editing the marked box: an edge, a corner or the move handle dragged by an amount, never collapsing it."""

import pytest

from document_lab.capture.desktop import Region, _edited, recall, remember, size_label

BOX = Region(100, 200, 300, 150)


@pytest.mark.parametrize(
    ('held', 'dx', 'dy', 'expected'),
    [
        ('e', 50, 99, Region(100, 200, 350, 150)),  # only the right edge moves, and only sideways
        ('w', -20, 0, Region(80, 200, 320, 150)),
        ('n', 0, 30, Region(100, 230, 300, 120)),
        ('s', 0, 40, Region(100, 200, 300, 190)),
        ('se', 10, 20, Region(100, 200, 310, 170)),
        ('nw', -10, -20, Region(90, 180, 310, 170)),
        ('move', 7, -9, Region(107, 191, 300, 150)),
    ],
)
def test_a_held_part_moves_by_the_drag(held, dx, dy, expected):
    assert _edited(held, BOX, dx, dy) == expected


def test_an_edge_dragged_past_the_other_stops_short_of_collapsing():
    assert _edited('e', BOX, -1000, 0).width == 8
    assert _edited('n', BOX, 0, 1000).height == 8


SCREEN = Region(-1920, 0, 3840, 1080)  # two monitors, the second left of the primary


def test_the_last_area_confirmed_is_recalled_while_it_is_on_the_screen(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert recall(SCREEN) is None  # nothing confirmed yet
    remember(Region(-500, 100, 800, 600))
    remember(BOX)  # the last one wins
    assert recall(SCREEN) == BOX
    assert recall(Region(0, 0, 350, 300)) is None  # a smaller screen no longer holds it: not opened on


def test_a_memory_this_code_did_not_write_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    (tmp_path / 'document-lab').mkdir()
    for junk in ('{', '{"left": 1}', '[1, 2]'):
        (tmp_path / 'document-lab' / 'capture-area.json').write_text(junk, encoding='utf-8')
        assert recall(SCREEN) is None


def test_the_size_shows_above_the_marks_corner_or_inside_it_at_the_screens_top():
    screen = Region(0, 0, 1920, 1080)
    assert size_label(BOX, screen) == ('300 x 150', 100, 174)
    assert size_label(Region(100, 10, 300, 150), screen) == ('300 x 150', 100, 14)
    assert size_label(Region(-1900, -1070, 40, 30), Region(-1920, -1080, 3840, 2160))[2] == -1066
