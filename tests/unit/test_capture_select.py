"""Editing the marked box: an edge, a corner or the move handle dragged by an amount, never collapsing it."""

import pytest

from document_lab.capture.desktop import Region, _edited

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
