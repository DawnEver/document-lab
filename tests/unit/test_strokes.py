"""Dashed thick strokes joined into curves, and a dash-dot curve told from a dashed one: synthetic PDFs, known truth."""

import numpy as np
import pymupdf
import pytest

from document_lab.ir import BBox
from document_lab.pdf import read_raster
from document_lab.strokes import Fragment, bands, centre_line, coverage, dots, fragments, link, opening, split_by_dots

STROKE = 1.5  # pt, the curves
HAIR = 0.3  # pt, the grid
CLIP = BBox(40, 40, 360, 260)
DASH = '[5 3] 0'
DASH_DOT = '[5 2.5 0 2.5] 0'  # a zero-length dash with a round cap is a dot of one stroke


def _line(a, b):
    return lambda x: a + b * x


def _pdf(tmp_path, curves, *, grid=True):
    """``curves``: ``(fn, x0, x1, dashes)`` in page pt; a hairline grid underneath."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    if grid:
        for v in range(50, 360, 20):
            page.draw_line((v, 40), (v, 260), width=HAIR)
        for v in range(50, 260, 20):
            page.draw_line((40, v), (360, v), width=HAIR)
    for fn, x0, x1, dashes in curves:
        xs = np.linspace(x0, x1, 200)
        page.draw_polyline([(float(x), float(fn(x))) for x in xs], width=STROKE, dashes=dashes, lineCap=1)
    path = tmp_path / 'strokes.pdf'
    doc.save(path)
    return read_raster(path, 1, CLIP, dpi=400)


def _owner(frag, fns):
    """The index of the curve every sample of ``frag`` lies on (within a stroke), else None."""
    on = [k for k, fn in enumerate(fns) if np.all(np.abs(frag.y - fn(frag.x)) < STROKE)]
    return on[0] if len(on) == 1 else None


def test_the_opening_keeps_a_square_that_fits_and_drops_a_thinner_line():
    ink = np.zeros((20, 20), bool)
    ink[2:8, 2:8] = True  # 6 x 6
    ink[12, :] = True  # a 1 px hairline
    out = opening(ink, 3)
    assert out[2:8, 2:8].all()
    assert not out[12].any()
    assert out.sum() == 36


def test_the_grid_is_opened_away_and_a_dashed_curve_comes_out_as_its_dashes(tmp_path):
    fn = _line(200, -0.3)
    frags = fragments(_pdf(tmp_path, [(fn, 60, 340, DASH)]), opening_pt=0.9, min_pt=1.0)
    assert len(frags) >= 30
    for f in frags:
        assert np.all(np.abs(f.y - fn(f.x)) < STROKE / 3)  # the centre line, not an edge (a round cap skews its end)
        assert f.stroke == pytest.approx(STROKE, abs=0.25)


def test_a_dashed_curve_links_into_one_chain_that_is_not_solid(tmp_path):
    fn = _line(200, -0.3)
    frags = fragments(_pdf(tmp_path, [(fn, 60, 340, DASH)]), opening_pt=0.9, min_pt=1.0)
    chains = link(frags)
    assert len(chains) == 1
    assert coverage(frags, chains[0]) < 0.8
    x, y = centre_line(frags, chains[0])
    assert np.all(np.abs(y - fn(x)) < 0.3)


def test_planted_control_two_crossing_dashed_curves_are_not_merged(tmp_path):
    fns = [_line(220, -0.5), _line(60, 0.5)]  # they cross at x = 160
    raster = _pdf(tmp_path, [(fns[0], 60, 340, DASH), (fns[1], 60, 340, DASH)])
    frags = fragments(raster, opening_pt=0.9, min_pt=1.0)
    owners = [_owner(f, fns) for f in frags]
    chains = link(frags)
    assert all(len({owners[i] for i in ch} - {None}) == 1 for ch in chains if len(ch) > 1)  # no chain crosses over
    # Away from the crossing (where the two strokes share ink), each curve's dashes on each side are ONE chain.
    sides = {}
    for ch in chains:
        for i in ch:
            if owners[i] is not None and abs(frags[i].x.mean() - 160) > 12:
                sides.setdefault((owners[i], frags[i].x.mean() > 160), set()).add(tuple(ch))
    assert sorted(sides) == [(0, False), (0, True), (1, False), (1, True)]
    assert all(len(v) == 1 for v in sides.values())
    # The check can say no: a chain spliced at the crossing (curve 0 on the left, curve 1 on the right) is caught.
    (left,), (right,) = sides[0, False], sides[1, True]
    spliced = [i for i in left if frags[i].x.mean() < 160] + [i for i in right if frags[i].x.mean() > 160]
    assert len({owners[i] for i in spliced} - {None}) > 1


def test_dashes_join_the_band_of_the_solid_curve_above_them(tmp_path):
    solid_fns = [_line(80, 0.2), _line(160, 0.2)]
    dash_fns = [_line(86, 0.2), _line(166, 0.2)]  # 6 pt below each solid
    raster = _pdf(tmp_path, [(f, 60, 340, None) for f in solid_fns] + [(f, 60, 340, DASH) for f in dash_fns])
    frags = fragments(raster, opening_pt=0.9, min_pt=1.0)
    chains = link(frags)
    solids = sorted((ch for ch in chains if coverage(frags, ch) > 0.9), key=lambda ch: frags[ch[0]].y.mean())
    assert len(solids) == 2
    members, orphans = bands(frags, solids, side='above')
    assert orphans == []
    for k, idx in enumerate(members):
        assert idx
        assert all(_owner(frags[i], dash_fns) == k for i in idx)
    # Planted control: asked for the solid BELOW, the upper dashes go to the lower band.
    wrong, _ = bands(frags, solids, side='below', reach_pt=100.0)
    assert any(_owner(frags[i], dash_fns) == 0 for i in wrong[1])


def _pair(tmp_path, apart):
    """A dashed and a dash-dot curve ``apart`` pt apart, read with the dash opening and with a dot opening."""
    fns = [_line(150, -0.25), _line(150 + apart, -0.25)]
    raster = _pdf(tmp_path, [(fns[0], 60, 340, DASH), (fns[1], 60, 340, DASH_DOT)])
    return fns, raster


def test_the_dash_opening_loses_nearly_every_dot_and_a_smaller_one_keeps_them(tmp_path):
    _, raster = _pair(tmp_path, 8.0)

    def inside(fs):  # a curve's last dash may be cut short at its end: not a dot
        return [f for f in fs if 65 < f.x.mean() < 335]

    assert len(inside(dots(fragments(raster, opening_pt=1.3, min_pt=0.5)))) <= 2
    kept = inside(dots(fragments(raster, opening_pt=0.6, min_pt=0.5)))
    assert len(kept) >= 20


@pytest.mark.parametrize('apart', [8.0, 3.0, -3.0])  # pt; 3 is two strokes, the N72 70/100 degC spacing
def test_a_dash_dot_curve_is_told_from_a_dashed_one_by_its_dots(tmp_path, apart):
    fns, raster = _pair(tmp_path, apart)  # fns[0] dashed, fns[1] dash-dot
    dashes = fragments(raster, opening_pt=1.3, min_pt=2.5)
    found = [f for f in dots(fragments(raster, opening_pt=0.6, min_pt=0.8)) if 65 < f.x.mean() < 335]
    on, off = split_by_dots(dashes, found)
    assert on
    assert off
    assert all(_owner(dashes[i], fns) == 1 for i in on)
    assert all(_owner(dashes[i], fns) == 0 for i in off if _owner(dashes[i], fns) is not None)
    # Planted control: the same split traced by the DASHED curve's line puts its dashes on, so the dots are what
    # names the curve, not the order of the two curves on the page.
    stand_in = [Fragment(np.array([x]), np.array([fns[0](x)]), STROKE) for x in (70.0, 200.0, 330.0)]
    swapped, _ = split_by_dots(dashes, stand_in)
    assert swapped
    assert all(_owner(dashes[i], fns) == 0 for i in swapped)


def test_a_split_needs_two_dots():
    with pytest.raises(ValueError, match='two or more'):
        split_by_dots([], [Fragment(np.array([1.0]), np.array([1.0]), STROKE)])


def test_a_dot_has_no_direction_so_a_dash_dot_curve_links_through_its_dots(tmp_path):
    """Read with the dot opening, a sloped dash-dot curve is dash, dot, dash ... at short gaps: one chain."""
    fn = _line(220, -0.5)
    frags = fragments(_pdf(tmp_path, [(fn, 60, 300, DASH_DOT)]), opening_pt=0.6, min_pt=0.8)  # a dot is ~1.1 pt
    inside = [f for f in frags if 65 < f.x.mean() < 295]
    assert len(dots(inside)) >= 15
    chains = link(inside, gap_pt=3.5, off_pt=1.0)
    assert len(chains) == 1


def test_an_end_is_placed_on_the_stroke_line_not_on_its_cap_corner(tmp_path):
    """A steep dashed curve with long gaps links at a tight miss: each end's y is the fitted line's, not the corner a
    column cut through the cap samples (that corner sits up to half a stroke off the line)."""
    fn = _line(250, -0.65)
    frags = fragments(_pdf(tmp_path, [(fn, 60, 300, '[5.3 7] 0')]), opening_pt=0.9, min_pt=1.0)
    inside = [f for f in frags if 65 < f.x.mean() < 295]
    assert len(inside) >= 15
    assert len(link(inside, gap_pt=8.5, off_pt=0.5)) == 1
