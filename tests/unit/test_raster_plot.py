"""Curves recovered from RASTER plots: a known vector plot rendered to pixels, so every expected value is known."""

import numpy as np
import pytest

from document_lab.glyphs import UnreadableGlyph
from document_lab.ir import BBox
from document_lab.pdf import read_raster
from document_lab.plot import NoPlotFrame, UnreadableTickLabels, raster_plot
from tests.synthetic import BLUE, RED, Axis, PlotSpec, raster_pdf

XS = [float(x) for x in range(0, 101, 5)]


def _raster(tmp_path, spec, dpi):
    path, clip = raster_pdf(tmp_path / 'raster.pdf', spec, dpi)
    return read_raster(path, 1, BBox(*clip))


def _within_pixels(curve, fn, slope):
    """Every recovered point lies on ``fn`` within a pixel in y plus a pixel in x carried through the slope."""
    tol = 1.5 * (curve.y_uncertainty + abs(slope) * curve.x_uncertainty)  # data units of y, from the pixel size
    return np.abs(curve.y - fn(curve.x)) <= tol


def test_a_rendered_curve_is_recovered_within_its_pixel_size(tmp_path):
    spec = PlotSpec(Axis([0, 20, 40, 60, 80, 100]), Axis([0, 50, 100, 150, 200, 250]), [(lambda x: 2 * x + 5, XS, RED)])
    raster = _raster(tmp_path, spec, dpi=300)  # 8 pt labels about 24 px high
    plot = raster_plot(raster)
    assert sorted(v for _, v in plot.x.ticks) == [0, 20, 40, 60, 80, 100]
    assert sorted(v for _, v in plot.y.ticks) == [0, 50, 100, 150, 200, 250]
    (curve,) = plot.curves
    assert curve.basis == 'raster'
    pixel = 72 / 300  # points per pixel
    assert curve.x_uncertainty == pytest.approx(pixel * 100 / 300, rel=0.05, abs=0)  # data units of x per pixel
    assert _within_pixels(curve, lambda x: 2 * x + 5, 2).all()
    assert curve.x.min() <= 1 and curve.x.max() >= 99  # data units of x: the whole curve
    assert curve.provenance.page == 1
    assert curve.provenance.bbox is not None


def test_a_vector_curve_says_its_basis_and_has_no_pixel_uncertainty(tmp_path):
    from document_lab.pdf import read_pdf  # noqa: PLC0415
    from document_lab.plot import plots  # noqa: PLC0415
    from tests.synthetic import plot_pdf  # noqa: PLC0415

    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    (plot,) = plots(read_pdf(plot_pdf(tmp_path / 'v.pdf', spec), tables=False).page(1))
    (curve,) = plot.curves
    assert (curve.basis, curve.x_uncertainty, curve.y_uncertainty) == ('vector', None, None)


def test_two_crossing_curves_of_two_colours_stay_separate(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED), (lambda x: 100 - x, XS, BLUE)], grid=True
    )
    plot = raster_plot(_raster(tmp_path, spec, dpi=200))
    assert len(plot.curves) == 2  # one per colour; the grid is not a curve
    by_hue = {int(np.argmax(c.stroke)): c for c in plot.curves}
    assert _within_pixels(by_hue[0], lambda x: x, 1).all()  # red
    assert _within_pixels(by_hue[2], lambda x: 100 - x, -1).all()  # blue


def test_labels_too_few_pixels_high_raise_by_name_unless_the_caller_states_them(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([-10, 0, 10]), [(lambda x: x / 5 - 10, XS, RED)])
    raster = _raster(tmp_path, spec, dpi=100)  # 8 pt labels about 8 px high
    with pytest.raises(UnreadableTickLabels, match='x axis') as caught:
        raster_plot(raster)
    assert isinstance(caught.value.__cause__, UnreadableGlyph)
    plot = raster_plot(raster, x_values=[0, 50, 100], y_values=[-10, 0, 10])
    (curve,) = plot.curves
    assert _within_pixels(curve, lambda x: x / 5 - 10, 0.2).all()


def test_stated_values_that_do_not_match_the_label_count_are_refused(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 10, 20]), [(lambda x: x / 5, XS, RED)])
    with pytest.raises(UnreadableTickLabels, match='3 labels'):
        raster_plot(_raster(tmp_path, spec, dpi=100), x_values=[0, 100], y_values=[0, 10, 20])


def test_a_blank_image_has_no_plot_frame(tmp_path):
    with pytest.raises(NoPlotFrame):
        raster_plot(_raster(tmp_path, None, dpi=100))


def test_an_axis_title_aligned_with_the_labels_far_edge_is_not_a_label(tmp_path):
    """3E10 p2 fig 4: the y title ``B`` is flush with ``500``'s left edge; the labels share the edge facing the axis."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 100, 200]),
        [(lambda x: 2 * x, XS, RED)],
        stray=[('B', 100 - 4 - 13.34, 153, 8)],  # flush with the left edge of '100' and '200', between them
    )
    plot = raster_plot(_raster(tmp_path, spec, dpi=100), x_values=[0, 50, 100], y_values=[0, 100, 200])
    (curve,) = plot.curves
    assert _within_pixels(curve, lambda x: 2 * x, 2).all()


def test_a_dark_curve_crossing_grid_lines_stays_one_curve(tmp_path):
    """3E10 p2: removing the grid cuts a black curve at every grid line; the cuts are bridged, not new curves."""
    black = (0, 0, 0)
    spec = PlotSpec(Axis([0, 25, 50, 75, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, black)], grid=True, grid_grey=0)
    plot = raster_plot(_raster(tmp_path, spec, dpi=150), x_values=[0, 25, 50, 75, 100], y_values=[0, 50, 100])
    (curve,) = plot.curves
    assert curve.x.min() <= 2 and curve.x.max() >= 98  # data units of x: the whole curve
    assert _within_pixels(curve, lambda x: x, 1).all()
