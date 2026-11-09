"""Curves digitised from synthetic vector plots whose data are known exactly (the planted controls)."""

import math

import numpy as np
import pymupdf
import pytest

from document_lab.pdf import read_pdf
from document_lab.plot import RasterPlot, plots
from tests.synthetic import BLUE, RED, Axis, PlotSpec, draw_plot, plot_pdf, to_page

XS = [float(x) for x in range(0, 101, 10)]


def _one_plot(tmp_path, spec, **kwargs):
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec), tables=False)
    (plot,) = plots(doc.page(1), **kwargs)
    return plot


def test_a_linear_plot_reads_back_its_curve(tmp_path):
    spec = PlotSpec(Axis([0, 20, 40, 60, 80, 100]), Axis([0, 50, 100, 150, 200, 250]), [(lambda x: 2 * x + 5, XS, RED)])
    plot = _one_plot(tmp_path, spec)
    assert (plot.x.scale, plot.y.scale) == ('linear', 'linear')
    (curve,) = plot.curves
    assert curve.x == pytest.approx(XS, abs=0.01)  # data units of x
    assert curve.y == pytest.approx([2 * x + 5 for x in XS], abs=0.02)  # data units of y
    assert curve.axis_scales == ('linear', 'linear')
    assert curve.stroke == RED


def test_a_log_y_axis_is_detected_and_inverted(tmp_path):
    spec = PlotSpec(
        Axis([0, 25, 50, 75, 100]), Axis([1, 10, 100, 1000], log=True), [(lambda x: 10 ** (x / 50), XS, RED)]
    )
    plot = _one_plot(tmp_path, spec)
    assert (plot.x.scale, plot.y.scale) == ('linear', 'log10')
    (curve,) = plot.curves
    assert np.log10(curve.y) == pytest.approx([x / 50 for x in XS], abs=1e-3)  # decades


def test_a_log_log_plot_with_power_of_ten_labels(tmp_path):
    xs = [10 ** (k / 4) for k in range(13)]  # 1 .. 1000
    spec = PlotSpec(
        Axis([1, 10, 100, 1000], log=True, power_labels=False),
        Axis([0.01, 0.1, 1, 10, 100], log=True, power_labels=True),
        [(lambda x: 0.05 * x, xs, BLUE)],
    )
    plot = _one_plot(tmp_path, spec)
    assert (plot.x.scale, plot.y.scale) == ('log10', 'log10')
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([0.01, 0.1, 1, 10, 100], rel=1e-12, abs=0)
    (curve,) = plot.curves
    assert np.log10(curve.x) == pytest.approx(np.log10(xs), abs=1e-3)
    assert np.log10(curve.y) == pytest.approx(np.log10([0.05 * x for x in xs]), abs=1e-3)


def test_each_stroked_curve_is_its_own_curve_and_grid_and_ticks_are_not_curves(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 100, 200]),
        [(lambda x: x, XS, RED), (lambda x: 200 - x, XS, BLUE)],
        grid=True,
    )
    plot = _one_plot(tmp_path, spec)
    by_colour = {c.stroke: c for c in plot.curves}
    assert set(by_colour) == {RED, BLUE}
    assert by_colour[BLUE].y == pytest.approx([200 - x for x in XS], abs=0.02)


def test_without_tick_marks_the_labels_alone_calibrate(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 10, 20]), [(lambda x: x / 5, XS, RED)], tick_marks=False)
    (curve,) = _one_plot(tmp_path, spec).curves
    assert curve.x == pytest.approx(XS, abs=0.05)
    assert curve.y == pytest.approx([x / 5 for x in XS], abs=0.2)  # a label's box centre is not its tick


def test_a_caller_can_force_the_scale(tmp_path):
    # Two labels cannot tell linear from log; the caller who knows says so.
    spec = PlotSpec(Axis([0, 100]), Axis([1, 100], log=True), [(lambda x: 10 ** (x / 50), XS, RED)])
    assert _one_plot(tmp_path, spec).y.scale == 'linear'
    (curve,) = _one_plot(tmp_path, spec, y_scale='log10').curves
    assert np.log10(curve.y) == pytest.approx([x / 50 for x in XS], abs=1e-3)


def test_a_curve_cites_its_page_and_box(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    (curve,) = _one_plot(tmp_path, spec).curves
    x0, y1 = to_page(spec, 0, 0)
    x1, y0 = to_page(spec, 100, 100)
    assert curve.provenance.page == 1
    assert tuple(curve.provenance.bbox) == pytest.approx((x0, y0, x1, y1), abs=0.01)


def test_a_page_without_a_plot_has_none(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), '12 34', fontsize=9)
    doc.save(tmp_path / 'p.pdf')
    assert plots(read_pdf(tmp_path / 'p.pdf').page(1)) == []


def test_a_raster_plot_is_refused_not_skipped(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]))
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    draw_plot(page, spec)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False)
    page.insert_image(
        pymupdf.Rect(spec.frame.x0 + 5, spec.frame.y0 + 5, spec.frame.x1 - 5, spec.frame.y1 - 5), pixmap=pix
    )
    doc.save(tmp_path / 'r.pdf')
    with pytest.raises(RasterPlot, match='raster'):
        plots(read_pdf(tmp_path / 'r.pdf').page(1))


def test_the_tick_values_are_the_label_values(tmp_path):
    spec = PlotSpec(Axis([0, 20, 40]), Axis([-1, 0, 1]), [(math.sin, XS[:5], RED)])
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.y.ticks) == [-1, 0, 1]


def test_minor_log_grid_lines_do_not_capture_a_decade_label(tmp_path):
    # Real sheets draw 2..9 x grid lines inside each decade; a label box centre sits nearer a minor line than its own.
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([10, 100, 1000, 10000], log=True, power_labels=True, label_drop=4.0),
        [(lambda x: 10 ** (1 + x / 50), XS, RED)],
        grid=True,
        minor_grid=True,
        tick_marks=False,
    )
    plot = _one_plot(tmp_path, spec)
    for pos, value in plot.y.ticks:
        assert pos == pytest.approx(to_page(spec, 0, value)[1], abs=1e-3)  # points
    (curve,) = plot.curves
    assert np.log10(curve.y) == pytest.approx([1 + x / 50 for x in XS], abs=1e-3)


def test_a_frame_drawn_as_four_lines_and_a_drawn_minus_in_an_exponent(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0.01, 0.1, 1, 10], log=True, power_labels=True, vector_minus=True),
        [(lambda x: 10 ** (x / 50 - 2), XS, RED)],
        frame_as_lines=True,
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([0.01, 0.1, 1, 10], rel=1e-12, abs=0)
    (curve,) = plot.curves
    assert np.log10(curve.y) == pytest.approx([x / 50 - 2 for x in XS], abs=1e-3)


def test_a_drawn_minus_before_a_plain_label_makes_it_negative(tmp_path):
    spec = PlotSpec(Axis([-50, 0, 50, 100], vector_minus=True), Axis([0, 50, 100]), [(lambda x: x / 2 + 25, XS, RED)])
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.x.ticks) == [-50, 0, 50, 100]
    (curve,) = plot.curves
    assert curve.x == pytest.approx(XS, abs=0.05)


def test_a_smaller_number_beside_the_y_axis_is_not_a_tick_label(tmp_path):
    # A unit's superscript ("kW/m3") sits where y labels sit, in a smaller size.
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=[('3', 90, 150, 5)])
    assert sorted(v for _, v in _one_plot(tmp_path, spec).y.ticks) == [0, 50, 100]


def test_grid_lines_cut_short_by_a_legend_are_not_curves(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], grid=True, grid_reach=0.75)
    assert [c.stroke for c in _one_plot(tmp_path, spec).curves] == [RED]


def test_a_broken_linear_axis_is_mapped_piecewise_at_its_scale_change(tmp_path):
    # -25..50 every 25 over the left half, 50..250 every 50 over the right: one straight line skews every value.
    x = Axis([-25, 0, 25, 50, 100, 150, 200, 250], at=[0, 1 / 6, 2 / 6, 3 / 6, 5 / 8, 6 / 8, 7 / 8, 1])
    xs = [-25.0, -10.0, 0.0, 30.0, 50.0, 75.0, 120.0, 250.0]
    spec = PlotSpec(x, Axis([0, 50, 100]), [(lambda v: v / 5 + 50, xs, RED)])
    (curve,) = _one_plot(tmp_path, spec).curves
    assert curve.x == pytest.approx(xs, abs=0.05)  # data units of x


def test_a_label_off_the_tick_row_or_outside_the_frame_is_not_a_tick(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 50, 100]),
        [(lambda x: x, XS, RED)],
        stray=[('7', 250, 322, 8), ('400', 404, 312, 8), ('9', 72, 180, 8)],  # below the row, past x1, off the column
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.x.ticks) == [0, 50, 100]
    assert sorted(v for _, v in plot.y.ticks) == [0, 50, 100]
    (curve,) = plot.curves
    assert curve.x == pytest.approx(XS, abs=0.01)


def test_ticks_that_are_not_monotone_along_the_axis_are_refused(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100], labels=['0', '100', '50']), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec), tables=False)
    with pytest.raises(ValueError, match='monotone'):
        plots(doc.page(1))


def test_a_minus_set_as_its_own_span_makes_the_label_negative(tmp_path):
    spec = PlotSpec(
        Axis([-25, 0, 25, 50], span_minus=True),
        Axis([-100, -50, 0, 50], span_minus=True),
        [(lambda x: 2 * x - 50, [-25.0, 0.0, 50.0], RED)],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.x.ticks) == [-25, 0, 25, 50]
    assert sorted(v for _, v in plot.y.ticks) == [-100, -50, 0, 50]
    (curve,) = plot.curves
    assert curve.y == pytest.approx([-100, -50, 50], abs=0.05)  # data units of y


def test_a_curve_carries_its_dash_array_and_a_solid_one_none(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 50, 100]),
        [(lambda x: x, XS, RED), (lambda x: 100 - x, XS, BLUE, '[3 1.5] 0')],
    )
    by_colour = {c.stroke: c for c in _one_plot(tmp_path, spec).curves}
    assert by_colour[RED].dashes == ()
    assert by_colour[BLUE].dashes == (3.0, 1.5)  # points: on, off


def test_fragments_meeting_end_to_end_in_one_style_are_one_curve(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 100, 200]),
        [(lambda x: 2 * x, XS, RED), (lambda x: 200 - x, XS, BLUE, '[3 1.5] 0')],
        fragments=4,
    )
    by_colour = {c.stroke: c for c in _one_plot(tmp_path, spec).curves}
    assert set(by_colour) == {RED, BLUE}
    assert sorted(by_colour[RED].x) == pytest.approx(XS, abs=0.01)  # data units of x, each sample once
    assert by_colour[RED].y == pytest.approx([2 * x for x in by_colour[RED].x], abs=0.02)
    assert by_colour[BLUE].dashes == (3.0, 1.5)


def test_fragments_of_different_styles_are_not_joined(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 50, 100]),
        [(lambda x: x, XS[:6], RED), (lambda x: x, XS[5:], RED, '[3 1.5] 0')],  # meet at x = 50
    )
    assert sorted(c.dashes for c in _one_plot(tmp_path, spec).curves) == [(), (3.0, 1.5)]


def test_a_short_sample_line_beside_a_label_is_a_legend_entry_not_a_curve(tmp_path):
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0, 100, 200]),
        [(lambda x: x, XS, RED), (lambda x: 2 * x, XS, BLUE, '[3 1.5] 0')],
        legend=[('25 C', RED, None), ('100 C', BLUE, '[3 1.5] 0')],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(c.stroke for c in plot.curves) == sorted([RED, BLUE])
    assert [(e.label, e.stroke, e.dashes) for e in plot.legend] == [('25 C', RED, ()), ('100 C', BLUE, (3.0, 1.5))]


def test_a_label_set_one_glyph_per_span_reads_as_one_number(tmp_path):
    """Handbook 2013 p99: ``250`` is three spans ``2``, ``5``, ``0`` abutting on one baseline -- one label."""
    spec = PlotSpec(Axis([0, 150, 250], glyph_spans=True), Axis([0, 50, 100]), [(lambda x: x / 5, [0.0, 250.0], RED)])
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.x.ticks) == [0, 150, 250]
    (curve,) = plot.curves
    assert curve.y == pytest.approx([0, 50], abs=0.05)  # data units of y


def test_a_drawn_minus_running_into_its_digits_box_still_makes_the_label_negative(tmp_path):
    """Handbook 2013 p126: the dash of ``-50`` ends 0.6 pt INSIDE the ``50`` span's box."""
    spec = PlotSpec(
        Axis([-50, 0, 50, 100], vector_minus=True, minus_gap=-0.6),
        Axis([0, 50, 100]),
        [(lambda x: x / 2 + 25, XS, RED)],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.x.ticks) == [-50, 0, 50, 100]


def test_an_exponent_minus_set_as_its_own_span_makes_the_power_negative(tmp_path):
    """Handbook 2013 p108: ``10`` then ``-`` then ``2``, the minus a span of its own -- 10^-2, not 10^2 or 10."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([0.01, 0.1, 1, 10], log=True, power_labels=True, span_minus=True),
        [(lambda x: 10 ** (x / 50 - 2), XS, RED)],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([0.01, 0.1, 1, 10], rel=1e-12, abs=0)


def test_an_underscore_minus_span_makes_a_label_and_a_power_negative(tmp_path):
    """TDK SIFERRIT N97 p3: ``10`` ``_`` ``2`` is 10^-2 and ``_`` ``60`` is -60, the minus an underscore of its own."""
    xs = [-60.0, -30.0, 0.0, 30.0, 60.0]
    spec = PlotSpec(
        Axis([-60, -20, 20, 60], span_minus=True, minus_glyph='_'),
        Axis([0.01, 0.1, 1, 10], log=True, power_labels=True, span_minus=True, minus_glyph='_'),
        [(lambda x: 10 ** (x / 40 - 0.5), xs, RED)],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.x.ticks) == [-60, -20, 20, 60]
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([0.01, 0.1, 1, 10], rel=1e-12, abs=0)


def test_a_bare_mantissa_between_power_of_ten_labels_is_that_multiple_of_the_decade_below(tmp_path):
    """TDK SIFERRIT N97 p4: ``10^0 5 10^1 5 10^2 5`` reads 1, 5, 10, 50, 100, 500 -- not a refusal, not 5 thrice."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([1, 5, 10, 50, 100, 500], log=True, power_labels=True),
        [(lambda x: 10 ** (x / 50), XS, RED)],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([1, 5, 10, 50, 100, 500], rel=1e-12, abs=0)
    (curve,) = plot.curves
    # decades; a quarter point on this frame (about 0.013 decade per point): the label boxes snap to the tick marks
    assert np.log10(curve.y) == pytest.approx([x / 50 for x in XS], abs=3e-3)


def test_a_bare_mantissa_past_a_missing_decade_label_reads_its_own_decade(tmp_path):
    """N97 p4 prints no 10^3: ``10^2 5 5 10^4`` reads 500 and 5000 by position, not 500 twice."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([1, 10, 100, 500, 5000, 10000], log=True, power_labels=True),
        [(lambda x: 10 ** (x / 25), XS, RED)],
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([1, 10, 100, 500, 5000, 10000], rel=1e-12, abs=0)


def test_a_bare_digit_without_power_of_ten_labels_still_refuses_a_non_monotone_axis(tmp_path):
    """The planted control: ``1 5 10 5 100`` with no power labels is not a log axis to repair."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([1, 3, 10, 30, 100], log=True, labels=['1', '5', '10', '5', '100']),
        [(lambda x: 10 ** (x / 50), XS, RED)],
    )
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec), tables=False)
    with pytest.raises(ValueError, match='not monotone'):
        plots(doc.page(1))


def test_an_outlined_exponent_set_a_point_apart_is_still_the_power(tmp_path):
    """TDK SIFERRIT N97 p5: outlines carry no side bearing; ``10`` and ``1`` set 1 pt apart are 2.6 pt apart in ink."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([1, 10, 100, 1000], log=True, power_labels=True, exponent_gap=1.0),
        [(lambda x: 10 ** (x / 50), XS, RED)],
    )
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False)
    (plot,) = plots(doc.page(1))
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([1, 10, 100, 1000], rel=1e-12, abs=0)


def test_a_non_uniform_axis_maps_linearly_between_each_pair_of_labelled_ticks(tmp_path):
    # 4S2F's B-H loop: -50..100 every 50 over 60 % of the axis, then 100, 500, 1000 over the rest.
    x = Axis([-50, 0, 50, 100, 500, 1000], at=[0, 0.2, 0.4, 0.6, 0.8, 1])
    xs = [-50.0, -20.0, 0.0, 70.0, 100.0, 300.0, 750.0, 1000.0]
    spec = PlotSpec(x, Axis([0, 50, 100]), [(lambda v: v / 20 + 50, xs, RED)])
    plot = _one_plot(tmp_path, spec)
    (curve,) = plot.curves
    assert curve.x == pytest.approx(xs, abs=0.05)  # data units of x
    assert len(plot.x.pieces) == 3  # 50 per fifth, 400 per fifth, 500 per fifth


def test_a_uniform_axis_still_reads_as_one_linear_piece(tmp_path):
    spec = PlotSpec(Axis([0, 20, 40, 60, 80, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    plot = _one_plot(tmp_path, spec)
    assert (plot.x.scale, len(plot.x.pieces)) == ('linear', 1)


def test_a_log_axis_still_reads_as_one_log10_piece(tmp_path):
    spec = PlotSpec(Axis([1, 10, 100, 1000], log=True), Axis([0, 50, 100]), [(lambda x: x / 20, [1.0, 1000.0], RED)])
    plot = _one_plot(tmp_path, spec)
    assert (plot.x.scale, len(plot.x.pieces)) == ('log10', 1)


def test_labels_set_as_glyph_outlines_calibrate_the_axes(tmp_path):
    """4S2F p5 fig 4: every tick label is a filled glyph outline, no text -- the plot read as having no labels."""
    x = Axis([-50, 0, 50, 100, 500, 1000], at=[0, 0.2, 0.4, 0.6, 0.8, 1])
    xs = [-50.0, 0.0, 70.0, 300.0, 1000.0]
    spec = PlotSpec(x, Axis([0, 100, 200, 300, 400, 500]), [(lambda v: v / 4 + 200, xs, RED)])
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False)
    assert not doc.page(1).spans  # the planted premise: nothing here is text
    (plot,) = plots(doc.page(1))
    assert sorted(v for _, v in plot.x.ticks) == [-50, 0, 50, 100, 500, 1000]
    assert sorted(v for _, v in plot.y.ticks) == [0, 100, 200, 300, 400, 500]
    (curve,) = plot.curves
    assert curve.x == pytest.approx(xs, abs=0.5)  # data units of x; a glyph box centre is not its tick
    assert curve.y == pytest.approx([v / 4 + 200 for v in xs], abs=1.0)  # data units of y


def test_outlined_powers_of_ten_and_a_drawn_minus_read_as_their_values(tmp_path):
    """4S2F p5 fig 1: ``10`` and its raised exponent are outlines a hair apart; a drawn dash makes it negative."""
    spec = PlotSpec(
        Axis([-50, 0, 50, 100], vector_minus=True),
        Axis([0.01, 0.1, 1, 10, 100], log=True, power_labels=True, vector_minus=True),
        [(lambda x: 10 ** (x / 50), [-50.0, 0.0, 100.0], RED)],
    )
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False)
    (plot,) = plots(doc.page(1))
    assert sorted(v for _, v in plot.x.ticks) == [-50, 0, 50, 100]
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([0.01, 0.1, 1, 10, 100], rel=1e-12, abs=0)
    (curve,) = plot.curves
    assert np.log10(curve.y) == pytest.approx([-1, 0, 2], abs=0.01)  # decades
