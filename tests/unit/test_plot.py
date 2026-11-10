"""Curves digitised from synthetic vector plots whose data are known exactly (the planted controls)."""

import math

import numpy as np
import pymupdf
import pytest

from document_lab.numbers import bound
from document_lab.pdf import read_pdf
from document_lab.plot import RasterPlot, UnreadFrame, UnrecognisedFrame, outline_text, plots
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


def test_the_grid_of_a_plot_above_is_not_a_mark_of_the_plot_below(tmp_path):
    """TDK SIFERRIT N97 p3: the log plot's grid lines, as long as the frame below but above it, snapped its labels."""
    upper = PlotSpec(
        Axis([1, 10, 100, 1000, 1e4, 1e5, 1e6], log=True),  # N97's ~40 grid lines over the frame's width
        Axis([0, 50, 100]),
        [(lambda x: 50.0, [1.0, 1e6], BLUE)],
        frame=pymupdf.Rect(100, 80, 400, 280),
        grid=True,
        minor_grid=True,
    )
    lower = PlotSpec(
        Axis([0, 100, 200, 300, 400, 500]),
        Axis([0, 50, 100]),
        [(lambda x: x / 5, [100.0, 300.0, 450.0], RED)],
        frame=pymupdf.Rect(111, 380, 411, 580),  # offset so the grid above lands near some labels (292 for 300)
        tick_marks=False,  # N97's grid is cut by its legend: no mark of its own near most labels
    )
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    draw_plot(page, upper)
    draw_plot(page, lower)
    doc.save(tmp_path / 'two.pdf')
    found = plots(read_pdf(tmp_path / 'two.pdf').page(1))
    (below,) = [p for p in found if p.frame.y0 > 300]
    (curve,) = below.curves
    assert curve.x == pytest.approx([100, 300, 450], abs=0.5)  # data units of x


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


def test_outlined_units_and_grades_read_as_text(tmp_path):
    """TDK SIFERRIT N96/N88: units, grades and legends are outlines; the letter-blind reader dropped every one."""
    words = ['mT', 'degC', 'N96', 'N88', 'kHz', 'oO0']
    stray = [(w, 450.0, 120.0 + 20 * k, 8.0) for k, w in enumerate(words)]
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=stray)
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    assert not page.spans  # the planted premise: nothing here is text
    text = outline_text(page)
    assert sorted(s.text for s in text) == sorted(words)  # tick labels are numbers, not text
    for span in text:
        assert span.provenance.bbox.x0 == pytest.approx(450.0, abs=1.5)  # points: the word's own ink


def test_an_outlined_legend_names_its_sample_line(tmp_path):
    legend = [('N96', RED, None), ('100mT', BLUE, None)]
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], legend=legend)
    (plot,) = plots(read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1))
    assert sorted(s.label for s in plot.legend) == ['100mT', 'N96']


def test_a_letter_the_reference_font_draws_as_another_is_refused_not_guessed(tmp_path):
    """``l`` and ``I`` are one bar in Helvetica: a word holding either is absent, never read as its twin."""
    stray = [('Il', 450.0, 120.0, 8.0), ('l', 450.0, 140.0, 8.0), ('mT', 450.0, 160.0, 8.0)]
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=stray)
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    assert [s.text for s in outline_text(page)] == ['mT']  # the control beside them still reads


@pytest.mark.parametrize('label', ['l00', '5l'])
def test_an_unreadable_outlined_tick_label_refuses_the_frame_by_name(tmp_path, label):
    """Older SIFERRIT sheets: a label the glyph reader cannot read was dropped and the axis fit from the rest."""
    spec = PlotSpec(Axis([0, 50, 100], labels=['0', '50', label]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    with pytest.raises(UnrecognisedFrame, match='unrecognised frame'):
        plots(page)


def test_a_recognised_outlined_frame_still_reads_beside_an_outlined_axis_title(tmp_path):
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=[('Il', 240.0, 325.0, 8.0)])
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    (plot,) = plots(page)  # an unreadable word below the label row is a title, not a label
    assert sorted(v for _, v in plot.x.ticks) == [0, 50, 100]


def test_outlined_lowercase_words_dotted_letters_and_primes_read_as_text(tmp_path):
    """TDK legends and drawings: ``max.``, ``min.`` (an ``i``'s dot is its own fill) and ``µ'``/``µ''`` were absent."""
    mu = chr(0xB5)
    words = ['max.', 'min.', mu + "'", mu + "''", 'mT']
    stray = [(w, 450.0, 120.0 + 20 * k, 8.0) for k, w in enumerate(words)]
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=stray)
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    assert sorted(s.text for s in outline_text(page)) == sorted(words)


def test_an_outlined_word_is_read_at_its_own_scale_and_a_lone_letter_by_shape_alone(tmp_path):
    """``ox`` tops at the x-height and ``OX`` at the ascent; a lone ``x`` has no baseline to tell ``x`` from ``X``."""
    stray = [('ox', 450.0, 120.0, 8.0), ('OX', 450.0, 140.0, 8.0), ('x', 450.0, 160.0, 8.0), ('B', 450.0, 180.0, 8.0)]
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=stray)
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    assert sorted(s.text for s in outline_text(page)) == ['B', 'OX', 'ox']


def test_an_outlined_number_before_min_or_max_is_one_bound_span(tmp_path):
    """TDK E-core drawings: a window span set ``12.5 min.`` is a lower bound, not a bare number beside a word."""
    stray = [('12.5', 420.0, 400.0, 8.0), ('min.', 440.0, 400.0, 8.0), ('3', 420.0, 430.0, 8.0)]
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], stray=stray)
    page = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True), tables=False).page(1)
    (span,) = outline_text(page)  # the lone 3 stays a number
    assert span.text == '12.5 min.'
    assert bound(span.text) == ('min', 12.5)


def test_a_curve_drawn_past_its_frame_is_read_inside_it(tmp_path):
    """DMEGC: curves are drawn past the frame and clipped there; a curve with any point outside was dropped whole."""
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: 1.5 * x - 25, XS, RED)])
    (curve,) = _one_plot(tmp_path, spec).curves
    assert curve.y == pytest.approx(1.5 * curve.x - 25, abs=0.02)  # data units of y
    assert curve.x.min() == pytest.approx(50 / 3, abs=0.05)  # data units of x: where it enters at y = 0
    assert curve.x.max() == pytest.approx(250 / 3, abs=0.05)  # and leaves at y = 100


@pytest.mark.parametrize('grid', [False, True])
def test_axes_drawn_through_the_corner_still_frame_the_plot(tmp_path, grid):
    """Axes that run on past the origin (an outward tick at the corner) meet nowhere at their ends.

    With a grid, the grid lines cross the axes near the corner too: the lowest and leftmost crossing lines frame it.
    """
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], overhang=6.0, grid=grid)
    (plot,) = _one_plot_list(tmp_path, spec)
    assert tuple(plot.frame) == pytest.approx((100, 100, 400, 300), abs=0.5)  # points: not a grid line's frame
    (curve,) = plot.curves
    assert curve.y == pytest.approx(curve.x, abs=0.05)  # data units of y


def _one_plot_list(tmp_path, spec):
    return plots(read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec), tables=False).page(1))


def test_a_page_whose_every_labelled_frame_fails_to_read_raises_naming_the_page_and_frame(tmp_path):
    """SIFERRIT N72 p3: a frame with one readable x label read as no plot and raised nothing."""
    spec = PlotSpec(Axis([0, 50, 100], labels=['', '50', '']), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    doc = read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec), tables=False)
    with pytest.raises(UnreadFrame, match=r'page 1: .*\(100\.0, 100\.0, 400\.0, 300\.0\)'):
        plots(doc.page(1))


def test_a_readable_frame_beside_an_unreadable_one_is_still_returned(tmp_path):
    """The control: the refusal is for a page that reads NO plot, not for a stray box beside one that reads."""
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    doc = pymupdf.open(plot_pdf(tmp_path / 'plot.pdf', spec))
    doc[0].draw_rect(pymupdf.Rect(450, 400, 550, 500), color=(0, 0, 0), width=0.8)
    doc[0].insert_text((440, 503), '7', fontname='helv', fontsize=8)  # one label: the box reads as no plot
    doc.save(tmp_path / 'two.pdf')
    (plot,) = plots(read_pdf(tmp_path / 'two.pdf', tables=False).page(1))
    assert tuple(plot.frame) == pytest.approx((100, 100, 400, 300), abs=0.5)  # points


def _with_stroke(tmp_path, spec, points):
    doc = pymupdf.open(plot_pdf(tmp_path / 'plot.pdf', spec))
    doc[0].draw_polyline(points, color=BLUE, width=1.0)
    doc.save(tmp_path / 'stroked.pdf')
    return read_pdf(tmp_path / 'stroked.pdf', tables=False).page(1)


def _left_of_50(spec):
    return to_page(spec, 50, 0)[0] - pymupdf.get_text_length('50', fontname='helv', fontsize=8) / 2


def test_a_curve_stroke_running_flat_into_a_tick_label_is_not_its_minus(tmp_path):
    """SIFERRIT N48 p3, DMEGC dmr96a/dmr52w: a curve continuing past the frame crossed the label row.

    Its short flat segments ending just left of ``50`` were read as a minus, so the axis read 0, -50, 100.
    """
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    x0 = _left_of_50(spec)
    stroke = [(x0 - 60, 250.0), (x0 - 30, 309.0), *((x0 - 30 + 2 * k, 309.0) for k in range(15))]
    plot = plots(_with_stroke(tmp_path, spec, stroke))[0]
    assert sorted(v for _, v in plot.x.ticks) == [0, 50, 100]


def test_a_lone_short_stroke_before_a_tick_label_is_still_its_minus(tmp_path):
    """The control: a minus drawn as a path of its own still negates the label the curve stroke must not."""
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    x0 = _left_of_50(spec)
    with pytest.raises(ValueError, match='monotone'):
        plots(_with_stroke(tmp_path, spec, [(x0 - 3.5, 309.0), (x0 - 1.0, 309.0)]))


def _with_multiplier(tmp_path, spec, cy):
    """``spec`` with a ``10^-6`` set right-aligned under the top y label at ``cy``, as TDK SIFERRIT N48 p3 does."""
    from tests.synthetic import _power_label  # noqa: PLC0415 -- the synthetic power-label drawing

    doc = pymupdf.open(plot_pdf(tmp_path / 'plot.pdf', spec))
    _power_label(doc[0], 1e-6, spec.frame.x0 - 4, cy, vector_minus=False, span_minus=True, minus_glyph='_')
    doc.save(tmp_path / 'multiplied.pdf')
    return read_pdf(tmp_path / 'multiplied.pdf', tables=False).page(1)


def test_a_power_of_ten_stacked_under_the_end_label_scales_the_axis(tmp_path):
    """SIFERRIT N48 p3: ``250`` over ``10^-6`` read as two ticks, so the axis read [250, 1e-06, 50, 0, -50]."""
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], grid=True)
    (plot,) = plots(_with_multiplier(tmp_path, spec, spec.frame.y0 + 10))
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([0, 50e-6, 100e-6], rel=1e-9, abs=1e-12)
    (curve,) = plot.curves
    assert curve.y == pytest.approx(curve.x * 1e-6, abs=1e-8)  # data units of y: the axis is in 10^-6


def test_a_power_of_ten_between_two_labels_is_a_tick_not_a_multiplier(tmp_path):
    """The control: a lone power label set between labels, not stacked at the end, stays a tick and is refused."""
    spec = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)], grid=True)
    with pytest.raises(ValueError, match='monotone'):
        plots(_with_multiplier(tmp_path, spec, to_page(spec, 0, 75)[1]))


def _outlined_with(tmp_path, spec, draw):
    doc = pymupdf.open(plot_pdf(tmp_path / 'plot.pdf', spec, outlined=True))
    draw(doc[0])
    doc.save(tmp_path / 'drawn.pdf')
    return read_pdf(tmp_path / 'drawn.pdf', tables=False).page(1)


def _log_grid(**kwargs):
    """A log y axis 1..100 with a minor grid, so a word anywhere in the label column sits on a grid line."""
    return PlotSpec(
        Axis([0, 50, 100]), Axis([1, 10, 100], log=True), [(lambda x: 1 + x, XS, RED)], minor_grid=True, **kwargs
    )


def _arrow(page, cx, apex):
    """An axis-title arrow: a filled head 3.8 x 9.5 points over a hairline shaft, as SIFERRIT N48 p4 draws it."""
    shape = page.new_shape()
    shape.draw_polyline([(cx + 0.15, apex), (cx + 1.9, apex + 9.5), (cx - 1.9, apex + 9.5), (cx - 0.15, apex)])
    shape.finish(fill=(0, 0, 0), color=None, closePath=True)
    shape.commit()
    page.draw_line((cx, apex + 8.2), (cx, apex + 28), color=(0, 0, 0), width=0.3)


def test_an_outlined_axis_title_arrowhead_in_the_label_column_is_no_tick_label(tmp_path):
    """SIFERRIT N48 p4: the y title's filled arrowhead read as a glyph of no digit, refusing the frame."""
    (plot,) = plots(_outlined_with(tmp_path, _log_grid(), lambda page: _arrow(page, 89.0, 135.0)))
    assert sorted(v for _, v in plot.y.ticks) == [1, 10, 100]


def test_an_unreadable_outlined_word_in_the_label_column_still_refuses(tmp_path):
    """The control: a real glyph the reader cannot read where the arrowhead sat still refuses the frame."""
    page = read_pdf(
        plot_pdf(tmp_path / 'plot.pdf', _log_grid(stray=[('Il', 85.0, 143.0, 8.0)]), outlined=True), tables=False
    ).page(1)
    with pytest.raises(UnrecognisedFrame, match='unrecognised frame'):
        plots(page)


def _stroked_digits(page, spec, *, where='labels'):
    """A ``1``-like stroke (a flag and a stem, as stroke-font lettering draws it) at each tick label's place."""
    f = spec.frame
    places = [(to_page(spec, v, 0)[0], f.y1 + 10) for v in spec.x.ticks]
    places += [(f.x0 - 8, to_page(spec, 0, v)[1]) for v in spec.y.ticks]
    if where == 'away':
        places = [(x + 150, y - 300) for x, y in places[:1]]
    for x, y in places:
        page.draw_polyline([(x - 1.5, y - 2.0), (x, y - 3.5), (x, y + 3.5)], color=(0, 0, 0), width=0.4)


def _stroked_lettered(tmp_path, where):
    blank = ['', '', '']
    spec = PlotSpec(Axis([0, 50, 100], labels=blank), Axis([0, 50, 100], labels=blank), [(lambda x: x, XS, RED)])
    doc = pymupdf.open(plot_pdf(tmp_path / 'plot.pdf', spec))
    _stroked_digits(doc[0], spec, where=where)
    doc.save(tmp_path / 'stroked.pdf')
    return read_pdf(tmp_path / 'stroked.pdf', tables=False).page(1)


def test_tick_labels_drawn_as_stroked_lettering_refuse_the_frame_by_name(tmp_path):
    """SIFERRIT N72 p3: labels drawn as stroke-font polylines read as nothing, and three plots read as no plot."""
    with pytest.raises(UnrecognisedFrame, match=r'stroked lines .*: \(100\.0, 100\.0, 400\.0, 300\.0\)'):
        plots(_stroked_lettered(tmp_path, 'labels'))


def test_a_stroked_mark_away_from_the_label_band_refuses_nothing(tmp_path):
    """The control: a frame with no labels and no lettering beside it is still no plot, not a refusal."""
    assert plots(_stroked_lettered(tmp_path, 'away')) == []


@pytest.mark.parametrize('gap', [1.8, 3.0])
def test_a_text_exponent_set_apart_from_its_ten_is_still_the_power(tmp_path, gap):
    """SIFERRIT N48 p4: ``10`` and a raised ``1`` 1.8 pt apart in text read as three 10s, a non-monotone axis."""
    spec = PlotSpec(
        Axis([0, 50, 100]),
        Axis([1, 10, 100, 1000], log=True, power_labels=True, exponent_gap=gap),
        [(lambda x: 10 ** (x / 50), XS, RED)],
    )
    (plot,) = plots(read_pdf(plot_pdf(tmp_path / 'plot.pdf', spec), tables=False).page(1))
    assert sorted(v for _, v in plot.y.ticks) == pytest.approx([1, 10, 100, 1000], rel=1e-12, abs=0)


def test_a_small_number_a_word_gap_past_a_ten_is_not_its_exponent(tmp_path):
    """The control: a smaller raised number farther than a word gap from ``10`` is no power; the axis stays 10."""
    spec = PlotSpec(
        Axis([0, 50, 100]), Axis([0, 5, 10]), [(lambda x: x / 10, XS, RED)], stray=[('3', 104.0, 97.0, 5.0)]
    )
    plot = _one_plot(tmp_path, spec)
    assert sorted(v for _, v in plot.y.ticks) == [0, 5, 10]


def test_a_readable_frame_beside_a_stroke_lettered_one_is_still_returned(tmp_path):
    """SIFERRIT N91 p6: one stroke-lettered inset must not refuse the page's readable plots; it is skipped."""
    blank = ['', '', '']
    lettered = PlotSpec(
        Axis([0, 50, 100], labels=blank),
        Axis([0, 50, 100], labels=blank),
        [(lambda x: x, XS, RED)],
        frame=pymupdf.Rect(100, 450, 400, 650),
    )
    readable = PlotSpec(Axis([0, 50, 100]), Axis([0, 50, 100]), [(lambda x: x, XS, RED)])
    doc = pymupdf.open(plot_pdf(tmp_path / 'plot.pdf', readable))
    draw_plot(doc[0], lettered)
    _stroked_digits(doc[0], lettered)
    doc.save(tmp_path / 'both.pdf')
    (plot,) = plots(read_pdf(tmp_path / 'both.pdf', tables=False).page(1))
    assert tuple(plot.frame) == pytest.approx((100, 100, 400, 300), abs=0.5)  # points
