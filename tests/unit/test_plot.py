"""Curves digitised from synthetic vector plots whose data are known exactly (the planted controls)."""

import math

import numpy as np
import pymupdf
import pytest

from document_lab.pdf import read_pdf
from document_lab.plot import RasterPlot, parse_number, plots
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


@pytest.mark.parametrize(
    ('text', 'value'),
    [(chr(0x2212) + '1.5', -1.5), ('2e3', 2000.0), ('.5', 0.5), ('10⁻³', 1e-3), ('10²', 100.0), ('1 000', 1000.0)],
)
def test_a_tick_label_spelling_parses_to_its_value(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize('text', ['Fig. 1', '', '1-2', 'kHz'])
def test_a_word_is_not_a_tick_label(text):
    assert parse_number(text) is None
