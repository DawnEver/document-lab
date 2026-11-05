"""The ``vendor`` part: a real document named by ``DOCUMENT_LAB_VENDOR_PDF`` (vendor PDFs are never committed).

Its own part, never tests/base; run where the file exists, and it FAILS without one rather than skip.

The expected values of a vendor sheet are not known exactly, so these are the PROPERTIES every calibration must
hold; the exact checks are the synthetic controls in tests/unit. Measured 2026-11-05 on a ferrite material sheet:
7 plots, each with ordered ticks -- before the label fixes the same sheet gave 0 plots, then repeated tick values.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from document_lab.pdf import read_pdf
from document_lab.plot import plots

pytestmark = pytest.mark.vendor

_PATH = os.environ.get('DOCUMENT_LAB_VENDOR_PDF')


@pytest.fixture(scope='module')
def found():
    if not _PATH:
        pytest.fail('the vendor part needs DOCUMENT_LAB_VENDOR_PDF naming a real PDF with vector plots')
    doc = read_pdf(Path(_PATH), tables=False)
    return [plot for page in doc.pages for plot in plots(page)]


def test_the_document_has_a_plot(found):
    assert found


def test_every_axis_reads_ordered_distinct_ticks(found):
    for plot in found:
        for axis, label in ((plot.x, 'x'), (plot.y, 'y')):
            values = [v for _, v in axis.ticks]  # in page order: x left->right, y top->bottom
            ordered = values if label == 'x' else values[::-1]
            assert ordered == sorted(ordered), f'{label} ticks out of order: {values}'
            assert len(set(values)) == len(values), f'{label} ticks repeat: {values}'


def test_every_curve_maps_to_finite_data(found):
    for plot in found:
        for curve in plot.curves:
            assert np.all(np.isfinite(curve.x))
            assert np.all(np.isfinite(curve.y))
