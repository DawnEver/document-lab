"""The reference part's comparison (`tests.reference.compare`) on a synthetic page whose paint is known."""

from tests.reference.compare import Plant, compare
from tests.synthetic import hatch_pdf


def test_a_tinted_tiling_pattern_fill_is_not_compared_as_a_solid_fill(tmp_path):
    """An uncolored tiling pattern tinted red (a TikZ hatch) is stated with its tint and ``pattern='tiling'``;
    PyMuPDF draws its cell, clipped, never one red fill -- so it is no solid fill to match, while the solid fill
    beside it still is."""
    (page,) = compare(hatch_pdf(tmp_path / 'hatch.pdf')).pages
    assert (page.paths, page.unmatched) == (1, 0)


def test_the_solid_fill_beside_a_pattern_is_still_compared(tmp_path):
    """The planted control: PyMuPDF's colors turned, the solid fill no longer matches -- skipping the pattern skips
    nothing else."""
    (page,) = compare(hatch_pdf(tmp_path / 'hatch.pdf'), plant=Plant(recolor=True)).pages
    assert (page.paths, page.unmatched) == (1, 1)
