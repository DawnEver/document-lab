"""The stitcher against a synthetic tall image cut into overlapping frames: re-stitched it must equal the image
pixel for pixel, and an overlap it cannot place uniquely it refuses."""

import numpy as np
import pytest

from document_lab.capture.stitch import AmbiguousOverlap, NoOverlap, offset, stitch

WIDTH = 64


def tall(height: int, seed: int = 0) -> np.ndarray:
    """A tall RGB image whose every row is distinct noise."""
    return np.random.default_rng(seed).integers(0, 256, (height, WIDTH, 3), dtype=np.uint8)


def frames(
    image: np.ndarray, view: int, steps: list[int], header: np.ndarray | None = None, footer: np.ndarray | None = None
) -> list[np.ndarray]:
    """The viewport of height ``view`` after each scroll in ``steps``, framed by fixed header/footer rows."""
    tops = [0]
    for step in steps:
        tops.append(min(tops[-1] + step, len(image) - view))
    out = []
    for top in tops:
        parts = [p for p in (header, image[top : top + view], footer) if p is not None]
        out.append(np.concatenate(parts))
    return out


def test_offset_is_the_scroll_in_rows():
    image = tall(300)
    a, b = frames(image, 100, [37])
    assert offset(a, b) == 37


def test_identical_frames_are_offset_zero():
    a = tall(100)
    assert offset(a, a.copy()) == 0


def test_stitch_restores_the_image_exactly():
    image = tall(1000)
    assert np.array_equal(stitch(frames(image, 200, [60, 80, 150, 70, 140, 120, 150, 150])), image)


def test_a_last_scroll_that_hits_the_end_is_shorter():
    image = tall(450)
    shots = frames(image, 200, [150, 150])  # the second scroll moves only 100: the end was reached
    assert offset(shots[1], shots[2]) == 100
    assert np.array_equal(stitch(shots), image)


def test_fixed_header_and_footer_are_kept_once():
    image = tall(600, seed=1)
    header, footer = tall(12, seed=2), tall(9, seed=3)
    shots = frames(image, 150, [50, 90, 100, 100, 100, 100], header, footer)
    assert np.array_equal(stitch(shots, top=12, bottom=9), np.concatenate([header, image, footer]))


def test_repeated_frames_add_nothing():
    image = tall(300)
    a, b = frames(image, 100, [40])
    assert np.array_equal(stitch([a, a, b, b]), image[:140])


def test_a_scroll_past_the_whole_view_refuses():
    image = tall(400)
    a, b = frames(image, 100, [100])
    with pytest.raises(NoOverlap):
        offset(a, b)


def test_a_periodic_overlap_refuses():
    stripe = tall(20)
    image = np.concatenate([stripe] * 20)
    a, b = frames(image, 100, [40])
    with pytest.raises(AmbiguousOverlap):
        offset(a, b)


def test_a_blank_overlap_refuses():
    image = np.full((400, WIDTH, 3), 255, dtype=np.uint8)
    a, b = frames(image, 100, [40])
    with pytest.raises(AmbiguousOverlap):
        offset(a, b)


def test_frames_of_different_shape_refuse():
    with pytest.raises(ValueError, match='shape'):
        offset(tall(100), tall(101))


def blink(frame: np.ndarray) -> np.ndarray:
    """``frame`` with a caret-sized block inverted: what a blinking cursor or a hover does between two shots."""
    out = frame.copy()
    out[30:44, 10:12] = 255 - out[30:44, 10:12]
    return out


def test_a_blinking_caret_breaks_the_exact_match_but_not_a_tolerant_one():
    image = tall(500)
    shots = frames(image, 200, [60, 70, 90])
    shots[2] = blink(shots[2])
    with pytest.raises(NoOverlap):
        stitch(shots)
    assert stitch(shots, tolerance=4).shape == image[:420].shape
    assert offset(shots[1], shots[2], tolerance=4) == 70


def test_a_tolerant_match_still_refuses_a_periodic_overlap():
    image = np.concatenate([tall(20)] * 20)
    a, b = frames(image, 100, [40])
    with pytest.raises(AmbiguousOverlap):
        offset(a, b, tolerance=4)
