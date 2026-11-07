"""The stitcher against a synthetic tall image cut into overlapping frames: re-stitched it must equal the image
pixel for pixel, and an overlap it cannot place uniquely it refuses."""

import itertools
import time

import numpy as np
import pytest

from document_lab.capture.stitch import AmbiguousOverlap, Match, NoOverlap, Stitcher, offset, stitch

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
    assert np.array_equal(stitch(shots, Match(top=12, bottom=9)), np.concatenate([header, image, footer]))


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


def test_a_blinking_caret_is_an_outlier_the_match_absorbs():
    image = tall(500)
    shots = frames(image, 200, [60, 70, 90])
    shots[2] = blink(shots[2])
    assert offset(shots[1], shots[2]) == 70
    assert stitch(shots).shape == image[:420].shape
    with pytest.raises(NoOverlap):
        offset(shots[1], shots[2], Match(tolerance=0, outliers=0))  # held to exact, the caret breaks it


def test_a_tolerant_match_still_refuses_a_periodic_overlap():
    image = np.concatenate([tall(20)] * 20)
    a, b = frames(image, 100, [40])
    with pytest.raises(AmbiguousOverlap):
        offset(a, b, Match(tolerance=4))


def test_the_stitcher_keeps_only_the_last_frame():
    image = tall(800)
    shots = frames(image, 200, [100] * 6)
    stitcher = Stitcher(shots[0])
    for shot in shots[1:]:
        stitcher.add(shot)
    assert stitcher.last is shots[-1]
    assert stitcher.height == 800
    assert np.array_equal(stitcher.image(), image)


def test_a_tolerant_match_of_a_tall_view_is_fast():
    image = tall(4000)
    a, b = frames(image, 1800, [500])
    start = time.perf_counter()
    assert offset(a, blink(b), Match(tolerance=4)) == 500
    assert time.perf_counter() - start < 5


# Overlays: parts of the view that do not scroll -- a sticky header or footer, a floating button, a scrollbar thumb.


def overlaid(image: np.ndarray, view: int, steps: list[int], mask: np.ndarray, paint: np.ndarray):
    """The viewport after each scroll with ``paint`` drawn over it where ``mask`` is set; the frames and their tops."""
    tops = [0]
    for step in steps:
        tops.append(min(tops[-1] + step, len(image) - view))
    shots = []
    for top in tops:
        shot = image[top : top + view].copy()
        shot[mask] = paint[mask]
        shots.append(shot)
    return shots, tops


def clean(image: np.ndarray, tops: list[int], view: int, mask: np.ndarray, paint: np.ndarray) -> np.ndarray:
    """What a clean stitch shows: the content wherever some frame shows it uncovered, else the first overlay over it."""
    out = image[: tops[-1] + view].copy()
    seen = np.zeros(out.shape[:2], dtype=bool)
    first = np.zeros(out.shape[:2], dtype=bool)
    for top in tops:
        seen[top : top + view] |= ~mask
        fresh = ~first[top : top + view] & mask
        out[top : top + view][fresh & ~seen[top : top + view]] = paint[fresh & ~seen[top : top + view]]
        first[top : top + view] = True
    out[seen] = image[: len(out)][seen]
    return out


def screen(view: int, rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros((view, WIDTH), dtype=bool)
    mask[rows, cols] = True
    return mask


def test_dropped_a_floating_button_is_found_and_painted_out():
    image, paint = tall(900, seed=4), tall(200, seed=5)
    mask = screen(200, slice(150, 175), slice(40, 60))
    shots, tops = overlaid(image, 200, [60, 80, 90, 70, 100, 90, 100, 100], mask, paint)
    got = stitch(shots, drop_overlays=True)
    assert np.array_equal(got, clean(image, tops, 200, mask, paint))
    assert (got[: tops[-1]] == image[: tops[-1]]).all()  # the button is left only where nothing else shows that spot


def test_dropped_a_sticky_header_and_footer_need_no_hint_and_are_kept_once():
    image, paint = tall(700, seed=6), tall(200, seed=7)
    mask = screen(200, slice(0, 14), slice(None)) | screen(200, slice(190, 200), slice(None))
    shots, tops = overlaid(image, 200, [70, 90, 100, 100, 100, 100], mask, paint)
    got = stitch(shots, drop_overlays=True)
    assert np.array_equal(got, clean(image, tops, 200, mask, paint))
    assert np.array_equal(got[:14], paint[:14])
    assert np.array_equal(got[-10:], paint[190:])


def test_by_default_the_frames_are_stitched_as_they_were():
    image, paint = tall(600, seed=8), tall(200, seed=9)
    mask = screen(200, slice(150, 175), slice(40, 60))
    shots, _ = overlaid(image, 200, [100, 100, 100, 100], mask, paint)
    got = stitch(shots)
    assert len(got) == 600
    for top in (0, 100, 200, 300, 400):  # every frame's button, where its new rows were taken
        assert np.array_equal(got[top + 150 : top + 175, 40:60], paint[150:175, 40:60])


def test_a_scrollbar_thumb_moving_on_its_own_does_not_break_the_match():
    image = tall(1200, seed=10)
    shots = frames(image, 300, [100, 120, 90, 110])
    for shot, top in zip(shots, [0, 100, 220, 310, 420], strict=True):
        shot[:, -3:] = 200
        thumb = top * 300 // 1200
        shot[thumb : thumb + 75, -3:] = 90
    assert [offset(a, b) for a, b in itertools.pairwise(shots)] == [100, 120, 90, 110]


def test_an_animation_alone_is_not_a_scroll():
    a = tall(300, seed=11)
    b = blink(a)
    assert offset(a, b) == 0


def page(height: int, width: int, seed: int) -> np.ndarray:
    """A sparse document: white, with lines of random words every 37 rows, a grey scrollbar track at the right."""
    rng = np.random.default_rng(seed)
    out = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in range(30, height - 30, 37):
        x = 40
        while x < width - 120 and rng.random() > 0.08:
            word = int(rng.integers(15, 60))
            out[y : y + 12, x : x + word] = rng.integers(0, 80, (12, word, 3))
            x += word + int(rng.integers(6, 20))
    out[:, -10:] = 230
    return out


def browsed(image: np.ndarray, view: int, step: int) -> tuple[list[np.ndarray], list[int]]:
    """``image`` scrolled in a browser: a sticky header, a floating button, a scrollbar thumb that moves on its own."""
    tops = [0]
    while tops[-1] < len(image) - view:
        tops.append(min(tops[-1] + step, len(image) - view))
    shots = []
    for top in tops:
        shot = image[top : top + view].copy()
        thumb = top * view // len(image)
        shot[thumb : thumb + view * view // len(image), -10:] = 120
        shot[view - 90 : view - 40, -80:-20] = (30, 120, 250)
        shot[:30] = (40, 40, 40)
        shots.append(shot)
    return shots, tops


@pytest.mark.parametrize('step', [23, 37, 111, 250])
def test_a_browsed_page_is_stitched_whole_and_its_overlays_dropped(step):
    image = page(2400, 600, seed=12)
    shots, _ = browsed(image, 500, step)
    got = stitch(shots, drop_overlays=True)
    assert len(got) == len(image)
    body = slice(30, len(image) - 90)  # under the header at the top, beside the button at the end: nothing else shows
    assert np.array_equal(got[body, :-10], image[body, :-10])
    assert np.array_equal(got[:30], shots[0][:30])


def test_a_browsed_page_by_default_keeps_what_each_frame_showed():
    image = page(2400, 600, seed=13)
    shots, tops = browsed(image, 500, 200)
    got = stitch(shots)
    assert len(got) == len(image)
    for top in tops[1:-1]:  # every frame's button, where its new rows were taken
        assert (got[top + 410 : top + 460, -80:-20] == (30, 120, 250)).all()


def test_drained_rows_are_final_and_memory_stays_one_view():
    image = page(6000, 300, seed=14)
    shots, _ = browsed(image, 400, 150)
    stitcher, parts = Stitcher(shots[0], drop_overlays=True), []
    for shot in shots[1:]:
        stitcher.add(shot)
        parts.append(stitcher.drain())
        assert stitcher.held <= 3 * 400  # what is kept is bounded by the view, not the page
    parts.append(stitcher.drain(final=True))
    streamed = np.concatenate(parts)
    assert np.array_equal(streamed, stitch(shots, drop_overlays=True))
    assert stitcher.height == len(streamed) == len(image)


def rendered(image: np.ndarray, top: float, view: int) -> np.ndarray:
    """The view at a fractional ``top``, as an application re-renders it: rows blended (175% scaling scrolls so)."""
    whole, part = int(top), top - int(top)
    rows = image[whole : whole + view + 1].astype(np.float32)
    return np.round(rows[:-1] * (1 - part) + rows[1:] * part).astype(np.uint8)


@pytest.mark.parametrize('step', [1.75 * 3, 1.75 * 41, 1.75 * 97, 1.75 * 160])
def test_a_fractional_scroll_is_placed_within_a_row_never_refused(step):
    image = page(3000, 600, seed=15)
    for top in np.arange(0, 3000 - 500 - step - 1, 3 * step):
        assert abs(offset(rendered(image, top, 500), rendered(image, top + step, 500)) - step) <= 1
