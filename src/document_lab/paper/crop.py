"""Figure and table crops, and an equation's context text, from a page read into the IR.

An equation's crop is `document_lab.regions.equation`'s.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

from document_lab import layout
from document_lab.ir import BBox, union

if TYPE_CHECKING:
    from document_lab.ir import Page


def crop_figure(page: Page, label: str, content_below: bool = False) -> BBox | None:
    """Return a tight crop rect for *label* on *page*, or None to fall back to full page.

    Strategy (works for typical IEEE 2-column layouts):
    1. Find the caption text block that starts with the label ("Fig. X." / "Table X.").
    2. Collect graphic content blocks (images + drawings):
       - content_below=False (figures): content ABOVE the caption
       - content_below=True  (tables):  content BELOW the caption
    3. The crop spans from the top of those blocks down to the bottom of the caption,
       with a small margin. Column width is inferred from the caption block's x-span.
    """
    page_rect = BBox(0.0, 0.0, page.width, page.height)
    blocks = layout.blocks(page)

    caption_rect = _find_caption_rect(blocks, label)
    if caption_rect is None:
        return None

    cap_top, cap_bottom = caption_rect.y0, caption_rect.y1
    cap_x0, cap_x1 = caption_rect.x0, caption_rect.x1
    half_page = page_rect.height / 2
    page_mid = page_rect.width / 2

    # Constrain crop to not bleed into adjacent captions in the same column
    upper_bound, lower_bound = _find_caption_bounds(blocks, cap_x0, cap_top, cap_bottom, page_mid)

    col_x0, col_x1 = _column_bounds(page_rect, cap_x0)

    if content_below:
        # Stop before the first full-column-width body paragraph (table has ended)
        body_lower = _find_body_text_lower(page, cap_bottom, col_x0, col_x1)
        lower_bound = min(lower_bound, body_lower)

    graphic_rects = _collect_graphic_rects(
        page,
        cap_top,
        cap_bottom,
        half_page,
        col_x0,
        col_x1,
        content_below,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
    )

    extra_x_rects: list = []
    if content_below:
        # Collect table row text rects: used for x-expansion and (if no drawings) y-bottom
        row_rects, row_bottom = _collect_table_row_bounds(page, cap_bottom, col_x0, col_x1, lower_bound)
        extra_x_rects = row_rects
        if not graphic_rects and row_bottom > cap_bottom:
            lower_bound = min(lower_bound, row_bottom)

    return _build_crop_rect(
        page_rect,
        cap_x0,
        cap_x1,
        cap_top,
        cap_bottom,
        half_page,
        graphic_rects,
        content_below,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        extra_x_rects=extra_x_rects,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _line_box(line: layout.Line) -> BBox:
    return BBox(line.x0, line.y0, line.x1, line.y1)


def _images(page: Page) -> list[BBox]:
    """The box of each image placed on the page."""
    return [f.provenance.bbox for f in page.figures if f.provenance.bbox is not None]


def _drawings(page: Page) -> list[BBox]:
    """The box of each painted path on the page (all its subpaths: one drawing)."""
    parts: dict[int, list[BBox]] = defaultdict(list)
    for path in page.paths:
        if path.provenance.bbox is not None:
            parts[path.group].append(path.provenance.bbox)
    return [union(boxes) for boxes in parts.values()]


_CAP_LABEL_RE = re.compile(r'(?:Fig\.|Figure|Table)\s+', re.IGNORECASE)
# Maximum block height for a caption (incl. multi-line subfigure layouts).
# Body-text paragraphs are typically 80-200 pt; IEEE captions are <=60 pt.
_CAP_MAX_HEIGHT = 45.0


def _find_caption_bounds(
    blocks: list[layout.Block], cap_x0: float, cap_top: float, cap_bottom: float, page_mid: float
) -> tuple[float, float]:
    """Return (upper_bound, lower_bound) y-coords from nearest same-column captions above/below.

    Uses a height threshold to distinguish compact caption blocks (≤60 pt) from
    taller body-text paragraphs that happen to contain a figure cross-reference.
    """
    upper_bound = 0.0
    lower_bound = float('inf')
    same_col_left = cap_x0 < page_mid

    for blk in blocks:
        blk_text = ''.join(ln.text for ln in blk.lines)
        bx0, y0, bx1, y1 = blk.bbox
        blk_height = y1 - y0
        # Only compact blocks that look like captions (short text, narrow height)
        if len(blk_text) > 350 or not _CAP_LABEL_RE.search(blk_text) or blk_height > _CAP_MAX_HEIGHT:
            continue
        blk_col_left = (bx0 + bx1) / 2 < page_mid
        if blk_col_left != same_col_left:
            continue  # different column — ignore
        if y1 <= cap_top and abs(y0 - cap_top) > 3:
            upper_bound = max(upper_bound, y1)
        elif y0 >= cap_bottom and abs(y0 - cap_top) > 3:
            lower_bound = min(lower_bound, y0)
    return upper_bound, lower_bound


_MIN_TABLE_HEIGHT = 40.0  # body text must start at least this far below the table caption


def _find_body_text_lower(page: Page, cap_bottom: float, col_x0: float, col_x1: float) -> float:
    """Return y0 of first full-column-width body-paragraph text block below the table.

    Body paragraphs span ≥70% of the column width.  We require a minimum gap of
    _MIN_TABLE_HEIGHT below cap_bottom so the table header is not mistaken for body text.
    """
    col_width = col_x1 - col_x0
    threshold = col_width * 0.70
    earliest_body_y = cap_bottom + _MIN_TABLE_HEIGHT

    def _in_col(r: BBox) -> bool:
        return col_x0 < (r.x0 + r.x1) / 2 < col_x1

    for blk in sorted(layout.blocks(page), key=lambda b: b.bbox.y0):
        r = blk.bbox
        if r.y0 < earliest_body_y or not _in_col(r):
            continue
        if r.width >= threshold:
            return r.y0
    return float('inf')


def _collect_table_row_bounds(
    page: Page, cap_bottom: float, col_x0: float, col_x1: float, lower_bound: float
) -> tuple[list[BBox], float]:
    """Return (row_rects, last_y1) for table-row text blocks below the caption.

    row_rects are used to expand the crop x-bounds.  last_y1 is the y1 of the
    last row block (used as content_bottom when no drawing elements are present).
    """
    col_width = col_x1 - col_x0
    threshold = col_width * 0.70
    limit = min(cap_bottom + 300, lower_bound)
    last_y1 = cap_bottom
    rects: list[BBox] = []

    def _in_col(r: BBox) -> bool:
        return col_x0 < (r.x0 + r.x1) / 2 < col_x1

    for blk in layout.blocks(page):
        r = blk.bbox
        if r.y0 < cap_bottom - 5 or r.y1 > limit or not _in_col(r):
            continue
        rects.append(r)
        if r.width < threshold:
            last_y1 = max(last_y1, r.y1)
    return rects, last_y1


def _find_caption_rect(blocks: list[layout.Block], label: str) -> BBox | None:
    """Return the best-match caption block for *label*.

    Collects all short blocks (≤350 chars) that contain "label." and returns
    the one with the SMALLEST height — caption blocks are compact (1-2 lines),
    while body-text blocks that contain a cross-reference to the same figure
    are typically much taller.

    Also tries the "Fig. N." short form when *label* is "Figure N", so that
    captions using the abbreviated form are found even when the caller passes
    the normalized form.
    """
    label = re.sub(r'\s+', ' ', label).strip()  # normalize newlines from 2-col extraction
    patterns = [re.compile(re.escape(label) + r'\.', re.IGNORECASE)]
    m = re.match(r'^Figure\s+(\d+)$', label, re.IGNORECASE)
    if m:
        patterns.append(re.compile(rf'Fig\.\s*{m.group(1)}\.', re.IGNORECASE))

    candidates: list[tuple[float, BBox]] = []
    for blk in blocks:
        blk_text = ''.join(ln.text for ln in blk.lines)
        if len(blk_text) > 350:
            continue
        for pat in patterns:
            if pat.search(blk_text):
                bbox = blk.bbox
                height = bbox[3] - bbox[1]
                candidates.append((height, bbox))
                break

    if not candidates:
        return None
    # Pick the most compact block -- true captions are 1-2 lines, body refs are paragraph-tall
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _column_bounds(page_rect: BBox, cap_x0: float) -> tuple[float, float]:
    page_mid = page_rect.width / 2
    if cap_x0 < page_mid:
        return 0.0, page_mid + 20
    return page_mid - 20, page_rect.width


def _collect_graphic_rects(
    page: Page,
    cap_top: float,
    cap_bottom: float,
    half_page: float,
    col_x0: float,
    col_x1: float,
    content_below: bool,
    upper_bound: float = 0.0,
    lower_bound: float = float('inf'),
) -> list[BBox]:
    def _in_col(r: BBox) -> bool:
        return col_x0 < (r.x0 + r.x1) / 2 < col_x1

    rects: list[BBox] = []

    for r in _images(page):
        if content_below:
            if r.y0 >= cap_bottom - 10 and r.y1 - cap_bottom <= half_page and _in_col(r) and r.y1 <= lower_bound:
                rects.append(r)
        elif r.y1 <= cap_top + 20 and cap_top - r.y0 <= half_page and _in_col(r) and r.y0 >= upper_bound:
            rects.append(r)

    for r in _drawings(page):
        if r.height < 3 or r.width < 3:
            continue
        if content_below:
            if r.y0 >= cap_bottom - 10 and r.y1 - cap_bottom <= half_page and _in_col(r) and r.y1 <= lower_bound:
                rects.append(r)
        elif r.y1 <= cap_top + 20 and cap_top - r.y0 <= half_page and _in_col(r) and r.y0 >= upper_bound:
            rects.append(r)

    return rects


def _build_crop_rect(
    page_rect: BBox,
    cap_x0: float,
    cap_x1: float,
    cap_top: float,
    cap_bottom: float,
    half_page: float,
    graphic_rects: list[BBox],
    content_below: bool,
    upper_bound: float = 0.0,
    lower_bound: float = float('inf'),
    extra_x_rects: list | None = None,
) -> BBox | None:
    margin = 6
    x0, x1 = cap_x0, cap_x1
    for r in list(graphic_rects) + list(extra_x_rects or []):
        x0 = min(x0, r.x0)
        x1 = max(x1, r.x1)

    eff_lower = lower_bound if lower_bound < float('inf') else page_rect.height

    if not graphic_rects:
        warnings.warn(
            f'no graphic elements found for {"table" if content_below else "figure"} crop '
            f'(cap_top={cap_top:.1f}); falling back to half-page estimate',
            stacklevel=4,
        )

    if content_below:
        content_top = cap_top
        content_bottom = max(
            (r.y1 for r in graphic_rects),
            default=min(page_rect.height, cap_bottom + half_page, eff_lower),
        )
        content_bottom = min(content_bottom, eff_lower)
    else:
        content_top = min(
            (r.y0 for r in graphic_rects),
            default=max(0.0, cap_top - half_page, upper_bound),
        )
        content_top = max(content_top, upper_bound)
        content_bottom = cap_bottom

    crop = BBox(
        max(0.0, x0 - margin),
        max(0.0, content_top - margin),
        min(page_rect.width, x1 + margin),
        min(page_rect.height, content_bottom + margin),
    )
    return crop if crop.height >= 40 else None


def _find_eq_number_pos(blocks: list[layout.Block], eq_num: str) -> tuple[float | None, float | None]:
    """Find the y/x position of equation number (N) on the page.

    Returns the y/x center of the specific LINE (not block) containing "(N)" so that
    the crop region is tight even when (N) appears at the bottom of a multi-line block.
    Matches both standalone "(N)" blocks and inline "formula... (N)" at end of block.
    """
    pattern = re.compile(rf'\({re.escape(eq_num)}\)\s*$')
    for blk in blocks:
        # Scan individual lines to find the one ending with (N)
        for line in blk.lines:
            line_text = line.text.strip()
            if pattern.search(line_text):
                lb = _line_box(line)
                return (lb[1] + lb[3]) / 2, (lb[0] + lb[2]) / 2
        # Fallback: check full block text (some renderers produce single-span blocks)
        blk_text = ''.join(ln.text for ln in blk.lines).strip()
        if pattern.search(blk_text):
            bbox = blk.bbox
            return (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    return None, None


_BODY_END_RE = re.compile(r'(?:is|are|as|by|of|for|with|from|follows):\s*$', re.IGNORECASE)


def _is_body_text_line(text: str) -> bool:
    """Return True if the text line looks like body prose rather than an equation formula.

    Two tests:
    1. Ends with a colon-phrase that closes a body sentence ("...calculated by:", "...set as:")
    2. Has 5+ tokens and high alphabetic density (>60%) — typical prose sentence fragment
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _BODY_END_RE.search(stripped):
        return True
    tokens = stripped.split()
    if len(tokens) < 5:
        return False
    total = sum(1 for c in stripped if not c.isspace())
    alpha = sum(1 for c in stripped if c.isalpha())
    return total > 0 and alpha / total > 0.60


def get_equation_context(page: Page, eq_num: str) -> tuple[str | None, str | None]:
    """Return (lead_in_text, formula_text) for equation *eq_num*.

    lead_in_text: the body-prose sentence immediately before the formula,
                  used to locate the insertion point in the markdown when
                  the explicit (N) reference is absent.
    formula_text: concatenated span text of formula lines (may contain unicode
                  math chars; useful as a plain-text hint for AI readers).

    Strategy: walk backward from the (N) line within its block (same logic as
    _crop_eq_text_region) until hitting a body-text line — that line is the
    lead-in.  For standalone (N) blocks that contain no body text, fall back
    to the last body-text LINE of the preceding same-column block.
    """
    page_rect = BBox(0.0, 0.0, page.width, page.height)
    page_mid = page_rect.width / 2
    blocks = layout.blocks(page)

    eq_num_y, eq_num_x = _find_eq_number_pos(blocks, eq_num)
    if eq_num_y is None:
        return None, None

    eq_col_left = (eq_num_x < page_mid) if eq_num_x is not None else True
    pattern = re.compile(rf'\({re.escape(eq_num)}\)\s*$')

    lead_in_text: str | None = None
    formula_parts: list[str] = []
    eq_block_y0: float | None = None  # y0 of the (N)-containing block

    # ── Step 1: within-block backward walk ────────────────────────────────────
    for blk in blocks:
        blk_col_left = (blk.bbox.x0 + blk.bbox.x1) / 2 < page_mid
        if blk_col_left != eq_col_left:
            continue
        lines = blk.lines
        eq_line_idx = None
        for i, line in enumerate(lines):
            lt = line.text.strip()
            if pattern.search(lt):
                eq_line_idx = i
                break
        if eq_line_idx is None:
            continue

        eq_block_y0 = blk.bbox.y0

        # Walk backward: collect formula lines, then ALL consecutive body-text lines
        # that form the lead-in sentence (may span multiple lines, e.g. "...can be\n"
        # "easily obtained in the cylindrical coordinate system as:").
        lead_in_parts: list[str] = []
        collecting_lead_in = False
        for i in range(eq_line_idx - 1, -1, -1):
            lt = lines[i].text.strip()
            if _is_body_text_line(lt):
                collecting_lead_in = True
                lead_in_parts.insert(0, lt)
            elif lt:
                if collecting_lead_in:
                    break  # formula line interrupts body → stop
                formula_parts.insert(0, lt)
        if lead_in_parts:
            lead_in_text = ' '.join(lead_in_parts)
        break  # found the block

    # ── Step 2: fallback — last body-text LINE of the preceding same-col block ─
    if lead_in_text is None and eq_block_y0 is not None:
        best_y1 = 0.0
        for blk in blocks:
            bx0, _by0, bx1, by1 = blk.bbox
            blk_col_left = (bx0 + bx1) / 2 < page_mid
            if blk_col_left != eq_col_left or by1 > eq_block_y0 + 5:
                continue
            if by1 <= best_y1:
                continue
            # Find the last body-text line in this block
            for line in reversed(blk.lines):
                lt = line.text.strip()
                if _is_body_text_line(lt):
                    best_y1 = by1
                    lead_in_text = lt
                    break

    formula_text = ' '.join(formula_parts).strip() or None
    return lead_in_text, formula_text
