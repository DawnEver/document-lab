---
name: plot-outlines-and-raster
description: 4S2F's "frameless" B-H loop was outlined glyph labels; a digit shape reader (glyphs.py) that refuses near ties; raster_plot traces pixels with per-point pixel uncertainty and raises UnreadableTickLabels unless labels are stated
created: 2026-11-08
---

# Plots with outlined labels, non-uniform axes, and raster plots

Branch `pr/plot-nonuniform-and-raster` (from `integrate/main` ed18d41, merging `pr/plot-split-glyph-labels` 14eb125).

- The premise "4S2F p5 fig 4 has no frame" was wrong: the frame is found; every tick label on that page is a FILLED
  GLYPH OUTLINE (no text), so each axis had < 2 labels and the plot was skipped. Frame-from-labels was therefore
  not built (no case needs it; an L-shaped axis already frames via the bottom-left corner rule).
- The non-uniform axis (-50, 0, 50, 100, 500, 1000) was already mapped by the piecewise fit; tests now pin it,
  with a uniform and a log axis as the planted controls (one linear / one log10 piece).
- `glyphs.read_glyph`: nearest Helvetica digit rendered at the glyph's own pixel height (3x3 sub-pixel phases),
  12x8 ink-fraction grid + 2x aspect difference; accept <= 2.5 with >= 0.75 margin; refuse under 14 px (measured:
  at 9-13 px a Helvetica 6 and 8 come within the margin). Other fonts (Times, Courier) are mostly refused, never
  misread in the sweep; an `O`/`o` reads as `0` (a shape reader cannot tell them).
- Outlined labels: subpaths whose centre lies in a larger fill are its holes; words by row and gap; a raised smaller
  glyph starts its own word, and `10` + raised integer word is a power (dash between = negative), as for text.
- `raster_plot`: ink = darkness > 0.15 (hairline grids at 100 dpi average ~0.2) or chroma > 0.3; frame edges from
  NEUTRAL ink only (an orange border is not a frame); of the corner frames, the largest with >= 2 label words below
  and left (a grey drop shadow frames none). `_aligned` ties now go to the edge facing the axis (a y title flush
  with the labels' far edge tied with them on 3E10).
- Curves: grid rows/columns (>= 90 % inked) are removed and bridged only for connectivity; coloured ink is one curve
  per 30-degree hue sector, dark ink one per component; a column run ending against other ink / the frame is not
  sampled (its centre is biased). Uncertainty = one pixel's extent in data units per point.
- Real 3E10/3E12 (3e1012.pdf p2, 100 dpi native): labels are 5 px high -> `UnreadableTickLabels` by name; with the
  printed values stated, axes calibrate (x piecewise: 2 pieces). B near H = 250 A/m: 3E10 25 C ~428 mT;
  3E12 25 C ~445, 100 C ~260. Table B is at 1200 A/m (beyond the axis): 460/270 and 470/290 mT, consistent as upper
  bounds. All four branches touch, so each plot's loop is ONE dark curve; legend samples come out as short curves.
