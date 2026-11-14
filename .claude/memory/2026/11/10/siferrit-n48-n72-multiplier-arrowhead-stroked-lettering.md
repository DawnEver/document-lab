---
name: siferrit-n48-n72-multiplier-arrowhead-stroked-lettering
description: TDK SIFERRIT N48 p3/p4 now read (axis multiplier 10^-6, filled arrowhead, exponent a space apart); N72 p3 draws its labels as stroke-font polylines and is refused by name (UnrecognisedFrame) instead of returning nothing
created: 2026-11-10
---

# SIFERRIT N48 p3, N48 p4, N72 p3 (branch pr/transformer-glyphs-and-raster)

Reproduced on the real sheets locally (vendor archive, never committed); each failure has a synthetic test plus a
planted control in `tests/unit/test_plot.py`. Decisions taken autonomously (user away 2026-11-10), for review:

- **N48 p3** `[250, 1e-06, 50, 0, -50]`: the `10^-6` is not a tick. TDK stacks it under the top label `250`: the
  axis is in units of 1e-6. Decision: `_multiplied` SCALES the other labels by it (dropping it would misstate every
  value by 1e6). Rule: the axis' only power label, its box abutting/overlapping the first or last other label's, on
  no mark. The page now reads 250e-6 .. -50e-6 (-150/-250 sit 2.3 pt off the label column and are not used; the fit
  puts them at -152e-6 / -252e-6, consistent).
- **N48 p4** glyph (72.5, 230.5, 76.3, 240.0): the FILLED ARROWHEAD of the y-title arrow (a triangle with a 0.3 pt
  cut apex, a zero-height sliver inside it, over a stroked shaft). Fix: a glyph whose one fill is a triangle is not a
  glyph (`_arrowhead`; a `4`'s triangular counter is a hole, so only hole-less fills), slivers under `SLIVER` are no
  fill. The page then exposed a second cause: `10` and a raised `1` set 1.8 pt apart in TEXT (TDK's space) read as
  `10, 10, 10`; the plain exponent gap is now `max(EDGE_TOL, WORD_GAP * size)`, checked AFTER the minus cases.
- **N72 p3**: the page has three plots, but every label is a stroke-font polyline (no text, no fill). A stroke-font
  reader was measured via the raster path (Helvetica references): 0-1 of 5-9 labels read, so it is NOT built.
  Decision: refuse by name. `_stroked_label` finds glyph-sized bent strokes on a mark in a frame's label band;
  `plots` raises `UnrecognisedFrame` naming those frames only when NO frame on the page reads (as `UnreadFrame`), so
  a stroke-lettered inset beside readable plots (N91 p6) does not refuse the page.
- Survey of 15 SIFERRIT PDFs before/after: no page that read changed; new refusals only on pages that returned
  nothing (N27 p3/p5, N41 p3, N72 p3/p5, N30 p3). Open: a stroke-font glyph reader would read all of them.
