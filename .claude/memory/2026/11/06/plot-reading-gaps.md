---
name: plot-reading-gaps
description: plot axes fitted piecewise at breaks with monotone refusal and aligned labels; minus as its own span; dash arrays on paths and curves; fragments joined, legend samples on Plot.legend
created: 2026-11-06
---

# Four plot reading gaps a consumer measured on vendor ferrite sheets

Branch `pr/plot-reading-gaps` (from `integrate/main` ed18d41). Each fix has a synthetic control in
tests/unit/test_plot.py that failed before it.

- Axis: `Axis(scale, pieces, ticks)` replaces `slope`/`intercept` (no alias). The fit is cut at every tick where the
  data-per-point rate between neighbouring intervals changes by more than `BREAK_TOL` (0.1, relative); each piece
  is a least-squares line through its own ticks, neighbours share the break tick, the end pieces extrapolate.
  Same rule for log10 (in decades) -- one mechanism, not a linear special case.
- Scale inference: fewer pieces wins, then smaller single-line misfit. A positive broken linear axis no longer
  reads as log10 because a single log line fitted it slightly better.
- Ticks not strictly monotone along the axis (incl. two labels at one position) raise `ValueError` -- the planted
  wrong-label case now refuses instead of skewing. This raise aborts `plots()` for the whole page; per-frame
  isolation was not added (no consumer asked).
- Labels: a tick label's box must overlap the frame's extent along its axis (was: centre within reach/3 of it),
  and only the largest set aligned on one row/column (low edge, high edge or centre, to `EDGE_TOL`) is kept.
- Minus: a span holding only `-`, U+2212, U+2013 or U+2010 ending within `DASH_GAP` left of the digits, at their
  height, negates them (the drawn-dash case already existed; both live in `_minus_before`).
- Dashes: `VectorPath.dashes` and `Curve.dashes` hold PyMuPDF's dash ARRAY in points; the phase is dropped;
  `()` is solid. `VectorPath` gained the field positionally before `closed`.
- Fragments: paths of one style (stroke, width, dashes) whose ends agree to `JOIN_DIGITS` (2) decimals are joined
  in chain order (reversing as needed); a point where more than two same-style ends meet is ambiguous and is not
  joined. Rounding (not a tolerance) may miss ends straddling a rounding boundary -- accepted, PDF fragments
  share exact coordinates.
- Legend: a joined path no longer than `SAMPLE_MAX` (40 pt), no taller than `EDGE_TOL`, with a span starting
  within `SAMPLE_GAP` (10 pt) right of it at its height is a `LegendSample(label, stroke, width, dashes,
  provenance)` on `Plot.legend`, never in `curves`. Matching a sample to its curve by style is left to callers.
- Full base verdict is forge-only (verify refuses a local full suite); tests/unit, architecture and paper pass
  locally (204). Not pushed.
