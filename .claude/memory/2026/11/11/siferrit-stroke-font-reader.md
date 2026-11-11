---
name: siferrit-stroke-font-reader
description: A stroke-font tick-label reader (templates from the SIFERRIT sheets, no OCR) reads N27 p3/p5, N30 p3, N41 p3, N72 p3/p5 and N22 p3's three silent frames; a C is no 0, arrowheads and unread small symbols no longer refuse, read lettering is never a curve
created: 2026-11-11
---

# SIFERRIT stroke-font lettering, read (branch pr/transformer-glyphs-and-raster)

Coordinator decision (full-day grant): build the reader the 11-10 record left open. Reproduced on the 15 real
SIFERRIT PDFs locally (vendor archive, never committed). Every change has a synthetic test with a planted control;
each control was checked to fail with its mechanism disabled.

- **What the "stroke font" is**: not single-line lettering but each glyph's OUTLINE stroked as 0.4 pt hairlines,
  broken into overlapping fragments (a `0` is five polylines; an `H` is two closed bars plus bare crossbars).
- **Segmentation** (`plot._stroke_groups`): fragments that bend or close, boxes within `STROKE_TOUCH` (0.3 pt), are
  one glyph; a straight fragment joins every glyph it touches (the `H`) and is dropped when it touches none (ticks).
- **Ink** (`glyphs.stroke`): polylines stroked at their own width over their box grown by half that width. The
  first version clipped the stroke at the centreline box and a `1` read 3.3 from its own template; now 0.0.
- **Templates** (`strokefont.TEMPLATES`, 23 chars: `%0-9ACFHLMRTWkmz`): one instance per character harvested from
  N27/N30 p3/p5 by shape clustering, normalised to unit height, RDP-simplified; place (top, bottom) in digit
  heights. No two are twins (`_twins(32, STROKE)` empty). An `O`, `I`, `B`, `/`, `°`, `µ` is absent: such a glyph
  is refused or read as its nearest only where it also passes the label rules (size, alignment, a tick).
- **Font abstraction** (`glyphs.Font`, `HELVETICA`, `STROKE`): every reader takes the font; Helvetica behaviour is
  unchanged (the 15-PDF survey reads every previously read page identically).
- **A letter nearer than any digit is no digit** (`read_glyph`): N22 p3's `C` of `°C` read `0` (2.23, next digit
  3.01) and broke the axis. Applied to both fonts unless the font draws the two as twins.
- **Arrowheads**: N22 p3's hatched y-title wedge joined `300` into one unread word. A stroke glyph that is a solid
  wedge (`_solid_triangle`) AND reads as no character is dropped (`1`, `L`, `T`, `7` are as solid, so the reading
  gate is needed).
- **Unread stroke words refuse only at label size**: the stroke font holds only what the sheets print, so N88 p6 and
  N96 p3/p5 (outline pages) were refused by a stroked degree ring; `A/m`, `µa` likewise. An unread stroked word now
  refuses a frame only within `SIZE_TOL` of its read labels' size; the `%` (label-sized) got a template instead.
- **Lettering is never a curve**: up to 69 "curves" per frame were hairlines of `300 mT`, `4300`, legends. Paths of
  stroke words that READ are withheld from curves; an unread stroke stays a curve (control).

## N22 p3: why 3 of 4 frames were silent

Frame 1 has text labels; mu_i(T) and both loops letter in the stroke font. The page refusal fires only when NO
frame reads (the N91 p6 inset decision of 11-10), so the three were skipped silently. They now read:
mu_i(T) x `[-60, 0, 20, 100, 160]`, y `[4000, 2000, 1000, 0]`; loops x `[-200, 0, 400, 800, 1400]`, y `[400..0]`.

## Survey (15 SIFERRIT PDFs, before -> after)

Newly read (previously `UnrecognisedFrame` or missing frames): N22 p3 (+3 frames), N27 p3 (2), N27 p4 (+2 loops),
N27 p5 (3), N30 p3 (4, incl. the broken log axis 10^0, 10^-3..10^-7), N30 p4 (+1), N41 p3 (2), N41 p4 (+2),
N41 p5 (+1), N72 p3 (3), N72 p4 (+2), N72 p5 (3). No page that read before changed.

## Open

- Curves on these sheets are drawn the same way (each curve's two edges as overlapping hairline fragments), so a
  curve comes back as dozens of pieces (N72 mu_i(T): 76). Values per piece are right; joining edge fragments into
  one centreline is a curve-reader change, not a label one.
- A frame whose stroke labels still do not read beside a frame that does is skipped, as before (N91 p6 inset rule).
