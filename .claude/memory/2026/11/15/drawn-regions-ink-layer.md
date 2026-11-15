---
name: drawn-regions-ink-layer
description: Regions also told by drawn ink -- document_lab.ink measures a page picture's drawings, per_page hands them to regions
---
# Drawn regions (user decision 2026-11-15)

User chose option A: per_page, which holds the page picture, hands "ink boxes" to regions; regions still reads no
pixels; the dependency arrow is `per_page -> ink, regions`; `ink -> ir`.

`document_lab.ink.drawings(pixels, scale, text, reach) -> [Drawing(box, rules, loose, stroke)]`:
- text boxes (OCR'd lines) wiped pixel by pixel; drawings FOUND on a grid of reach/3 cells (a 1-unit grid took
  9 min per document on 1500x2100-unit pages) and MEASURED on their own pixels (a coarse grid smeared a table's
  words into 13 'rules').
- rules: rows/columns inked across >= half the drawing; loose: largest piece off the rules; stroke: longest
  straight run. per_page passes reach = 0.5 body (1 body merged text lines into drawings).

regions(page, body, drawings=()) -- `_drawn`, each rule from a corpus case:
- in the text column wholly (cover bands to the page edge, the sideways licence note are not);
- >= 1.5 body high and wide; stroke >= 2 body (words OCR missed are glyphs, no long line);
- not a frame round words: shares half a line's band, across or beside it, <= 3 line heights, the line wordy or
  filling 80% of the box (link frames, framed running headers);
- crosses no prose; not inside an earlier region;
- frame + inner rule ((>=2, >=3) rules) and loose <= body -> table (IEC voting table, its 7 pt words unread);
  (2, 2) with only words -> a bare frame, no region;
- other pieces no prose line runs between -> one figure (side-by-side views); <= 4 body high with no wordy line
  inside -> 'Untitled equation N' (60205 formulas without numbers: fraction bars are drawn).
Tried and dropped: 2-cell text rows as tables (caught subclause headings 'D.3 / D.3.1' and diagram labels).

Result, 38 IEC documents: 1413 regions (was 1118): 200 untitled figures, 25 untitled equations; 18-1 has
tables p5, p6, p9 and two cover logos (p3).
Open: logos and cover art count as untitled figures (they are drawings); a framed formula with OCR'd words is a
figure.

Also this session: markdown clause numbers have 1-2 digits (#35, year-led rows were headings); tests run on
Actions alone (#34); remote branches pr/capture, pr/the-ops-app-may-break-glass, pr/transformer-glyphs-and-raster
archived as tags archive/* -- branchprune refuses them (commits only reach main rebased), so the user deletes them.
