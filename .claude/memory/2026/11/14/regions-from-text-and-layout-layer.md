---
name: regions-from-text-and-layout-layer
description: "Decision log cont.: layout layer, regions from text geometry, pages cuts figures/tables/equations; OCR noise headings"
from:
  - 2026/11/13/ocr-pages-architecture-and-decisions.md
metadata:
  type: project
---

2026-11-14, continuing the authorised run (user away; review later).

Why: the user asked why paper's figure/equation cropping was not wired into pages. paper/crop.py finds a figure by DRAWN content (vector paths, embedded image blocks) above its caption; a capture's page is one picture, so paper deliberately links it whole (is_full_page_image) -- crop cannot work there. crop_equation is text-based but entangled with paper's IEEE heuristics.

Decisions:
1. document_lab.layout: Line/lines/ordered/wordy/body_size moved out of markdown (behaviour unchanged) so a second reader of page geometry does not reach into markdown. markdown, regions -> layout -> ir.
2. document_lab.regions: figure = caption 'Figure N <dash>' below it, region up to nearest prose above; table = caption above, down to next prose or caption (else page foot); equation = '(N)'/'(A.1)' alone right of 60% width, grown over adjacent non-prose lines (OCR'd formulas read as noise) within 2 body sizes. Prose = wordy, size <= 1.4 body, width >= half the text width. Captions need a dash after the number (rejects 'Table 7, item 1d' mentions); four dots (leaders) reject. Spans the text width.
3. pages cuts each region from the page's own pixels (page-NNN-<kind>-<label>.png), links it after the paragraph holding the label, marks '(continues on the next page)' at the page foot (captures cut pages off the original pagination), INDEX lists clauses | regions.
4. markdown on unweighted (OCR'd) docs: headings must be wordy and numbered ones not all caps; four dots head nothing anywhere (garbled contents leaders, figure labels '1 H', big lettering 'NN I ERE Ee' were headings).
5. paper is NOT switched to regions for full-page-image pages yet -- recorded as the next step; its vector crop stays for born-digital papers (keep the option).

Verified on real pages: BS-EN-60034-1 p20 Figure 2 (whole figure + key + caption; one short 'Example' line above included), p33 Table 1 (cut at page foot, continues), BS-EN-60034-26 p12 equations A.1-A.4 (OCR read A.2 as '(2)'; the label follows OCR, the picture is right).
Batch: all 37 standards OCR'd (60034-15 failed once under memory pressure -- 'Tesseract not installed' from a failed spawn; rerun with user consent via Telegram; batch.sh now sets TESSDATA_PREFIX).
