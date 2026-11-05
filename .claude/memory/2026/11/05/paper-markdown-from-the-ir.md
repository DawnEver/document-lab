---
name: paper-markdown-from-the-ir
description: pymupdf4llm dropped -- paper markdown written from the IR, structural e2e checks with planted wrong truths, venv back on Python 3.15 with pymupdf<2, gpu and vendor as unproduced resource parts
created: 2026-11-05
corrects: document-lab-design
---

# Paper markdown from the IR; structural checks; Python 3.15

Corrects document-lab-design's pymupdf4llm pin, Python <= 3.14 venv, and "open: verify inconclusive" lines.

- paper: pymupdf4llm was first pinned to 1.27.2.3 (the byte goldens reproduced only on it;
  its onnxruntime has no cp315 wheel, so the venv went to 3.13). Owner decision (coordinator, user autonomy grant):
  DROP pymupdf4llm. `paper/markdown.py` writes the markdown from the IR: lines by baseline + word gap (a wider gap is a
  column gutter), column by extent, bands between full-width lines (left then right), median span top/bottom (a tall
  bracket glyph must not reorder lines); title = text >= 1.6x body on any page (a bound-in paper's title is on a later
  page); `##` = Roman upper-case or REFERENCES/ACKNOWLEDGMENT/APPENDIX/CONCLUSION, or bold numbered; an upper-case line
  right under a heading wraps into it; lines join within 0.6 line heights (0.3 overlap allowed); a lower-case block
  rejoins the nearest unterminated paragraph across a column/page break (footers between); body text starting `#` is
  escaped. Formula augmentation went with it (it keyed on pymupdf4llm image names).
- e2e: the byte/pixel goldens are gone; `tests/paper/structure.py::problems` checks title, section headings in order,
  each captioned figure/table number -> an image file (and INDEX.md pointing at it), no image for an uncaptioned
  number, appended-paper count. Truth read off the PDFs. Planted: 10 falsified truths x 3 papers must each be
  reported, plus liang+hairpin bound into one PDF must split into 1 appended paper (and is reported if undeclared).
  The checks found real defects, fixed: images rendered for a bare mention ("Table IV") -> page map is captions only;
  a caption as a line opening with its label (a 405-char block, "TABLE I" over its title) was missed; a one-line
  ACKNOWLEDGMENT was dropped as noise -> recognised headings are kept; a bound paper split twice (title, then its
  I. INTRODUCTION) -> a paper just opened by its title is not re-split.
- Environment: back on the family Python 3.15 (door bootstrap with ai-lab's interpreter). `pymupdf<2`, NOT bare:
  PyMuPDF 2.x (2.1, measured) itself requires pymupdf4llm + pymupdf-layout + onnxruntime, uninstallable on cp315.
  Locked pymupdf 1.28.2. The 3.13 venv is parked in output/logs/26/11/05/venv-py313 (delete refused by the guard).
- Parts: `gpu` and `vendor` are `[tool.lab_commons.tests] resources`, `[tool.dev_lab.parts] unproduced` (neither
  exists on a hosted runner); base selects `not gpu and not vendor`; each FAILS without its resource (ai-lab `live`
  convention), never skips. tests/architecture/test_resource_tests_are_their_own_parts.py pins it.
- Measured 2026-11-05: base selection (tests -m "not gpu and not vendor") 196 passed, verify pass; vendor part with a
  ferrite sheet 3 passed; gpu part fails here ("no CUDA GPU with >= 4 GB VRAM"), as declared.
- Also dropped from paper_pdf_ingest: pymupdf4llm itself and the formula augmentation keyed on its image names.
