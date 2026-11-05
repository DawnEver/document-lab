---
name: document-lab-design
description: why document-lab is shaped as it is -- provenance IR, edge-based plot frames, one number grammar, pymupdf4llm pinned to 1.27.2.3 on Python <= 3.14, what was dropped from paper_pdf_ingest
created: 2026-11-05
---

# document-lab: design decisions

- IR is frozen values; every extracted thing carries `Provenance(sha256, uri, page 1-based, bbox in PDF points,
  y down | locator for HTML)`. A table cell without its own box cites the table's.
- `extract.Value` keeps the source `text` beside the typed value (qualifiers like `≈ ≤ ±25%` stay legible) and a
  `Basis` (`rule` | `model`). Unstated -> absent, never None; two matching columns -> `AmbiguousField`.
- `ai.ask` asks the model for value + verbatim quote per field; a found quote cites its span/cell, else the page or
  table -- never a guessed box. Optional extra; base imports without ai_lab (subprocess test).
- `numbers`: one grammar for tick labels (`parse_number`, whole text) and cells (`first_number`). Plain spaces do
  NOT join digits: legend text `100     100` read as 100100 before.
- `plot`, measured on a vendor ferrite sheet (0 plots -> 7): frames from axis-parallel EDGES meeting at a bottom-left
  corner (sheets draw frames as separate segments and rules as thin filled rects); labels snap to marks by ONE
  shared offset (nearest-mark snapping let log minor grid lines capture decade labels); a drawn dash negates a
  label or exponent; labels off the axis' common font size (unit superscripts) are dropped; a long rule anchored
  on a frame edge is a grid line even when a legend cuts it. Raster plots raise `RasterPlot`. Each shape has a
  synthetic control in tests/unit/test_plot.py; tests/vendor checks properties on a real PDF (opt-in env var).
- paper: pymupdf4llm pinned to 1.27.2.3. The e2e goldens reproduce byte-for-byte on it; 1.28.2, 0.3.4 and 0.2.9
  split the title off differently and shift every section by one (measured; the port's markdown was identical to
  the original's on the same dependency). 1.x needs pymupdf-layout -> onnxruntime with no cp315 wheel, so this
  venv is Python 3.13 (bootstrapped through the door with motronics-studio's interpreter).
- Dropped from paper_pdf_ingest: pdfplumber (unused), Pillow use (`pix.save` writes PNG), the `0-raw.pdf` +
  `1-paper-text/` layout (the out dir is the caller's; raw-image regex generalised to `*.pdf-NNNN`), the torch and
  macOS system_profiler GPU probes (nvidia-smi only; macOS -> 0), the swallow-all try/except around formula
  augmentation, choose_tool monkeypatching in tests (`convert(..., tool=)`), `make_test_pdf` (unused), the
  "PDF present" skips (the three PDFs are committed), hatch/pre-commit/coverage config, `ingest` script
  (now `document-lab paper`).
- Open: the full verify is INCONCLUSIVE by the 10% skip ceiling -- 51 marker (GPU) variants + 3 vendor tests skip
  on this box; they want their own part (resource) rather than skips in tests/base.
