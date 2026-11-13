# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance; the absorbed paper
ingester (`document-lab paper`); and the screen capture that makes such a PDF (`document-lab capture`: any window
or region, scrolled to its end and stitched into one long PDF -- the long screenshot). Generic and domain-free: no consumer's vocabulary here -- schemas live in
consumers. Modules and usage: `README.md`.

## Rules here

- Dependency arrow: `paper`, `ai` -> `extract`; `paper` -> `pdf`; `extract` -> `plot`, `pdf`, `html` -> `numbers`, `ir`;
  `plot` -> `glyphs`; `strokes` -> `plot`, `ir`; `capture` is apart (`cli` -> `session` -> `stitch`; `cli` -> `desktop`,
  `export`, `join` -> `export`, `stitch`), imported only by `__main__`. The base imports
  without `ai_lab` (`tests/unit/test_ai.py` proves it in a subprocess).
- A test that needs expected values builds a SYNTHETIC PDF (`tests/synthetic.py`); vendor PDFs are never
  committed. The three real papers are checked STRUCTURALLY against what each PDF says
  (`tests/paper/structure.py`), every check paired with a planted wrong truth it must report.
- `windows` (`tests/windows`: a real window, `viewer.py`, scrolled and captured) is an OS part: CI runs it on the
  hosted `windows-latest` runner. `gpu` (marker-pdf), `vendor` (`DOCUMENT_LAB_VENDOR_PDF=<pdf>`) and `ocr` (Tesseract)
  are declared resource parts, unproduced on
  hosted runners: base never selects them, and they FAIL where their resource is missing, never skip.
- A value the document does not state is absent, a raster plot raises, an ambiguous match refuses: never a
  silent default.
- No pymupdf4llm: paper markdown is written from the IR (`paper/markdown.py`). `pymupdf<2` because 2.x requires
  pymupdf4llm and onnxruntime (no cp315 wheel); the venv is the family's Python 3.15.
- Verdicts: `.venv/Scripts/python.exe -m dev_lab.dev.verify -- <paths>`. Family files are rendered
  (`tests/architecture/_famconfig.py`, `python -m dev_lab.dev.famfiles`, `python -m dev_lab.dev.ciworkflows render`).
- Design record: `.claude/memory/2026/11/05/`.
