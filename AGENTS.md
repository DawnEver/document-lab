# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance; the absorbed paper
ingester (`document-lab paper`). Generic and domain-free: no consumer's vocabulary here -- schemas live in
consumers. Modules and usage: `README.md`.

## Rules here

- Dependency arrow: `paper`, `ai` -> `extract` -> `plot`, `pdf`, `html` -> `numbers`, `ir`. The base imports
  without `ai_lab` (`tests/unit/test_ai.py` proves it in a subprocess).
- A test that needs expected values builds a SYNTHETIC PDF (`tests/synthetic.py`); vendor PDFs are never
  committed. `DOCUMENT_LAB_VENDOR_PDF=<pdf>` runs `tests/vendor` against a real one.
- A value the document does not state is absent, a raster plot raises, an ambiguous match refuses: never a
  silent default.
- The venv runs Python <= 3.14: the `paper` extra pins pymupdf4llm 1.27.2.3, whose layout model needs onnxruntime
  (no cp315 wheel); `tests/paper/e2e` goldens hold only on that version (`pyproject.toml`).
- Verdicts: `.venv/Scripts/python.exe -m dev_lab.dev.verify -- <paths>`. Family files are rendered
  (`tests/architecture/_famconfig.py`, `python -m dev_lab.dev.famfiles`, `python -m dev_lab.dev.ciworkflows render`).
- Design record: `.claude/memory/2026/11/05/`.
