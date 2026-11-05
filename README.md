# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance: every value can say which
file (by sha256), page and box -- or HTML locator -- it was read from.

| module | does |
|---|---|
| `ir` | frozen IR: `Provenance`, `TextSpan`, `VectorPath`, `Table`/`Cell`, `Figure`, `Page`, `Document` |
| `pdf` | `read_pdf(path)`: spans, vector paths, ruled tables with per-cell boxes, image figures and captions (PyMuPDF) |
| `html` | `read_html(path)`: every `<table>` as a `Table` (standard-library parser) |
| `plot` | `plots(page)`: curves digitised from VECTOR plots, axes linear or log10; a raster plot raises `RasterPlot` |
| `numbers` | the one number grammar: `parse_number` (whole text) and `first_number` (a cell's leading value) |
| `extract` | typed values against the caller's `Schema` from key-value (`from_rows`) or column (`from_columns`) tables |
| `ai` | `ask(responder, schema, page_or_table)`: the same values read by a model through ai-lab (`[ai]` extra) |
| `paper` | `document-lab paper <pdf> <out>`: a paper as per-section markdown plus cropped figures/tables/equations |

```python
from document_lab.pdf import read_pdf
from document_lab.plot import plots

doc = read_pdf('sheet.pdf', tables=False)
for page in doc.pages:
    for plot in plots(page):  # x_scale= / y_scale= force 'linear' or 'log10'
        for curve in plot.curves:
            print(curve.axis_scales, curve.x, curve.y, curve.provenance.page, curve.provenance.bbox)
```

A `Value` keeps the source `text` beside the typed `value` (a `≈`, `≤` or `±25%` is never lost) and a `basis`:
`rule` (deterministic code here) or `model`. A field the document does not state is absent, never `None`.

## Install

`pip install "document_lab @ git+https://github.com/DawnEver/document-lab.git"`; extras: `paper`
(pymupdf4llm, Python <= 3.14), `gpu` (marker-pdf), `ai` (ai-lab). Domain schemas live in the consumers.
