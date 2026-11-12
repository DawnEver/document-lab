# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance: every value can say which
file (by sha256), page and box -- or HTML locator -- it was read from.

| module | does |
|---|---|
| `ir` | frozen IR: `Provenance`, `TextSpan`, `VectorPath`, `Table`/`Cell`, `Figure`, `Raster`, `Page`, `Document` |
| `pdf` | `read_pdf(path)`: spans, vector paths, ruled tables with per-cell boxes, image figures and captions (PyMuPDF); `read_raster(path, page, bbox)`: a region's pixels at its images' native resolution |
| `html` | `read_html(path)`: every `<table>` as a `Table` (standard-library parser) |
| `plot` | `plots(page)`: curves digitised from VECTOR plots (axes linear, log10, or piecewise between labelled ticks; labels as text or glyph outlines; fragments joined; legend samples apart); a raster plot raises `RasterPlot`. `raster_plot(raster)`: one plot traced from pixels, each point with its pixel uncertainty; unreadable labels raise `UnreadableTickLabels` unless stated (`x_values=`, `y_values=`) |
| `strokes` | thick strokes in a `Raster`: `fragments` (a morphological opening drops hairlines), `link` (dashes joined end to start along their own direction; crossing curves stay apart), `bands` (fragments to the nearest solid curve), `attach_dots` + `pattern` (a smaller opening keeps dash-dot dots, which name a curve as `dash-dot` beside a `dashed` one) |
| `glyphs` | digits read from glyph shapes (outlines or pixels) against Helvetica; a near tie or a glyph under 14 px raises `UnreadableGlyph` |
| `numbers` | the one number grammar: `parse_number` (whole text) and `first_number` (a cell's leading value) |
| `extract` | typed values against the caller's `Schema` from key-value (`from_rows`) or column (`from_columns`) tables |
| `ai` | `ask(responder, schema, page_or_table)`: the same values read by a model through ai-lab (`[ai]` extra) |
| `paper` | `document-lab paper <pdf> <out>`: a paper as per-section markdown (written from the IR, or marker-pdf on a GPU) plus cropped figures/tables/equations |

```python
from document_lab.pdf import read_pdf
from document_lab.plot import plots

doc = read_pdf('sheet.pdf', tables=False)
for page in doc.pages:
    for plot in plots(page):  # x_scale= / y_scale= force 'linear' or 'log10'
        for curve in plot.curves:
            print(curve.axis_scales, curve.x, curve.y, curve.dashes, curve.provenance.page, curve.provenance.bbox)
        for sample in plot.legend:  # a legend's sample line: its label and the style of the curve it names
            print(sample.label, sample.stroke, sample.dashes)

# A raster plot: its pixels, read at the images' native resolution; labels too coarse to read are stated.
raster = read_raster('sheet.pdf', 2, figure_bbox)  # from document_lab.pdf
plot = raster_plot(raster, x_values=[-25, 0, 25, 50, 150, 250], y_values=[0, 100, 200, 300, 400, 500])
for curve in plot.curves:  # curve.basis == 'raster'
    print(curve.x, curve.y, curve.x_uncertainty, curve.y_uncertainty)  # one pixel, in data units, per point
```

A `Value` keeps the source `text` beside the typed `value` (a `≈`, `≤` or `±25%` is never lost) and a `basis`:
`rule` (deterministic code here) or `model`. A field the document does not state is absent, never `None`.

## Install

`pip install "document_lab @ git+https://github.com/DawnEver/document-lab.git"`; extras: `gpu` (marker-pdf),
`ai` (ai-lab). Domain schemas live in the consumers.
