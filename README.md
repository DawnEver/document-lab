# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance: every value can say which
file (by sha256), page and box -- or HTML locator -- it was read from.

| module | does |
|---|---|
| `ir` | frozen IR: `Provenance`, `TextSpan`, `VectorPath`, `Table`/`Cell`, `Figure`, `Page`, `Document` |
| `pdf` | `read_pdf(path)`: spans, vector paths, ruled tables with per-cell boxes, image figures and captions (PyMuPDF) |
| `html` | `read_html(path)`: every `<table>` as a `Table` (standard-library parser) |
| `plot` | `plots(page)`: curves digitised from VECTOR plots (axes linear or log10, piecewise at a break; fragments joined; legend samples apart); a raster plot raises `RasterPlot` |
| `numbers` | the one number grammar: `parse_number` (whole text) and `first_number` (a cell's leading value) |
| `extract` | typed values against the caller's `Schema` from key-value (`from_rows`) or column (`from_columns`) tables |
| `ai` | `ask(responder, schema, page_or_table)`: the same values read by a model through ai-lab (`[ai]` extra) |
| `paper` | `document-lab paper <pdf> <out>`: a paper as per-section markdown (written from the IR, or marker-pdf on a GPU) plus cropped figures/tables/equations |
| `capture` | `document-lab capture <out.pdf>`: any window or screen region, scrolled to its end and stitched losslessly, to a PDF -- the stage before `read_pdf` (Windows, `[capture]` extra) |

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
```

A `Value` keeps the source `text` beside the typed `value` (a `≈`, `≤` or `±25%` is never lost) and a `basis`:
`rule` (deterministic code here) or `model`. A field the document does not state is absent, never `None`.

## Capture

```
document-lab capture --list                        # the capturable windows and their handles
document-lab capture out.pdf                       # click a window or drag a rectangle, then Enter/Start
document-lab capture out.pdf --no-scroll           # the same, one frame
document-lab capture out.pdf --window "Notepad" --tolerance 4 --ignore-top 48 --ocr chi_sim+eng
```

Targets: none marks one on a dimmed still of the screen (the mark shows clear) (a click: the window under it; a drag: a rectangle; mark again to
change it; Enter or Start confirms, Esc cancels), or `--window TITLE` (must match one), `--hwnd`, `--rect L,T,W,H`. Scrolling:
`--method wheel|message|keys`, `--notches`, `--settle`, `--max-frames`, `--max-height`; Esc stops and keeps what
was captured. A window is read through `PrintWindow`, so another window over it does not show; the wheel, which
does land on whatever covers it, refuses then (bring it forward, or `--method message`). A view that seems not to have
moved is looked at again before it is taken for the end, so a late-handled scroll does not cut the capture short.

Frames are stitched where their shared rows agree -- exactly by default; `--tolerance` (mean grey difference,
0-255) absorbs a blinking caret or hover highlight. `--ignore-top/--ignore-bottom` keep fixed chrome once. An overlap
that fits more than one offset, or none, stops the capture with an error rather than a wrong image: scroll by fewer
`--notches`, or raise `--min-overlap`. PDF: `--page-height` (pixels; default A4-shaped, `0` one page), `--dpi`,
`--ocr [LANGS]` (Tesseract languages, default `eng`), `--png`. Page size and OCR are checked before the capture starts,
and a PDF that still fails to write leaves the capture as a PNG beside it. In Python: `capture.session.capture(grab, scroll)` takes any source and
holds one frame at a time (`capture.stitch.Stitcher`); `capture.desktop.Source` is the Windows one.

## Install

`pip install "document_lab @ git+https://github.com/DawnEver/document-lab.git"`; extras: `gpu` (marker-pdf),
`ai` (ai-lab), `capture` (no packages beyond the base). Domain schemas live in the consumers.
