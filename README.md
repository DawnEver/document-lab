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
document-lab capture                               # mark an area, Confirm, Start, scroll it yourself, Finish
document-lab capture out.pdf --drop-overlays       # paint out floating buttons and sticky bars where they recur
document-lab capture out.pdf --auto                # it scrolls for you, to the end
document-lab capture out.pdf --no-scroll           # one frame
document-lab capture out.pdf --window "Notepad" --ocr chi_sim+eng
document-lab capture --list                        # the capturable windows and their handles
```

Targets: none dims every monitor, live -- the window under the pointer is outlined; a click marks it, a drag marks a
rectangle (shown clear, the rest dim) -- a new drag, inside it or out, marks again; its edges, corners and top handle
drag to adjust it, arrows nudge it (Shift: 10 px, Ctrl: resize); Enter or Confirm sets it, Esc cancels -- or
`--window TITLE` (must match one), `--hwnd`, `--rect L,T,W,H`. Once set, nothing covers the desktop: open or arrange
other windows freely while the area waits, framed in blue, until Start on the panel beside it begins the capture
(Cancel quits, nothing written).

Scrolling: by default you scroll -- wheel, scrollbar, keys, at your own pace -- and it follows, grabbing a frame every
`--every` seconds and placing each new position; pauses end nothing. A view that does not follow the last one -- a
flick too far, or back up -- is looked for in the last three views captured and picks up wherever it overlaps them;
only a view outside all of them is skipped, the frame turning red with which way to go back. The Finish button on
the panel ends it (not a key, which is too easily hit). Frames placed to a fraction of a row add up without drift. `--auto` scrolls for you
(`--method wheel|message|keys`, `--notches`) until the view stops moving -- `--end-wait` (default 0.8 s) gives a feed
that loads more time to grow -- and says so, exit 1, if its first scroll moved nothing. Neither has a length cap unless
`--max-frames`/`--max-height`; the height so far shows while it runs. A window is read
through `PrintWindow`, so another window over it does not show; the wheel, which does land on whatever covers it,
refuses then (bring it forward, or `--method message` -- which some applications act on only some of the time, so it
may stop early); `keys` refuses a window it cannot bring forward.

Stitching (`capture.stitch`): what stays put while the page scrolls -- a sticky header, a floating button, a scrollbar
thumb -- is left out of the match, so no setting is needed for it. The scroll is the offset at which the frames are
most alike -- not exactly alike: at 175% scaling an application scrolls by fractions of a pixel and re-renders the text,
so no offset ever matches pixel for pixel. Its mean grey error, the worst `--outliers` share (default 5%) left out, must
be at most `--tolerance` (default 8) and clearly below every offset that is not its neighbour; candidates come from the
mean squared difference of every row at every offset, by FFT. An offset that is not unique (a repeating view), or none, stops the capture there: what was stitched is written, the reason is printed and the exit status is 1.
`--drop-overlays` (off by default) also removes the overlays from where they would recur: each spot of the page is
taken from the frame that showed it least covered. `--ignore-top/--ignore-bottom` exclude chrome that changes as it
scrolls. PDF: `--page-height` (pixels; default A4-shaped, `0` one page), `--dpi`, `--ocr [LANGS]` (Tesseract, default
`eng`), `--png`; page size and OCR are checked before the capture starts.

Any length: frames only move down, so every stitched row above the current view is final and is written at once --
PDF pages as they fill (saved incrementally), the PNG compressed as it grows -- and a capture holds about one view in
memory however long the page (`--page-height 0`, one page, is the exception: it is held whole). Both files are written
as `<name>.part` and renamed into place when the capture ends, however it ends, so a reader never sees half a file.
Everything runs in one thread, in order -- grab, scroll, stitch, write -- so nothing races. In Python:
`capture.session.capture(grab, scroll, sink)` takes any source and any sink (`Rows` keeps the image in memory;
`capture.export.PdfWriter`/`PngWriter` stream it) and returns a `Capture` (frames, height, why it stopped);
`capture.desktop.Source` is the Windows source.

## Install

`pip install "document_lab @ git+https://github.com/DawnEver/document-lab.git"`; extras: `gpu` (marker-pdf),
`ai` (ai-lab), `capture` (no packages beyond the base). Domain schemas live in the consumers.
