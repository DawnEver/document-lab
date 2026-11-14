# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance: every value can say which
file (by sha256), page and box -- or HTML locator -- it was read from.

| module | does |
|---|---|
| `ir` | frozen IR: `Provenance`, `TextSpan`, `VectorPath`, `Table`/`Cell`, `Figure`, `Raster`, `Page`, `Document` |
| `pdf` | `read_pdf(path)`: spans, vector paths, ruled tables with per-cell boxes, image figures and captions (PyMuPDF); `read_raster(path, page, bbox)`: a region's pixels at its images' native resolution |
| `html` | `read_html(path)`: every `<table>` as a `Table` (standard-library parser) |
| `plot` | `plots(page)`: curves digitised from VECTOR plots (axes linear, log10, or piecewise between labelled ticks; labels as text or glyph outlines; fragments joined; legend samples apart); a raster plot raises `RasterPlot`. `raster_plot(raster)`: one plot traced from pixels, each point with its pixel uncertainty; unreadable labels raise `UnreadableTickLabels` unless stated (`x_values=`, `y_values=`) |
| `strokes` | thick strokes in a `Raster`: `fragments` (a morphological opening drops hairlines), `link` (dashes joined end to start along their own direction; crossing curves stay apart), `bands` (fragments to the nearest solid curve), `dots` + `split_by_dots` (a smaller opening keeps dash-dot dots; they trace the dash-dot curve, telling it from a dashed one two strokes away) |
| `glyphs` | digits read from glyph shapes (outlines or pixels) against Helvetica; a near tie or a glyph under 14 px raises `UnreadableGlyph` |
| `numbers` | the one number grammar: `parse_number` (whole text) and `first_number` (a cell's leading value) |
| `extract` | typed values against the caller's `Schema` from key-value (`from_rows`) or column (`from_columns`) tables |
| `ai` | `ask(responder, schema, page_or_table)`: the same values read by a model through ai-lab (`[ai]` extra) |
| `ocr` | `document-lab ocr <pdf> [-o out] [--lang eng]`: a Tesseract text layer over every page that is one image and has no text; the pixels kept exactly, a page never laid twice |
| `layout` | `lines(page)` / `ordered(lines)`: a page's spans as lines in reading order (columns, a clause number with its title); `wordy(text)` tells words from OCR noise |
| `markdown` | `to_markdown(doc)` / `page_markdowns(doc)`: the IR as markdown -- headings, paragraphs -- whole, or each page standing alone |
| `regions` | `regions(page, body)`: a page's figures, tables and equations from its text alone -- a caption and the prose around it, an equation number and the noise beside it -- so an OCR'd capture has them too |
| `per_page` | `document-lab pages <pdf> <out>`: a PDF as `page-NNN.md` (its text) and `page-NNN.png` (the page as seen, a captured page's own pixels) per page, each region cut out (`page-NNN-figure-2.png`, `-table-1`, `-equation-a-1`) and linked where its caption or number is, and `INDEX.md` -- for standards, manuals and data sheets, cited by clause on their page. OCR'd text is for finding; a formula's or a table's values are read off its picture |
| `paper` | `document-lab paper <pdf> <out>`: a paper as per-section markdown (written from the IR, or marker-pdf on a GPU) plus cropped figures/tables/equations; a table-of-contents page opens no sections, and a page that is one picture (a screen-captured or scanned PDF) is linked whole, once, never cropped into copies |
| `capture` | `document-lab capture <out.pdf>`: any window or screen region, scrolled to its end and stitched losslessly, to a PDF of page images -- the stage before `ocr` (Windows, `[capture]` extra; `--ocr LANGS` runs `ocr` on the result); `--resume` and `capture-merge` join captures where they overlap |

A screen-captured standard, end to end: `capture std.pdf` -> `ocr std.pdf` -> `pages std.pdf std/`.

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

## Capture

```
document-lab capture                               # mark an area, Confirm, Start, scroll it yourself, Finish
document-lab capture out.pdf --drop-overlays       # paint out floating buttons and sticky bars where they recur
document-lab capture out.pdf --auto                # it scrolls for you, to the end
document-lab capture out.pdf --no-scroll           # one frame
document-lab capture out.pdf --window "Notepad" --ocr chi_sim+eng
document-lab capture --list                        # the capturable windows and their handles
document-lab capture out.pdf --page-height auto    # pages cut on the reader's own page separators
document-lab capture --resume part.pdf             # go on from where part.pdf ended, into part.pdf
document-lab capture-merge a.pdf b.pdf -o ab.pdf   # join captures where each overlaps the one before
document-lab capture out.pdf --open                # open the PDF once written
```

Requires Windows and a tkinter that starts: Python 3.13.0 in a venv cannot (cpython #125235, `Can't find a usable
init.tcl`); use Python >= 3.13.1, or copy the base install's `tcl` folder into `.venv\tcl`. The capture checks this
before it opens any window and prints that remedy.

Targets: none dims every monitor, live, the last area confirmed already marked (kept in
`%LOCALAPPDATA%\document-lab\capture-area.json`; not if it is off the screen now) -- the window under the pointer
is outlined; a click marks it, a drag marks a
rectangle (shown clear, the rest dim, its W x H above its corner) -- a new drag, inside it or out, marks again; its edges, corners and top handle
drag to adjust it, arrows nudge it (Shift: 10 px, Ctrl: resize); Enter or Confirm sets it, Esc cancels -- or
`--window TITLE` (must match one), `--hwnd`, `--rect L,T,W,H`. Once set, nothing covers the desktop: open or arrange
other windows freely while the area waits, framed in blue, until the panel beside it starts the capture:
**Start (I scroll)** follows your own scrolling, **Auto scroll** scrolls for you to the end (as `--auto`); either way
Finish ends it, and **Pause** holds it: nothing is grabbed or scrolled -- scroll anywhere, use any window -- until
Resume, which picks up wherever the view overlaps what was captured (red, with the way back, until it does).
Adjust reopens the selection with the area still marked, to drag or redo it (Esc there keeps it as it was); Cancel
quits, nothing written.

Scrolling: by default you scroll -- wheel, scrollbar, keys, at your own pace -- and it follows, grabbing a frame every
`--every` seconds and placing each new position; pauses end nothing. A view that does not follow the last one -- a
flick too far, or back up -- is looked for in the last three views captured and picks up wherever it overlaps them;
only a view outside all of them is skipped, the frame turning red with which way to go back. The Finish button on
the panel ends it (not a key, which is too easily hit). Frames placed to a fraction of a row add up without drift. `--auto` scrolls for you
(`--method wheel|message|keys`) until the view stops moving, each step about half the view: a notch is measured, not
assumed, and the notches grow to `--stride` of the view (default 0.5; 0 keeps `--notches`, the first step, every step);
every `--streak` steps placed in a row (default 3) the step grows a quarter, up to `--stride-max` (default 0.75), and
the settle gap halves back toward `--settle` -- a step that loses the view drops back to `--stride` (`--stride-max`
at `--stride`: a fixed pace) -- `--end-wait` (default 0.8 s) gives a feed
that loads more time to grow -- and says so, exit 1, if its first scroll moved nothing. Neither has a length cap unless
`--max-frames`/`--max-height`; the height so far shows while it runs. A frame `--auto` cannot place -- caught
mid-render, or a step past the overlap -- is looked at again, then found by scrolling back, the steps halved; only
one it cannot recover ends it (kept by `--keep-lost`). A view is settled when two grabs `--settle` seconds apart
(default 0.12) are equal; a frame taken for settled that a second look then places was still rendering, so the gap
doubles for every later step, up to `--settle-max` (default 1.0) -- a slow renderer (Acrobat in reading mode) gets
the wait it needs instead of bouncing between its first pages; raise `--settle` to start there. A reader in
single-page / page-flip mode shares nothing between frames: the capture stops and says to switch it to continuous
scrolling (no `--tolerance` helps). A scroll Windows refuses (a UAC prompt, an elevated window in front, the lock
screen) ends it the same way: what was stitched is written, the reason printed, exit 1. An output that exists is refused before anything unless
`--overwrite`: it may be the very document on screen; so is a name Windows cannot hold (`a:b.pdf` would be written as
a hidden stream of a file `a`), with one it can. A window is read
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
scrolls. PDF: `--page-height` (pixels; default A4-shaped, `0` one page; `auto` finds the reader's page separators --
rows that recur, equally spaced, from the first page to the end, not a viewer border repeating at the scroll step --
and starts each page on one, A4-shaped if there are none, and says which; `capture.pages.page_rhythm`), `--dpi`, `--ocr [LANGS]` (the finished
PDF through `ocr`, default `eng`), `--png`; page size and Tesseract are checked before the capture starts.

Resuming: a capture that stopped part way -- Finish, a lost view, a crash -- goes on with `--resume part.pdf`: bring
the view to anywhere overlapping where it ended and Start; a start whose area is not as wide, or whose view shows
nothing of that end, is refused on the panel with what to do, and it waits again. What is captured is joined to it
(into `part.pdf`, or the output named). `capture-merge a.pdf b.pdf [...] -o out.pdf` joins finished captures the same
way (`capture.join.join`): each one's first rows are found in the end of the one before and the rows both show kept
once, matched at `--join-tolerance` (`capture-merge --tolerance`; default 20, as the document reopened renders the same
text ~15 grey levels apart); the search reaches back as far as the earlier capture goes; only the seam is redrawn,
every other page copied as it is (its OCR layer too). Captures that share nothing,
fit more than one place, or differ in width are refused, never butted together.

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
