---
name: iec-corpus-batch
description: 2026-11-15 IEC corpus batch -- 60072/60317 captured, 52 standards sorted out of ToSort, run-all supervisor, the defects found on real DRM PDFs
---
# IEC corpus batch (2026-11-15, user away a day, full authority, decisions logged here)

## What the user asked
`BS-EN-IEC-60072` and `BS-EN-IEC-60317` from Downloads: scanned version (image only -- the original text/DRM content
is not kept) plus `extracted`, into
`...\260306-Motronics_Studio\archived\261105-IEC\IEC-<family>/{scanned,extracted}`, following IEC-60034.
Then (later the same evening): classify and scan every PDF under `Downloads\ToSort`. Then: shrink the PDFs
losslessly. Everything recorded here for review; the user is away until 2026-11-16.

## Decisions taken (and why)
1. **Layout** `IEC-<family>/scanned/<BS-EN-....pdf>` + `extracted/<same stem>.zip` -- as IEC-60034. `scanned` is the
   capture (page images, no text layer); `extracted` holds the OCR'd copy + page-NNN.md/png + regions + INDEX.md.
2. **Names**: the download name with spaces as dashes (`BS EN IEC 60317-0-2_2020.pdf` -> `BS-EN-IEC-60317-0-2_2020.pdf`);
   a download-time stamp `--[2026-11-14--03-47-43 PM]` dropped, and `-2020` becomes `_2020` so a re-download and its
   original are one standard. 98 files in BS-EN-IEC-60317 were 68 standards (the folder was downloaded twice).
3. **One capture-files run per family**, not one per PDF: the mouse is parked once for the run and given back at its
   end (`Park`). A crash loses at most the PDF in hand (`scanned/*.part` is quarantined at the next run's start).
4. **ToSort**: sorted into `Downloads\ToSort\IEC-<family>\` by the standard's own number (60404, 60216, 60455, 60464,
   60626, 60349, 60085), names as in (2); the originals are moved, not copied, and never deleted. All 52 are
   FOPN_foweb-secured and none opens in PyMuPDF, so the screen-capture path is the only one.
5. **Lossless shrink** of every `scanned` PDF: `save(garbage=4, deflate=True)` with every page's image bytes compared
   before the rewrite is kept. The gain is large because the writer's *incremental* saves stored each later page's
   image uncompressed (BS-EN-60317-0-1: 252 MB -> 7 MB, pixels equal). Fixed at the source too (export._slim).
6. **Safe word**: `output\iec2\STOP` (or the panel's Stop). The PDF in hand stops at its page, its pages are kept as
   `<name>.partial.pdf` in `output/iec2/partials/`, the batch stops (exit 2). `rm` the file to go on.
7. **Interference**: the viewer takes the foreground for every key, so the capture holds while another window is in
   front and the machine was used within 2 s; the mouse is only moved when it rests on the captured window, and it
   is given back where the user left it. `--panel` is a small Pause/Resume/Stop window on the other monitor.

## Defects found on the first real DRM run (all fixed, each named so the next reader knows)
- `open_in` looked for a window titled with the *file name*; Acrobat titles a secured standard by its own
  (`BS EN IEC 60072-1:2022 (SECURED)`), so every PDF was refused. Now: the window that appeared since the viewer
  started and holds its page view.
- `ctrl+l` was judged cleared after one `REDRAW`; a slow apply was re-pressed and toggled back off. Now polled.
- A bare backdrop (viewer still drawing) was refused; now waited for (`PAGE_WAIT`).
- Acrobat remembers a document's zoom; fit-width cut the page's head and foot. `ctrl+0` added to the chords.
- The mouse was parked at the virtual screen's *top right* -- the close button of the window there. Now bottom
  corners, 40 px inset, and given back afterwards.
- Two `_report` in capture/cli.py: the scroll capture's won, so every capture-files run died at its summary with
  `TypeError` (capture written, exit status wrong).
- Incremental saves left page images uncompressed (see 5).

## Where things are
- Worktree `.claude/worktrees/parse`, branch `pr/capture-files` (PR #38). Scripts (scratch, gitignored):
  `output/iec2/{classify,left,finish,rename,audit,shrink,close}.py`, `{capture4,extract,run-all}.sh`,
  `<family>.map.json` (source -> archived name), logs `capture-<fam>.log`, `extract-<fam>.log`, `run-all.log`,
  `audit.log`, `shrink.log`; `partials/` holds what was cut short.
- Commits this session: window by title (84b1c54), hold/stop-file/speed (5188d0c), bottom-corner parking (f7eaae2),
  --panel (f4e1b0e), mouse once per run (4c5d0b1), `_report` shadowing (23674c1), settle reuse (fa02368),
  deflated images (aebe1fd).

## Two more defects, found only by auditing the captures themselves
- **Acrobat's crash-recovery bar is a window of its own** (`Adobe Acrobat`, 886x216) over the page view, so a screen
  grab caught it and the capture kept it as part of a page -- 26 of 60317's first captures. It is put up when the
  previous session ended abruptly, which a window closed from outside (WM_CLOSE) is to Acrobat. Fixed twice over:
  the document is shut with the viewer's own key (`ctrl+w`, `Viewer.close`), and a window that appears after the
  viewer started and holds no page view is a dialog, and is closed (`_wait_for_document`).
- **A modal dialog blocks its window completely**: BS EN 60317-59 opens with "Cannot find or create the font
  'BSsymb10'", modal. The window cannot be brought to the front and takes no key, so the PDF was refused every
  time (and each attempt left a window behind). `_press` now answers the dialog (Enter goes to whatever holds the
  keyboard, `desktop.key`) and presses again -- at open and at every page turn.

## Audit, and what it is
`output/iec2/audit.py <families>`: every source has a capture and an archive; no page repeats the one before; the
recovery bar is in no page; page counts; a page without text is named. Plus the completeness check that mattered
most: **every archive's last page is the BSI back page** (NO COPYING WITHOUT BSI PERMISSION / bsi group headquarters
/ Chiswick). `fixup.py` removes a capture that fails a check (and its line in `<family>.done`, or the extraction
never rebuilds it), `fixup-run.sh` takes them again in up to three rounds. Found by these: 27 captures of 60317 with
the bar (twice), 0-8 (5 -> 34 pages), 17 (6 -> 16), 12 (22 -> 18+back page), 18 (9 -> 16).

## Final state (2026-11-16 05:00)
- 123 standards across 9 families, every one captured, OCR'd, paged into one archive each, shrunk, and audited:
  `audit.py` 0 findings, and every archive ends on the BSI back page.
- IEC-60072 3, IEC-60317 68, IEC-60404 23, IEC-60216 12, IEC-60455 8, IEC-60464 4, IEC-60626 3, IEC-60349 1,
  IEC-60085 1. New `scanned`: 1.01 GB, new `extracted`: 2.18 GB (the page PNGs are most of it; the zips hold them
  and the OCR'd PDF, and are derived).
- Lossless shrink on `scanned`: 252 MB -> 7 MB on the worst (BS-EN-60317-0-1), pixels compared page by page.
- Commits: window by title (84b1c54), hold/stop-file/speed (5188d0c), bottom-corner parking (f7eaae2), --panel
  (f4e1b0e), mouse once per run (4c5d0b1), `_report` shadowing (23674c1), settle reuse (fa02368), deflated images
  (aebe1fd), ctrl+w close (685fa32), dialogs closed (7a3f1da), modal answered (c85187c).
- PR: branch pr/capture-files pushed; **draft PR #40** opened (a draft merges nothing until the user marks it ready).
- Resume or re-check: `bash output/iec2/run-all.sh` (skips what exists), `bash output/iec2/fixup-run.sh`,
  `python -I output/iec2/audit.py <family>...`.
