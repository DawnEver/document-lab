---
name: handoff-capture-files
description: Handoff -- capture-files (PR 38) state, open issues and next steps; regions/ink and corpus work around it
---
# Handoff: document-lab capture-files, regions from drawn ink

## Goal
Turn PDFs (secured standards, vendor whitepapers) into page images and per-page zips with their figures, tables and
equations cut out -- with no AI in the loop -- through `document-lab capture-files` (screen capture via Acrobat) and
`document-lab pages` (parse to `<stem>.zip`).

## Done (merged to integrate/main unless said)
- regions (`src/document_lab/regions.py`): untitled tables by rows of cells (#29), drawn-ink regions via new
  `src/document_lab/ink.py` handed in by `per_page` (#36: ruled tables, uncaptioned figures, unnumbered equations),
  page furniture dropped -- a drawing at the same place on >= half the pages (#37, `per_page._without_furniture`).
- markdown: a 4-digit year leading a line heads no clause (#35). Tests run on GitHub Actions alone (#34).
- plot: rotated y-axis title is no tick label (#32, closes issue 30); raster frame survives a curve over the x axis
  (#33); transformer-lane glyph/plot commits landed (#31).
- dev-lab #585: `branchprune delete <b> --approved "<words>"` deletes a branch whose tip a tag on origin holds; used to
  delete three old document-lab branches (archived as tags `archive/capture`, `archive/the-ops-app-may-break-glass`,
  `archive/transformer-glyphs-and-raster` -- they are the only copy of those pre-rebase commits; keep).
- Corpus outputs regenerated: 38 IEC zips under `...\261105-IEC\IEC-60034\extracted` and `IEC-60205\extracted`
  (1413 regions); 61 JMAG whitepaper zips under `...\261112-JMAG-Library\official\whitepapers\extracted` (529 regions).
- **capture-files (PR #38, branch `pr/capture-files`, worktree `.claude/worktrees/parse`) -- open, CI running.**
  `document-lab capture-files <pdf|dir|list.txt>... -o <dir> [--monitor N] [--area mark|view|L,T,W,H] [--ocr]`.
  Per PDF: Acrobat `/n` -> `desktop.wait_for_window` -> `maximize(window, monitor)` -> chords `ctrl+l` (full screen),
  `home` -> per page: `park_cursor` (mouse off the window), `_settled` grab, `page_box` (view mode) or the marked
  area, shape check vs PDF page aspect (`page_shapes`), `PageWriter.add`, `pagedown` (re-pressed `TURN_TRIES` times if
  the page did not turn) -> close. Files: `src/document_lab/capture/{files.py,cli.py(add_files_parser, run_files),
  desktop.py,export.py(PageWriter)}`, tests `tests/unit/test_capture_files.py`. 548 tests pass locally. Real run on
  the user's 1600x2560 portrait monitor 1: two whitepapers, 4+4 pages, each 1600x2263, correct.

## Current state
- PR #38 open, auto-merge on, a Monitor was watching it; if red, read the run record (see dev-lab `forge watch`).
- Uncommitted in the parse worktree: this handoff file only. `output/` there holds scratch (trial PDFs, scripts).

## Left to do (priority order)
1. See PR #38 merged; fix any CI red (environment-class errors: rerun the failed job, ack with reason).
2. Batch-run capture-files for real on the user's target set (ask which folder); spot-check page counts vs source
   and a few pages per file. Use `--monitor 1 --area view` (black full-screen backdrop makes view mode reliable).
3. Hardening of capture-files, each with a unit test where it is pure logic:
   - secured PDFs (no page count): end = a page turn that changes nothing -> two identical consecutive pages
     (blank) stop early; consider confirming the end with `ctrl+end` + compare, or a second turn.
   - `open_in` refuses when another window of the same PDF is open (find_window sees several); decide: refuse with a
     clear message (now) or close our own leftovers.
   - `--area view` relies on `page_box` (corner backdrop, gap closing `PAGE_GAP`); untested on a landscape page or
     a PDF whose pages differ in size.
   - `REDRAW`/`SETTLE` timings are guesses from one machine.
4. regions gaps (lower): JMAG-style captions `Fig. 1 Title` (no dash) are not recognised -- figures come out as
   `Untitled figure`; side-by-side figures without prose between merge into one. Publisher logos on 1-2 cover pages
   still count as untitled figures (furniture rule needs >= half the pages).

## Key decisions (do not relitigate)
- Page-by-page in full screen, not scroll-and-stitch: stitching lost its place at large zoom, blank stretches read
  as the end, Acrobat's panes/AI bar/floating toolbar polluted frames; full screen shows one page, fitted, on black.
- The mouse is parked OFF the viewer window: in read/full-screen mode Acrobat shows its toolbar under the mouse and
  its panes when the mouse nears the top.
- `--start` for `capture` was added then removed (no caller); `capture` scroll mode unchanged.
- Page furniture is judged in `per_page` (sees all pages); `regions` never reads pixels (`ink` gives it boxes).
- Remote branch deletion only through `branchprune` (LOSSLESS-DELETE, user ruling); `--approved` needs a holding tag.

## Open questions for the user
- Which PDFs to batch-capture first, and where outputs go.
- Whether `--area view` should become the default once proven on the target set (user preferred marking once).
