---
name: desktop-capture-to-pdf
description: "document-lab capture: window/region auto-scroll stitched to PDF, the stage before read_pdf"
metadata:
  type: project
---

Added on pr/capture (3d4ccf5, local, not pushed): src/document_lab/capture/{stitch,session,desktop,export}.py and `document-lab capture`. User ruling: lives in document-lab as the `[capture]` extra (empty: ctypes+tkinter+PyMuPDF suffice) until further iteration justifies a split. Stitching is exact by default; optional --tolerance (mean grey diff on 64 column bins) absorbs carets/hovers (user asked for it); ambiguous/no overlap raises, never guesses. New unproduced resource part `desktop` (tests/desktop, Tk viewer fixture, pixel-exact, passes locally). Untested: --ocr, method=message, interactive --pick/--region. Open: push+PR awaits user confirmation; 3 architecture tests fail on clean integrate/main (stale family renders, not this change); hook install was permission-denied; leftover worktree .claude/worktrees/base.
