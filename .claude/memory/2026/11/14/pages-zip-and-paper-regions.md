---
name: pages-zip-and-paper-regions
description: "pages writes one .zip per document with the PDF inside; extracted/ converted; paper cuts picture-page figures via regions"
from:
  - 2026/11/14/regions-from-text-and-layout-layer.md
metadata:
  type: project
---

2026-11-14, user asked: every document under extracted/ must be one archive; and paper should use regions on picture pages.

1. document-lab pages <pdf> <out>: out ending .zip writes ONE archive (PNG stored, text deflated, .part then move into place); a directory otherwise. Either holds the PDF it was read from (skipped when already at out/<name>), so the layout stands alone and dir == zip content.
2. IEC extracted/: 38 folders (37 under IEC-60034, 1 under IEC-60205) converted to <stem>.zip by a script that zipped, re-read every member and compared bytes, and only then deleted the files. The empty folders refused deletion: OneDrive marks them ReadOnly+ReparsePoint -- cleared ReadOnly, then removed (0 left). extracted/ now holds only .zip files.
3. paper/render.py: on a full-page-image page, a figure/table label is cut where regions locates it (Figure N <dash> captions), an equation by its (N) label; else the whole page once, as before. Paper-style 'Figure 1.' captions do not match regions' dash rule, so captured papers still link whole pages -- deliberate, not widened untested.
4. Background jobs were reaped 3x for low system memory (Everything, node, VS Code, Chrome hold most of 32 GB); the last regeneration ran in foreground slices of <= 7 min instead.
