---
name: capture-not-on-integrate-main
description: "document-lab capture (long screenshot) lives on pr/capture-2, unmerged; run via .venv"
metadata:
  type: project
---

As of 2026-11-13 `document-lab capture` (window/region scrolled and stitched to one long PDF) exists only on pr/capture-2 (ac02f6e), not merged into integrate/main; on integrate/main src/document_lab/capture/ holds only __pycache__, so an agent there wrongly concludes no screenshot feature exists. AGENTS.md intro omitted capture -- added on pr/capture-2 (uncommitted at session end). The entry point is not on PATH: use .venv\Scripts\document-lab.exe capture out.pdf (editable install, works only while the worktree is on pr/capture-2).
