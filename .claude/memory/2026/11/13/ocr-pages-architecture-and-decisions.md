---
name: ocr-pages-architecture-and-decisions
description: "Decision log (user away, full authority): OCR and markdown as their own stages, document-lab ocr/pages, IEC batch extraction"
metadata:
  type: project
---

2026-11-13, user away for a day with full authority to decide; this is the log for their review.

Survey: document-lab's only OCR was capture --ocr inside capture/export.PdfWriter (Tesseract per page while writing), so a capture made without it could never get a text layer -- why all 36 IEC-60034 captures (and BS-EN-60205) had none. paper's IR path needs a text layer; paper --tool marker (marker-pdf 2.x + surya-ocr 0.23) now needs an inference server (vllm via Docker, or llama.cpp) -- neither here, so marker failed after 9 min (docker binary not found). The user's own earlier workflow (261105-IEC/feedback-for-author.md) used Tesseract 5.5.3 and a per-page export (md + page image per page) by an outside script, because paper's sections misread standards.

Decisions (branch pr/ocr-pages, worktree .claude/worktrees/ocr):
1. OCR is one stage, document_lab.ocr (document-lab ocr): a Tesseract layer over every page that is one image and has no text; the original page and image untouched (Tesseract's page shown over it with its image stripped from content and resources); a page never laid twice; check() only once there is work. PdfWriter writes images only; capture --ocr kept (a real option) but runs ocr on the finished PDF -- one source of OCR.
2. markdown leaves paper (document_lab.markdown) so a second layout uses it without reaching into paper; page_markdowns beside to_markdown.
3. document_lab.per_page (document-lab pages): page-NNN.md + page-NNN.png (a captured page's own pixels) + INDEX.md naming each page by the clause headings it opens. Exit 1 only when no page has text.
4. Headings in OCR'd text: a document with NO bold span is judged unweighted -- a short numbered line (<= 8 words, no sentence end) heads its clause, and a bare clause number or bullet joins the text set apart beside it (CLAUSE_GAP 8 font sizes). Weighted documents unchanged: a first, ungated version broke the three real papers' structure checks (figure labels like '2 A3' became headings), so it was gated.
5. marker kept as an option (keep-valuable-options) though it needs a server now; not installed Docker/llama.cpp.
6. Tesseract: winget install by the agent was refused by the permission classifier; the user ran it (5.4.0, per-user at %LOCALAPPDATA%/Programs/Tesseract-OCR, eng+osd only, not on this shell's PATH -- prefixed per run).
7. Layout of results: scanned/ stays the source; extracted/<stem>/ holds <stem>.pdf (capture + text layer), page-NNN.md/png and INDEX.md -- derived, rebuildable.
8. scanned/*.pdf recompressed in place (garbage=4, deflate) only after every page's pixels compared equal: e.g. 1345 -> 20 MB, 13 GB -> few hundred MB.

Known limits: Tesseract garbles table-of-contents leader dots, formulas and some tables; prose is near verbatim. Trial: BS-EN-60034-26 (16 pages) ocr 44 s, pages 12 s.
