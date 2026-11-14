---
name: untitled-tables-18-1
description: Why 60034-18-1 had no region, and the rules that find a table without a caption
---
# Untitled tables (60034-18-1)

Cause: 60034-18-1 has no numbered figure, table or equation; its only tables are the Annex ZA normative-reference
tables, set without a caption. Regions were told by captions and equation numbers only, so it had none.

Decision: `regions` also finds a table by its rows (`_untitled_tables`), labelled `Untitled table N`, file
`page-NNN-table-untitled-N.png`. Rules, each from a corpus case:
- row: >= ROW_CELLS (3) cells near the body size, none prose or a leader; a lone '-' may read small.
- a short cell (<= SHORT_CELL words) in the row: columns of prose, however OCR splits them, have none.
- no speck cell (< CELL_CHARS alphanumerics, not a dash): BSI back pages read 'oo', '.', 'i' as cells.
- a line of >= PROSE_WORDS (12) words is prose at any width: two-column back matter is under half width.
  Column alignment was tried and dropped: justified prose columns align as a table's do.
- a cell's wrapped line continues the row only if wordy and starting within a cell: a formula ends the table.
- the text width is taken over every body-sized line (not only wordy ones); the box is padded sideways.
Result over the 38 documents: 1118 regions, no untitled table on a last (publisher) page; 18-1 has pages 5 and 6.
All 38 zips regenerated. Open: page markdown takes a table row starting with a year ('2011 Evaluation ...') for a
clause heading in INDEX.md -- a markdown heading rule, not regions.
