"""Label → page and page → section mapping utilities, over a document read into the IR."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from document_lab import layout
from document_lab.paper.text import RN_PATTERN as _RN

if TYPE_CHECKING:
    from document_lab.ir import Document

FIG_TABLE_RE = re.compile(rf'(Figure\s+\d+|Fig\.\s*\d+|Table\s+(?:\d+|{_RN}))', re.IGNORECASE)


def build_figure_page_map(doc: Document) -> dict[str, int]:
    """Normalized figure/table labels -> the 0-based page of their CAPTION.

    A caption is a LINE opening with its label (``Fig. 3. ...``, ``TABLE I`` over its title), or a compact block
    holding ``label.``; of several, the most compact block wins (a body cross-reference sits in a tall paragraph).
    A label only mentioned, never captioned (``see Table IV``), has no figure to crop and is absent.
    """
    captions: dict[str, tuple[float, int]] = {}
    for page_num, page in enumerate(doc.pages):
        for blk in layout.blocks(page):
            lines = [ln.text.strip() for ln in blk.lines]
            text = ''.join(lines)
            height = blk.bbox.height
            for m in FIG_TABLE_RE.finditer(text):
                label = re.sub(r'\s+', ' ', m.group(1)).strip()
                opens = re.compile(re.escape(label) + r'(?![\w])', re.IGNORECASE)
                is_caption = any(opens.match(line) for line in lines) or (
                    len(text) <= 350 and re.search(re.escape(label) + r'\.', text, re.IGNORECASE)
                )
                normalized = re.sub(r'(?i)^Fig\.\s*', 'Figure ', label)
                if is_caption and (normalized not in captions or height < captions[normalized][0]):
                    captions[normalized] = (height, page_num)
    return {norm: page for norm, (_, page) in captions.items()}


def build_page_section_map(
    figure_page_map: dict[str, int],
    body_sections: list[tuple[str, str]],
    doc: Document | None = None,
) -> dict[int, int]:
    """Map page number -> 1-based section index based on where figure captions appear.

    A figure caption (label + period, e.g. "Fig. 2.") indicates the section
    that *owns* that figure's page.  Plain mentions ("Fig. 2") are
    cross-references and do not confer ownership.

    If *doc* is provided, also scans each page for section headings
    from *body_sections* and propagates ownership to all pages between the
    first and last mapped page.  This ensures equation-only pages and other
    pages without figure captions still receive correct section assignment.
    """
    page_to_section: dict[int, int] = {}
    for idx, (_heading, body) in enumerate(body_sections, 1):
        for fig_label, page_num in figure_page_map.items():
            if page_num in page_to_section:
                continue
            if re.compile(re.escape(fig_label) + r'\.', re.IGNORECASE).search(body):
                page_to_section[page_num] = idx

    # Detect section heading pages from the PDF to cover pages without figure
    # captions.  This is essential for 2-column IEEE papers where section
    # boundaries may fall on pages that have no figure/table captions.
    if doc is not None:
        seen_pages = set(page_to_section)
        for idx, (_heading, _body) in enumerate(body_sections, 1):
            if idx in page_to_section.values():
                continue  # already owns a page via caption
            heading_text = re.sub(r'^#+\s*', '', _heading).strip()
            if not heading_text:
                continue
            # Match numbered section headings (Roman numerals) to avoid false
            # positives from short subsection letters (A., B., etc.).
            m = re.match(r'^(I{1,3}V?|IV|V|VI{0,3})\.', heading_text)
            if not m:
                continue
            numeral = m.group(1)
            for page_num in range(len(doc.pages)):
                if page_num in seen_pages:
                    continue
                page_text = '\n'.join(
                    ln.text for ln in layout.ordered(layout.lines(doc.pages[page_num], weighted=True))
                )
                if re.search(rf'\b{re.escape(numeral)}\.\s+[A-Z]', page_text):
                    page_to_section[page_num] = idx
                    seen_pages.add(page_num)
                    break

    # Fill unmapped pages by forward propagation: each unmapped page inherits
    # the section of the nearest lower-numbered mapped page.
    if page_to_section:
        sorted_pages = sorted(page_to_section)
        last_sec = page_to_section[sorted_pages[0]]
        for page in range(sorted_pages[0], sorted_pages[-1] + 1):
            ps = page_to_section.get(page)
            if ps is not None:
                last_sec = ps
            else:
                page_to_section[page] = last_sec

    return page_to_section
