"""Figure and equation rendering: disk I/O, markdown insertion, cross-section fixup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from document_lab import formula, layout, png
from document_lab.ir import BBox
from document_lab.paper.crop import crop_figure, equation_context
from document_lab.paper.pagemap import FIG_TABLE_RE, build_figure_page_map, build_page_section_map
from document_lab.paper.text import RN_PATTERN as _RN
from document_lab.paper.text import slug
from document_lab.pdf import read_pdf, read_raster
from document_lab.regions import Region, Unnumbered, equations, regions

if TYPE_CHECKING:
    import numpy as np

    from document_lab.formula import Transcribe
    from document_lab.ir import Document, Page

_CAPTION_RE = re.compile(rf'(?P<label>Figure\s+\d+|Fig\.\s*\d+|Table\s+(?:\d+|{_RN}))\.(?P<rest>.+)', re.IGNORECASE)


# ── Low-level render helpers ──────────────────────────────────────────────────


_FULL_PAGE_IMAGE = 0.9  # share of a page one image covers for the page to be that image (a screen capture, a scan)


def is_full_page_image(page: Page) -> bool:
    """Is ``page`` one picture of a page -- a screen capture or a scan -- with nothing drawn beside it?

    Its figures and tables are pixels inside that picture, so there is nothing to crop them from.
    """
    area = page.width * page.height
    covered = [_covered(f.provenance.bbox, page) for f in page.figures if f.provenance.bbox is not None]
    return len(covered) == 1 and covered[0] >= _FULL_PAGE_IMAGE * area and not page.paths


def _covered(box: BBox, page: Page) -> float:
    """The area of ``box`` on the page."""
    width = min(box.x1, page.width) - max(box.x0, 0.0)
    height = min(box.y1, page.height) - max(box.y0, 0.0)
    return max(width, 0.0) * max(height, 0.0)


class _Located:
    """The regions a picture page's text locates (`document_lab.regions`), per page of a document read once.

    ``read`` is the document as its default view shows it; ``include_hidden``, ``include_in_background_color``:
    what it reads beside (see `document_lab.ir`), told to `document_lab.regions` alike.
    """

    def __init__(self, read: Document, *, include_hidden: bool, include_in_background_color: bool) -> None:
        self.read = read
        self.face = layout.body_face(read)
        self.include_hidden = include_hidden
        self.include_in_background_color = include_in_background_color

    def crop(self, page_num: int, label: str) -> BBox | None:
        """The region labeled ``label`` (``Figure 2``, ``Table 1``, ``(3)``) on the page, or None."""
        page = self.read.pages[page_num]
        found = regions(
            page,
            self.face,
            include_hidden=self.include_hidden,
            include_in_background_color=self.include_in_background_color,
        )
        found = [r for r in found if r.label == label]
        return found[0].bbox if len(found) == 1 else None


#: The resolution figures, tables and equations are rendered at: twice the page's size.
RENDER_DPI = 144.0


def _render(pdf: Path, read: Document, page_num: int, img_path: Path, crop: BBox | None) -> np.ndarray:
    """The page, or its ``crop``, rendered (`document_lab.pdf.read_raster`) at :data:`RENDER_DPI` to ``img_path``.

    Always rendered and written, a file already there replaced (the render is deterministic, so a rerun writes the
    same bytes): returns the pixels written -- a PNG is lossless, so they are the file's.
    """
    page = read.pages[page_num]
    region = crop or BBox(0.0, 0.0, page.width, page.height)
    pixels = read_raster(pdf, page_num + 1, region, dpi=RENDER_DPI).pixels
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(png.encode(pixels))
    return pixels


@dataclass(frozen=True)
class _Equation:
    """A display equation of the paper (`document_lab.regions.equations`), numbered or not.

    Its 0-based ``page`` and its ``region``, the section that owns it (``owning``) and the text around it
    (``lead_in``, ``formula``: `document_lab.paper.crop.equation_context`).
    """

    page: int
    region: Region
    owning: int
    lead_in: str | None
    formula: str | None

    @property
    def number(self) -> str | None:
        """Its number as printed (``3``, ``16a``, ``A.1``), or None: a display without one."""
        return self.region.label.strip('()') if self.region.captioned else None

    @property
    def alt(self) -> str:
        """Its link's text: ``Equation 3``, or ``Untitled equation 2`` (the second without a number on its page)."""
        return f'Equation {self.number}' if self.number is not None else self.region.label

    @property
    def image(self) -> str:
        """Its image, relative to the paper's directory: ``img/sec02/page-03-equation-3.png``.

        A display without a number is named by its page and its order there, as `document_lab.per_page` names it:
        ``page-03-equation-untitled-2.png``.
        """
        label = self.number if self.number is not None else 'untitled-' + self.region.label.rsplit(' ', 1)[-1]
        name = f'page-{self.page + 1:02d}-equation-{label.lower()}.png'
        return f'img/sec{self.owning:02d}/{name}'


def _equations(
    read: Document,
    paper: Document,
    face: tuple[str, float],
    body_sections: list[tuple[str, str]],
    page_section_map: dict[int, int],
    *,
    unnumbered: Unnumbered,
    include_hidden: bool,
    include_in_background_color: bool,
) -> list[_Equation]:
    """Every display equation of the paper, numbered or not, in reading order, each with the section that owns it.

    Read off ``paper`` (``read`` without its running lines) by `document_lab.regions.equations`; on a page that is one
    picture (:func:`is_full_page_image`), where its text locates them (`document_lab.regions.regions`, which sees the
    running lines: they set the text's width). A page no section owns goes to the section that prints the number, else
    to the nearest page's section; with none at all, nothing is written.
    """
    out = []
    for number, page in enumerate(paper.pages):
        if is_full_page_image(read.pages[number]):
            found = regions(
                read.pages[number],
                face,
                unnumbered=unnumbered,
                include_hidden=include_hidden,
                include_in_background_color=include_in_background_color,
            )
            found = [r for r in found if r.kind == 'equation']
        else:
            found = equations(page, face, unnumbered=unnumbered)
        for region in found:
            owning = page_section_map.get(number)
            printed = region.label if region.captioned else None
            if owning is None and printed is not None:
                pattern = re.compile(r'\( ?' + re.escape(printed.strip('()')) + r' ?\)')
                owning = next((i for i, (_, body) in enumerate(body_sections, 1) if pattern.search(body)), None)
            if owning is None and page_section_map:
                owning = page_section_map[min(page_section_map, key=lambda p, n=number: abs(p - n))]
            if owning is not None:
                out.append(_Equation(number, region, owning, *equation_context(page, region.bbox)))
    return out


# ── Markdown insertion helpers ────────────────────────────────────────────────


def _insert_figures_inline(
    section_file: Path,
    section_idx: int,
    figure_images: dict[str, str],
    figure_page_map: dict[str, int],
    page_section_map: dict[int, int],
) -> int:
    """Insert markdown image links before each figure/table caption in *section_file*.

    Returns number of figures inserted.
    """
    content = section_file.read_text(encoding='utf-8')
    insertions: list[tuple[int, str]] = []
    seen: set[str] = set()

    for m in _CAPTION_RE.finditer(content):
        label = m.group('label')
        normalized = re.sub(r'(?i)^Fig\.\s*', 'Figure ', label)
        if normalized in seen:
            continue
        img_rel = figure_images.get(normalized)
        if img_rel is None:
            continue

        # Skip mid-sentence references ("shown in Fig. 2.")
        preceding_start = content.rfind('\n', 0, m.start()) + 1
        prefix = content[preceding_start : m.start()].strip()
        if prefix and prefix[-1].isalpha() and len(prefix.split()) >= 2:
            continue

        seen.add(normalized)
        owning = page_section_map.get(figure_page_map.get(normalized, -1), section_idx)
        img_path = f'../img/sec{owning:02d}/{Path(img_rel).name}' if owning != section_idx else f'../{img_rel}'
        insertions.append((m.start(), f'\n\n![{label}]({img_path})\n\n'))

    for pos, img_md in reversed(insertions):
        content = content[:pos] + img_md + content[pos:]

    if insertions:
        section_file.write_text(content, encoding='utf-8')
    return len(insertions)


def _definition(content: str, number: str) -> int | None:
    """Where ``(number)`` stands as its equation's own mark in ``content``, else None.

    Nothing after it on its line, no letter before it: ``from (5) and (6)`` cites it.
    """
    for m in re.finditer(rf'\(\s*{re.escape(number)}\s*\)', content):
        start = content.rfind('\n', 0, m.start()) + 1
        end = content.find('\n', m.end())
        after = content[m.end() : end if end != -1 else len(content)].strip()
        if not after and not re.search(r'[a-zA-Z]', content[start : m.start()]):
            return m.start()
    return None


def _after_lead_in(content: str, lead_in: str, start: int) -> int | None:
    """Where the paragraph gap after ``lead_in`` is in ``content``, at or past ``start``, else None.

    Its last 40, 28 or 16 characters, found exactly, else allowing ``_x_`` italic marks round short words.
    """
    found = -1
    for suffix_len in (40, 28, 16):
        key = lead_in.strip()[-suffix_len:]
        if not key:
            continue
        p = content.lower().find(key.lower())
        if p != -1:
            found = p + len(key)
            break
        tokens = re.split(r'(\w+)', key)
        pattern = ''.join(
            rf'_?{re.escape(t)}_?\s*' if re.match(r'^\w{1,4}$', t) and not t.isdigit() else re.escape(t) for t in tokens
        )
        if m := re.search(pattern, content, re.IGNORECASE):
            found = m.end()
            break
    if found == -1 or (line_end := content.find('\n', found)) == -1:
        return None
    search_from = max(line_end, start)
    gap = re.search(r'\n+', content[search_from : search_from + 300])
    return search_from + gap.start() + 1 if gap else search_from + 1


def _insert_equations_inline(
    section_file: Path, found: list[_Equation], placed: set[int], latex: dict[int, str], *, cited: bool
) -> None:
    """Link each equation of ``found`` not yet ``placed`` where ``section_file`` sets it, in reading order.

    Where its number stands as its own mark (:func:`_definition`); else after the paragraph its lead-in sentence ends
    (:func:`_after_lead_in`), with its formula's text and its number -- a display without a number is placed so alone.
    ``cited``: the last resort for a numbered one placed nowhere else, at its first mention ``(N)`` in its own section.
    An equation placed is added to ``placed``: it is linked once, in one section. Its ``latex`` (by its id), when
    read, follows its image.
    """
    content = section_file.read_text(encoding='utf-8')
    changed = False
    start = 0  # the next one goes after the last placed: reading order holds
    for eq in found:
        if id(eq) in placed:
            continue
        if Path(eq.image).name in content:
            placed.add(id(eq))
            continue
        link = f'![{eq.alt}](../{eq.image})' + (f'\n\n{latex[id(eq)]}' if id(eq) in latex else '')
        if cited:
            m = re.search(rf'\(\s*{re.escape(eq.number)}\s*\)', content) if eq.number is not None else None
            at, insertion = (m.start(), f'\n\n{link}\n\n') if m else (None, '')
        elif eq.number is not None and (at := _definition(content, eq.number)) is not None:
            insertion = f'\n\n{link}\n\n'
        elif eq.lead_in and (at := _after_lead_in(content, eq.lead_in, start)) is not None:
            text = f'\n*{eq.formula}*' if eq.formula else ''
            number = f'\n({eq.number})' if eq.number is not None else ''
            insertion = f'\n{link}{text}{number}\n'
        else:
            continue
        if at is None:
            continue
        at = max(at, start)
        content = content[:at] + insertion + content[at:]
        start = at + len(insertion)
        placed.add(id(eq))
        changed = True
    if changed:
        section_file.write_text(content, encoding='utf-8')


def _fix_cross_section_image_refs(
    out_dir: Path,
    body_sections: list[tuple[str, str]],
    page_section_map: dict[int, int],
) -> None:
    """Rewrite flat-image references that belong to a different section."""
    flat_pattern = re.compile(r'!\[([^\]]*)\]\((?:\.\./img/flat/)?([^)\s]+)\)')

    for idx, (heading, _) in enumerate(body_sections, 1):
        sec_slug = slug(heading) or 'body'
        sec_file = out_dir / 'md' / f'{idx:02d}-{sec_slug}.md'
        if not sec_file.exists():
            continue
        content = sec_file.read_text(encoding='utf-8')
        changed = False

        def _rewrite(m: re.Match, _idx: int = idx) -> str:
            nonlocal changed
            alt, fname = m.group(1), m.group(2)
            page_m = re.search(r'\.pdf-(\d{4})', fname)  # the converter names an image <pdf name>-<page>-<n>
            if page_m:
                page_num = int(page_m.group(1)) - 1
                owning = page_section_map.get(page_num)
                if owning is not None and owning != _idx:
                    changed = True
                    return f'![{alt}](../img/sec{owning:02d}/{fname})'
            return m.group(0)

        new_content = flat_pattern.sub(_rewrite, content)
        if changed:
            sec_file.write_text(new_content, encoding='utf-8')


# ── Public render entry points ────────────────────────────────────────────────


def render_figure_pages(
    pdf_path: Path,
    out_dir: Path,
    body_sections: list[tuple[str, str]],
    *,
    include_hidden: bool = False,
    include_in_background_color: bool = False,
) -> tuple[dict[str, str], dict[int, int]]:
    """Render cropped figure images (`document_lab.pdf.read_raster`) and insert them inline at caption positions.

    A label on a page that is one picture (:func:`is_full_page_image`: a screen capture, a scan) has no drawn
    content to crop by: it is cut out where the page's text locates it (`document_lab.regions`: an OCR layer's
    caption and the prose around it), and else links the whole page, rendered once a run as ``page-NN.png`` however many
    such labels the page holds.

    Captions are found, and figures cropped, on the paper as its default view shows it (`document_lab.ir.Page.shown`;
    ``include_hidden``, ``include_in_background_color``: what it reads beside) without its running heads and feet
    (`document_lab.layout.without_running`).

    Returns (figure_images, page_section_map) where figure_images maps normalized
    figure label → relative image path (for INDEX.md).
    """
    read = read_pdf(pdf_path).shown(
        include_hidden=include_hidden, include_in_background_color=include_in_background_color
    )
    paper = layout.without_running(read)
    figure_page_map = build_figure_page_map(paper)
    page_section_map = build_page_section_map(figure_page_map, body_sections, paper)
    figure_images: dict[str, str] = {}

    # Collect unique figure entries (label, normalized, page, owning_section)
    seen: dict[str, int] = {}
    unique_entries: list[tuple[str, str, int, int]] = []
    for idx, (heading, _) in enumerate(body_sections, 1):
        sec_slug = slug(heading) or 'body'
        sec_file = out_dir / 'md' / f'{idx:02d}-{sec_slug}.md'
        if not sec_file.exists():
            continue
        content = sec_file.read_text(encoding='utf-8')
        for m in FIG_TABLE_RE.finditer(content):
            label = re.sub(r'\s+', ' ', m.group(1)).strip()
            normalized = re.sub(r'^Fig\.\s*', 'Figure ', label)
            page_num = figure_page_map.get(normalized)
            if page_num is None or normalized in seen:
                continue
            owning = page_section_map.get(page_num, idx)
            seen[normalized] = owning
            unique_entries.append((label, normalized, page_num, owning))

    located = (
        _Located(read, include_hidden=include_hidden, include_in_background_color=include_in_background_color)
        if any(is_full_page_image(read.pages[e[2]]) for e in unique_entries)
        else None
    )
    rendered: set[Path] = set()
    for label, normalized, page_num, owning in unique_entries:
        page = paper.pages[page_num]
        own = f'page-{page_num + 1:02d}-{normalized.lower().replace(" ", "-")}.png'
        if not is_full_page_image(page):
            crop, img_name = crop_figure(page, label, content_below=label.upper().startswith('TABLE')), own
        elif located is not None and (crop := located.crop(page_num, normalized)) is not None:
            img_name = own  # where the picture page's text locates it
        else:  # nothing locates it: the page itself, once, shared by every such label on it
            crop, img_name = None, f'page-{page_num + 1:02d}.png'
        img_path = out_dir / 'img' / f'sec{owning:02d}' / img_name
        if img_path not in rendered:  # a whole page several labels share is rendered once a run
            _render(pdf_path, read, page_num, img_path, crop)
            rendered.add(img_path)
        figure_images[normalized] = f'img/sec{owning:02d}/{img_name}'

    _fix_cross_section_image_refs(out_dir, body_sections, page_section_map)

    for idx, (heading, _) in enumerate(body_sections, 1):
        sec_slug = slug(heading) or 'body'
        sec_file = out_dir / 'md' / f'{idx:02d}-{sec_slug}.md'
        if sec_file.exists():
            _insert_figures_inline(sec_file, idx, figure_images, figure_page_map, page_section_map)

    return figure_images, page_section_map


def render_equation_images(
    pdf_path: Path,
    out_dir: Path,
    body_sections: list[tuple[str, str]],
    figure_images: dict[str, str],
    page_section_map: dict[int, int],
    *,
    transcribe: Transcribe | None = None,
    unnumbered: Unnumbered = 'strict',
    include_hidden: bool = False,
    include_in_background_color: bool = False,
) -> dict[str, str]:
    """Render every display equation of the paper, numbered or not, and link each in its section.

    ``transcribe`` (`document_lab.formula.transcriber`): each rendered crop's LaTeX follows its image as a display
    block. ``unnumbered``: how displays without a number are found (`document_lab.regions.Unnumbered`).

    The equations are found on the paper as its default view shows it (`document_lab.ir.Page.shown`;
    ``include_hidden``, ``include_in_background_color``: what it reads beside), without its running heads and feet
    (`document_lab.layout.without_running`, :func:`_equations`), each cut out with its number when one is printed, and
    linked once (:func:`_insert_equations_inline`): first where any section sets its number or its lead-in, then a
    numbered one still unplaced at its first mention in its own section. Returns ``figure_images`` with each
    equation's image added, under ``Equation 3`` -- one without a number under ``Untitled equation 2, page 3``.
    """
    read = read_pdf(pdf_path).shown(
        include_hidden=include_hidden, include_in_background_color=include_in_background_color
    )
    paper, face = layout.without_running(read), layout.body_face(read)
    found = _equations(
        read,
        paper,
        face,
        body_sections,
        page_section_map,
        unnumbered=unnumbered,
        include_hidden=include_hidden,
        include_in_background_color=include_in_background_color,
    )
    latex: dict[int, str] = {}
    for eq in found:
        written = _render(pdf_path, read, eq.page, out_dir / eq.image, eq.region.bbox)
        key = eq.alt if eq.number is not None else f'{eq.alt}, page {eq.page + 1}'
        figure_images[key] = eq.image
        if transcribe is not None:
            latex[id(eq)] = formula.beside(transcribe(written), eq.number)

    files = [
        (idx, path)
        for idx, (heading, _) in enumerate(body_sections, 1)
        if (path := out_dir / 'md' / f'{idx:02d}-{slug(heading) or "body"}.md').exists()
    ]
    placed: set[int] = set()
    for _, path in files:
        _insert_equations_inline(path, found, placed, latex, cited=False)
    for idx, path in files:
        _insert_equations_inline(path, [eq for eq in found if eq.owning == idx], placed, latex, cited=True)
    return figure_images
