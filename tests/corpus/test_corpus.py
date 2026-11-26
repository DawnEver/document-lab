"""The ``corpus`` part: real public PDFs (:mod:`tests.corpus.sources`) fetched at test time, read against in-code truth.

Every source is read page by page against its golden pages (:mod:`tests.corpus.goldens`; how they are written:
:mod:`tests.corpus.reading`), and against the feature it was chosen for, each truth read off the PDF itself -- its
rendering, its own objects -- not off the reader. The IRS forms are encrypted here, with each security handler
revision the reader decrypts, and must read as their originals. Every check is paired with a planted wrong truth it
must report.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from document_lab import _pdf
from document_lab.pdf import read_pdf
from tests.corpus.goldens import GOLDENS
from tests.corpus.reading import Page, problems, summarize
from tests.corpus.sources import SOURCES

pytestmark = pytest.mark.corpus


def test_every_source_has_goldens():
    assert set(GOLDENS) == set(SOURCES)


@pytest.mark.parametrize('pdf', sorted(SOURCES), indirect=True)
def test_a_source_reads_as_its_golden_pages(pdf: Path, request: pytest.FixtureRequest):
    assert problems(summarize(pdf), GOLDENS[request.node.callspec.params['pdf']]) == []


def _one_page_changed(golden: tuple[Page, ...], **change: Any) -> tuple[Page, ...]:
    middle = len(golden) // 2
    return (*golden[:middle], golden[middle]._replace(**change), *golden[middle + 1 :])


#: The planted wrong goldens: each falsifies one fact of a source's true pages.
WRONG_GOLDEN: dict[str, Callable[[tuple[Page, ...]], tuple[Page, ...]]] = {
    'a page dropped': lambda g: g[:-1],
    'a page wider': lambda g: _one_page_changed(g, width=g[len(g) // 2].width + 0.01),
    'other text': lambda g: _one_page_changed(g, text_sha256='0' * 64),
    'a span more': lambda g: _one_page_changed(g, spans=g[len(g) // 2].spans + 1),
    'a path more': lambda g: _one_page_changed(g, paths=g[len(g) // 2].paths + 1),
    'an image more': lambda g: _one_page_changed(g, images=g[len(g) // 2].images + 1),
}


@pytest.mark.parametrize('wrong', sorted(WRONG_GOLDEN))
@pytest.mark.parametrize('pdf', sorted(SOURCES), indirect=True)
def test_a_wrong_golden_is_reported(pdf: Path, wrong: str, request: pytest.FixtureRequest):
    golden = GOLDENS[request.node.callspec.params['pdf']]
    assert problems(summarize(pdf), WRONG_GOLDEN[wrong](golden)), f'{wrong}: the golden check could not say no'


# --- the features each source was chosen for -------------------------------------------------------------------------


@cache
def _rows(pdf: Path) -> dict[int, tuple]:
    """What the reader states on each page, as the extension's rows (render modes and image objects included)."""
    reader = _pdf.open(pdf.read_bytes(), b'')
    return {number: reader.page(number, 8) for number, _w, _h in reader.pages}


def _text(pdf: Path, page: int) -> str:
    return ' '.join(' '.join(span.text for span in read_pdf(pdf).pages[page - 1].spans).split())


def phrases_stated(pdf: Path, page: int, phrases: tuple[str, ...]) -> list[str]:
    text = _text(pdf, page)
    return [f'page {page} does not state {p!r}' for p in phrases if p not in text]


def phrases_left_out(pdf: Path, page: int, phrases: tuple[str, ...]) -> list[str]:
    text = _text(pdf, page)
    return [f'page {page} states {p!r}' for p in phrases if p in text]


def font_text(pdf: Path, font: str, text: str) -> list[str]:
    found = ''.join(row[0] for rows in _rows(pdf).values() for row in rows[0] if row[1] == font)
    return [] if found == text else [f'the {font} spans state {found!r}, not {text!r}']


def page_size(pdf: Path, page: int, size: tuple[float, float]) -> list[str]:
    found = next((round(w, 2), round(h, 2)) for n, w, h in _pdf.open(pdf.read_bytes(), b'').pages if n == page)
    return [] if found == size else [f'page {page} is {found} pt, not {size}']


def ocr_layer(pdf: Path, modes: frozenset[int]) -> list[str]:
    """Every page one scan under text in render modes ``modes`` (3: invisible)."""
    out = []
    for number, (spans, _paths, images, _warnings) in _rows(pdf).items():
        found = {row[6] for row in spans}
        if found != modes:
            out.append(f'page {number} renders its text in modes {sorted(found)}, not {sorted(modes)}')
        if not images:
            out.append(f'page {number} places no image')
    return out


def jpx_images(pdf: Path, page: int, objects: frozenset[int]) -> list[str]:
    """The image XObjects ``page`` places that the reader hands over still JPX-encoded (D3: not decoded)."""
    reader = _pdf.open(pdf.read_bytes(), b'')
    images = _rows(pdf)[page][2]
    found = {obj for _box, _w, _h, obj in images if obj is not None and reader.image(obj)[6] == 'JPXDecode'}
    return [] if found == objects else [f'page {page} places JPX images {sorted(found)}, not {sorted(objects)}']


#: The narrowest a vertical glyph's box may be, in ems: a CJK glyph is about one em wide.
VERTICAL_MIN_WIDTH = 0.5


def vertical_run(pdf: Path, page: int, area: tuple[float, float, float, float], text: str) -> list[str]:
    """The spans of Identity-V fonts centered in ``area`` (x0, y0, x1, y1) are stated vertical and, read top to
    bottom (a space between spans), state ``text``.
    """
    x0, y0, x1, y1 = area
    spans = sorted(
        (row[5][1], row[0], row[5][2] - row[5][0], row[2], row[7])
        for row in _rows(pdf)[page][0]
        if row[1].endswith('-Identity-V')
        and x0 <= (row[5][0] + row[5][2]) / 2 <= x1
        and y0 <= (row[5][1] + row[5][3]) / 2 <= y1
    )
    found = ' '.join(t for _y, t, _w, _s, _v in spans)
    out = [] if found == text else [f'page {page} states {found!r} down the column, not {text!r}']
    flat = [t for _y, t, _w, _s, vertical in spans if not vertical]
    out += [f'page {page}: Identity-V spans {flat!r} not stated vertical'] if flat else []
    narrow = [t for _y, t, width, size, _v in spans if width < VERTICAL_MIN_WIDTH * size]
    return out + ([f'page {page}: vertical spans {narrow!r} under half an em wide'] if narrow else [])


@dataclasses.dataclass(frozen=True)
class Feature:
    """A truth about ``source``, read off the PDF; ``check(pdf, *truth)`` lists what the reader departs from it."""

    source: str
    check: Callable[..., list[str]]
    truth: tuple
    wrong: tuple  # a planted wrong truth the check must report
    known: str = ''  # a known failure: what the reader states instead (the residual), until it is fixed


FEATURES = {
    'type3-glyphs-read': Feature(
        'latex-type3-arxiv-2204.08094v1',
        font_text,
        ('DejaVuSansMono', 'EADGBe' * 8),  # the string names under the tablature figures (pages 4 and 6), as drawn
        ('DejaVuSansMono', 'EADGBE' * 8),
    ),
    'cropped-page-is-its-crop-box': Feature(
        'cjk-zh-vertical-arxiv-2301.01621v3',
        page_size,
        (24, (595.68, 841.92)),  # page 24's CropBox; its MediaBox is 603.84 x 847.92
        (24, (603.84, 847.92)),
    ),
    'annotation-text-left-out-by-default': Feature(
        'cjk-zh-vertical-arxiv-2301.01621v3',
        phrases_left_out,
        (1, ('Work supported by the National Natural Science Foundation of China',)),  # page 1's FreeText /Contents
        (1, ('Grammar construction methods for extended deterministic expressions',)),  # page 1's title
    ),
    'vertical-cjk-reads-top-to-bottom': Feature(
        'cjk-zh-vertical-arxiv-2301.01621v3',
        vertical_run,
        # Page 3's contents: the page numbers of its fifteen entries, set in an Identity-V font down the right-hand
        # column (x 499-506 pt of ink, y 174-488 pt), as the page renders them: 1, 2, 2, 3, 3, 5, 6, 6, 8, 8, 9,
        # then the units of 12, 14, 16, 17 (their tens are glyphs set beside them, x 494-499 pt, which the font's
        # ToUnicode names U+006C 'l').
        (3, (499.0, 170.0, 520.0, 500.0), '1 2 2 3 3 5 6 6 8 8 9 2 4 6 7'),
        (3, (499.0, 170.0, 520.0, 500.0), '7 6 4 2 9 8 8 6 6 5 3 3 2 2 1'),
    ),
    'japanese-in-an-embedded-cid-font': Feature(
        'cjk-ja-euc-h-arxiv-2608.20674v1',
        phrases_stated,
        (4, ('最初のステップとして，以下の要旨を体言止め，箇条書きのリストに変換してください。',)),  # noqa: RUF001
        (4, ('最初のステップとして，以下の要旨を要約してください。',)),  # noqa: RUF001
    ),
    'japanese-powerpoint-and-its-license': Feature(
        'cjk-ja-powerpoint-jst-survey2024',
        phrases_stated,
        (1, ('情報サービス利用者の満足度調査', 'https://creativecommons.org/licenses/by/4.0/deed.ja')),
        (1, ('情報サービス利用者の満足度調査', 'https://creativecommons.org/licenses/by-sa/4.0/deed.ja')),
    ),
    'word-truetype-text': Feature(
        'word-truetype-arxiv-2105.10459v1',
        phrases_stated,
        (1, ('基于夜光遥感对于区域和城市群的发展的研究', 'night-light remote sensing')),
        (1, ('基于日光遥感对于区域和城市群的发展的研究', 'night-light remote sensing')),
    ),
    'scan-under-an-invisible-ocr-layer': Feature(
        'scan-ocr-rotated-naca-tn670',
        ocr_layer,
        (frozenset({3}),),
        (frozenset({0}),),
    ),
    'rotated-page-is-stated-as-displayed': Feature(
        'scan-ocr-rotated-naca-tn670',
        page_size,
        (10, (744.0, 565.0)),  # page 10: /Rotate 90 over a 565 x 744 MediaBox
        (10, (565.0, 744.0)),
    ),
    'jpx-images-handed-over-encoded': Feature(
        'devicen-jpx-usgs-fs20213001',
        jpx_images,
        (1, frozenset({288, 295, 299, 303})),  # page 1's image XObjects filtered /JPXDecode
        (1, frozenset({288, 295, 299})),
    ),
}


def _feature_cases() -> list:
    return [
        pytest.param(
            feature.source,
            name,
            id=name,
            marks=[pytest.mark.xfail(strict=True, reason=feature.known)] if feature.known else [],
        )
        for name, feature in FEATURES.items()
    ]


@pytest.mark.parametrize(('pdf', 'name'), _feature_cases(), indirect=['pdf'])
def test_a_source_has_the_feature_it_was_chosen_for(pdf: Path, name: str):
    feature = FEATURES[name]
    assert feature.check(pdf, *feature.truth) == []


@pytest.mark.parametrize(('pdf', 'name'), [(f.source, n) for n, f in FEATURES.items()], indirect=['pdf'])
def test_a_wrong_feature_truth_is_reported(pdf: Path, name: str):
    feature = FEATURES[name]
    assert feature.check(pdf, *feature.wrong), f'{name}: the check could not say no'


# --- encryption: the IRS forms, encrypted here (the bytes are not reproducible: random salts), read as originals --


#: Each form's encryption: PyMuPDF's method, and the /V and /R the security handler must then state.
ENCRYPTIONS = {
    'form-irs-f1040-2025': (pymupdf.PDF_ENCRYPT_AES_256, 5, 6),
    'form-irs-fw4-2026': (pymupdf.PDF_ENCRYPT_AES_128, 4, 4),
    'form-irs-fw9-2024': (pymupdf.PDF_ENCRYPT_RC4_128, 2, 3),
}
OWNER = 'document-lab-owner'
USER = 'document-lab-user'
WRONG_PASSWORD = 'not-the-password'  # noqa: S105 -- a password the files do not have


@pytest.fixture
def encrypted(pdf: Path, request: pytest.FixtureRequest, tmp_path: Path) -> Callable[[str], Path]:
    """``pdf`` encrypted as :data:`ENCRYPTIONS` names, with user password ``user`` (empty: anyone opens it)."""
    method, _v, _r = ENCRYPTIONS[request.node.callspec.params['pdf']]

    def make(user: str) -> Path:
        out = tmp_path / f'encrypted-{len(user)}.pdf'
        with pymupdf.open(pdf) as doc:
            doc.save(out, encryption=method, owner_pw=OWNER, user_pw=user, permissions=pymupdf.PDF_PERM_PRINT)
        return out

    return make


def handler(path: Path, version: int, revision: int) -> list[str]:
    """The file's standard security handler is /V ``version`` /R ``revision`` (PyMuPDF says 'Standard V5 R6 ...')."""
    with pymupdf.open(path) as doc:
        said = doc.metadata['encryption'] or 'none'
    found = re.match(r'Standard V(\d+) R(\d+) ', said)
    if found is None or (int(found[1]), int(found[2])) != (version, revision):
        return [f'encrypted {said!r}, not Standard V{version} R{revision}']
    return []


FORMS = sorted(ENCRYPTIONS)


@pytest.mark.parametrize('pdf', FORMS, indirect=True)
def test_the_form_is_encrypted_as_named(encrypted, request: pytest.FixtureRequest):
    _method, version, revision = ENCRYPTIONS[request.node.callspec.params['pdf']]
    path = encrypted('')
    assert handler(path, version, revision) == []
    assert handler(path, version, revision + 1), 'the planted control: another revision must be reported'


@pytest.mark.parametrize('pdf', FORMS, indirect=True)
def test_an_encrypted_form_reads_as_its_original_with_the_empty_password(encrypted, request: pytest.FixtureRequest):
    assert problems(summarize(encrypted('')), GOLDENS[request.node.callspec.params['pdf']]) == []


@pytest.mark.parametrize('pdf', FORMS, indirect=True)
def test_a_wrong_password_is_refused_and_the_owners_opens(encrypted, request: pytest.FixtureRequest):
    path = encrypted('')
    with pytest.raises(ValueError, match='password'):
        read_pdf(path, password=WRONG_PASSWORD)
    # The planted control: the refusal is the password's, not the file's -- the owner's password opens it.
    assert problems(summarize(path, OWNER), GOLDENS[request.node.callspec.params['pdf']]) == []


@pytest.mark.parametrize('pdf', FORMS, indirect=True)
def test_a_user_password_once_set_is_required(encrypted, request: pytest.FixtureRequest):
    path = encrypted(USER)
    with pytest.raises(ValueError, match='password'):
        read_pdf(path)
    assert problems(summarize(path, USER), GOLDENS[request.node.callspec.params['pdf']]) == []
