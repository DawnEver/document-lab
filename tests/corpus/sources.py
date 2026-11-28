"""The ``corpus`` part's real PDFs: where each is published, what it is pinned to, its license and its attribution.

The repo is public, so no real PDF is committed: :func:`fetch` downloads one from its ORIGINAL public URL into the
git-ignored cache (``output/cache/corpus/<sha256>.pdf``) and hands it over only when its bytes are the ones pinned
here. Only files with no licensing or exposure risk are listed (user rulings D29, D30, D157 of task
document-lab-pdf-reader): CC BY 4.0 works, US Government works (public domain, 17 U.S.C. 105) without third-party
content, and Apache-2.0 and BSD test files (:data:`JBIG2`).

So that fetching does not make CI fragile: every URL is IMMUTABLE (each source states why, :attr:`Source.permanence`);
a request is retried with backoff on a transient failure; when the original fails or serves other bytes, the source's
one pinned Internet Archive snapshot (:attr:`Source.fallback`, an ``id_`` URL serving the bytes as archived) is tried,
accepted only with the same sha256. Each fallback was checked to serve the pinned bytes on 2026-11-26.

A FETCH FAILURE IS NOT A REGRESSION. When no URL gives the pinned bytes, :class:`FetchRefused` says so, naming each
URL and what it did. In CI the part's setup step fetches (``python -m tests.corpus.sources``, exit
:data:`FETCH_FAILED`), so a fetch failure ends the job before any test runs and the family posts the part ``error``
``class=environment`` (``dev_lab.dev.jobparts.absent``), never the code's red. Where the tests fetch for themselves (a
local run), they fetch in a fixture: a fetch failure is a test ERROR at setup, never an assertion FAILED.

ATTRIBUTION. Each :class:`Source` below carries its work's attribution, license and license URL -- the attribution
of record, which CC BY requires. The documents are read, never redistributed or changed; the tests encrypt the IRS
forms in a temporary directory only.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ['CACHE', 'FETCH_FAILED', 'JBIG2', 'SOURCES', 'FetchRefused', 'Source', 'fetch', 'fetch_into']

#: Where fetched documents are kept between runs, each named by its sha256 (git-ignored ``output/``).
CACHE: Final = Path(__file__).resolve().parents[2] / 'output' / 'cache' / 'corpus'
#: The User-Agent of every request: the project, generically; nothing about who runs it.
USER_AGENT: Final = 'document-lab-test-corpus/1.0'
#: Attempts per URL before it is given up (a hosted runner's network has transient failures).
ATTEMPTS: Final = 3
#: Seconds a request may take.
TIMEOUT: Final = 120
#: The exit status of ``python -m tests.corpus.sources`` when a source cannot be fetched (EX_TEMPFAIL).
FETCH_FAILED: Final = 75
#: HTTP statuses worth another attempt: the server's own failures and its rate limit.
_TRANSIENT: Final = frozenset({408, 425, 429, 500, 502, 503, 504})

CC_BY_4: Final = ('CC BY 4.0', 'https://creativecommons.org/licenses/by/4.0/')
US_PUBLIC_DOMAIN: Final = (
    'public domain (a work of the US Government, 17 U.S.C. 105)',
    'https://www.law.cornell.edu/uscode/text/17/105',
)
ARXIV_VERSIONED: Final = 'an arXiv versioned link (<id>v<N>): a version, once announced, never changes'
IRS_PRIOR: Final = (
    'the IRS prior-year archive (irs-prior/<form>--<year>.pdf): one revision, kept as filed -- never the '
    'current-year irs-pdf/ URL, which IRS replaces'
)
USGS_PUBLICATION: Final = 'a USGS series publication (DOI-registered): published once, corrected only as a new version'


@dataclass(frozen=True)
class Source:
    """A real PDF at its original public URL, pinned by ``sha256`` and ``size``, with its license and attribution."""

    name: str
    url: str
    #: Why ``url`` keeps serving the same bytes.
    permanence: str
    #: An Internet Archive snapshot of the same bytes, tried only when ``url`` fails; None: none exists.
    fallback: str | None
    sha256: str
    size: int
    license: str
    license_url: str
    attribution: str
    covers: str


SOURCES: Final = {
    source.name: source
    for source in (
        Source(
            'latex-type3-arxiv-2204.08094v1',
            'https://arxiv.org/pdf/2204.08094v1',
            ARXIV_VERSIONED,
            'https://web.archive.org/web/20240718044946id_/https://arxiv.org/pdf/2204.08094v1',
            'bd524a1debdf38d7a1d1d4dde2ddb2775c343dfb8c57cff6cbfbe2149417fdd0',
            373_964,
            *CC_BY_4,
            'Frank Cwitkowitz, Jonathan Driedger, Zhiyao Duan. "A Data-Driven Methodology for Considering Feasibility '
            'and Pairwise Likelihood in Deep Learning Based Guitar Tablature Transcription Systems." '
            'arXiv:2204.08094v1, 2022. CC BY 4.0 (arXiv record).',
            'a LaTeX paper (pdfTeX): Type3 fonts, embedded and non-embedded Type1, math fonts, ligatures, ruled '
            'tables, Indexed images, object streams; the paper layout end to end',
        ),
        Source(
            'cjk-zh-vertical-arxiv-2301.01621v3',
            'https://arxiv.org/pdf/2301.01621v3',
            ARXIV_VERSIONED,
            None,  # no Wayback snapshot of v3 exists (CDX searched 2026-11-26): the user's to decide,
            '793058942416f992126bd2dc9d8ff4ef990458db649d84dc4678efe9137ad8ae',
            3_364_635,
            *CC_BY_4,
            'Xiaoying Mou, Haiming Chen. "Grammar construction methods for extended deterministic expressions." '
            'arXiv:2301.01621v3 (in Chinese). CC BY 4.0 (arXiv record).',
            'Chinese text in CID-keyed CFF Type0 fonts, Identity-H and Identity-V (vertical writing); optional content '
            '(one layer, on); Text and FreeText annotations; an AcroForm; a cropped page (24); DCT images',
        ),
        Source(
            'cjk-ja-euc-h-arxiv-2608.20674v1',
            'https://arxiv.org/pdf/2608.20674v1',
            ARXIV_VERSIONED,
            'https://web.archive.org/web/20260825185335id_/https://arxiv.org/pdf/2608.20674v1',
            'dca13ae01daca569efbba00d8b46a39e5273a3f2329b283f000e06663eb747db',
            878_062,
            *CC_BY_4,
            'Hitoshi Koshiba, Miki Ida-Kimura. "Analysis of Potential Generative AI Use in Abstracts of KAKENHI-Funded '
            'Projects." arXiv:2608.20674v1, 2026. CC BY 4.0 (arXiv record).',
            'a LaTeX preprint (arXiv GenPDF): Japanese text in an embedded CID-keyed Type0 font (Identity-H); '
            'non-embedded standard-14 fonts; ICCBased color; the paper layout end to end. Its non-embedded '
            "GothicBBB-Medium fonts on the predefined CMap EUC-H sit in an R figure's resources and draw nothing",
        ),
        Source(
            'cjk-ja-powerpoint-jst-survey2024',
            'https://www.jstage.jst.go.jp/static/files/ja/pub_survey2024_society.pdf',
            'a report of one fiscal year (2024), published once under its own file name; not an alias of the latest',
            'https://web.archive.org/web/20261021102312id_/https://www.jstage.jst.go.jp/static/files/ja/pub_survey2024_society.pdf',
            'f6078bfeae0e96969552acf7edd6e63f6a0e8c0e07212d0f43a1b3669afb25c4',
            1_261_129,
            'CC BY 4.0 (stated on its first page)',
            'https://creativecommons.org/licenses/by/4.0/deed.ja',
            # The publisher's name and the document's title as it states them (its full-width parentheses kept),
            # escaped: no CJK character is written in tracked source (NO-CJK-IN-TRACKED-SOURCE).
            '(c) 2025 Japan Science and Technology Agency ('
            '\u56fd\u7acb\u7814\u7a76\u958b\u767a\u6cd5\u4eba'  # spelling-ignore-line
            '\u79d1\u5b66\u6280\u8853\u632f\u8208\u6a5f\u69cb). '  # spelling-ignore-line
            '"\u4ee4\u548c6\u5e74\u5ea6JST\u60c5\u5831\u30b5\u30fc\u30d3\u30b9'  # spelling-ignore-line
            '\u5229\u7528\u8005\u306e\u6e80\u8db3\u5ea6\u8abf\u67fb'  # spelling-ignore-line
            '\uff08J-STAGE \u5229\u7528\u6a5f\u95a2\uff09". CC BY 4.0.',  # spelling-ignore-line
            'Microsoft PowerPoint slides: Japanese in TrueType Type0 Identity-H (Meiryo), TrueType WinAnsi (Calibri), '
            'charts, SMask images',
        ),
        Source(
            'word-truetype-arxiv-2105.10459v1',
            'https://arxiv.org/pdf/2105.10459v1',
            ARXIV_VERSIONED,
            # the snapshot is of the unversioned URL, then serving v1 (the only version): the same bytes
            'https://web.archive.org/web/20240609135646id_/https://arxiv.org/pdf/2105.10459',
            'fcde8a5e4a689ebd34bae1decdf3fdd38d4c39b5dd64e758e515896d7503040b',
            1_072_944,
            *CC_BY_4,
            'Jiongyan Zhang. "Research on Regional Urban Economic Development by Nightlight-time Remote Sensing." '
            'arXiv:2105.10459v1 (in Chinese). CC BY 4.0 (arXiv record).',
            'Microsoft Word 2016: TrueType fonts, simple (WinAnsi) and Type0 Identity-H; Chinese and English text',
        ),
        Source(
            'scan-ocr-rotated-naca-tn670',
            'https://ntrs.nasa.gov/api/citations/19930081478/downloads/19930081478.pdf',
            'the NTRS download of one citation id, a 1938 report: the archive file of a fixed record',
            # the snapshot is of NTRS's former address of the same file: the same bytes
            'https://web.archive.org/web/20170228120608id_/https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19930081478.pdf',
            '81e35565a10518e4f34d44767c5a968e087626209ed17332248d334c80c3e5ff',
            7_334_058,
            *US_PUBLIC_DOMAIN,
            'F. L. Thompson. "NACA Stall-Warning Indicator." NACA Technical Note 670, October 1938. NASA Technical '
            'Reports Server, https://ntrs.nasa.gov/citations/19930081478. US Government work, public domain.',
            'scanned pages (one DCT image each) under an invisible OCR text layer (render mode 3); pages 10-11 '
            '/Rotate 90',
        ),
        Source(
            'separation-usgs-fs20213010',
            'https://pubs.usgs.gov/fs/2021/3010/fs20213010.pdf',
            USGS_PUBLICATION,
            'https://web.archive.org/web/20250413222108id_/https://pubs.usgs.gov/fs/2021/3010/fs20213010.pdf',
            '076c0367394190f90f2309f9c61c6957c83943b868abe2b88a41a3b62888e76e',
            2_595_242,
            *US_PUBLIC_DOMAIN,
            'MaryLynn Musgrove. "Groundwater quality in the Edwards-Trinity aquifer system." U.S. Geological Survey '
            'Fact Sheet 2021-3010, https://doi.org/10.3133/fs20213010. US Government work, public domain.',
            'Adobe InDesign: a Separation color space, a shading, an AcroForm, Indexed and SMask images, TrueType, CFF '
            'and Type0 fonts',
        ),
        Source(
            'devicen-jpx-usgs-fs20213001',
            'https://pubs.usgs.gov/fs/2021/3001/fs20213001.pdf',
            USGS_PUBLICATION,
            'https://web.archive.org/web/20250414032256id_/https://pubs.usgs.gov/fs/2021/3001/fs20213001.pdf',
            'c7505a9a073161ee6e5f20e16aa2df2033c13fcc0b8db758077358d1de68861c',
            2_141_032,
            *US_PUBLIC_DOMAIN,
            'Jeffrey J. Love, Kristen A. Lewis. "Geomagnetic monitoring in the mid-Atlantic United States." U.S. '
            'Geological Survey Fact Sheet 2021-3001, https://doi.org/10.3133/fs20213001. US Government work, public '
            'domain.',
            'Adobe InDesign: a DeviceN color space, a shading, JPX (JPEG 2000) images, SMask and Indexed images, an '
            'AcroForm, CID-keyed Type0 fonts',
        ),
        Source(
            'form-irs-f1040-2025',
            'https://www.irs.gov/pub/irs-prior/f1040--2025.pdf',
            IRS_PRIOR,
            'https://web.archive.org/web/20260823235551id_/https://www.irs.gov/pub/irs-prior/f1040--2025.pdf',
            '3d31c226df0d189ced80e039d01cf0f8820c1019681a0f0ca6264de277b7e982',
            220_237,
            *US_PUBLIC_DOMAIN,
            'Internal Revenue Service, U.S. Department of the Treasury. Form 1040 (2025). US Government work, public '
            'domain.',
            'an AcroForm of text and checkbox widgets; CFF fonts; encrypted by the tests with AES-256 (R6)',
        ),
        Source(
            'form-irs-fw4-2026',
            'https://www.irs.gov/pub/irs-prior/fw4--2026.pdf',
            IRS_PRIOR,
            'https://web.archive.org/web/20260923031243id_/https://www.irs.gov/pub/irs-prior/fw4--2026.pdf',
            '92444d8856ce55d9e25dca8b6d1420634fc68b11e1ab1f760916ea29ddd312b2',
            208_845,
            *US_PUBLIC_DOMAIN,
            'Internal Revenue Service, U.S. Department of the Treasury. Form W-4 (2026). US Government work, public '
            'domain.',
            'form widgets; CID-keyed Type0 Identity-H; encrypted by the tests with AES-128 (R4, crypt filters)',
        ),
        Source(
            'form-irs-fw9-2024',
            'https://www.irs.gov/pub/irs-prior/fw9--2024.pdf',
            IRS_PRIOR,
            'https://web.archive.org/web/20260428193413id_/https://www.irs.gov/pub/irs-prior/fw9--2024.pdf',
            '2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53',
            140_815,
            *US_PUBLIC_DOMAIN,
            'Internal Revenue Service, U.S. Department of the Treasury. Form W-9 (Rev. March 2024). US Government '
            'work, public domain.',
            'form widgets; six pages of dense small text; encrypted by the tests with RC4-128 (R3)',
        ),
    )
}

#: Where pdf.js's test PDFs are fetched from: the repository at one commit, so each name's bytes never change.
PDFJS_COMMIT: Final = (
    'https://raw.githubusercontent.com/mozilla/pdf.js/9548d30dd48ac1e23cc4bd7267d987d58d3cda28/test/pdfs/'
)
PDFJS_PERMANENCE: Final = (
    'a file of a git repository at one commit (raw.githubusercontent.com/<owner>/<repo>/<sha>/...)'
)
APACHE_2: Final = ('Apache-2.0', 'https://www.apache.org/licenses/LICENSE-2.0')
#: The ``bitmap-*`` files came to pdf.js from SerenityOS's JBIG2 test suite.
SERENITY_BSD_2: Final = (
    'Apache-2.0 (pdf.js) over BSD-2-Clause (its SerenityOS origin)',
    'https://github.com/SerenityOS/serenity/blob/master/LICENSE',
)
PDFJS_ATTRIBUTION: Final = 'Mozilla pdf.js contributors, test/pdfs/{} at commit 9548d30. Apache-2.0.'
SERENITY_ATTRIBUTION: Final = (
    'Mozilla pdf.js contributors, test/pdfs/{} at commit 9548d30, Apache-2.0; from the SerenityOS JBIG2 test suite '
    '(c) the SerenityOS developers, BSD-2-Clause.'
)

#: pdf.js's JBIG2 test PDFs (user ruling D157 of task document-lab-pdf-reader admits Apache-2.0 and BSD files):
#: one page each, every coding T.88 names -- generic templates 0-3, TPGDON, custom AT, MMR, Huffman symbol and text
#: regions with standard and custom tables, halftone, generic and text refinement, standalone refinement, stripes.
#: Drawn, not read against goldens (:data:`SOURCES`): ``tests/corpus/test_jbig2.py``.
JBIG2: Final = {
    name: Source(
        name,
        PDFJS_COMMIT + name,
        PDFJS_PERMANENCE,
        None,  # the commit is the pin; no Wayback snapshot is needed
        sha256,
        size,
        *(SERENITY_BSD_2 if name.startswith('bitmap-') else APACHE_2),
        (SERENITY_ATTRIBUTION if name.startswith('bitmap-') else PDFJS_ATTRIBUTION).format(name),
        'JBIG2Decode images (pdf.js test suite)',
    )
    for name, sha256, size in (
        (
            'bitmap-composite-and-xnor-halftone.pdf',
            'a88d42b10bd7c860abd3bf16432378d23a9fcf267a7008be047b608e1a332f4e',
            1_537,
        ),
        (
            'bitmap-composite-and-xnor-refine.pdf',
            'c2046037b2617ffdde875d523bf56a428339d7a6f753c25f4f957317bc9ecc9a',
            1_891,
        ),
        (
            'bitmap-composite-and-xnor-text.pdf',
            'aba6249b83158532ad87ccf47ba69737c9ba2b787c2f815767d239e910564235',
            1_243,
        ),
        ('bitmap-composite-and-xnor.pdf', '4432b73b7efcae8eb960b409de37af218f366bd548709d74773dae4ac7c257ef', 1_314),
        (
            'bitmap-composite-or-xor-replace-halftone.pdf',
            '3ffb92c8a77140d06ea628514a1d44c116a5067d40e3cd3887455b8d40918202',
            1_849,
        ),
        (
            'bitmap-composite-or-xor-replace-refine.pdf',
            'cc9510f5b113c8a2053fbc3891e0c8275bf0f59e1ca44522832e77d6a26cc145',
            2_343,
        ),
        (
            'bitmap-composite-or-xor-replace-text.pdf',
            'f4fb077e7e541f01ce21983f7822bd2b43d98f8554bf9cb421a03c7daa6442a5',
            1_578,
        ),
        (
            'bitmap-composite-or-xor-replace.pdf',
            'ba8dfa7f3879d2c851287769c0503cf84441e871a93e312e46253e9238dca737',
            1_630,
        ),
        ('bitmap-customat-tpgdon.pdf', '91b35c6963b8135dde1f509b424d3cfe5b02d9cc023c802c412916c6b76cc938', 1_057),
        ('bitmap-customat.pdf', '9be6eb11af2e4078eb209cbf680bb415960f2275f7e120660d22e7edc09e2ede', 1_051),
        ('bitmap-halftone-10bpp-mmr.pdf', 'c164623c7745aa431f59fa4a088e5adf3ff640687469576c64c5b044102ddae0', 1_710),
        ('bitmap-halftone-10bpp.pdf', '8a4577a1cdbdf342e2a61e18a239423452355c13733075af76fcd008f12eac5f', 1_311),
        ('bitmap-halftone-composite.pdf', 'e52ea26a9cadf0a1fbcd58c3ae411cf2bf14b05f172966283a6254296c4825a0', 1_599),
        ('bitmap-halftone-grid.pdf', '5ba3f0ca836c6df256415b2ddd61286b819f5b6a3f5598e146601839302c0158', 1_622),
        ('bitmap-halftone-refine.pdf', 'f2efff25bcf1ee5d8ee87e3b7c54c7b75e8a1ad33c9e5be22386ed35648dfd7d', 1_399),
        ('bitmap-halftone-skip-dummy.pdf', '6cdcfefd9200ddfbd37a15ce66ef50421d2fbcc9b396b2a9714b316f7c0a0ff4', 1_287),
        (
            'bitmap-halftone-skip-grid-template1.pdf',
            'bafb6edc883b156dbc337b74b0230fbcc5c45f48e08123f3040872edb1c1ec69',
            1_606,
        ),
        (
            'bitmap-halftone-skip-grid-template2.pdf',
            '4aedf60701a9b35e17766e4c26a7326adb0b618c87fb487e7cdde8c909efdd11',
            1_597,
        ),
        (
            'bitmap-halftone-skip-grid-template3.pdf',
            'ad156587b727404d16794dafcc738319b8a2a601374895634b0d5ee8d8b34f1b',
            1_600,
        ),
        ('bitmap-halftone-skip-grid.pdf', 'cfd77f057c98bb1af01fc108c5a9e0b09b438e0e4f1a31aa8b07ce7b275cc348', 1_624),
        ('bitmap-halftone-template1.pdf', 'f509758d869cc08a1b45b0fcee33794ae1f45887105ba6fa5a2c8f52598518fc', 1_265),
        ('bitmap-halftone-template2.pdf', '6c3d7bdc9e1012cce3c80b27592fdcc7864734c628184e1b55245b678d0ec47f', 1_299),
        ('bitmap-halftone-template3.pdf', '4f249e0d12ad232b96de05592c250498dd3fc661607a8fcbd721bdb5a79eb425', 1_300),
        ('bitmap-halftone.pdf', 'c276d6cbc7d0709af448b532232c9f8b92424859974b38feaef0d8983bd6d7c0', 1_287),
        (
            'bitmap-initially-unknown-size.pdf',
            '97c0fe6757d24bc2469a8b3a1fab346550ae5e7289f6855e21ebc2ce421dbe6f',
            1_030,
        ),
        ('bitmap-mmr.pdf', '32b70aeb63ca60f60c25c2d797b8f5af16043d66dce1bcb0b0f08a78acee95ed', 1_122),
        ('bitmap-p32-eof.pdf', '6c959401b0a43e94b4abc580d95b8373adcea87cb13612a3b9eece18767de576', 1_490),
        ('bitmap-randomaccess.pdf', '82079253a8c5af9683e5dc341e7416d8eb5901c48d6af850a79413e6f11711ec', 1_026),
        (
            'bitmap-refine-customat-tpgron.pdf',
            '7a17d8f6d26061f3110efd54f297a7e0e8d4652e3086cc99e3c171d31fd0810f',
            1_146,
        ),
        ('bitmap-refine-customat.pdf', '161d25d77c9bf788515dd0fc1aaaf6e35e1c23495acffe9cf99a98d567d1c5de', 1_146),
        ('bitmap-refine-lossless.pdf', 'f30056cb8e30203e3c16f00ffe28bc90bdb6af8aa72d58c14f19e71ec9c7cecf', 1_128),
        ('bitmap-refine-page-subrect.pdf', '74114e6f3925faf7145c0738168ecd3041c45c0ea69b2a91e7f105cdf7045bd0', 1_110),
        ('bitmap-refine-page.pdf', '7dbeb64018e183c77333996eeb5c2b9bcc90caaca8381ad42fb98a64c14025eb', 1_127),
        ('bitmap-refine-refine.pdf', '2ebb942ac5be0e066ab4ce204384a39b61692ca75fab0a4df4f36dae925e1f4d', 1_203),
        (
            'bitmap-refine-template1-tpgron.pdf',
            '921c1eab4ece655029bb524eb22136d21f305d13779f86fbe81086b62a22d4b3',
            1_230,
        ),
        ('bitmap-refine-template1.pdf', '019b26e3de36cbdd829e2faef67866f4140abb17040cd0b626e5bceab272b68c', 1_116),
        ('bitmap-refine-tpgron.pdf', '6b17840d81375faafbe48e2150706f1a70271865ec18b9f2e6700e06dadbe86c', 1_251),
        ('bitmap-refine.pdf', 'ec9e2f9601167ccb00f961e33dec5cc46a23bf08e132f9dce5d3aecb3f1cd9f9', 1_128),
        (
            'bitmap-stripe-initially-unknown-height.pdf',
            'fc2a20fc1cb5b4fd780e68be7b7ed95e63aa9b58ef290d7501b97212c206ff0f',
            1_271,
        ),
        ('bitmap-stripe-last-implicit.pdf', 'ff3b3a2465c764fd1705db05edfb3f11336e6a93535bad956e5d5aaebaecf4ae', 1_256),
        (
            'bitmap-stripe-single-no-end-of-stripe.pdf',
            '8bf3bbb2a48b0bc689b920fece52080582b15415eef3615e811ffb2cc8932966',
            1_026,
        ),
        ('bitmap-stripe-single.pdf', 'b6d9b0b9348e9f4c33ba192a32c913ecc446d212dc3df0108ed732cc399b5482', 1_041),
        ('bitmap-stripe.pdf', 'b1e3c558a34537aeace4334cbcea6c42f82f4b815c8232df462ee8956e6faf82', 1_271),
        ('bitmap-symbol-big-segmentid.pdf', '2fb59d0fbb8396bc812d5ed9af8bc6b2b8a69051519b8449b360cb304ea4e0b8', 1_176),
        ('bitmap-symbol-context-reuse.pdf', 'a6f05b48046b132054efe967c68ad5292b6835f78cce14bf2a613c9bce39b2c1', 1_192),
        ('bitmap-symbol-empty.pdf', 'adf661486871b31773fc85750372dbd3051db664b17128791e1bf667becde7e2', 1_133),
        (
            'bitmap-symbol-negative-sbdsoffset.pdf',
            '5aa48933bc980565e1c0ea93d91c4fe5370e96d4a461198481aa86af908793e5',
            1_077,
        ),
        ('bitmap-symbol-refine.pdf', '5e2ae6875ea8d0f1d2aecfaeff6aeac6acb8ff76eadef42e5f0fa7fc76c818b0', 1_185),
        (
            'bitmap-symbol-symbolrefine-textrefine.pdf',
            '2902534243c382770456f958fe7ba835f1916ea85f6219a3ba2b122f63db34c6',
            1_301,
        ),
        (
            'bitmap-symbol-symbolrefineone-customat.pdf',
            'cd8792e011e41f56b07dd0ba14df10b89c36d4492937032c60656177f5d27da5',
            1_167,
        ),
        (
            'bitmap-symbol-symbolrefineone-template1.pdf',
            'f175b6edddc6d40cbacc447b46a628c164a9a5ea0185b20f09f48bedbe7516c3',
            1_155,
        ),
        (
            'bitmap-symbol-symbolrefineone.pdf',
            '670053e1dbd980602b44f052bc9c0ffa7e7835521fc0ee324261a4143b52f66c',
            1_164,
        ),
        (
            'bitmap-symbol-symbolrefineseveral.pdf',
            '705133ec80d7e755ca206be508ad92840f11555520a5c117a3fe2886c30e2435',
            1_129,
        ),
        (
            'bitmap-symbol-symhuff-texthuff.pdf',
            'af8f3c70dbe1c0f0da090c829cea6da5015ed9978dd13fd2d0b9b44ac715db2b',
            1_209,
        ),
        (
            'bitmap-symbol-symhuff-texthuffB10B13.pdf',
            '1e0ad39f563c79b7e650347265dafebb1ccbfb20c9fbce62fc64575af9a13aa9',
            1_209,
        ),
        (
            'bitmap-symbol-symhuffB5B3-texthuffB7B9B12.pdf',
            '7777647082ccdde57b94bbdbcf1e0a743537adb02faabd662414634c92815d57',
            1_213,
        ),
        (
            'bitmap-symbol-symhuffcustom-texthuffcustom.pdf',
            'c0799af446d978d843ee0788a6dddb8c572a5e2b57414de4e7b2a742ec2d34a2',
            1_415,
        ),
        (
            'bitmap-symbol-symhuffrefine-textrefine.pdf',
            '3e4eecb287e59b630d9799bf00d2085f890d23bbc6ae2bebfb86b10c00b8a6a3',
            1_417,
        ),
        (
            'bitmap-symbol-symhuffrefineone.pdf',
            'a38677054cb800d0a8b1bac32d21f9ad9e6102bcd650fc3735015d05e1778cc9',
            1_298,
        ),
        (
            'bitmap-symbol-symhuffrefineseveral.pdf',
            '6d3050b223ce7bda0c948635c9bf67c04eac6c21b6e214d988cda9e1599e257a',
            1_244,
        ),
        (
            'bitmap-symbol-symhuffuncompressed-texthuff.pdf',
            '347a83cce9504ff8414fe028d8db2b311ebe800cab0116469c90456dd889218b',
            6_177,
        ),
        ('bitmap-symbol-textbottomleft.pdf', '0bc02085b750d0413fa3360825cc23b5e5a449b2e3f08fa084c3b00b64e03683', 1_112),
        (
            'bitmap-symbol-textbottomlefttranspose.pdf',
            'a9c552ce38b76935843861f8491bcb2df3af3833baa354e035e2026c286b7f98',
            1_097,
        ),
        (
            'bitmap-symbol-textbottomright.pdf',
            'e50632eb44e2dc3dc411e4265844ccf6fdd146b496b8c84e3d81a5c9adf858a5',
            1_112,
        ),
        (
            'bitmap-symbol-textbottomrighttranspose.pdf',
            '27b036ccba3f9e068677ef8bf203cf66bc6fef1eb091188119a1b7f474344d26',
            1_097,
        ),
        ('bitmap-symbol-textcomposite.pdf', 'a3f3985d4adcff6f5fbd2df7fad5382f43ac1dd2f4f868cb854a7a067bdeaf86', 1_319),
        ('bitmap-symbol-texthuffrefine.pdf', '945dc0c5ad52ff055c857baa35bfb83e1283ec5ac8f9a031d81e320d41f47a89', 1_270),
        (
            'bitmap-symbol-texthuffrefineB15.pdf',
            '0668b10ff9061f028185a8b2c014184ceaa3c23cddaf058c91f2d068b2e20def',
            1_270,
        ),
        (
            'bitmap-symbol-texthuffrefinecustom.pdf',
            'd150d22c02ae175c031e77aea704813cce57c33c4dd6ed3f22fa7494bb29e453',
            1_396,
        ),
        (
            'bitmap-symbol-texthuffrefinecustomdims.pdf',
            'fef10e63ce72c3285a804767499a0a814a75d24f919659e8e69596cd2095114f',
            1_318,
        ),
        (
            'bitmap-symbol-texthuffrefinecustompos.pdf',
            '03b17dfc51931f8f9f35a2fabfe98e36bb61fc7e604e455c6c6d6fca3d6e6603',
            1_318,
        ),
        (
            'bitmap-symbol-texthuffrefinecustomposdims.pdf',
            '342d06ad5f3d6843ea77d9a03117f5a3df663d22375a68e50a71109eaea6c4c1',
            1_371,
        ),
        (
            'bitmap-symbol-texthuffrefinecustomsize.pdf',
            '1a2a7861127869e6c779b65b898ec41e5045f1107242923aa8109c8c412cdec1',
            1_295,
        ),
        (
            'bitmap-symbol-textrefine-customat.pdf',
            '2c86c9cde4e623f2ba3c94d6b9e067298b25adeebd96ab704cbe908c22de5c8b',
            1_132,
        ),
        (
            'bitmap-symbol-textrefine-negative-delta-width.pdf',
            '53fa2b3ccef2999986369ea19c245dbada17c17f959e0c0d98f926e2fdf2fb1a',
            1_130,
        ),
        ('bitmap-symbol-textrefine.pdf', 'c6dc6a57d164c86c8aa59e6136f8b166a4697213ef6d5669bed6a7f6227bed0b', 1_128),
        ('bitmap-symbol-texttopright.pdf', 'e5b6eb678be01e99db7a6316fa99070bebf29489ae61054bdde04c9c26c6b9dd', 1_109),
        (
            'bitmap-symbol-texttoprighttranspose.pdf',
            '2e6522bf6fbbf01fbd8c05b00915b9f5c2783cbdf1815b53b027f49fb91cdad7',
            1_097,
        ),
        ('bitmap-symbol-texttranspose.pdf', '926887bcb2397304842c777a703219296d9b396d924922d492103327ab99a661', 1_097),
        ('bitmap-symbol.pdf', '39bf73b128947979f9a6080791ad03e206ea5756f4b4d37b6609e154d4f875b2', 1_109),
        (
            'bitmap-template1-customat-tpgdon.pdf',
            '071fd70fdad79ea30a2bfbaafcee63f7da0e2b723ffeef5949d583bae39ffd58',
            1_020,
        ),
        ('bitmap-template1-customat.pdf', '9fbdff065b72818b2e8cef051586b4807afefc3f68773117a0d7fca217c7eb42', 1_015),
        ('bitmap-template1-tpgdon.pdf', 'e8e715223bceac2b95b186e7220594793a57333c8d0da64cde2060027b9e91d4', 1_206),
        ('bitmap-template1.pdf', '25ab482ef6ba07335fd4927ac3ab22b231044345b48e024b5ed2399b048bc677', 1_015),
        (
            'bitmap-template2-customat-tpgdon.pdf',
            'f062d3e4f6bed408216ed042a113aca71ff30cb859b314c26b3d7259cfa5ca07',
            1_065,
        ),
        ('bitmap-template2-customat.pdf', '389e2719d8432f7a4d0a579e49f2fba3a6e085a007c046023666e66432105554', 1_055),
        ('bitmap-template2-tpgdon.pdf', 'f54b9f8a913f7f9389ea3fbe6ff3afd148753be651d5f6ed8caebd67acbc6271', 1_176),
        ('bitmap-template2.pdf', 'f37d525a05dfbfc5936b56c0fb7b7af382d0fa2da56a1a305c99850d5982912b', 1_018),
        (
            'bitmap-template3-customat-tpgdon.pdf',
            'a536e4f1dbe09f27f5e355bfad252e96d223cce761a9f58cf47fcfb507133f4d',
            1_079,
        ),
        ('bitmap-template3-customat.pdf', '8db3cd4ebd4d75a0a45c685ce243af13855bb6a3642b3a3810b0e7ce29786389', 1_070),
        ('bitmap-template3-tpgdon.pdf', '1d8514a3c8664ac84b1434ccac539f30fec3a02dccbd1da612a7925394cb2a48', 1_197),
        ('bitmap-template3.pdf', '98b5e78016e3415990e9f3c99b5ed2940ed869826d9ddf04385f8abdf543cf13', 1_033),
        ('bitmap-tpgdon.pdf', '7d9c22dcee6a357e6290de04b57a19ffc42a6fd5d802d62a4cbd75a59b078924', 1_211),
        (
            'bitmap-trailing-7fff-stripped-harder-refine.pdf',
            '17e08dc35c1fe2c6a9ce5db25ee53805248ccb9a8ae528a339d37d764f29dc9a',
            1_887,
        ),
        (
            'bitmap-trailing-7fff-stripped-harder.pdf',
            '47fa9873c6250a75552b60771fed2494f14eb4a20bdacf558ee00f9136835be9',
            15_181,
        ),
        (
            'bitmap-trailing-7fff-stripped.pdf',
            '1fb303a7eaa11933f3f1e5c3b554375649ea4a5f1226efa97a7534cac768b8fb',
            1_049,
        ),
        ('jbig2_file_header.pdf', '5edb6045fc86ee5fa8e837685a35a57014764d1f72518e136e5e8f13fcf92c8d', 774),
        ('jbig2_symbol_offset.pdf', '305b71c67829ef0bb8bddab9fa3ee763dcecd173f5fd996740d17212888fa97d', 1_157),
        ('perf_codecs_jbig2.pdf', 'c7cdfb036282939f6a5461d67c34c895910856266c598f118cc41c9b7ca1d82e', 9_909),
    )
}


class FetchRefused(RuntimeError):
    """No URL of a source gave its pinned bytes -- an outage or a changed source, never the reader's fault."""


def fetch(name: str) -> Path:
    """The verified local copy of source ``name`` (:data:`SOURCES`), downloaded into :data:`CACHE` when absent.

    Raises:
        FetchRefused: neither the URL nor the fallback gave the pinned bytes.

    """
    return fetch_into({**SOURCES, **JBIG2}[name], CACHE)


def fetch_into(source: Source, cache: Path) -> Path:
    """``source``'s verified copy in ``cache`` (``<sha256>.pdf``): kept when its bytes are right, else downloaded.

    The original URL first, the fallback only when it fails; a download is written beside its place and renamed into
    it only once verified, so a reader never sees half a file.

    Raises:
        FetchRefused: neither the URL nor the fallback gave the pinned bytes.

    """
    path = cache / f'{source.sha256}.pdf'
    if path.is_file() and _digest(path.read_bytes()) == source.sha256:
        return path
    said = []
    for url in (source.url, source.fallback):
        if url is None:
            said.append('no fallback snapshot exists')
            continue
        try:
            data = _download(url)
        except _Unreachable as error:
            said.append(str(error))
            continue
        if _digest(data) != source.sha256 or len(data) != source.size:
            said.append(f'{url} served {len(data)} bytes, sha256 {_digest(data)}')
            continue
        cache.mkdir(parents=True, exist_ok=True)
        part = cache / f'{source.sha256}.{os.getpid()}.part'  # this process's own; renamed whole, never read half
        part.write_bytes(data)
        try:
            part.replace(path)
        except PermissionError:  # Windows: a sibling worker renamed its copy in first and holds it open
            if not path.is_file() or _digest(path.read_bytes()) != source.sha256:
                raise
            part.unlink()
        return path
    msg = (
        f'FETCH FAILED, not a reader regression: {source.name} is pinned to {source.size} bytes, sha256 '
        f'{source.sha256}, and no URL gave them -- {"; ".join(said)}. An outage passes with time; a changed source '
        f'is surveyed again (license included) and re-pinned in tests/corpus/sources.py, or dropped'
    )
    raise FetchRefused(msg)


class _Unreachable(Exception):
    """A URL that gave no response body after every attempt."""


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310 -- the https URLs above
    cause = ''
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 -- the https URLs above
                return response.read()
        except urllib.error.HTTPError as error:
            cause = f'HTTP {error.code} {error.reason}'
            if error.code not in _TRANSIENT:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            cause = str(getattr(error, 'reason', error))
        if attempt < ATTEMPTS:
            time.sleep(2**attempt)
    msg = f'{url} failed: {cause}'
    raise _Unreachable(msg)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    """Fetch every source into :data:`CACHE`, printing each path; :data:`FETCH_FAILED`, naming each that failed."""
    failed = []
    for name in {**SOURCES, **JBIG2}:
        try:
            print(f'{name}: {os.fspath(fetch(name))}')  # noqa: T201 -- the command's output
        except FetchRefused as error:
            failed.append(str(error))
    for line in failed:
        print(line, file=sys.stderr)  # noqa: T201 -- the command's output
    return FETCH_FAILED if failed else 0


if __name__ == '__main__':
    sys.exit(main())
