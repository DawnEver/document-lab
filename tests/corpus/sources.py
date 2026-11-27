"""The ``corpus`` part's real PDFs: where each is published, what it is pinned to, its license and its attribution.

The repo is public, so no real PDF is committed: :func:`fetch` downloads one from its ORIGINAL public URL into the
git-ignored cache (``output/cache/corpus/<sha256>.pdf``) and hands it over only when its bytes are the ones pinned
here. Only files with no licensing or exposure risk are listed (user rulings D29, D30 of task document-lab-pdf-reader):
CC BY 4.0 works and US Government works (public domain, 17 U.S.C. 105) without third-party content.

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

__all__ = ['CACHE', 'FETCH_FAILED', 'SOURCES', 'FetchRefused', 'Source', 'fetch', 'fetch_into']

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


class FetchRefused(RuntimeError):
    """No URL of a source gave its pinned bytes -- an outage or a changed source, never the reader's fault."""


def fetch(name: str) -> Path:
    """The verified local copy of source ``name`` (:data:`SOURCES`), downloaded into :data:`CACHE` when absent.

    Raises:
        FetchRefused: neither the URL nor the fallback gave the pinned bytes.

    """
    return fetch_into(SOURCES[name], CACHE)


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
        part.replace(path)
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
    for name in SOURCES:
        try:
            print(f'{name}: {os.fspath(fetch(name))}')  # noqa: T201 -- the command's output
        except FetchRefused as error:
            failed.append(str(error))
    for line in failed:
        print(line, file=sys.stderr)  # noqa: T201 -- the command's output
    return FETCH_FAILED if failed else 0


if __name__ == '__main__':
    sys.exit(main())
