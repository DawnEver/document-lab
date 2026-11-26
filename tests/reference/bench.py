"""How fast and how lean ``read_pdf`` reads real PDFs, against the reference, PyMuPDF, document by document.

``python -m tests.reference.bench [--jobs N] [--json OUT] [PATH ...]`` -- PATHs are PDFs or directories searched for
them, ``DOCUMENT_LAB_REFERENCE_PDFS`` when none is given (real documents are never committed: the reference part's
corpus). A local measuring script, not a test: it prints a table and judges nothing.

Each document is read in a fresh process per reader, so a reading's peak memory is its own: ``ours`` reads every
page through ``read_pdf`` (``Document.pages``), PyMuPDF extracts the same per page through its fastest calls by
default -- ``get_texttrace()`` (every glyph with its box, font and color), ``get_cdrawings()``,
``get_image_info()``; ``--text`` and ``--drawings`` select its structured calls instead (``rawdict`` builds a dict
per character, ``get_drawings`` one per path; they cost it about ten times more). Time is wall time from the path to
the last page (imports excluded); memory is the process's peak working set (resident set) over its working set
once both readers were imported. A document PyMuPDF asks a password for is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab_commons.log import emit

#: Seconds a reading may take before it is reported as timed out.
TIMEOUT = 600

#: PyMuPDF's path readers: ``get_cdrawings`` returns its raw tuples, ``get_drawings`` builds a dict per path on them.
DRAWINGS = ('get_cdrawings', 'get_drawings')
#: PyMuPDF's text readers: ``texttrace`` its raw glyph trace, ``rawdict``/``dict`` blocks of lines of spans built on it.
TEXTS = ('texttrace', 'rawdict', 'dict')


def _resident() -> tuple[int, int]:
    """This process's working set now and at its peak, in bytes."""
    if sys.platform == 'win32':
        import ctypes  # noqa: PLC0415 -- Windows only
        from ctypes import wintypes  # noqa: PLC0415

        class Counters(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL('psapi')
        kernel = ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    import resource  # noqa: PLC0415 -- POSIX only

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    with Path('/proc/self/statm').open() as statm:
        now = int(statm.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    return now, peak


def _ours(pdf: Path) -> int:
    from document_lab.pdf import read_pdf  # noqa: PLC0415 -- imported before the clock starts, see _worker

    doc = read_pdf(pdf)
    return sum(1 for _ in doc.pages)


def _theirs(pdf: Path, text: str, drawings: str) -> int:
    import pymupdf  # noqa: PLC0415 -- imported before the clock starts, see _worker

    pages = 0
    with pymupdf.open(pdf) as doc:
        for page in doc:
            page.get_texttrace() if text == 'texttrace' else page.get_text(text)
            getattr(page, drawings)()
            page.get_image_info()
            pages += 1
    return pages


def _worker(reader: str, pdf: Path, text: str, drawings: str) -> dict:
    """One reading in this process: pages, seconds, and peak working set over the one after imports (MB)."""
    import pymupdf  # noqa: PLC0415

    import document_lab.pdf  # noqa: F401, PLC0415

    with pymupdf.open(pdf) as doc:
        if doc.needs_pass:
            return {'locked': True}
    base, _ = _resident()
    start = time.perf_counter()
    pages = _ours(pdf) if reader == 'ours' else _theirs(pdf, text, drawings)
    seconds = time.perf_counter() - start
    _, peak = _resident()
    return {'pages': pages, 'seconds': seconds, 'peak_mb': max(peak - base, 0) / 2**20, 'base_mb': base / 2**20}


def _measure(reader: str, pdf: Path, text: str, drawings: str) -> dict:
    cmd = [sys.executable, '-m', 'tests.reference.bench', '--worker', reader, '--text', text, '--drawings', drawings]
    cmd.append(str(pdf))
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {'error': f'timed out after {TIMEOUT} s'}
    if done.returncode != 0:
        return {'error': (done.stderr.strip().splitlines() or ['no output'])[-1]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def _pdfs(paths: list[str]) -> list[Path]:
    roots = [Path(p) for p in paths or os.environ.get('DOCUMENT_LAB_REFERENCE_PDFS', '').split(os.pathsep) if p]
    if not roots:
        sys.exit('name PDFs or directories, or set DOCUMENT_LAB_REFERENCE_PDFS')
    return sorted({pdf for root in roots for pdf in ([root] if root.is_file() else root.rglob('*.pdf'))})


def _report(rows: list[dict], text: str, drawings: str) -> None:
    read = [r for r in rows if 'seconds' in r['ours'] and 'seconds' in r['theirs']]
    failed = [r for r in rows if 'error' in r['ours'] or 'error' in r['theirs']]
    ours, theirs = (sum(r[k]['seconds'] for r in read) for k in ('ours', 'theirs'))
    pages = sum(r['ours']['pages'] for r in read)
    locked = sum(1 for r in rows if r['ours'].get('locked'))
    emit(
        f'{len(read)} documents, {pages} pages; {locked} locked, {len(failed)} failed; PyMuPDF by {text}, {drawings}()'
    )
    emit(f'total time: ours {ours:.2f} s, PyMuPDF {theirs:.2f} s (ours/PyMuPDF {ours / max(theirs, 1e-9):.2f})')
    peak = max(read, key=lambda r: r['ours']['peak_mb'], default=None)
    if peak is not None:
        their_peak = max(r['theirs']['peak_mb'] for r in read)
        emit(f'peak memory: ours {peak["ours"]["peak_mb"]:.0f} MB ({peak["pdf"].name}), PyMuPDF {their_peak:.0f} MB')
        emit(f'working set after imports: {peak["ours"]["base_mb"]:.0f} MB')
    emit('\nslowest of ours (ours s / PyMuPDF s, ours MB / PyMuPDF MB, pages):')
    for r in sorted(read, key=lambda r: -r['ours']['seconds'])[:10]:
        o, t = r['ours'], r['theirs']
        emit(
            f'  {o["seconds"]:7.2f} / {t["seconds"]:6.2f}  {o["peak_mb"]:6.0f} / {t["peak_mb"]:5.0f}  '
            f'{o["pages"]:4d}  {r["pdf"].name}'
        )
    emit('\nworst ratio of ours to PyMuPDF (over 0.05 s):')
    for r in sorted(
        (r for r in read if r['theirs']['seconds'] > 0.05),
        key=lambda r: -r['ours']['seconds'] / r['theirs']['seconds'],
    )[:10]:
        o, t = r['ours'], r['theirs']
        emit(f'  {o["seconds"] / t["seconds"]:5.2f}x  {o["seconds"]:7.2f} / {t["seconds"]:6.2f}  {r["pdf"].name}')
    for r in failed:
        emit(f'FAILED {r["pdf"].name}: ours {r["ours"].get("error", "")} | PyMuPDF {r["theirs"].get("error", "")}')


def main(argv: list[str] | None = None) -> None:
    """Measure the documents named (as the module says) and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', help='PDFs or directories (default: DOCUMENT_LAB_REFERENCE_PDFS)')
    parser.add_argument('--jobs', type=int, default=1, help='documents measured at once (1: timings undisturbed)')
    parser.add_argument('--json', type=Path, help='also write every measurement here')
    parser.add_argument(
        '--drawings',
        choices=DRAWINGS,
        default=DRAWINGS[0],
        help='the PyMuPDF call that reads paths (default: its fastest)',
    )
    parser.add_argument(
        '--text', choices=TEXTS, default=TEXTS[0], help='the PyMuPDF call that reads text (default: its fastest)'
    )
    parser.add_argument('--worker', choices=('ours', 'theirs'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        emit(json.dumps(_worker(args.worker, Path(args.paths[0]), args.text, args.drawings)))
        return
    pdfs = _pdfs(args.paths)

    def one(pdf: Path) -> dict:
        return {
            'pdf': pdf,
            'ours': _measure('ours', pdf, args.text, args.drawings),
            'theirs': _measure('theirs', pdf, args.text, args.drawings),
        }

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, pdfs))
    _report(rows, args.text, args.drawings)
    if args.json:
        args.json.write_text(json.dumps([{**r, 'pdf': str(r['pdf'])} for r in rows], indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
