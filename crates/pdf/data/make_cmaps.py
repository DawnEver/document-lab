"""Write ``src/cmaps.rs`` and ``src/cmaps/``: the predefined CJK ``CMap``s of PDF, each code's CID.

ISO 32000-2 section 9.7.5.2 (Table 116) names the predefined ``CMap``s a PDF may use by name alone (``NAMES``:
``GBK-EUC-H``, ``90ms-RKSJ-V``, ``UniJIS-UTF16-H`` ...; ``Identity-H``/``-V`` are read as such). The source is Adobe's
``cmap-resources`` pinned at ``REVISION``. Each is resolved whole -- its ``usecmap`` parent first, its
``notdefrange``s under its ``cidrange``s and ``cidchar``s, a later definition winning -- then stored as its difference
from a BASE: the earlier ``CMap`` of its collection whose difference is smallest (``GBK-EUC-V`` is ``GBK-EUC-H`` and
36 vertical forms), or none when the whole is smaller.

A ``CMap``'s own mappings (code to CID; CID 0 unmaps a code its base maps) are SEGMENTS of consecutive codes, sorted
by code, each stored the smaller way: a RUN to consecutive CIDs, or DENSE, one CID per code. A code in no segment is
the base's. What is written:
- ``src/cmaps/segments.bin``: every segment, twelve bytes -- its first and last code, and either its first CID or
  (top bit set) the index of its first CID in ``cids.bin`` -- each a little-endian u32;
- ``src/cmaps/cids.bin``: the dense segments' CIDs, little-endian u16;
- ``src/cmaps.rs``: per ``CMap``, in ``NAMES``' order, its collection's ``/Ordering``, writing mode, code space, base
  and segment span;
- ``src/cmaps/LICENSE.txt``: Adobe's BSD-3-Clause notice of the source, which a redistribution must carry.

A code is its bytes as a big-endian number, as ``crate::cmap`` reads it; codes of two lengths with one number refuse.
The downloads are kept under the repository's ``output/cmaps/<REVISION>/``. Run from anywhere:
``python crates/pdf/data/make_cmaps.py``; the output is a function of the pinned files alone.
"""

import json
import re
import urllib.request
from pathlib import Path

REPOSITORY = 'adobe-type-tools/cmap-resources'
#: The tag 20231115.
REVISION = 'f5cf3bca7fdfeaceb77aa82847e974f2306c20b4'
#: ISO 32000-2 Table 116, Identity's aside: Chinese (simplified, traditional), Japanese, Korean.
NAMES = (
    *('GB-EUC-H', 'GB-EUC-V', 'GBpc-EUC-H', 'GBpc-EUC-V', 'GBK-EUC-H', 'GBK-EUC-V', 'GBKp-EUC-H', 'GBKp-EUC-V'),
    *('GBK2K-H', 'GBK2K-V', 'UniGB-UCS2-H', 'UniGB-UCS2-V', 'UniGB-UTF16-H', 'UniGB-UTF16-V'),
    *('B5pc-H', 'B5pc-V', 'HKscs-B5-H', 'HKscs-B5-V', 'ETen-B5-H', 'ETen-B5-V', 'ETenms-B5-H', 'ETenms-B5-V'),
    *('CNS-EUC-H', 'CNS-EUC-V', 'UniCNS-UCS2-H', 'UniCNS-UCS2-V', 'UniCNS-UTF16-H', 'UniCNS-UTF16-V'),
    *('83pv-RKSJ-H', '90ms-RKSJ-H', '90ms-RKSJ-V', '90msp-RKSJ-H', '90msp-RKSJ-V', '90pv-RKSJ-H'),
    *('Add-RKSJ-H', 'Add-RKSJ-V', 'EUC-H', 'EUC-V', 'Ext-RKSJ-H', 'Ext-RKSJ-V', 'H', 'V'),
    *('UniJIS-UCS2-H', 'UniJIS-UCS2-V', 'UniJIS-UCS2-HW-H', 'UniJIS-UCS2-HW-V', 'UniJIS-UTF16-H', 'UniJIS-UTF16-V'),
    *('KSC-EUC-H', 'KSC-EUC-V', 'KSCms-UHC-H', 'KSCms-UHC-V', 'KSCms-UHC-HW-H', 'KSCms-UHC-HW-V', 'KSCpc-EUC-H'),
    *('UniKS-UCS2-H', 'UniKS-UCS2-V', 'UniKS-UTF16-H', 'UniKS-UTF16-V'),
)

CRATE = Path(__file__).resolve().parent.parent
CACHE = CRATE.parent.parent / 'output' / 'cmaps' / REVISION
HEX = r'<([0-9a-fA-F]+)>'
#: rustfmt's `array_width`: an array whose items fit in it stays on one line.
ARRAY_WIDTH = 60
#: Bytes of one segment, and of one dense CID.
SEGMENT, CID = 12, 2
#: The flag of a dense segment's value.
DENSE = 1 << 31


def fetch(name: str) -> bytes:
    """The pinned revision's file ``name``, downloaded once into the cache."""
    path = CACHE / name
    if not path.exists():
        url = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_bytes()


def paths() -> dict[str, str]:
    """Every ``CMap``'s name to its path in the repository, from the pinned tree."""
    path = CACHE / 'tree.json'
    if not path.exists():
        url = f'https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    tree = json.loads(path.read_text(encoding='utf-8'))
    return {p.split('/')[-1]: p for p in (item['path'] for item in tree['tree']) if '/CMap/' in p}


class CMap:
    """One ``CMap`` resolved: its ordering, writing mode, code space and each code's CID."""

    def __init__(self, name: str, where: dict[str, str], done: dict[str, 'CMap']) -> None:
        """Read ``name`` (and its ``usecmap`` parent, through ``done``)."""
        text = fetch(where[name]).decode('latin-1')
        body = text[text.index('begincmap') :]
        self.name = name
        self.ordering = re.search(r'/Ordering \((\w+)\)', body).group(1)
        self.vertical = re.search(r'/WMode (\d)', body).group(1) == '1'
        self.space: list[tuple[str, str]] = []
        self.cids: dict[int, int] = {}
        self.lengths: dict[int, int] = {}
        parent = re.search(r'/([\w-]+) usecmap', body)
        if parent:
            base = done.get(parent.group(1)) or CMap(parent.group(1), where, done)
            self.space, self.cids, self.lengths = list(base.space), dict(base.cids), dict(base.lengths)
        for block in re.findall(r'begincodespacerange(.*?)endcodespacerange', body, re.DOTALL):
            self.space += re.findall(rf'{HEX}\s*{HEX}', block)
        for kind in ('notdefrange', 'cidrange'):
            for block in re.findall(rf'begin{kind}(.*?)end{kind}', body, re.DOTALL):
                for lo, hi, cid in re.findall(rf'{HEX}\s*{HEX}\s*(\d+)', block):
                    for offset in range(int(hi, 16) - int(lo, 16) + 1):
                        self.put(int(lo, 16) + offset, int(cid) + (offset if kind == 'cidrange' else 0), len(lo))
        for block in re.findall(r'begincidchar(.*?)endcidchar', body, re.DOTALL):
            for code, cid in re.findall(rf'{HEX}\s*(\d+)', block):
                self.put(int(code, 16), int(cid), len(code))
        done[name] = self

    def put(self, code: int, cid: int, digits: int) -> None:
        """Map ``code`` (``digits`` hex digits long) to ``cid``; a number already a code of another length refuses."""
        if self.lengths.setdefault(code, digits) != digits:
            msg = f'{self.name}: code {code:#x} has two lengths'
            raise ValueError(msg)
        self.cids[code] = cid


def segments(mapping: dict[int, int]) -> list[tuple[int, int, int | list[int]]]:
    """``mapping`` (code to CID, 0 for unmapped) as segments: (first code, last code, first CID or every CID)."""
    out: list[tuple[int, int, int | list[int]]] = []
    stretch: list[int] = []
    for code in [*sorted(mapping), None]:
        if stretch and (code is None or code != stretch[-1] + 1):
            out += cheaper([mapping[c] for c in stretch], stretch[0])
            stretch = []
        if code is not None:
            stretch.append(code)
    return out


def cheaper(cids: list[int], first: int) -> list[tuple[int, int, int | list[int]]]:
    """Consecutive codes from ``first`` with their ``cids``: as runs, or dense, whichever is smaller."""
    runs: list[list[int]] = []
    for offset, cid in enumerate(cids):
        if runs and cid == (runs[-1][2] and runs[-1][2] + first + offset - runs[-1][0]):
            runs[-1][1] = first + offset
        else:
            runs.append([first + offset, first + offset, cid])
    if len(runs) * SEGMENT <= SEGMENT + len(cids) * CID:
        return [(lo, hi, cid) for lo, hi, cid in runs]
    return [(first, first + len(cids) - 1, cids)]


def size(parts: list[tuple[int, int, int | list[int]]]) -> int:
    """The bytes ``parts`` take."""
    return sum(SEGMENT + (len(v) * CID if isinstance(v, list) else 0) for _, _, v in parts)


def difference(cmap: CMap, base: CMap) -> dict[int, int]:
    """What ``cmap`` maps otherwise than ``base``: its own CIDs, and 0 for the codes it leaves unmapped."""
    out = {code: cid for code, cid in cmap.cids.items() if base.cids.get(code) != cid}
    out.update({code: 0 for code in base.cids if code not in cmap.cids})
    return out


def rust_bytes(hex_text: str) -> str:
    """A code's hex digits as a Rust byte-string literal."""
    return 'b"' + ''.join(f'\\x{b:02X}' for b in bytes.fromhex(hex_text)) + '"'


def entry(cmap: CMap, base: int | None, first: int, count: int) -> str:
    """One ``Predefined`` literal, laid as rustfmt lays it."""
    tuples = [f'({rust_bytes(lo)}, {rust_bytes(hi)})' for lo, hi in cmap.space]
    if len(', '.join(tuples)) <= ARRAY_WIDTH:
        space = f'&[{", ".join(tuples)}]'
    else:
        # A tuple a line: a bound is at most four bytes, so a tuple stays within rustfmt's call width.
        pairs = ''.join(f'            {t},\n' for t in tuples)
        space = f'&[\n{pairs}        ]'
    return (
        f'    Predefined {{\n        name: "{cmap.name}",\n        ordering: "{cmap.ordering}",\n'
        f'        vertical: {str(cmap.vertical).lower()},\n        space: {space},\n'
        f'        base: {"None" if base is None else f"Some({base})"},\n'
        f'        first: {first},\n        count: {count},\n    }},\n'
    )


HEADER = f"""\
//! The predefined CJK `CMap`s of PDF: each code's CID. GENERATED by `data/make_cmaps.py`; edit that, not this.
//!
//! From <https://github.com/{REPOSITORY}> at `{REVISION}`:
//! Copyright Adobe, BSD-3-Clause, the notice in `cmaps/LICENSE.txt`.

/// Every `CMap`'s segments, twelve bytes each -- first and last code, then the first CID, or (top bit set) the index
/// of the first in [`CIDS`] -- each a little-endian u32. CID 0: codes the `CMap` does not map though its base does.
pub static SEGMENTS: &[u8] = include_bytes!("cmaps/segments.bin");

/// The CIDs of the dense segments, little-endian u16.
pub static CIDS: &[u8] = include_bytes!("cmaps/cids.bin");

/// One predefined `CMap`; [`crate::cmap`] reads it.
#[derive(Debug)]
pub struct Predefined {{
    /// Its name.
    pub name: &'static str,
    /// The `/Ordering` of its collection (`/Registry (Adobe)`).
    pub ordering: &'static str,
    /// Whether its writing mode is vertical.
    pub vertical: bool,
    /// Its code space ranges: (low, high), each as bytes of one length.
    pub space: &'static [(&'static [u8], &'static [u8])],
    /// The `CMap` (its index in [`PREDEFINED`]) that maps every code its segments do not.
    pub base: Option<usize>,
    /// Its first segment in [`SEGMENTS`].
    pub first: usize,
    /// How many segments it has, sorted by code.
    pub count: usize,
}}

"""

if __name__ == '__main__':
    where = paths()
    done: dict[str, CMap] = {}
    segments_bin, cids_bin = bytearray(), bytearray()
    entries = ''
    for index, name in enumerate(NAMES):
        cmap = done.get(name) or CMap(name, where, done)
        whole = segments(cmap.cids)
        best = (size(whole), -1, whole)
        for earlier, other in enumerate(NAMES[:index]):
            if done[other].ordering == cmap.ordering:
                parts = segments(difference(cmap, done[other]))
                best = min(best, (size(parts), earlier, parts), key=lambda b: b[:2])
        _, base, own = best
        entries += entry(cmap, None if base < 0 else base, len(segments_bin) // SEGMENT, len(own))
        for lo, hi, value in own:
            if isinstance(value, list):
                dense = DENSE | len(cids_bin) // CID
                cids_bin += b''.join(cid.to_bytes(CID, 'little') for cid in value)
                value = dense  # noqa: PLW2901 -- the index stands for the list from here
            segments_bin += b''.join(v.to_bytes(4, 'little') for v in (lo, hi, value))
    (CRATE / 'src' / 'cmaps').mkdir(exist_ok=True)
    (CRATE / 'src' / 'cmaps' / 'segments.bin').write_bytes(bytes(segments_bin))
    (CRATE / 'src' / 'cmaps' / 'cids.bin').write_bytes(bytes(cids_bin))
    notice = fetch('LICENSE.md').decode('utf-8')
    (CRATE / 'src' / 'cmaps' / 'LICENSE.txt').write_text(notice, encoding='utf-8', newline='\n')
    table = (
        '// spelling-off: the code bytes below are data, not words.\n'
        f'/// Every predefined `CMap`.\npub static PREDEFINED: [Predefined; {len(NAMES)}] = [\n{entries}];\n'
        '// spelling-on\n'
    )
    (CRATE / 'src' / 'cmaps.rs').write_text(HEADER + table, encoding='utf-8', newline='\n')
