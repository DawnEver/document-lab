"""The own renderer's text accepted against PyMuPDF's renders (``tests/rendering.py`` says how), one case or more per
font kind, each paired with a planted wrong truth -- the same text with its glyph codes rotated, every glyph drawn as
another -- that the comparison must report.

The programs are ``tests/fonts.py``'s, whose four glyphs (a square, a triangle, a ring, an L) cannot pass for one
another, so a glyph selected wrong is a difference: simple TrueType fonts through each ``cmap`` route ISO 32000-2
9.6.5.4 gives ((3,1) by the glyph name's character, (3,0) by the code, (1,0) by the name's Mac OS Roman code), a
composite one through its ``/CIDToGIDMap``, CFF programs by their own encoding, by ``/Differences`` names and,
CID-keyed, through their charset, an OpenType program, a Type 1 program (eexec), a Type 3 font's procedures, glyphs
stroked, filled and stroked, and clipped to, and a vertical font placed by its position vectors.

A standard font the file does not embed is drawn with its substitute, PDFium's Foxit font (``crates/pdf/src/fonts``),
where PyMuPDF draws URW's: the two designs differ, so such text is held to PyMuPDF drawing the very substitute,
embedded under the same name and widths -- which glyph each code selects, its width and place are compared, not the
design.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from tests import fonts
from tests.rendering import agrees, disagreement, own, page_pdf, reference

DPI = 150
#: The reader's substitutes for the standard fonts.
SUBSTITUTES = Path(__file__).resolve().parents[2] / 'crates' / 'pdf' / 'src' / 'fonts'

DESCRIPTOR = (
    '<< /Type /FontDescriptor /FontName /{name} /Flags {flags} /FontBBox [0 -200 700 800] /ItalicAngle 0 /Ascent 800 '
    '/Descent -200 /CapHeight 700 /StemV 80 {file} >>'
)
WIDTHS = '/FirstChar 1 /LastChar 255 /Widths [' + ' '.join(['700'] * 255) + ']'
#: Text in two sizes (under 24 and over 48 pixels an em at DPI, MuPDF placing each on its own grid) and two lines.
TEXT = 'BT /F1 30 Tf 20 90 Td ({0}) Tj ET BT /F1 11 Tf 20 40 Td ({0}{0}) Tj 0 -16 Td 0 0.4 0.8 rg ({0}) Tj ET'


def _simple(subtype: str, encoding: str, flags: int, program: tuple[str, str, bytes]) -> dict:
    """The objects of a simple font F1 of ``subtype`` embedding ``program`` -- its key in the descriptor, its
    stream's entries, its bytes."""
    file_key, file_entries, data = program
    return {
        'F1': (
            f'<< /Type /Font /Subtype /{subtype} /BaseFont /Synthetic {WIDTHS} {encoding} /FontDescriptor {{D}} >>',
            None,
        ),
        'D': (DESCRIPTOR.format(name='Synthetic', flags=flags, file=f'/{file_key} {{P}}'), None),
        'P': (f'<< {file_entries} >>', data),
    }


def _composite(cid_font: str, encoding: str, program: tuple[str, str, bytes], extra: str) -> dict:
    program_key, program_entries, data = program
    return {
        'F1': (
            '<< /Type /Font /Subtype /Type0 /BaseFont /Synthetic /Encoding /' + encoding + ' /DescendantFonts [{C}] >>',
            None,
        ),
        'C': (
            (
                f'<< /Type /Font /Subtype /{cid_font} /BaseFont /Synthetic /CIDSystemInfo << /Registry (Adobe) '
                f'/Ordering (Identity) /Supplement 0 >> /FontDescriptor {{D}} /DW 700 {extra} >>'
            ),
            None,
        ),
        'D': (DESCRIPTOR.format(name='Synthetic', flags=4, file=f'/{program_key} {{P}}'), None),
        'P': (f'<< {program_entries} >>', data),
    }


def _cid_map(cids_to_gids: dict[int, int]) -> bytes:
    size = max(cids_to_gids) + 1
    return b''.join(cids_to_gids.get(c, 0).to_bytes(2, 'big') for c in range(size))


FONT = '/Font << /F1 {F1} >>'
LATIN = {0x41: 1, 0x42: 2, 0x43: 3, 0x44: 4}
CFF = fonts.cff()
CID_CFF = fonts.cff(cid_keyed=True)
#: CIDs 1-4 select glyphs 3, 4, 1, 2.
CID_TO_GID = {1: 3, 2: 4, 3: 1, 4: 2}

#: name: (objects, text, the text of the planted wrong render: its glyph codes rotated).
CASES: dict[str, tuple[dict, str, str]] = {
    'truetype-unicode-cmap': (
        _simple('TrueType', '/Encoding /WinAnsiEncoding', 32, ('FontFile2', '', fonts.truetype({(3, 1): LATIN}))),
        'ABCD',
        'BCDA',
    ),
    'truetype-differences-by-name': (
        _simple(
            'TrueType',
            '/Encoding << /Differences [1 /A /B /C /D] >>',
            32,
            ('FontFile2', '', fonts.truetype({(3, 1): LATIN}, post_names=False)),
        ),
        '\\001\\002\\003\\004',
        '\\002\\003\\004\\001',
    ),
    'truetype-symbolic-cmap': (
        _simple(
            'TrueType', '', 4, ('FontFile2', '', fonts.truetype({(3, 0): {0xF000 + c: g for c, g in LATIN.items()}}))
        ),
        'ABCD',
        'BCDA',
    ),
    'truetype-mac-roman-cmap': (
        # WinAnsi's C4 is Adieresis, Mac OS Roman's 80: the (1,0) subtable is read by the name's Mac code.
        _simple(
            'TrueType',
            '/Encoding /WinAnsiEncoding',
            32,
            ('FontFile2', '', fonts.truetype({(1, 0): {0x80: 1, 0x42: 2, 0x43: 3, 0x44: 4}}, post_names=False)),
        ),
        '\\304BCD',
        'BCD\\304',
    ),
    'truetype-composite-cid-to-gid': (
        {
            **_composite(
                'CIDFontType2', 'Identity-H', ('FontFile2', '', fonts.truetype({(3, 1): LATIN})), '/CIDToGIDMap {M}'
            ),
            'M': ('<< >>', _cid_map(CID_TO_GID)),
        },
        '<0003000400010002>',
        '<0004000100020003>',
    ),
    'cff-own-encoding': (_simple('Type1', '', 32, ('FontFile3', '/Subtype /Type1C', CFF)), 'ABCD', 'BCDA'),
    'cff-differences': (
        _simple('Type1', '/Encoding << /Differences [97 /A /B /C /D] >>', 32, ('FontFile3', '/Subtype /Type1C', CFF)),
        'abcd',
        'bcda',
    ),
    'cff-cid-keyed-charset': (
        _composite('CIDFontType0', 'Identity-H', ('FontFile3', '/Subtype /CIDFontType0C', CID_CFF), ''),
        '<000A0014001E0028>',
        '<0014001E0028000A>',
    ),
    'opentype': (
        _simple(
            'Type1', '/Encoding /WinAnsiEncoding', 32, ('FontFile3', '/Subtype /OpenType', fonts.opentype(CFF, LATIN))
        ),
        'ABCD',
        'BCDA',
    ),
}


def _type1_case() -> tuple[dict, str, str]:
    program, *lengths = fonts.type1()
    length = ' '.join(f'/Length{i + 1} {n}' for i, n in enumerate(lengths))
    return _simple('Type1', '', 32, ('FontFile', length, program)), 'ABCD', 'BCDA'


CASES['type1-eexec'] = _type1_case()

#: A Type 3 font of the four glyphs, each a procedure in a 1000-unit glyph space; `C` paints its own color after `d0`.
TYPE3 = {
    'F1': (
        (
            '<< /Type /Font /Subtype /Type3 /FontBBox [0 0 700 700] /FontMatrix [0.001 0 0 0.001 0 0] '
            '/CharProcs << /A {A} /B {B} /C {C} /D {D} >> '
            '/Encoding << /Type /Encoding /Differences [65 /A /B /C /D] >> '
            '/FirstChar 65 /LastChar 68 /Widths [700 700 700 700] >>'
        ),
        None,
    ),
    'A': ('<< >>', b'700 0 100 0 600 700 d1 100 0 500 700 re f'),
    'B': ('<< >>', b'700 0 100 0 600 700 d1 100 0 m 350 700 l 600 0 l h f'),
    'C': ('<< >>', b'700 0 d0 1 0 0 rg 100 0 500 700 re 250 200 m 250 500 l 450 500 l 450 200 l h f'),
    'D': ('<< >>', b'700 0 100 0 600 700 d1 100 0 m 100 700 l 250 700 l 250 150 l 600 150 l 600 0 l h f'),
}
CASES['type3-procedures'] = (TYPE3, 'ABCD', 'BCDA')

#: Render modes, by the CFF font: stroked (1), filled and stroked (2), clipped to (7, a fill after it).
MODES = {
    'stroked': ('BT /F1 40 Tf 1 Tr 1.5 w 0 0 1 RG 20 80 Td ({0}) Tj ET', 'ABCD', 'BCDA'),
    'filled-and-stroked': ('BT /F1 40 Tf 2 Tr 2 w 1 0 0 RG 0 0.5 0 rg 20 80 Td ({0}) Tj ET', 'ABCD', 'BCDA'),
    'clipped-to': ('BT /F1 60 Tf 7 Tr 20 40 Td ({0}) Tj ET 0 0 1 rg 0 0 200 150 re f', 'ABCD', 'BCDA'),
}


def _vertical() -> tuple[dict, str, str]:
    objects = _composite(
        'CIDFontType2',
        'Identity-V',
        ('FontFile2', '', fonts.truetype({(3, 1): LATIN})),
        '/W2 [3 [-1000 350 900]] /DW2 [880 -1000]',
    )
    return objects, '<0001000200030004>', '<0002000300040001>'


CASES['vertical-position-vectors'] = _vertical()


def _content(name: str, text: str) -> str:
    if name in MODES:
        return MODES[name][0].format(text)
    if name == 'vertical-position-vectors':
        return f'BT /F1 24 Tf 60 130 Td {text} Tj ET BT /F1 11 Tf 140 130 Td {text} Tj ET'
    return TEXT.format(text) if not text.startswith('<') else TEXT.replace('({0})', '{0}').format(text)


def _objects(name: str) -> dict:
    return _simple('Type1', '', 32, ('FontFile3', '/Subtype /Type1C', CFF)) if name in MODES else CASES[name][0]


def _texts(name: str) -> tuple[str, str]:
    return (MODES[name][1], MODES[name][2]) if name in MODES else (CASES[name][1], CASES[name][2])


NAMES = [*CASES, *MODES]


def _case(name: str, *, wrong: bool):
    text, rotated = _texts(name)
    truth = page_pdf(_content(name, text), _objects(name), resources=FONT)
    drawn = page_pdf(_content(name, rotated), _objects(name), resources=FONT) if wrong else truth
    pixels, undrawn = own(drawn, DPI)
    return pixels, reference(truth, DPI), undrawn


@pytest.mark.parametrize('name', NAMES)
def test_the_own_renderer_draws_text_as_the_reference_does(name):
    pixels, truth, undrawn = _case(name, wrong=False)
    assert undrawn == {}
    assert (truth.min(axis=2) < 128).mean() > 0.003, f'{name}: the reference draws (almost) nothing'
    mean, share = disagreement(pixels, truth)
    assert agrees(mean, share), f'{name}: mean {mean:.3f}, {share:.4%} of the inked pixels off'


@pytest.mark.parametrize('name', NAMES)
def test_a_planted_wrong_text_render_is_reported(name):
    pixels, truth, _ = _case(name, wrong=True)
    mean, share = disagreement(pixels, truth)
    assert not agrees(mean, share), name


#: The standard fonts, as PyMuPDF writes them (not embedded), and the Foxit substitute each is drawn with.
STANDARD = {
    'helv': 'FoxitSans',
    'hebo': 'FoxitSansBold',
    'tiro': 'FoxitSerif',
    'tiit': 'FoxitSerifItalic',
    'cour': 'FoxitFixed',
    'symb': 'FoxitSymbol',
    'zadb': 'FoxitDingbats',
}
#: Codes of Symbol's and Zapf Dingbats' built-in encodings (ISO 32000-2 Annex D.5, D.6) and the glyphs they name.
BUILT_IN = {
    'symb': dict(
        zip(
            b'abgdeABGD123+-= ',
            [
                'alpha',
                'beta',
                'gamma',
                'delta',
                'epsilon',
                'Alpha',
                'Beta',
                'Gamma',
                'Delta',
                'one',
                'two',
                'three',
                'plus',
                'minus',
                'equal',
                'space',
            ],
            strict=True,
        )
    ),
    'zadb': dict(zip(b'!"#$%& ', ['a1', 'a2', 'a202', 'a3', 'a4', 'a5', 'space'], strict=True)),
}
#: The standard names of the symbol fonts.
SYMBOL_NAMES = {'symb': 'Symbol', 'zadb': 'ZapfDingbats'}


def _standard_page(short: str, *, shifted: bool = False) -> bytes:
    """A page of text in the standard font ``short``, not embedded: PyMuPDF's for a Latin one (WinAnsi codes); for
    a symbol font, its built-in codes in a font naming no encoding (PyMuPDF writes other codes than the standard's)."""
    if short in BUILT_IN:
        codes = bytes(BUILT_IN[short]).decode('latin-1')
        content = f'BT /F1 11 Tf {10 + 2 * shifted} 80 Td ({codes}) Tj ET BT /F1 26 Tf 10 30 Td ({codes}) Tj ET'
        font = f'<< /Type /Font /Subtype /Type1 /BaseFont /{SYMBOL_NAMES[short]} >>'
        return page_pdf(content, {'F1': (font, None)}, resources=FONT)
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=120)
    text = 'Hamburgefonstiv 0123456789 (AaBbCc)'
    page.insert_text((10 + 2 * shifted, 40), text, fontsize=11, fontname=short)
    page.insert_text((10, 90), text[:14], fontsize=26, fontname=short)
    return doc.tobytes()


def _embedded_substitute(data: bytes, short: str) -> bytes:
    """``data`` with its standard font embedding the substitute the own renderer draws it with, under the same
    name and widths, its codes naming the glyphs the standard says they show (WinAnsi's; Symbol's and Zapf
    Dingbats' built-in ones', by ``/Differences``): PyMuPDF then draws the very glyphs the own renderer should."""
    doc = pymupdf.open(stream=data, filetype='pdf')
    program = (SUBSTITUTES / f'{STANDARD[short]}.cff').read_bytes()
    substitute = pymupdf.Font(fontbuffer=program)
    for xref, _ext, _kind, base, _name, _enc in doc.get_page_fonts(0):
        stream = doc.get_new_xref()
        doc.update_object(stream, '<< /Subtype /Type1C >>')
        doc.update_stream(stream, program, compress=False)
        descriptor = doc.get_new_xref()
        symbolic = short in BUILT_IN
        doc.update_object(
            descriptor, DESCRIPTOR.format(name=base, flags=4 if symbolic else 32, file=f'/FontFile3 {stream} 0 R')
        )
        if symbolic:
            differences = ' '.join(f'{code} /{name}' for code, name in sorted(BUILT_IN[short].items()))
            doc.xref_set_key(xref, 'Encoding', f'<< /Type /Encoding /Differences [{differences}] >>')
        else:
            widths = ' '.join(
                str(round(substitute.glyph_advance(ord(bytes([c]).decode('cp1252', errors='replace'))) * 1000))
                for c in range(32, 256)
            )
            doc.xref_set_key(xref, 'FirstChar', '32')
            doc.xref_set_key(xref, 'LastChar', '255')
            doc.xref_set_key(xref, 'Widths', f'[{widths}]')
        doc.xref_set_key(xref, 'FontDescriptor', f'{descriptor} 0 R')
    return doc.tobytes()


@pytest.mark.parametrize('short', STANDARD)
def test_a_standard_font_not_embedded_is_drawn_as_its_substitute_embedded_is(short):
    data = _standard_page(short)
    pixels, undrawn = own(data, DPI)
    assert undrawn == {}
    truth = reference(_embedded_substitute(data, short), DPI)
    mean, share = disagreement(pixels, truth)
    assert agrees(mean, share), f'{short}: mean {mean:.3f}, {share:.4%} of the inked pixels off'
    wrong, _ = own(_standard_page(short, shifted=True), DPI)
    assert not agrees(*disagreement(wrong, truth)), f'{short}: a line moved 2 points passed'


#: Text in a CJK font the file does not embed: two lines of characters of Adobe-GB1, by their UCS-2 codes.
CJK_TEXT = 'BT /F1 28 Tf 10 100 Td <4E2D658775284F8B> Tj ET BT /F1 11 Tf {x} 40 Td <78C1602796F64EF6002C0041> Tj ET'


def _cjk_page(flags: int, *, shifted: bool = False) -> bytes:
    """A page of GB1 text in a composite font that embeds no program, its descriptor's flags ``flags``."""
    objects = {
        'F1': (
            '<< /Type /Font /Subtype /Type0 /BaseFont /FZSSK--GBK1-0 /Encoding /UniGB-UCS2-H /DescendantFonts [{C}] >>',
            None,
        ),
        'C': (
            (
                '<< /Type /Font /Subtype /CIDFontType0 /BaseFont /FZSSK--GBK1-0 /CIDSystemInfo << /Registry (Adobe) '
                '/Ordering (GB1) /Supplement 2 >> /FontDescriptor {D} /DW 1000 >>'
            ),
            None,
        ),
        'D': (DESCRIPTOR.format(name='FZSSK--GBK1-0', flags=flags, file=''), None),
    }
    return page_pdf(CJK_TEXT.format(x=10 + 2 * shifted), objects, resources=FONT)


@pytest.mark.parametrize(('flags', 'fallback'), [(4, 'noto-sans-sc'), (6, 'noto-serif-sc')])
def test_a_cjk_font_not_embedded_is_drawn_from_its_fallback_as_mupdf_draws_it(flags, fallback):
    """The fallback (sans or serif by the descriptor's flags, as MuPDF chooses) is fetched once, by name. PyMuPDF
    draws such a font with its own Droid Sans Fallback: served as the fallback here, the very program is drawn on
    both sides, so where each glyph is placed and which one, through its CID's character, are compared, not the
    design; a line moved 2 points is reported."""
    droid = bytes(pymupdf.Font('cjk').buffer)
    asked = []

    def served(name: str) -> bytes:
        asked.append(name)
        return droid

    data = _cjk_page(flags)
    pixels, undrawn = own(data, DPI, fonts=served)
    assert (undrawn, asked) == ({}, [fallback])
    truth = reference(data, DPI)
    mean, share = disagreement(pixels, truth)
    assert agrees(mean, share), f'mean {mean:.3f}, {share:.4%} of the inked pixels off'
    wrong, _ = own(_cjk_page(flags, shifted=True), DPI, fonts=served)
    assert not agrees(*disagreement(wrong, truth)), 'a line moved 2 points passed'


def test_a_fallback_that_cannot_be_had_leaves_its_glyphs_undrawn_by_name():
    def offline(name: str) -> bytes:
        msg = f'{name} cannot be had: offline'
        raise RuntimeError(msg)

    _, undrawn = own(_cjk_page(4), DPI, fonts=offline)
    assert undrawn == {'font FZSSK--GBK1-0 is not embedded and its fallback noto-sans-sc cannot be had: offline': 10}
