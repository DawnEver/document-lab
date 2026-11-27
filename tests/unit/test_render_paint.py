"""The own renderer's shadings, patterns, soft masks and transparency groups (stage 3) accepted against PyMuPDF's
renders on synthetic pages (``tests/rendering.py`` says how), each case paired with a planted wrong render -- the same
page drawn a little wrong -- that the comparison must report.

Kept to the standard where MuPDF differs, so no case holds the difference: a radial shading's circles are exact,
where MuPDF draws them as 32-sided polygons (its edge is off by up to half a pixel at a radius of 125 pixels), so
the radial cases extend past their outer circle or clip inside it; a patch's edges are its curves, where MuPDF cuts
a patch into a few straight pieces, so the patch cases are clipped inside their edges; a tiling cell is clipped to
its ``/BBox`` (8.7.3.1), where MuPDF clips a turned cell only to the pixels around it, so the turned case's cell
stays inside its box; a group blends in sRGB whatever its ``/CS``. The stencil case's samples have their edges on
pixel edges: where a pixel's center falls on a stencil sample's edge, the own renderer and MuPDF take different
samples whatever the stencil paints (stage 1's image placement, not stage 3's painting).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tests.rendering import agrees, disagreement, own, page_pdf, reference

DPI = 150

RED_BLUE = '<< /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >>'
STITCHED = (
    '<< /FunctionType 3 /Domain [0 1] /Bounds [0.4] /Encode [0 1 0 1] /Functions ['
    '<< /FunctionType 2 /Domain [0 1] /C0 [1 1 0] /C1 [0 0.6 0.3] /N 1 >> '
    '<< /FunctionType 2 /Domain [0 1] /C0 [0 0.6 0.3] /C1 [0.4 0 0.8] /N 2 >>] >>'
)


def _axial(coords: str, extend: str = '[true true]', function: str = RED_BLUE) -> str:
    return f'<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [{coords}] /Function {function} /Extend {extend} >>'


def _radial(coords: str, extend: str = '[true true]') -> str:
    return f'<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [{coords}] /Function {STITCHED} /Extend {extend} >>'


def _mesh(entries: str, data: bytes) -> tuple[str, bytes]:
    return f'<< /ColorSpace /DeviceRGB /BitsPerCoordinate 16 /BitsPerComponent 8 {entries} >>', data


def _vertex(x: float, y: float, *values: float, flag: int | None = 0) -> bytes:
    """A mesh vertex: its flag (8 bits, none for a lattice), x over 0..200 and y over 0..150 in 16 bits, values in
    8."""
    head = b'' if flag is None else bytes([flag])
    return (
        head
        + struct.pack('>HH', round(x / 200 * 65535), round(y / 150 * 65535))
        + bytes(round(v * 255) for v in values)
    )


def _patch(points: list[tuple[float, float]], colors: list[tuple[float, ...]]) -> bytes:
    """A flag-0 patch: its points (12 for Coons, 16 for a tensor) and four corners' colors."""
    out = bytes([0])
    for x, y in points:
        out += struct.pack('>HH', round(x / 200 * 65535), round(y / 150 * 65535))
    for color in colors:
        out += bytes(round(v * 255) for v in color)
    return out


MESH_DECODE = '/Decode [0 200 0 150 0 1 0 1 0 1]'
TRIANGLES = (
    _vertex(20, 20, 1, 0, 0)
    + _vertex(180, 30, 0, 1, 0)
    + _vertex(60, 130, 0, 0, 1)
    + _vertex(170, 140, 1, 1, 0, flag=1)
    + _vertex(110, 145, 0, 1, 1, flag=2)
)
LATTICE = b''.join(
    _vertex(20 + 80 * i, 20 + 55 * j + 10 * i, (i + j) / 4, i / 2, 1 - j / 2, flag=None)
    for j in range(3)
    for i in range(3)
)
COONS = [
    (20, 20),
    (60, 0),
    (100, 40),
    (180, 20),
    (190, 60),
    (160, 100),
    (180, 130),
    (120, 150),
    (60, 110),
    (20, 130),
    (40, 90),
    (10, 50),
]
TENSOR = [*COONS, (70, 60), (120, 50), (130, 100), (70, 100)]
CORNERS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
TILE = (
    '<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 12 12] /XStep {step} /YStep 12 /Matrix {matrix} '
    '/Resources << >> >>'
)
TILE_CELL = b'1 0 0 rg 0 0 6 6 re f 0 0 1 RG 1 w 6 6 m 12 12 l S'
#: A cell whose marks stay inside its box: turned, it is the same clipped to its box or not (see the module).
INSIDE_CELL = b'1 0 0 rg 0 0 6 6 re f 0 0 1 RG 1 w 6.5 6.5 m 11 11 l S'
UNCOLORED = '<< /PatternType 1 /PaintType 2 /TilingType 2 /BBox [0 0 8 8] /XStep 8 /YStep 8 /Resources << >> >>'
UNCOLORED_CELL = b'0 0 m 8 8 l 0 8 l h f'
BIG_TILE = (
    '<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [-10 -10 50 50] /XStep 70 /YStep 60 '
    '/Matrix [0.8 0.6 -0.6 0.8 40 10] /Resources << >> >>'
)
BIG_CELL = b'0 0.5 0 rg 0 0 40 40 re f 1 0.5 0 rg -10 -10 m 50 20 l 20 50 l h f'
BACKDROP = '0.9 0.6 0.2 rg 0 0 100 150 re f 0.1 0.4 0.8 rg 100 0 100 150 re f 0.5 g 50 40 100 70 re f'


def _form(group: str, resources: str = '') -> str:
    """A Form over the page that is a transparency group with ``group`` in its group dictionary."""
    return (
        f'<< /Type /XObject /Subtype /Form /BBox [0 0 200 150] /Group << /S /Transparency {group} >> '
        f'/Resources << {resources} >> >>'
    )


GROUP_MARKS = b'1 0 0 rg 20 30 90 70 re f 0 0.8 0.2 rg 70 50 90 70 re f'
MASK_GROUP = (
    f'<< /Type /XObject /Subtype /Form /BBox [0 0 200 150] /Group << /S /Transparency /CS /DeviceRGB >> '
    f'/Resources << /Shading << /Sh {_axial("30 0 170 0", "[false false]")} >> >> >>'
)
MASK_CONTENT = b'1 g 10 10 80 130 re f /Sh sh'
ALPHA_MASK = (
    '<< /Type /XObject /Subtype /Form /BBox [0 0 200 150] /Group << /S /Transparency >> '
    '/Resources << /ExtGState << /H << /ca 0.4 >> >> >> >>'
)
ALPHA_CONTENT = b'0 g 10 10 120 100 re f /H gs 80 60 110 80 re f'
STAR = '100 140 m 130 15 l 20 95 l 180 95 l 70 15 l h'

#: name: (content, objects, resources, the planted wrong content, its objects or None for the same, its resources
#: or None for the same).
CASES = {
    'axial': (
        'q 20 20 160 110 re W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_axial("40 0 160 40")} >>',
        'q 20 20 160 110 re W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_axial("40 0 170 40")} >>',
    ),
    'axial-not-extended': (
        'q 10 10 180 130 re W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_axial("50 0 150 0", "[false false]", STITCHED)} >>',
        'q 10 10 180 130 re W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_axial("50 0 150 0", "[true false]", STITCHED)} >>',
    ),
    'radial': (
        'q 15 15 170 120 re W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_radial("70 60 10 120 80 70")} >>',
        'q 15 15 170 120 re W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_radial("70 60 10 120 80 60")} >>',
    ),
    'radial-inside': (
        f'q {STAR} W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_radial("100 75 0 100 75 90", "[false false]")} >>',
        f'q {STAR} W n /Sh sh Q',
        None,
        f'/Shading << /Sh {_radial("100 80 0 100 75 90", "[false false]")} >>',
    ),
    'function-based': (
        '/Sh sh',
        {'F': ('<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1] >>', b'{ 0 exch }')},
        (
            '/Shading << /Sh << /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] /Matrix [160 0 0 110 20 20] '
            '/Function {F} >> >>'
        ),
        '/Sh sh',
        None,
        (
            '/Shading << /Sh << /ShadingType 1 /ColorSpace /DeviceRGB /Domain [0 1 0 1] /Matrix [110 0 0 160 20 20] '
            '/Function {F} >> >>'
        ),
    ),
    'free-form-mesh': (
        '/Sh sh',
        {'M': _mesh(f'/ShadingType 4 /BitsPerFlag 8 {MESH_DECODE}', TRIANGLES)},
        '/Shading << /Sh {M} >>',
        '/Sh sh',
        {'M': _mesh(f'/ShadingType 4 /BitsPerFlag 8 {MESH_DECODE}', TRIANGLES[:-5] + b'\x00\x00\x00\xff\xff')},
        None,
    ),
    'mesh-through-a-function': (
        '/Sh sh',
        {
            'M': (
                (
                    f'<< /ShadingType 4 /ColorSpace /DeviceRGB /BitsPerCoordinate 16 /BitsPerComponent 8 '
                    f'/BitsPerFlag 8 /Decode [0 200 0 150 0 1] /Function {STITCHED} >>'
                ),
                _vertex(20, 20, 0) + _vertex(180, 40, 1) + _vertex(90, 140, 0.5),
            )
        },
        '/Shading << /Sh {M} >>',
        '/Sh sh',
        {
            'M': (
                (
                    f'<< /ShadingType 4 /ColorSpace /DeviceRGB /BitsPerCoordinate 16 /BitsPerComponent 8 '
                    f'/BitsPerFlag 8 /Decode [0 200 0 150 0 1] /Function {STITCHED} >>'
                ),
                _vertex(20, 20, 0) + _vertex(180, 40, 0.5) + _vertex(90, 140, 1),
            )
        },
        None,
    ),
    'lattice-mesh': (
        '/Sh sh',
        {'M': _mesh(f'/ShadingType 5 /VerticesPerRow 3 {MESH_DECODE}', LATTICE)},
        '/Shading << /Sh {M} >>',
        '/Sh sh',
        {'M': _mesh(f'/ShadingType 5 /VerticesPerRow 3 {MESH_DECODE}', LATTICE[:-7] + bytes(7))},
        None,
    ),
    'coons-patch': (
        'q 30 30 140 90 re W n /Sh sh Q',
        {'M': _mesh(f'/ShadingType 6 /BitsPerFlag 8 {MESH_DECODE}', _patch(COONS, CORNERS))},
        '/Shading << /Sh {M} >>',
        'q 30 30 140 90 re W n /Sh sh Q',
        {'M': _mesh(f'/ShadingType 6 /BitsPerFlag 8 {MESH_DECODE}', _patch(COONS[6:] + COONS[:6], CORNERS))},
        None,
    ),
    'tensor-patch': (
        'q 30 30 140 90 re W n /Sh sh Q',
        {'M': _mesh(f'/ShadingType 7 /BitsPerFlag 8 {MESH_DECODE}', _patch(TENSOR, CORNERS))},
        '/Shading << /Sh {M} >>',
        'q 30 30 140 90 re W n /Sh sh Q',
        {
            'M': _mesh(
                f'/ShadingType 7 /BitsPerFlag 8 {MESH_DECODE}',
                _patch([*COONS, (20, 140), (180, 140), (180, 10), (20, 10)], CORNERS),
            )
        },
        None,
    ),
    'shading-pattern-fill': (
        f'/Pattern cs /P scn {STAR} f',
        None,
        f'/Pattern << /P << /PatternType 2 /Matrix [1 0 0 1 20 10] /Shading {_axial("0 0 120 100")} >> >>',
        f'/Pattern cs /P scn {STAR} f',
        None,
        f'/Pattern << /P << /PatternType 2 /Matrix [1 0 0 1 0 10] /Shading {_axial("0 0 120 100")} >> >>',
    ),
    'shading-pattern-stroke-in-a-form': (
        'q 0.5 0 0 0.5 50 40 cm /Fm1 Do Q',
        {
            'Fm1': (
                '<< /Type /XObject /Subtype /Form /BBox [0 0 200 150] /Resources << /Pattern << /P {P} >> >> >>',
                b'/Pattern CS /P SCN 12 w 20 20 m 180 130 l 180 20 l S',
            ),
            'P': (f'<< /PatternType 2 /Shading {_axial("0 0 200 0")} >>', None),
        },
        '',
        'q 0.5 0 0 0.5 50 40 cm /Fm1 Do Q',
        {
            'Fm1': (
                '<< /Type /XObject /Subtype /Form /BBox [0 0 200 150] /Resources << /Pattern << /P {P} >> >> >>',
                b'/Pattern CS /P SCN 12 w 20 20 m 180 130 l 180 20 l S',
            ),
            'P': (f'<< /PatternType 2 /Shading {_axial("200 0 0 0")} >>', None),
        },
        None,
    ),
    'tiling-pattern': (
        '/Pattern cs /P scn 10 10 180 130 re f',
        {'T': (TILE.format(step=12, matrix='[1 0 0 1 3 2]'), TILE_CELL)},
        '/Pattern << /P {T} >>',
        '/Pattern cs /P scn 10 10 180 130 re f',
        {'T': (TILE.format(step=13, matrix='[1 0 0 1 3 2]'), TILE_CELL)},
        None,
    ),
    'tiling-pattern-turned': (
        f'/Pattern cs /P scn {STAR} f',
        {'T': (TILE.format(step=12, matrix='[0.93 0.41 -0.41 0.93 0.7 0.3]'), INSIDE_CELL)},
        '/Pattern << /P {T} >>',
        f'/Pattern cs /P scn {STAR} f',
        {'T': (TILE.format(step=12, matrix='[0.93 0.45 -0.45 0.93 0.7 0.3]'), INSIDE_CELL)},
        None,
    ),
    'tiling-pattern-few-cells': (
        '/Pattern cs /P scn 5 5 190 140 re f',
        {'T': (BIG_TILE, BIG_CELL)},
        '/Pattern << /P {T} >>',
        '/Pattern cs /P scn 5 5 190 140 re f',
        {'T': (BIG_TILE.replace('/XStep 70', '/XStep 75'), BIG_CELL)},
        None,
    ),
    'uncolored-tiling-pattern': (
        '/C cs 0.2 0.5 0.9 /P scn 10 10 180 130 re f',
        {'T': (UNCOLORED, UNCOLORED_CELL)},
        '/Pattern << /P {T} >> /ColorSpace << /C [/Pattern /DeviceRGB] >>',
        '/C cs 0.9 0.5 0.2 /P scn 10 10 180 130 re f',
        {'T': (UNCOLORED, UNCOLORED_CELL)},
        None,
    ),
    'stencil-mask-painting-a-pattern': (
        '/Pattern cs /P scn q 153.6 0 0 115.2 19.2 14.64 cm /Im1 Do Q',
        {
            'Im1': (
                '<< /Type /XObject /Subtype /Image /Width 8 /Height 6 /ImageMask true >>',
                bytes([0b10101010, 0b01010101] * 3),
            ),
            'P': (f'<< /PatternType 2 /Shading {_axial("20 0 180 0")} >>', None),
        },
        '/Pattern << /P {P} >>',
        '/Pattern cs /P scn q 153.6 0 0 115.2 19.2 14.64 cm /Im1 Do Q',
        {
            'Im1': (
                '<< /Type /XObject /Subtype /Image /Width 8 /Height 6 /ImageMask true /Decode [1 0] >>',
                bytes([0b10101010, 0b01010101] * 3),
            ),
            'P': (f'<< /PatternType 2 /Shading {_axial("20 0 180 0")} >>', None),
        },
        None,
    ),
    'luminosity-soft-mask': (
        f'{BACKDROP} /M gs 0 0.3 0 rg 0 0 200 150 re f',
        {'G': (MASK_GROUP, MASK_CONTENT)},
        '/ExtGState << /M << /SMask << /S /Luminosity /G {G} >> >> >>',
        f'{BACKDROP} /M gs 0 0.3 0 rg 0 0 200 150 re f',
        {'G': (MASK_GROUP, MASK_CONTENT.replace(b'1 g', b'0.5 g'))},
        None,
    ),
    'soft-mask-backdrop-and-transfer': (
        f'{BACKDROP} /M gs 0.6 0 0.6 rg 0 0 200 150 re f',
        {'G': (MASK_GROUP.replace('/CS /DeviceRGB', '/CS /DeviceGray'), b'0 g 10 10 80 130 re f')},
        (
            '/ExtGState << /M << /SMask << /S /Luminosity /G {G} /BC [0.8] '
            '/TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1.5 >> >> >> >>'
        ),
        f'{BACKDROP} /M gs 0.6 0 0.6 rg 0 0 200 150 re f',
        {'G': (MASK_GROUP.replace('/CS /DeviceRGB', '/CS /DeviceGray'), b'0 g 10 10 80 130 re f')},
        '/ExtGState << /M << /SMask << /S /Luminosity /G {G} /BC [0.8] >> >> >>',
    ),
    'alpha-soft-mask-on-an-image': (
        f'{BACKDROP} /M gs q 160 0 0 120 20 15 cm /Im1 Do Q',
        {
            'G': (ALPHA_MASK, ALPHA_CONTENT),
            'Im1': (
                '<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8 >>',
                bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]),
            ),
        },
        '/ExtGState << /M << /SMask << /S /Alpha /G {G} >> >> >>',
        f'{BACKDROP} /M gs q 160 0 0 120 20 15 cm /Im1 Do Q',
        {
            'G': (ALPHA_MASK.replace('/ca 0.4', '/ca 0.8'), ALPHA_CONTENT),
            'Im1': (
                '<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8 >>',
                bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]),
            ),
        },
        None,
    ),
    'group-alpha': (
        f'{BACKDROP} /H gs /Fm1 Do',
        {'Fm1': (_form(''), GROUP_MARKS)},
        '/ExtGState << /H << /ca 0.5 >> >>',
        f'{BACKDROP} /H gs {GROUP_MARKS.decode()}',
        None,
        None,
    ),
    'group-soft-mask': (
        f'{BACKDROP} /M gs /Fm1 Do',
        {'Fm1': (_form(''), GROUP_MARKS), 'G': (MASK_GROUP, MASK_CONTENT)},
        '/ExtGState << /M << /SMask << /S /Luminosity /G {G} >> >> >>',
        f'{BACKDROP} /Fm1 Do',
        None,
        None,
    ),
    'isolated-group': (
        f'{BACKDROP} /Fm1 Do',
        {'Fm1': (_form('/I true', '/ExtGState << /M << /BM /Multiply >> >>'), b'/M gs ' + GROUP_MARKS)},
        '',
        f'{BACKDROP} /Fm1 Do',
        {'Fm1': (_form('', '/ExtGState << /M << /BM /Multiply >> >>'), b'/M gs ' + GROUP_MARKS)},
        None,
    ),
    'knockout-group': (
        f'{BACKDROP} /Fm1 Do',
        {'Fm1': (_form('/K true', '/ExtGState << /H << /ca 0.6 >> >>'), b'/H gs ' + GROUP_MARKS)},
        '',
        f'{BACKDROP} /Fm1 Do',
        {'Fm1': (_form('', '/ExtGState << /H << /ca 0.6 >> >>'), b'/H gs ' + GROUP_MARKS)},
        None,
    ),
    'isolated-knockout-group-with-alpha': (
        f'{BACKDROP} /H gs /Fm1 Do',
        {'Fm1': (_form('/I true /K true', '/ExtGState << /H << /ca 0.6 >> >>'), b'/H gs ' + GROUP_MARKS)},
        '/ExtGState << /H << /ca 0.7 >> >>',
        f'{BACKDROP} /H gs {GROUP_MARKS.decode()}',
        None,
        None,
    ),
}

#: The blend modes (11.3.5), each a group composited by it onto the backdrop; the planted wrong render uses the
#: next.
BLENDS = [
    'Normal',
    'Multiply',
    'Screen',
    'Overlay',
    'Darken',
    'Lighten',
    'ColorDodge',
    'ColorBurn',
    'HardLight',
    'SoftLight',
    'Difference',
    'Exclusion',
    'Hue',
    'Saturation',
    'Color',
    'Luminosity',
]
BLEND_MARKS = b'0.9 0.2 0.3 rg 20 30 90 70 re f 0.2 0.7 0.4 rg 70 50 90 70 re f 0.95 0.9 0.1 rg 40 10 40 130 re f'


def _blended(mode: str) -> tuple[str, dict, str]:
    return (
        f'{BACKDROP} /B gs /Fm1 Do',
        {'Fm1': (_form(''), BLEND_MARKS)},
        f'/ExtGState << /B << /BM /{mode} /ca 0.85 >> >>',
    )


for _i, _mode in enumerate(BLENDS):
    _wrong = BLENDS[(_i + 1) % len(BLENDS)]
    CASES[f'blend-{_mode}'] = (*_blended(_mode), *_blended(_wrong))


def _case(name: str, *, wrong: bool) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    content, objects, resources, wrong_content, wrong_objects, wrong_resources = CASES[name]
    truth = page_pdf(content, objects, resources=resources)
    drawn = (
        page_pdf(
            wrong_content, wrong_objects or objects, resources=resources if wrong_resources is None else wrong_resources
        )
        if wrong
        else truth
    )
    pixels, undrawn = own(drawn, DPI)
    return pixels, reference(truth, DPI), undrawn


@pytest.mark.parametrize('name', CASES)
def test_the_own_renderer_paints_as_the_reference_does(name):
    pixels, truth, undrawn = _case(name, wrong=False)
    assert undrawn == {}
    mean, share = disagreement(pixels, truth)
    assert agrees(mean, share), f'{name}: mean {mean:.3f}, {share:.4%} of the inked pixels off'


@pytest.mark.parametrize('name', CASES)
def test_a_planted_wrong_paint_is_reported(name):
    pixels, truth, _ = _case(name, wrong=True)
    mean, share = disagreement(pixels, truth)
    assert not agrees(mean, share), name
