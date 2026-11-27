//! The renderer on synthetic pages whose every pixel is known: geometry, clips, strokes, compositing, images.

#[path = "../../pdf/tests/common/mod.rs"]
mod common;

use pdf::File;
use pdf::geometry::Rect;
use render::{Options, Raster, render};

/// The page (200 x 100 points) of `content` with `page_extra` in its dictionary and `extra` objects (5, 6, ...),
/// drawn at `scale` over `region`.
fn drawn(
    page_extra: &str,
    content: &str,
    extra: &[&str],
    scale: f64,
    region: Option<Rect>,
) -> Raster {
    let file = File::open(common::one_page(page_extra, content, extra), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    let options = Options {
        region,
        scale,
        text: false,
        annotations: false,
        alpha: false,
    };
    render(&file, &page, &options).unwrap()
}

/// The page of `content` at one pixel a point.
fn simple(content: &str) -> Raster {
    drawn("", content, &[], 1.0, None)
}

/// Pixel `(x, y)`'s RGB.
fn at(r: &Raster, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * r.width + x) * u32::from(r.channels)) as usize;
    [r.data[i], r.data[i + 1], r.data[i + 2]]
}

const WHITE: [u8; 3] = [255; 3];
const BLACK: [u8; 3] = [0; 3];
const RED: [u8; 3] = [255, 0, 0];

/// The page of `content` with `page_extra` and `extra` objects, text drawn, at one pixel a point.
fn texted(page_extra: &str, content: &str, extra: &[&str]) -> Raster {
    let file = File::open(common::one_page(page_extra, content, extra), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    let options = Options {
        region: None,
        scale: 1.0,
        text: true,
        annotations: false,
        alpha: false,
    };
    render(&file, &page, &options).unwrap()
}

/// A Type 3 font (object 5) of two glyphs, each a square across its 1000-unit glyph space: `a` (object 6) after
/// `d1`, setting red, which it may not; `b` (object 7) after `d0`, which keeps its red.
fn squares() -> [String; 3] {
    [
        "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] /FontMatrix [0.001 0 0 0.001 0 0] \
         /CharProcs << /a 6 0 R /b 7 0 R >> /Encoding << /Type /Encoding /Differences [97 /a /b] >> \
         /FirstChar 97 /LastChar 98 /Widths [1000 1000] >>"
            .to_string(),
        common::stream("", "1000 0 0 0 1000 1000 d1 1 0 0 rg 0 0 1000 1000 re f"),
        common::stream("", "1000 0 d0 1 0 0 rg 0 0 1000 1000 re f"),
    ]
}

const BLUE: [u8; 3] = [0, 0, 255];

#[test]
fn a_type3_glyph_is_drawn_by_its_procedure_in_the_text_color_after_d1() {
    let [font, a, b] = squares();
    let content = "BT /T3 40 Tf 0 0 1 rg 10 10 Td (a) Tj 60 0 Td (b) Tj 3 Tr 60 0 Td (a) Tj ET";
    let r = texted(
        "/Resources << /Font << /T3 5 0 R >> >>",
        content,
        &[&font, &a, &b],
    );
    // User (10, 10)-(50, 50) is page columns 10..50, rows 50..90.
    assert_eq!(at(&r, 10, 50), BLUE);
    assert_eq!(at(&r, 49, 89), BLUE);
    assert_eq!(at(&r, 9, 70), WHITE);
    assert_eq!(at(&r, 50, 70), WHITE);
    assert_eq!(at(&r, 90, 70), RED, "d0: the procedure's own color");
    assert_eq!(at(&r, 150, 70), WHITE, "render mode 3 paints nothing");
    assert!(r.undrawn.is_empty(), "{:?}", r.undrawn);
    let off = drawn(
        "/Resources << /Font << /T3 5 0 R >> >>",
        content,
        &[&font, &a, &b],
        1.0,
        None,
    );
    assert!(
        off.data.iter().all(|&v| v == 255),
        "text off draws no glyph"
    );
}

const HELVETICA: &str =
    "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>";

/// The columns of row `y` that are dark.
fn dark(r: &Raster, y: u32) -> Vec<u32> {
    (0..r.width).filter(|&x| at(r, x, y)[0] < 128).collect()
}

#[test]
fn a_glyph_is_its_outline_filled_stroked_or_clipped_to() {
    // Helvetica's I (its substitute's), 100 points tall from user y 10: one bar.
    let filled = texted(HELVETICA, "BT /F1 100 Tf 10 10 Td (I) Tj ET", &[]);
    let bar = dark(&filled, 50);
    let (left, right) = (bar[0], *bar.last().unwrap());
    assert!(
        right - left >= 6 && u32::try_from(bar.len()) == Ok(right - left + 1),
        "{bar:?}"
    );
    assert!(
        left > 10 && right < 10 + 28,
        "{bar:?}: within the I's advance"
    );
    assert!(filled.undrawn.is_empty(), "{:?}", filled.undrawn);
    // Stroked thin, the bar's edges ink and its middle does not.
    let stroked = texted(
        HELVETICA,
        "BT /F1 100 Tf 1 Tr 0.5 w 10 10 Td (I) Tj ET",
        &[],
    );
    let middle = u32::midpoint(left, right);
    assert_eq!(at(&stroked, middle, 50), WHITE);
    assert!(at(&stroked, left, 50)[0] < 200 && at(&stroked, right, 50)[0] < 200);
    // As a clip (mode 7), a fill after it shows inside the bar alone.
    let clipped = texted(
        HELVETICA,
        "BT /F1 100 Tf 7 Tr 10 10 Td (I) Tj ET 1 0 0 rg 0 0 200 100 re f",
        &[],
    );
    assert_eq!(at(&clipped, middle, 50), RED);
    assert_eq!(at(&clipped, left - 2, 50), WHITE);
    assert_eq!(at(&clipped, right + 2, 50), WHITE);
    assert!(clipped.undrawn.is_empty(), "{:?}", clipped.undrawn);
}

#[test]
fn a_glyph_whose_font_has_no_program_and_no_substitute_is_counted_by_its_font() {
    let font = "<< /Type /Font /Subtype /Type0 /BaseFont /MSMincho /Encoding /Identity-H /DescendantFonts [<< \
                /Type /Font /Subtype /CIDFontType2 /BaseFont /MSMincho /CIDSystemInfo << /Registry (Adobe) \
                /Ordering (Japan1) /Supplement 2 >> /FontDescriptor << /Type /FontDescriptor /FontName /MSMincho \
                /Flags 4 >> >>] >>";
    let r = texted(
        "/Resources << /Font << /F2 5 0 R >> >>",
        "BT /F2 20 Tf 10 10 Td <00010002> Tj ET",
        &[font],
    );
    assert!(r.data.iter().all(|&v| v == 255));
    assert_eq!(
        r.undrawn
            .get("font MSMincho is not embedded and has no substitute"),
        Some(&2),
        "{:?}",
        r.undrawn
    );
}

#[test]
fn a_rectangle_on_whole_pixels_fills_exactly_them_and_a_half_pixel_edge_half() {
    // User space is y up: 10..20 across, 50..60 up is page rows 40..50.
    let r = simple("1 0 0 rg 10 50 10 10 re f 0 0 1 rg 30.5 50 10 10 re f");
    assert_eq!((r.width, r.height, r.channels), (200, 100, 3));
    assert_eq!(at(&r, 10, 40), RED);
    assert_eq!(at(&r, 19, 49), RED);
    assert_eq!(at(&r, 9, 45), WHITE);
    assert_eq!(at(&r, 20, 45), WHITE);
    assert_eq!(at(&r, 10, 50), WHITE);
    let edge = at(&r, 30, 45);
    assert!((120..=135).contains(&edge[0]) && edge[2] == 255, "{edge:?}");
    assert!(r.undrawn.is_empty());
}

#[test]
fn a_region_is_drawn_on_the_whole_pages_pixels_there() {
    let content =
        "0 0 1 RG 3 w 1 J 12.3 7.7 m 150.2 88.1 l S 0.2 0.6 0.1 rg 40 20 m 90 95 120 5 160 60 c f";
    let whole = drawn("", content, &[], 2.0, None);
    let region = Rect {
        x0: 30.25,
        y0: 10.0,
        x1: 101.0,
        y1: 70.6,
    };
    let part = drawn("", content, &[], 2.0, Some(region));
    // The pixels the region touches: columns 60..202, rows 20..142.
    assert_eq!((part.width, part.height), (142, 122));
    let mut worst = 0;
    let mut off = 0;
    for y in 0..part.height {
        for x in 0..part.width {
            let (a, b) = (at(&part, x, y), at(&whole, x + 60, y + 20));
            let d = (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap();
            worst = worst.max(d);
            off += usize::from(d > 0);
        }
    }
    // Curve edges are anti-aliased as they are cut by the canvas's edge: a few of the 16 coverage steps apart.
    assert!(worst <= 64, "{worst}");
    assert!(off * 100 < (part.width * part.height) as usize, "{off}");
}

#[test]
fn a_rectangle_clip_takes_the_whole_pixels_it_touches_and_q_restores_the_clip() {
    let r =
        simple("q 10.5 50.5 10 10 re W n 1 0 0 rg 0 0 200 100 re f Q 0 0 1 rg 100 0 10 10 re f");
    assert_eq!(at(&r, 10, 39), RED, "the clip's half pixel is whole");
    assert_eq!(at(&r, 20, 49), RED);
    assert_eq!(at(&r, 9, 45), WHITE);
    assert_eq!(at(&r, 21, 45), WHITE);
    assert_eq!(
        at(&r, 105, 95),
        [0, 0, 255],
        "drawn after Q, outside the old clip"
    );
}

#[test]
fn an_even_odd_clip_leaves_its_hole_and_a_curved_clip_its_disc() {
    let r = simple(
        "q 0 0 100 100 re 25 25 50 50 re W* n 1 0 0 rg 0 0 100 100 re f Q \
         q 150 50 m 150 61 141 70 130 70 c 119 70 110 61 110 50 c 110 39 119 30 130 30 c 141 30 150 39 150 50 c W n \
         0 0 0 rg 100 0 100 100 re f Q",
    );
    assert_eq!(at(&r, 10, 10), RED);
    assert_eq!(at(&r, 50, 50), WHITE, "the hole");
    assert_eq!(at(&r, 130, 50), BLACK, "inside the disc");
    assert_eq!(at(&r, 112, 32), WHITE, "outside the disc, inside its box");
}

#[test]
fn a_form_draws_only_inside_its_bbox() {
    let form = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 10 10]",
        "1 0 0 rg -50 -50 200 200 re f",
    );
    let r = drawn(
        "/Resources << /XObject << /Fm1 5 0 R >> >>",
        "q 1 0 0 1 20 20 cm /Fm1 Do Q",
        &[&form],
        1.0,
        None,
    );
    assert_eq!(at(&r, 25, 75), RED);
    assert_eq!(at(&r, 35, 75), WHITE);
}

#[test]
fn caps_extend_a_line_by_half_its_width_and_butt_ends_it() {
    let r = simple("10 w 0 J 50 80 m 100 80 l S 2 J 50 50 m 100 50 l S 1 J 50 20 m 100 20 l S");
    assert_eq!(at(&r, 52, 20), BLACK);
    assert_eq!(at(&r, 47, 20), WHITE, "butt: no further than its end");
    assert_eq!(at(&r, 47, 50), BLACK, "square: half the width further");
    assert_eq!(at(&r, 45, 45), BLACK, "square corners");
    assert_eq!(at(&r, 45, 75), WHITE, "round: no corner");
    assert_eq!(at(&r, 47, 80), BLACK, "round: its end");
}

#[test]
fn a_dash_array_of_odd_count_repeats_and_its_phase_shifts_it() {
    let r = simple("2 w [10] 0 d 0 90 m 200 90 l S [10] 5 d 0 50 m 200 50 l S");
    assert_eq!(at(&r, 5, 10), BLACK);
    assert_eq!(at(&r, 15, 10), WHITE);
    assert_eq!(at(&r, 25, 10), BLACK);
    assert_eq!(at(&r, 2, 50), BLACK, "phase 5: the first dash's last half");
    assert_eq!(at(&r, 7, 50), WHITE);
}

#[test]
fn a_stroke_under_an_uneven_matrix_is_stroked_in_user_space() {
    // 2 units wide in user space, scaled 4 across and 1 down: a vertical line 8 points wide, a horizontal one 2.
    let r = simple("4 0 0 1 0 0 cm 2 w 10 10 m 10 90 l S 20 50 m 40 50 l S");
    assert_eq!(at(&r, 37, 50), BLACK);
    assert_eq!(at(&r, 43, 50), BLACK);
    assert_eq!(at(&r, 35, 50), WHITE);
    assert_eq!(at(&r, 100, 48), WHITE);
    assert_eq!(at(&r, 100, 50), BLACK);
}

#[test]
fn hidden_and_inkless_content_marks_nothing() {
    let r = drawn(
        "/Resources << /ColorSpace << /CS0 [/Separation /None /DeviceGray << /FunctionType 2 /Domain [0 1] /N 1 >>] >> >>",
        "/CS0 cs 1 scn 0 0 200 100 re f",
        &[],
        1.0,
        None,
    );
    assert_eq!(at(&r, 100, 50), WHITE);
}

#[test]
fn a_shading_and_a_pattern_are_counted_not_drawn() {
    let r = drawn(
        "/Resources << /Shading << /Sh1 << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 1 0] \
         /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >> >> \
         /Pattern << /P1 << /PatternType 2 /Shading << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 1 0] \
         /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >> >> >> >>",
        "q 0 0 50 50 re W n /Sh1 sh Q /Pattern cs /P1 scn 100 0 50 50 re f",
        &[],
        1.0,
        None,
    );
    assert_eq!(at(&r, 25, 75), WHITE);
    assert_eq!(at(&r, 125, 75), WHITE);
    assert_eq!(r.undrawn.get("shading (stage 3)"), Some(&1));
    assert_eq!(r.undrawn.get("pattern fill (stage 3)"), Some(&1));
}

#[test]
fn constant_alpha_and_a_blend_mode_composite_onto_the_backdrop() {
    let r = drawn(
        "/Resources << /ExtGState << /Half << /ca 0.5 >> /Mul << /BM /Multiply >> >> >>",
        "q /Half gs 1 0 0 rg 0 0 50 50 re f Q 0 0 1 rg 100 0 50 50 re f q /Mul gs 1 1 0 rg 100 0 50 50 re f Q",
        &[],
        1.0,
        None,
    );
    let half = at(&r, 25, 75);
    assert!(half[0] == 255 && (126..=129).contains(&half[1]), "{half:?}");
    assert_eq!(at(&r, 125, 75), BLACK, "yellow multiplied onto blue");
}

#[test]
fn a_soft_mask_is_counted_as_drawn_without_it() {
    let r = drawn(
        "/Resources << /ExtGState << /G1 << /SMask << /S /Luminosity /G 5 0 R >> >> >> >>",
        "/G1 gs 0 0 10 10 re f",
        &[&common::stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 1 1] /Group << /S /Transparency >>",
            "",
        )],
        1.0,
        None,
    );
    assert_eq!(at(&r, 5, 95), BLACK);
    assert_eq!(r.undrawn.len(), 1, "{:?}", r.undrawn);
}

/// A page placing image object 5 over user space (10, 10)-(10 + w, 10 + h), drawn at one pixel a point.
fn placed(image: &str, w: u32, h: u32, before: &str) -> Raster {
    drawn(
        "/Resources << /XObject << /Im1 5 0 R >> >>",
        &format!("{before} q {w} 0 0 {h} 10 10 cm /Im1 Do Q"),
        &[image],
        1.0,
        None,
    )
}

#[test]
fn an_image_drawn_larger_shows_each_sample_as_a_block() {
    let image = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[0, 255, 255, 0],
    );
    let r = placed(&image, 20, 20, "");
    // Its first row is the top: page rows 70..80.
    assert_eq!(at(&r, 10, 70), BLACK);
    assert_eq!(at(&r, 19, 79), BLACK);
    assert_eq!(at(&r, 20, 70), WHITE);
    assert_eq!(at(&r, 20, 80), BLACK);
}

#[test]
fn an_image_off_the_pixel_grid_is_widened_to_the_whole_pixels_it_touches() {
    let image = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[0],
    );
    let r = drawn(
        "/Resources << /XObject << /Im1 5 0 R >> >>",
        "q 9.4 0 0 9.4 10.3 10.3 cm /Im1 Do Q",
        &[&image],
        1.0,
        None,
    );
    assert_eq!(at(&r, 10, 80), BLACK);
    assert_eq!(at(&r, 19, 89), BLACK);
    assert_eq!(at(&r, 20, 85), WHITE);
}

#[test]
fn an_image_drawn_smaller_averages_its_samples() {
    let image = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 4 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[0, 255, 0, 255],
    );
    let r = placed(&image, 2, 1, "");
    assert!(
        (126..=129).contains(&at(&r, 10, 89)[0]),
        "{:?}",
        at(&r, 10, 89)
    );
}

#[test]
fn decode_inverts_and_bit_depths_spread_their_levels() {
    let inverted = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Decode [1 0]",
        &[0, 255],
    );
    let r = placed(&inverted, 2, 1, "");
    assert_eq!((at(&r, 10, 89), at(&r, 11, 89)), (WHITE, BLACK));
    let deep = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 16",
        &[0x80, 0x00, 0xff, 0xff],
    );
    let r = placed(&deep, 2, 1, "");
    assert_eq!((at(&r, 10, 89)[0], at(&r, 11, 89)), (128, WHITE));
    let two = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 4 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 2",
        &[0b0001_1011],
    );
    let r = placed(&two, 4, 1, "");
    let levels: Vec<u8> = (10..14).map(|x| at(&r, x, 89)[0]).collect();
    assert_eq!(levels, [0, 85, 170, 255]);
}

#[test]
fn an_indexed_image_looks_its_samples_up_past_the_table_at_its_last_entry() {
    let image = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 3 /Height 1 /BitsPerComponent 8 /ColorSpace [/Indexed /DeviceRGB 1 <ff000000ff00>]",
        &[0, 1, 7],
    );
    let r = placed(&image, 3, 1, "");
    assert_eq!(
        (at(&r, 10, 89), at(&r, 11, 89), at(&r, 12, 89)),
        (RED, [0, 255, 0], [0, 255, 0])
    );
}

#[test]
fn a_stencil_mask_paints_the_fill_color_where_its_samples_are_0() {
    let image = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 8 /Height 1 /ImageMask true",
        &[0b0101_0101],
    );
    let r = placed(&image, 8, 1, "1 0 0 rg");
    let row: Vec<[u8; 3]> = (10..14).map(|x| at(&r, x, 89)).collect();
    assert_eq!(row, [RED, WHITE, RED, WHITE]);
}

#[test]
fn a_soft_mask_a_stencil_mask_and_color_keys_give_samples_their_alpha() {
    let soft = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8",
        &[255, 0],
    );
    let image = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /SMask 6 0 R",
        &[0, 0],
    );
    let r = drawn(
        "/Resources << /XObject << /Im1 5 0 R >> >>",
        "q 2 0 0 1 10 10 cm /Im1 Do Q",
        &[&image, &soft],
        1.0,
        None,
    );
    assert_eq!((at(&r, 10, 89), at(&r, 11, 89)), (BLACK, WHITE));
    let keyed = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Mask [0 10]",
        &[5, 100],
    );
    let r = placed(&keyed, 2, 1, "");
    assert_eq!((at(&r, 10, 89), at(&r, 11, 89)), (WHITE, [100; 3]));
    let stencil = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ImageMask true",
        &[0b0100_0000],
    );
    let masked = common::hex_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Mask 6 0 R",
        &[0, 0],
    );
    let r = drawn(
        "/Resources << /XObject << /Im1 5 0 R >> >>",
        "q 2 0 0 1 10 10 cm /Im1 Do Q",
        &[&masked, &stencil],
        1.0,
        None,
    );
    assert_eq!(
        (at(&r, 10, 89), at(&r, 11, 89)),
        (BLACK, WHITE),
        "a mask sample of 1 masks"
    );
}

#[test]
fn an_inline_image_names_its_color_space_in_the_resources_in_force() {
    let r = drawn(
        "/Resources << /ColorSpace << /CS0 [/Indexed /DeviceRGB 1 <ff000000ff00>] >> >>",
        "q 2 0 0 1 10 10 cm BI /W 2 /H 1 /CS /CS0 /BPC 8 ID \x01\x00 EI Q",
        &[],
        1.0,
        None,
    );
    assert_eq!((at(&r, 10, 89), at(&r, 11, 89)), ([0, 255, 0], RED));
}

#[test]
fn a_fax_image_decodes_its_runs() {
    // Group 3, one-dimensional: a row of 8 white (10011), then white 2 (0111), black 4 (011), white 2 (0111).
    let params = "/DecodeParms [null << /K 0 /Columns 8 /Rows 2 >>]"; // spelling-ignore-line: the PDF name
    let image = common::hex_stream(
        &format!(
            "/Type /XObject /Subtype /Image /Width 8 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 1 \
             /Filter [/ASCIIHexDecode /CCITTFaxDecode] {params}"
        ),
        &[0x9B, 0xB7],
    );
    // hex_stream states its own /Filter: replace it with the chain above.
    let image = image.replacen(" /Filter /ASCIIHexDecode", "", 1);
    let r = placed(&image, 8, 2, "");
    let first: Vec<u8> = (10..18).map(|x| at(&r, x, 88)[0]).collect();
    let second: Vec<u8> = (10..18).map(|x| at(&r, x, 89)[0]).collect();
    assert_eq!(first, [255; 8]);
    assert_eq!(second, [255, 255, 0, 0, 0, 0, 255, 255]);
}

#[test]
fn an_image_in_a_codec_not_decoded_is_counted_by_its_name() {
    let image = common::stream(
        "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /JPXDecode",
        "x",
    );
    let r = placed(&image, 10, 10, "");
    assert_eq!(at(&r, 15, 85), WHITE);
    assert_eq!(
        r.undrawn.get("JPXDecode image (not decoded in stage 1)"),
        Some(&1)
    );
}

#[test]
fn a_line_of_width_0_or_under_a_tenth_of_a_pixel_is_one_pixel_wide() {
    // Centered on page row 49 (user y 50.5): that row inked, its neighbors not.
    let r = simple("0 w 10 50.5 m 90 50.5 l S 0.05 w 110 50.5 m 190 50.5 l S");
    for x in [50, 150] {
        assert!(at(&r, x, 49)[0] < 64, "{:?}", at(&r, x, 49));
        assert_eq!(at(&r, x, 47), WHITE);
        assert_eq!(at(&r, x, 51), WHITE);
    }
}
