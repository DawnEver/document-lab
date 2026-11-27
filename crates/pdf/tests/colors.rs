//! Colors in every space a page paints in, read to RGB: `Separation` and `DeviceN` through each type of tint
//! transform, `Lab`, `Indexed`, and the colorants `None` and `All`.

mod common;

use pdf::interpret::PathEvent;
use pdf::{Event, File};

/// The events of a page whose resources name the color space `/CS0` (object 5) and whose content is `content`;
/// `extra` objects follow as objects 6, 7, ...
fn painted(space: &str, content: &str, extra: &[&str]) -> pdf::Content {
    let mut objects = vec![space];
    objects.extend(extra);
    let pdf = common::one_page(
        "/Resources << /ColorSpace << /CS0 5 0 R >> >>",
        content,
        &objects,
    );
    let file = File::open(pdf, b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    file.interpret(&page, false).unwrap()
}

fn paths(content: &pdf::Content) -> Vec<&PathEvent> {
    content
        .events
        .iter()
        .filter_map(|e| {
            if let Event::Path(p) = e {
                Some(p)
            } else {
                None
            }
        })
        .collect()
}

/// Each fill's components and RGB, in painting order.
fn fills(content: &pdf::Content) -> Vec<(Vec<f64>, [f64; 3])> {
    paths(content)
        .iter()
        .map(|p| {
            let color = &p.fill.as_ref().unwrap().color;
            (color.components.clone(), color.rgb.unwrap())
        })
        .collect()
}

fn near(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
}

/// A fill of the unit square in each of `tints` (`sc` operands), in `/CS0`.
fn squares(tints: &[&str]) -> String {
    tints
        .iter()
        .flat_map(|t| ["/CS0 cs ", t, " sc 0 0 1 1 re f "])
        .collect()
}

fn rgbs(content: &pdf::Content) -> Vec<[f64; 3]> {
    fills(content).into_iter().map(|(_, rgb)| rgb).collect()
}

fn assert_rgbs(content: &pdf::Content, expected: &[[f64; 3]]) {
    let found = rgbs(content);
    assert!(
        found.len() == expected.len() && found.iter().zip(expected).all(|(a, b)| near(*a, *b)),
        "{found:?} != {expected:?}; warnings {:?}",
        content.warnings
    );
}

#[test]
fn a_separation_converts_through_an_exponential_tint_transform() {
    // MuPDF (color management off) states the same: (1, 0.5, 0) and (1, 0.75, 0.5).
    let content = painted(
        "[/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>]",
        &squares(&["1", "0.5"]),
        &[],
    );
    assert_rgbs(&content, &[[1.0, 0.5, 0.0], [1.0, 0.75, 0.5]]);
    assert_eq!(fills(&content)[1].0, [0.5], "the tint is kept");
    let space = &paths(&content)[0].fill.as_ref().unwrap().color.space;
    assert_eq!(space, b"Separation");
}

#[test]
fn a_separation_converts_through_a_sampled_tint_transform() {
    let table = common::stream(
        "/FunctionType 0 /Domain [0 1] /Range [0 1 0 1 0 1] /Size [3] /BitsPerSample 8 /Filter /ASCIIHexDecode",
        "FFFFFF FF0000 0000FF>",
    );
    let content = painted(
        "[/Separation /Spot /DeviceRGB 6 0 R]",
        &squares(&["0", "0.25", "0.75", "1"]),
        &[&table],
    );
    assert_rgbs(
        &content,
        &[[1.0; 3], [1.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]],
    );
}

#[test]
fn a_sampled_transform_reads_encode_decode_and_narrow_samples() {
    // Four 4-bit samples over 0..1, /Encode reversing the table, /Decode onto 0.2..0.6.
    let table = common::stream(
        "/FunctionType 0 /Domain [0 1] /Range [0 1] /Size [4] /BitsPerSample 4 /Encode [3 0] /Decode [0.2 0.6] \
         /Filter /ASCIIHexDecode",
        "05AF>",
    );
    let content = painted(
        "[/Separation /Spot /DeviceGray 6 0 R]",
        &squares(&["0", "1", "0.5"]),
        &[&table],
    );
    let at = |sample: f64| 0.2 + sample / 15.0 * 0.4;
    let (last, first) = (at(15.0), at(0.0));
    let middle = f64::midpoint(at(5.0), at(10.0));
    assert_rgbs(&content, &[[last; 3], [first; 3], [middle; 3]]);
}

#[test]
fn a_separation_converts_through_a_stitching_tint_transform() {
    let content = painted(
        "[/Separation /Spot /DeviceGray << /FunctionType 3 /Domain [0 1] /Bounds [0.5] /Encode [0 1 0 1] \
         /Functions [<< /FunctionType 2 /Domain [0 1] /N 1 >> << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>] >>]",
        &squares(&["0.25", "0.5", "0.75", "1"]),
        &[],
    );
    assert_rgbs(&content, &[[0.5; 3], [1.0; 3], [0.5; 3], [0.0; 3]]);
}

#[test]
fn a_separation_converts_through_a_calculator_tint_transform() {
    let program = common::stream(
        "/FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1]",
        "{ dup 0.5 gt { pop 1 0 0 } { 2 mul 0 exch 0 } ifelse }",
    );
    let content = painted(
        "[/Separation /Spot /DeviceRGB 6 0 R]",
        &squares(&["0.8", "0.25"]),
        &[&program],
    );
    assert_rgbs(&content, &[[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]);
}

/// Each fill's components and RGB (None: no RGB), in painting order.
fn stated(content: &pdf::Content) -> Vec<(Vec<f64>, Option<[f64; 3]>)> {
    paths(content)
        .iter()
        .map(|p| {
            let color = &p.fill.as_ref().unwrap().color;
            (color.components.clone(), color.rgb)
        })
        .collect()
}

#[test]
fn a_tint_transform_that_overflows_its_stack_gives_no_rgb_and_says_why() {
    // Below 0.5 the program pushes 100 more values than its stack holds.
    let program = common::stream(
        "/FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1]",
        &format!("{{ dup 0.5 lt {{ {} }} if dup dup }}", "dup ".repeat(100)),
    );
    let content = painted(
        "[/Separation /Spot /DeviceRGB 6 0 R]",
        "/CS0 cs 0 0 1 1 re f 0.2 sc 0 0 1 1 re f",
        &[&program],
    );
    assert_eq!(
        stated(&content),
        [(vec![1.0], Some([1.0; 3])), (vec![0.2], None)]
    );
    assert_eq!(content.warnings.len(), 1, "{:?}", content.warnings);
    assert!(
        content.warnings[0].contains("stack grows past 100"),
        "{:?}",
        content.warnings
    );
}

#[test]
fn a_device_n_converts_its_colorants_together() {
    // MuPDF (color management off) states (0.5, 0, 1) for (0.5, 1) through { 0 0 } into CMYK.
    let program = common::stream(
        "/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1]",
        "{ 0 0 }",
    );
    let grid = common::stream(
        "/FunctionType 0 /Domain [0 1 0 1] /Range [0 1] /Size [2 2] /BitsPerSample 8 /Filter /ASCIIHexDecode",
        "00FF FF00>",
    );
    let content = painted(
        "[/DeviceN [/Cyan /Spot] /DeviceCMYK 6 0 R << /Colorants << /Spot [/Separation /Spot /DeviceCMYK 6 0 R] >> >>]",
        &format!(
            "{} [/DeviceN [/A /B] /DeviceGray 7 0 R] cs 0.5 0.5 sc 0 0 1 1 re f 0.25 0 sc 0 0 1 1 re f",
            squares(&["0.5 1"])
        ),
        &[&program, &grid],
    );
    assert_rgbs(&content, &[[0.5, 0.0, 1.0], [0.5; 3], [0.25; 3]]);
    assert_eq!(fills(&content)[0].0, [0.5, 1.0]);
}

#[test]
fn a_lab_color_converts_to_srgb() {
    // Lab D50 (50, 20, -30): sRGB (0.5212, 0.4237, 0.6685) by the CIE and IEC formulas.
    let content = painted(
        "[/Lab << /WhitePoint [0.9642 1 0.8249] >>]",
        &squares(&["50 20 -30", "100 0 0"]),
        &[],
    );
    let found = rgbs(&content);
    assert!(
        found[0]
            .iter()
            .zip([0.5212, 0.4237, 0.6685])
            .all(|(a, b)| (a - b).abs() < 1e-4),
        "{found:?}"
    );
    assert!(found[1].iter().all(|v| (v - 1.0).abs() < 1e-4), "{found:?}");
    assert_eq!(fills(&content)[0].0, [50.0, 20.0, -30.0]);
}

#[test]
fn a_separation_converts_into_lab_and_an_indexed_color_through_a_separation() {
    let separation = "[/Separation /Spot [/Lab << /WhitePoint [0.9642 1 0.8249] >>] \
                      << /FunctionType 2 /Domain [0 1] /C0 [100 0 0] /C1 [50 20 -30] /N 1 >>]";
    let content = painted(
        &format!("[/Indexed {separation} 1 <00FF>]"),
        &format!("{} {separation} cs 1 sc 0 0 1 1 re f", squares(&["0", "1"])),
        &[],
    );
    let found = rgbs(&content);
    assert!(found[0].iter().all(|v| (v - 1.0).abs() < 1e-4), "{found:?}");
    assert!(near(found[1], found[2]), "{found:?}");
    assert!((found[2][0] - 0.5212).abs() < 1e-4, "{found:?}");
}

#[test]
fn the_colorant_none_paints_nothing() {
    let none = "[/Separation /None /DeviceRGB << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [1 0 0] /N 1 >>]";
    let content = painted(
        none,
        "/CS0 cs 1 sc 0 0 1 1 re f 0 0 1 RG 0 0 2 2 re B \
         [/DeviceN [/None /None] /DeviceRGB 6 0 R] cs 1 1 sc 0 0 3 3 re f",
        &[&common::stream(
            "/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]",
            "{ pop 0 0 }",
        )],
    );
    let found = paths(&content);
    assert_eq!(found.len(), 1, "only the stroke paints: {found:?}");
    assert!(found[0].fill.is_none());
    assert_eq!(
        found[0].stroke.as_ref().unwrap().color.rgb,
        Some([0.0, 0.0, 1.0])
    );
    assert_eq!(content.warnings, Vec::<String>::new());
}

#[test]
fn the_colorant_all_paints_its_tint_transforms_color() {
    let content = painted(
        "[/Separation /All /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>]",
        &squares(&["1", "0.3"]),
        &[],
    );
    assert_rgbs(&content, &[[0.0; 3], [0.4; 3]]);
}

#[test]
fn a_space_that_does_not_read_keeps_its_colors_components_and_is_named() {
    // Not the gray the stream left before: a color set after it is its components alone.
    let content = painted(
        "[/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0] /C1 [1 1 1] /N 1 >>]",
        "0.5 g /CS0 cs 0.7 sc 0 0 1 1 re f [/Lab << >>] cs 0 0 1 1 re f",
        &[],
    );
    assert_eq!(stated(&content), [(vec![0.7], None), (vec![0.0], None)]);
    let space = &paths(&content)[1].fill.as_ref().unwrap().color.space;
    assert_eq!(space, b"Lab");
    assert_eq!(content.warnings.len(), 2, "{:?}", content.warnings);
    assert!(content.warnings[0].contains("gives 3, for 1 colorants into 4 components"));
    assert!(content.warnings[1].contains("no valid /WhitePoint"));
}
