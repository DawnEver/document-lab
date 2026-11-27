//! `ICCBased` colors converted through their embedded profiles (made here by `moxcms`, written into the file) at the
//! rendering intent in force, `CalRGB` and `CalGray` through their parameters, and a profile that does not read.

mod common;

use moxcms::{
    ColorProfile, DataColorSpace, LutDataType, LutStore, LutType, LutWarehouse, Matrix3d,
    ProfileClass,
};
use pdf::interpret::PathEvent;
use pdf::{Event, File};

/// The Adobe RGB (1998) primaries' CIE XYZ (D65) per unit of linear red, green and blue (its specification, 4.3.4.3).
const ADOBE_TO_XYZ: [[f64; 3]; 3] = [
    [0.576_730_9, 0.185_554_0, 0.188_185_2],
    [0.297_376_9, 0.627_349_1, 0.075_274_1],
    [0.027_034_3, 0.070_687_2, 0.991_108_5],
];
/// CIE XYZ (D65) to linear sRGB (IEC 61966-2-1).
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];
const ADOBE_GAMMA: f64 = 563.0 / 256.0;

fn times(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

fn encoded(linear: f64) -> f64 {
    let v = linear.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// The sRGB of an Adobe RGB color, by the two specifications' formulas: the planted truth, apart from any CMM.
fn adobe_in_srgb(rgb: [f64; 3]) -> [f64; 3] {
    times(
        &XYZ_TO_SRGB,
        times(&ADOBE_TO_XYZ, rgb.map(|c| c.powf(ADOBE_GAMMA))),
    )
    .map(encoded)
}

/// The events of a page whose resources name `/CS0` (object 5, `space`) and `/GS0` (object 6, `/RI` `gs_intent`),
/// with `extra` objects from 7 on.
fn painted(space: &str, gs_intent: &str, content: &str, extra: &[&str]) -> pdf::Content {
    let gs = format!("<< /Type /ExtGState /RI /{gs_intent} >>");
    let mut objects = vec![space, &gs];
    objects.extend(extra);
    let pdf = common::one_page(
        "/Resources << /ColorSpace << /CS0 5 0 R >> /ExtGState << /GS0 6 0 R >> >>",
        content,
        &objects,
    );
    let file = File::open(pdf, b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    file.interpret(&page, false).unwrap()
}

fn fills(content: &pdf::Content) -> Vec<(Vec<f64>, Option<[f64; 3]>)> {
    content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Path(PathEvent { fill: Some(f), .. }) => {
                Some((f.color.components.clone(), f.color.rgb))
            }
            _ => None,
        })
        .collect()
}

fn assert_near(found: Option<[f64; 3]>, expected: [f64; 3], tolerance: f64) {
    let found = found.expect("an RGB");
    assert!(
        found
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= tolerance),
        "{found:?} != {expected:?}"
    );
}

const COLORS: [[f64; 3]; 3] = [[0.2, 0.6, 0.4], [0.1, 0.8, 0.3], [0.9, 0.3, 0.1]];

fn squares(colors: &[[f64; 3]]) -> String {
    colors.iter().fold(String::new(), |out, [r, g, b]| {
        out + &format!("{r} {g} {b} sc 0 0 1 1 re f ")
    })
}

#[test]
fn an_icc_based_color_converts_through_its_embedded_profile() {
    let profile = ColorProfile::new_adobe_rgb().encode().unwrap();
    let content = painted(
        "[/ICCBased 7 0 R]",
        "Perceptual",
        &format!("/CS0 cs {}", squares(&COLORS)),
        &[&common::hex_stream("/N 3", &profile)],
    );
    let found = fills(&content);
    assert_eq!(found.len(), 3, "{:?}", content.warnings);
    for ((components, rgb), color) in found.into_iter().zip(COLORS) {
        assert_eq!(components, color);
        let truth = adobe_in_srgb(color);
        // The planted premise: read as device RGB, the color would be off by more than the tolerance.
        assert!(truth.iter().zip(color).any(|(a, b)| (a - b).abs() > 0.02));
        assert_near(rgb, truth, 2e-3);
    }
    assert_eq!(content.warnings, Vec::<String>::new());
}

#[test]
fn an_srgb_profile_keeps_the_components_and_a_gray_profile_converts_its_gray() {
    let srgb = ColorProfile::new_srgb().encode().unwrap();
    let content = painted(
        "[/ICCBased 7 0 R]",
        "Perceptual",
        &format!(
            "/CS0 cs {} [/ICCBased 8 0 R] cs 0.5 sc 0 0 1 1 re f",
            squares(&COLORS)
        ),
        &[
            &common::hex_stream("/N 3", &srgb),
            &common::hex_stream(
                "/N 1",
                &ColorProfile::new_gray_with_gamma(2.2).encode().unwrap(),
            ),
        ],
    );
    let found = fills(&content);
    for ((_, rgb), color) in found.iter().zip(COLORS) {
        assert_near(*rgb, color, 1e-3);
    }
    // A gray of gamma 2.2: 0.5 is a linear 0.5^2.2 (its white the profile's, mapped to sRGB's).
    assert_near(found[3].1, [encoded(0.5_f64.powf(2.2)); 3], 2e-3);
}

/// A CMYK profile (PCS XYZ) whose perceptual table gives white for every color and whose colorimetric table gives
/// black: which table a color went through is plain to see. It has no saturation table.
fn intents_profile() -> Vec<u8> {
    let table = |xyz: [u16; 3]| {
        LutWarehouse::Lut(LutDataType {
            num_input_channels: 4,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix: Matrix3d::IDENTITY,
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            input_table: LutStore::Store16([0, 65535].repeat(4)),
            clut_table: LutStore::Store16(xyz.repeat(16)),
            output_table: LutStore::Store16([0, 65535].repeat(3)),
            lut_type: LutType::Lut16,
        })
    };
    // XYZ in lut16 is u1Fixed15 (1.0 at 32768): D50 white is (0.9642, 1, 0.8249).
    let white = [31_595, 32_768, 27_030]; // 0.9642, 1, 0.8249
    let mut profile = ColorProfile::default();
    profile.color_space = DataColorSpace::Cmyk;
    profile.pcs = DataColorSpace::Xyz;
    profile.profile_class = ProfileClass::OutputDevice;
    profile.lut_a_to_b_perceptual = Some(table(white));
    profile.lut_a_to_b_colorimetric = Some(table([0, 0, 0]));
    profile.encode().unwrap()
}

#[test]
fn a_profile_converts_at_the_rendering_intent_in_force() {
    let content = painted(
        "[/ICCBased 7 0 R]",
        "Perceptual",
        "/CS0 cs 0 0 0 0 sc 0 0 1 1 re f /Perceptual ri 0 0 1 1 re f /Saturation ri 0 0 1 1 re f \
         /RelativeColorimetric ri 0 0 1 1 re f q /GS0 gs 0 0 1 1 re f Q 0 0 1 1 re f \
         /Perceptual ri /CS0 cs 0 0 1 1 re f /Unknown ri 0 0 1 1 re f",
        &[&common::hex_stream("/N 4", &intents_profile())],
    );
    let found: Vec<_> = fills(&content)
        .into_iter()
        .map(|(_, rgb)| rgb.unwrap())
        .collect();
    let (white, black) = ([1.0; 3], [0.0; 3]);
    let expected = [
        black, // relative colorimetric until an intent is set
        white, // `ri`: perceptual
        white, // saturation: the profile has no table for it, so its perceptual one (ICC.1)
        black, white, // a graphics state's /RI, until `Q`
        black, white, // a color set after `ri` converts at it too
        black, // an intent the standard does not define: the default
    ];
    assert_eq!(found.len(), expected.len(), "{:?}", content.warnings);
    for (rgb, truth) in found.into_iter().zip(expected) {
        assert_near(Some(rgb), truth, 2e-3);
    }
}

#[test]
fn a_profile_is_read_once_per_profile_object() {
    let profile = ColorProfile::new_adobe_rgb().encode().unwrap();
    let pdf = common::one_page(
        "/Resources << /ColorSpace << /A 5 0 R /B 6 0 R >> >>",
        "",
        &[
            "[/ICCBased 7 0 R]",
            "[/Indexed [/ICCBased 7 0 R] 0 <000000>]",
            &common::hex_stream("/N 3", &profile),
        ],
    );
    let file = File::open(pdf, b"").unwrap();
    let resources = file.pages().unwrap().remove(0).resources;
    let read = |name: &[u8]| {
        file.color_space(&pdf::Object::Name(name.to_vec()), &resources)
            .unwrap()
    };
    let (pdf::color::Space::Icc(a), pdf::color::Space::Indexed(base, _)) = (read(b"A"), read(b"B"))
    else {
        panic!("not read as ICCBased");
    };
    let pdf::color::Space::Icc(b) = *base else {
        panic!("not read as ICCBased");
    };
    assert!(std::rc::Rc::ptr_eq(&a, &b));
}

#[test]
fn a_profile_that_does_not_read_keeps_the_components_and_says_why() {
    let content = painted(
        "[/ICCBased 7 0 R]",
        "Perceptual",
        "/CS0 cs 0.1 0.2 0.3 sc 0 0 1 1 re f [/ICCBased 8 0 R] cs 0.5 sc 0 0 1 1 re f",
        &[
            &common::stream("/N 3", "not a profile"),
            &common::hex_stream("/N 1", &ColorProfile::new_adobe_rgb().encode().unwrap()),
        ],
    );
    assert_eq!(
        fills(&content),
        [(vec![0.1, 0.2, 0.3], None), (vec![0.5], None)]
    );
    assert_eq!(content.warnings.len(), 2, "{:?}", content.warnings);
    assert!(
        content.warnings[0].contains("its ICC profile does not read"),
        "{:?}",
        content.warnings
    );
    assert!(
        content.warnings[1].contains("not the 1 its /N states"),
        "{:?}",
        content.warnings
    );
}

#[test]
fn cal_rgb_and_cal_gray_convert_through_their_parameters() {
    // CalRGB with Adobe RGB's white, gamma and matrix is Adobe RGB: its profile gives the same sRGB.
    let m = ADOBE_TO_XYZ;
    let adobe = format!(
        "[/CalRGB << /WhitePoint [0.95047 1 1.08883] /Gamma [{ADOBE_GAMMA} {ADOBE_GAMMA} {ADOBE_GAMMA}] \
         /Matrix [{} {} {} {} {} {} {} {} {}] >>]",
        m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]
    );
    let content = painted(
        &adobe,
        "Perceptual",
        &format!(
            "/CS0 cs {} [/CalGray << /WhitePoint [0.9505 1 1.089] /Gamma 2.2 >>] cs 0.5 sc 0 0 1 1 re f \
             [/CalRGB << /WhitePoint [0.95047 1 1.08883] >>] cs 0.475235 0.5 0.544415 sc 0 0 1 1 re f",
            squares(&COLORS)
        ),
        &[],
    );
    let found = fills(&content);
    assert_eq!(found.len(), 5, "{:?}", content.warnings);
    for ((_, rgb), color) in found.iter().zip(COLORS) {
        assert_near(*rgb, adobe_in_srgb(color), 1e-3);
    }
    assert_near(found[3].1, [encoded(0.5_f64.powf(2.2)); 3], 1e-3);
    // No /Gamma, no /Matrix: each 1 and the identity, so the components are CIE XYZ -- half of D65 white's here,
    // which is sRGB's linear 0.5.
    assert_near(found[4].1, [encoded(0.5); 3], 1e-3);
    assert_eq!(content.warnings, Vec::<String>::new());
}
