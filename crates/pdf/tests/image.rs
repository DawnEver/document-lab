//! An image's samples say what more than its samples decides what shows: its `/Decode` and whether it is masked.

mod common;

use pdf::File;

/// The samples of a 2 x 1 gray image whose dictionary also states `extra`, with an 8-bit gray image as object 6.
fn samples(extra: &str) -> pdf::image::Samples {
    let image = common::stream(
        &format!(
            "/Type /XObject /Subtype /Image /Width 2 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray {extra}"
        ),
        "ab",
    );
    let other = common::stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray",
        "cd",
    );
    let file = File::open(common::one_page("", "", &[&image, &other]), b"").unwrap();
    file.image(5).unwrap()
}

#[test]
fn plain_samples_state_no_decode_and_no_mask() {
    let plain = samples("");
    assert_eq!((plain.masked(), plain.decode), (false, None));
    assert_eq!(plain.data, b"ab");
}

#[test]
fn a_decode_array_is_handed_over_as_stated() {
    assert_eq!(samples("/Decode [1 0]").decode, Some(vec![1.0, 0.0]));
}

#[test]
fn a_soft_mask_or_a_mask_says_the_image_is_masked() {
    assert!(samples("/SMask 6 0 R").masked());
    assert!(samples("/Mask 6 0 R").masked());
    assert!(samples("/Mask [0 127]").masked());
    // A JPEG 2000 taking its data's opacity as its soft mask.
    assert!(samples("/SMaskInData 1").masked());
    assert!(!samples("/SMaskInData 0").masked());
}

#[test]
fn a_fax_image_states_its_coding_each_parameter_defaulting_as_the_standard_says() {
    let fax = |params: &str| {
        let codec = samples(&format!("/Filter /CCF {params}")).codec.unwrap();
        assert_eq!(codec.name, b"CCITTFaxDecode");
        codec.fax()
    };
    let stated = fax(
        "/DecodeParms << /K -1 /Columns 2 /BlackIs1 true /EncodedByteAlign true /EndOfLine true >>", // spelling-ignore-line: the PDF name
    );
    assert_eq!(
        stated.unwrap(),
        Some(pdf::filter::Fax {
            k: -1,
            columns: 2,
            end_of_line: true,
            byte_align: true,
            black_is_1: true,
        })
    );
    assert_eq!(
        fax("").unwrap(),
        Some(pdf::filter::Fax {
            k: 0,
            columns: 1728,
            end_of_line: false,
            byte_align: false,
            black_is_1: false,
        })
    );
    let error = fax("/DecodeParms << /BlackIs1 1 >>").unwrap_err(); // spelling-ignore-line: the PDF name
    assert!(error.message.contains("/BlackIs1 is a boolean"), "{error}");
}

#[test]
fn another_codec_states_no_fax_coding() {
    let codec = samples("/Filter /DCTDecode").codec.unwrap();
    assert_eq!(codec.fax().unwrap(), None);
}

/// The samples of a `width` x 1 image stating `dict` (its size and filter given here), its data `hex`, with
/// `extra` objects from 6 on.
fn image(width: usize, dict: &str, hex: &str, extra: &[&str]) -> pdf::image::Samples {
    let image = common::stream(
        &format!("/Subtype /Image /Width {width} /Height 1 /Filter /ASCIIHexDecode {dict}"),
        &format!("{hex}>"),
    );
    let mut objects = vec![image.as_str()];
    objects.extend(extra);
    let file = File::open(common::one_page("", "", &objects), b"").unwrap();
    file.image(5).unwrap()
}

#[test]
fn gray_and_rgb_samples_are_their_own_srgb() {
    let rgb = image(
        2,
        "/BitsPerComponent 8 /ColorSpace /DeviceRGB",
        "102030405060",
        &[],
    );
    assert_eq!(rgb.rgb(), Some(vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60]));
    // Two-bit gray: levels 0..=3 spread over 0..=255; the row's last byte padded.
    let gray = image(3, "/BitsPerComponent 2 /ColorSpace /DeviceGray", "6c", &[]);
    assert_eq!(
        gray.rgb(),
        Some(
            [0x55; 3]
                .into_iter()
                .chain([0xaa; 3])
                .chain([0xff; 3])
                .collect()
        )
    );
}

#[test]
fn icc_based_samples_convert_through_their_profile() {
    let adobe = moxcms::ColorProfile::new_adobe_rgb().encode().unwrap();
    let profile = common::hex_stream("/N 3", &adobe);
    let samples = image(
        2,
        "/BitsPerComponent 8 /ColorSpace [/ICCBased 6 0 R]",
        "33996619e64d",
        &[&profile],
    );
    let rgb = samples.rgb().unwrap();
    // The same colors as fills: the profile's conversion, to the nearest level.
    let file = File::open(common::one_page("", "", &[&profile]), b"").unwrap();
    let space = file
        .color_space(
            &pdf::Object::Array(vec![
                pdf::Object::Name(b"ICCBased".to_vec()),
                pdf::Object::Reference(pdf::ObjRef {
                    num: 5,
                    generation: 0,
                }),
            ]),
            &pdf::Dict::new(),
        )
        .unwrap();
    let expected: Vec<f64> = [[0x33, 0x99, 0x66], [0x19, 0xe6, 0x4d]]
        .iter()
        .flat_map(|s| {
            let color = space
                .color(
                    s.iter().map(|&v| f64::from(v) / 255.0).collect(),
                    pdf::icc::Intent::default(),
                )
                .unwrap();
            color.rgb.unwrap().map(|v| v * 255.0)
        })
        .collect();
    assert!(
        rgb.iter()
            .zip(&expected)
            .all(|(a, b)| (f64::from(*a) - b).abs() <= 1.0),
        "{rgb:?} != {expected:?}"
    );
    assert_ne!(rgb, [0x33, 0x99, 0x66, 0x19, 0xe6, 0x4d]); // the planted premise: the profile changes them
}

#[test]
fn what_the_samples_alone_do_not_show_has_no_srgb() {
    let gray = "/BitsPerComponent 8 /ColorSpace /DeviceGray";
    for (dict, extra) in [
        (format!("{gray} /Decode [1 0]"), None),
        (
            format!("{gray} /SMask 6 0 R"),
            Some(common::stream(
                "/Subtype /Image /Width 2 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray",
                "ab",
            )),
        ),
        (
            "/BitsPerComponent 8 /ColorSpace /DeviceCMYK".to_string(),
            None,
        ),
        (
            "/BitsPerComponent 16 /ColorSpace /DeviceGray".to_string(),
            None,
        ),
        (
            "/BitsPerComponent 8 /ColorSpace [/Indexed /DeviceRGB 0 <000000>]".to_string(),
            None,
        ),
        (
            "/BitsPerComponent 8 /ColorSpace [/ICCBased 6 0 R]".to_string(),
            Some(common::stream("/N 1", "not a profile")),
        ),
    ] {
        let extra: Vec<&str> = extra.iter().map(String::as_str).collect();
        assert_eq!(
            image(2, &dict, "0011223344556677", &extra).rgb(),
            None,
            "{dict}"
        );
    }
    let short = image(4, gray, "0011", &[]);
    assert_eq!(short.rgb(), None, "a row missing");
}
