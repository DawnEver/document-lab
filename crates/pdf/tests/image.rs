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
    assert_eq!((plain.decode, plain.masked), (None, false));
    assert_eq!(plain.data, b"ab");
}

#[test]
fn a_decode_array_is_handed_over_as_stated() {
    assert_eq!(samples("/Decode [1 0]").decode, Some(vec![1.0, 0.0]));
}

#[test]
fn a_soft_mask_or_a_mask_says_the_image_is_masked() {
    assert!(samples("/SMask 6 0 R").masked);
    assert!(samples("/Mask 6 0 R").masked);
    assert!(samples("/Mask [0 127]").masked);
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
