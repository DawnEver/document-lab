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

/// The headers of JPEG data up to its first scan: a frame header (SOF0, or SOF2 when `progressive`) of `components`,
/// the first sampled `sampling`, behind Adobe's marker with `adobe` as its transform when given.
fn jpeg(components: u8, sampling: u8, adobe: Option<u8>, progressive: bool) -> Vec<u8> {
    let mut data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 7, b'J', b'F', b'I', b'F', 0];
    if let Some(transform) = adobe {
        data.extend([0xFF, 0xEE, 0, 14]);
        data.extend(b"Adobe");
        data.extend([0, 100, 0, 0, 0, 0, transform]);
    }
    let length = 8 + 3 * u16::from(components);
    data.extend([0xFF, if progressive { 0xC2 } else { 0xC0 }]);
    data.extend(length.to_be_bytes());
    data.extend([8, 0, 1, 0, 2, components]);
    for id in 1..=components {
        data.extend([id, if id == 1 { sampling } else { 0x11 }, 0]);
    }
    data.extend([0xFF, 0xDA, 0, 2]);
    data
}

fn dct(data: &[u8], params: Option<&str>) -> pdf::error::Result<Option<pdf::filter::Dct>> {
    let params = params.map(|p| {
        let file = File::open(common::one_page("", "", &[p]), b"").unwrap();
        file.object(5).unwrap().as_dict().unwrap().clone()
    });
    let codec = pdf::filter::Codec {
        name: b"DCTDecode".to_vec(),
        params,
    };
    codec.dct(data)
}

#[test]
fn a_jpeg_states_its_coding_and_whether_its_colors_are_transformed() {
    let coded = |components, sampling, transform| pdf::filter::Dct {
        precision: 8,
        components,
        sampling,
        transform,
    };
    // Three components are YCbCr unless Adobe's marker, else /ColorTransform, says they are not.
    assert_eq!(
        dct(&jpeg(3, 0x22, None, false), None).unwrap(),
        Some(coded(3, (2, 2), true))
    );
    assert_eq!(
        dct(&jpeg(3, 0x11, Some(0), true), None).unwrap(),
        Some(coded(3, (1, 1), false))
    );
    let untransformed = Some("<< /ColorTransform 0 >>");
    assert_eq!(
        dct(&jpeg(3, 0x21, None, false), untransformed).unwrap(),
        Some(coded(3, (2, 1), false))
    );
    assert_eq!(
        dct(&jpeg(3, 0x22, Some(1), false), untransformed).unwrap(),
        Some(coded(3, (2, 2), true))
    );
    // Gray is never transformed; four components only when said to be (YCCK).
    assert_eq!(
        dct(&jpeg(1, 0x11, None, false), None).unwrap(),
        Some(coded(1, (1, 1), false))
    );
    assert_eq!(
        dct(&jpeg(4, 0x11, Some(2), false), None).unwrap(),
        Some(coded(4, (1, 1), true))
    );
}

#[test]
fn a_jpeg_without_a_frame_header_or_with_a_wrong_parameter_is_refused() {
    let headless = [0xFF, 0xD8, 0xFF, 0xDA, 0, 2];
    let error = dct(&headless, None).unwrap_err();
    assert!(error.message.contains("no frame header"), "{error}");
    let cut = &jpeg(3, 0x22, None, false)[..15];
    assert!(dct(cut, None).unwrap_err().message.contains("ends inside"));
    let error = dct(
        &jpeg(3, 0x22, None, false),
        Some("<< /ColorTransform /No >>"),
    )
    .unwrap_err();
    assert!(
        error.message.contains("/ColorTransform is an integer"),
        "{error}"
    );
    let other = pdf::filter::Codec {
        name: b"JPXDecode".to_vec(),
        params: None,
    };
    assert_eq!(other.dct(b"").unwrap(), None);
}
