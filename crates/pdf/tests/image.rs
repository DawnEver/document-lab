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
