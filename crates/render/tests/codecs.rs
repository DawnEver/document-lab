//! JBIG2 images drawn against a truth their decoder (`hayro-jbig2`) does not give: known pixels encoded at test time
//! by `jbig2enc-rust`, an independent encoder, and drawn one sample a pixel -- they must come back as those pixels,
//! or, for a halftone region (lossy by design), as the encoder's own reconstruction of what its stream holds. Each
//! case first checks its stream holds the segments it claims to exercise (T.88 7.2), and is paired with a planted
//! wrong decode the comparison must report. (JPEG 2000's truth is `OpenJPEG`'s encoding: `tests/unit/test_jpx.py`.)

#[path = "../../pdf/tests/common/mod.rs"]
mod common;

use jbig2enc_rust::jbig2halftone::{encode_halftone_pdf_fragment_auto, halftone_bilevel_preview};
use jbig2enc_rust::jbig2structs::{GenericRegionParams, PdfDictionaryPolicy};
use jbig2enc_rust::jbig2sym::BitImage;
use jbig2enc_rust::{Jbig2Config, Jbig2Context, encode_single_image_with_config};
use pdf::File;
use render::{Options, Raster, render};

/// A bilevel image: `w` x `h` samples, rows top-down, true black.
#[derive(Clone)]
struct Bits {
    w: u32,
    h: u32,
    black: Vec<bool>,
}

impl Bits {
    fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            black: vec![false; (w * h) as usize],
        }
    }

    /// `glyph` (rows of `#` black and `.` white) set with its top-left at (x, y).
    fn stamp(&mut self, glyph: &[&str], x: u32, y: u32) {
        for (row, gy) in glyph.iter().zip(y..) {
            for (c, gx) in row.bytes().zip(x..) {
                self.black[(gy * self.w + gx) as usize] = c == b'#';
            }
        }
    }

    /// One byte a pixel, 1 black: the encoder's input.
    fn bytes(&self) -> Vec<u8> {
        self.black.iter().map(|&b| u8::from(b)).collect()
    }
}

const A: [&str; 10] = [
    "...##...", "..####..", ".##..##.", "##....##", "##....##", "########", "##....##", "##....##",
    "##....##", "##....##",
];
const E: [&str; 10] = [
    "########", "##......", "##......", "##......", "######..", "######..", "##......", "##......",
    "##......", "########",
];
const O: [&str; 10] = [
    "..####..", ".##..##.", "##....##", "##....##", "##....##", "##....##", "##....##", "##....##",
    ".##..##.", "..####..",
];

/// Lines of text: `A`, `E` and `O` repeated, so a symbol coder finds each shape many times.
fn text(w: u32, h: u32) -> Bits {
    let mut bits = Bits::new(w, h);
    let glyphs = [A, E, O];
    for (line, y) in (2..h - 12).step_by(14).enumerate() {
        for (k, x) in (2..w - 10).step_by(11).enumerate() {
            bits.stamp(&glyphs[(k + line) % 3], x, y);
        }
    }
    bits
}

/// The page placing image object 5 (`dict`, its `/Filter` chain ending in `/JBIG2Decode` over hex `data`, its
/// parameters `params`) at its own size in the page's bottom-left corner, with `extra` objects 6, 7, ... -- drawn one
/// pixel a point.
fn drawn(dict: &str, data: &[u8], params: &str, extra: &[String], bits: &Bits) -> Raster {
    let mut objects = vec![jbig2_image(dict, data, params, bits)];
    objects.extend(extra.iter().cloned());
    placed(&objects, bits)
}

/// An image `XObject` of `bits`'s size, `dict`, its `/Filter` chain ending in `/JBIG2Decode` over hex `data`, its
/// parameters `params`.
fn jbig2_image(dict: &str, data: &[u8], params: &str, bits: &Bits) -> String {
    common::hex_stream(
        &format!(
            "/Type /XObject /Subtype /Image /Width {} /Height {} {dict} /DecodeParms [null {params}]", // spelling-ignore-line: the PDF name
            bits.w, bits.h
        ),
        data,
    )
    .replacen(
        "/Filter /ASCIIHexDecode",
        "/Filter [/ASCIIHexDecode /JBIG2Decode]",
        1,
    )
}

/// The page placing image object 5 of `objects` (5, 6, ...) at `bits`'s size in its bottom-left corner, drawn one
/// pixel a point.
fn placed(objects: &[String], bits: &Bits) -> Raster {
    let objects = objects.iter().map(String::as_str).collect::<Vec<_>>();
    let content = format!("1 0 0 rg q {} 0 0 {} 0 0 cm /Im1 Do Q", bits.w, bits.h);
    let file = File::open(
        common::one_page(
            "/Resources << /XObject << /Im1 5 0 R >> >>",
            &content,
            &objects,
        ),
        b"",
    )
    .unwrap();
    let page = file.pages().unwrap().remove(0);
    let options = Options {
        region: None,
        scale: 1.0,
        text: false,
        annotations: false,
        alpha: false,
        jpx: render::image::Jpx::default(),
    };
    render(&file, &page, &options).unwrap()
}

/// A gray image's dictionary entries.
const GRAY: &str = "/ColorSpace /DeviceGray /BitsPerComponent 1";

/// How many of the image's pixels are drawn otherwise than `truth` says (dark where black), and the image was drawn
/// with nothing left undrawn; None when something was.
fn wrong(r: &Raster, truth: &Bits) -> Option<usize> {
    if !r.undrawn.is_empty() {
        return None;
    }
    let top = r.height - truth.h;
    let mut count = 0;
    for y in 0..truth.h {
        for x in 0..truth.w {
            let i = (((top + y) * r.width + x) * u32::from(r.channels)) as usize;
            let dark = r.data[i + 1] < 128; // green: dark for black and for the red fill alike
            count += usize::from(dark != truth.black[(y * truth.w + x) as usize]);
        }
    }
    Some(count)
}

/// Whether the comparison reports a drawing: something left undrawn, or a pixel drawn wrong.
fn reported(r: &Raster, truth: &Bits) -> bool {
    wrong(r, truth) != Some(0)
}

/// The segments of an embedded JBIG2 stream (T.88 7.2): each one's type and its data.
fn segments(mut data: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut out = Vec::new();
    while data.len() >= 11 {
        let number = u32::from_be_bytes(data[..4].try_into().unwrap());
        let flags = data[4];
        let mut at = 5;
        let mut count = usize::from(data[at] >> 5);
        if count == 7 {
            count =
                (u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) & 0x1FFF_FFFF) as usize;
            at += 4 + (count + 1).div_ceil(8);
        } else {
            at += 1;
        }
        at += count
            * match number {
                0..=256 => 1,
                257..=65536 => 2,
                _ => 4,
            };
        at += if flags & 0x40 == 0 { 1 } else { 4 };
        let length = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        out.push((flags & 0x3F, data[at..at + length].to_vec()));
        data = &data[at + length..];
    }
    out
}

/// The types of `data`'s segments.
fn types(data: &[u8]) -> Vec<u8> {
    segments(data).into_iter().map(|(t, _)| t).collect()
}

/// Whether a text region segment of `data` refines its symbols (`SBREFINE`, 7.4.3.1.1, after the 17 bytes of its
/// region segment information).
fn refines(data: &[u8]) -> bool {
    segments(data)
        .iter()
        .any(|(t, d)| matches!(t, 4 | 6 | 7) && d[18] & 0x02 != 0)
}

/// The globals object (6) of `globals`.
fn globals_object(globals: &[u8]) -> String {
    common::hex_stream("", globals)
}

#[test]
fn a_generic_region_draws_its_known_pixels_and_inverted_is_reported() {
    let bits = text(150, 60);
    let config = Jbig2Context::with_config(Jbig2Config::lossless(), true);
    let encoded = encode_single_image_with_config(&bits.bytes(), bits.w, bits.h, config).unwrap();
    assert_eq!(encoded.global_data, None);
    // Page information (48), an immediate lossless generic region (38).
    assert!(
        types(&encoded.page_data).contains(&38),
        "{:?}",
        types(&encoded.page_data)
    );
    let r = drawn(GRAY, &encoded.page_data, "null", &[], &bits);
    assert_eq!(wrong(&r, &bits), Some(0));
    // Planted wrong decode: the samples read inverted.
    let inverted = drawn(
        &format!("{GRAY} /Decode [1 0]"),
        &encoded.page_data,
        "null",
        &[],
        &bits,
    );
    assert!(reported(&inverted, &bits));
}

/// `bits` symbol-coded, exactly, with every symbol in the globals.
fn symbol_coded(bits: &Bits) -> jbig2enc_rust::Jbig2EncodeResult {
    let mut config = Jbig2Config::text_lossless();
    config.pdf_dictionary_policy = PdfDictionaryPolicy::AllGlobal;
    let context = Jbig2Context::with_config(config, true);
    encode_single_image_with_config(&bits.bytes(), bits.w, bits.h, context).unwrap()
}

#[test]
fn a_text_region_draws_its_symbols_from_the_globals_and_others_globals_are_reported() {
    let bits = text(180, 70);
    let encoded = symbol_coded(&bits);
    let globals = encoded
        .global_data
        .expect("a symbol dictionary in the globals");
    // The symbol dictionary (0) in the globals; the page's text region (6 or 7) uses it.
    assert!(types(&globals).contains(&0), "{:?}", types(&globals));
    assert!(
        types(&encoded.page_data).iter().any(|t| matches!(t, 6 | 7)),
        "{:?}",
        types(&encoded.page_data)
    );
    let params = "<< /JBIG2Globals 6 0 R >>";
    let r = drawn(
        GRAY,
        &encoded.page_data,
        params,
        &[globals_object(&globals)],
        &bits,
    );
    assert_eq!(wrong(&r, &bits), Some(0));
    // Planted wrong decode: the globals of another page, whose dictionary holds other symbols.
    let mut other = Bits::new(180, 70);
    other.stamp(&O, 3, 3);
    other.stamp(&A, 40, 20);
    let foreign = symbol_coded(&other).global_data.unwrap();
    let r = drawn(
        GRAY,
        &encoded.page_data,
        params,
        &[globals_object(&foreign)],
        &bits,
    );
    assert!(reported(&r, &bits));
}

#[test]
fn a_text_region_draws_its_symbols_from_a_dictionary_of_its_page() {
    let bits = text(180, 70);
    let config = Jbig2Context::with_config(Jbig2Config::text_lossless(), true);
    let encoded = encode_single_image_with_config(&bits.bytes(), bits.w, bits.h, config).unwrap();
    // No globals: the symbol dictionary (0) is the page's own, before its text region (6 or 7).
    assert_eq!(encoded.global_data, None);
    let found = types(&encoded.page_data);
    assert!(
        found.contains(&0) && found.iter().any(|t| matches!(t, 6 | 7)),
        "{found:?}"
    );
    let r = drawn(GRAY, &encoded.page_data, "null", &[], &bits);
    assert_eq!(wrong(&r, &bits), Some(0));
    // Planted wrong decode: drawn a row low.
    let mut low = bits.clone();
    low.black.rotate_right(bits.w as usize);
    assert!(reported(&r, &low));
}

#[test]
fn refined_symbols_draw_their_known_pixels_and_their_references_are_reported() {
    // Large glyphs, each copy differing from the first by a pixel: cheaper refined than stored again.
    let ring: Vec<String> = (0..24)
        .map(|y: i32| {
            (0..24)
                .map(|x: i32| {
                    let d = (x - 12).pow(2) + (y - 12).pow(2);
                    if (49..=121).contains(&d) { '#' } else { '.' }
                })
                .collect()
        })
        .collect();
    let mut bits = Bits::new(190, 90);
    let mut references = Bits::new(190, 90);
    for (k, (x, y)) in (0u32..).zip(
        (0..6)
            .flat_map(|row| (0..7).map(move |col| (3 + col * 26, 3 + row * 14)))
            .filter(|&(_, y)| y + 24 <= 90),
    ) {
        let rows: Vec<&str> = ring.iter().map(String::as_str).collect();
        bits.stamp(&rows, x, y);
        references.stamp(&rows, x, y);
        // Each copy but the first has a notch of its own on the ring.
        if k > 0 {
            let notch = (x + 12 + k % 5 - 2, y + 2);
            bits.black[(notch.1 * bits.w + notch.0) as usize] ^= true;
        }
    }
    let config = Jbig2Context::with_config(Jbig2Config::text_lossless_max_compression(), true);
    let encoded = encode_single_image_with_config(&bits.bytes(), bits.w, bits.h, config).unwrap();
    assert!(
        refines(&encoded.page_data),
        "no refining text region: {:?}",
        types(&encoded.page_data)
    );
    let (params, extra) = match &encoded.global_data {
        Some(globals) => ("<< /JBIG2Globals 6 0 R >>", vec![globals_object(globals)]),
        None => ("null", vec![]),
    };
    let r = drawn(GRAY, &encoded.page_data, params, &extra, &bits);
    assert_eq!(wrong(&r, &bits), Some(0));
    // Planted wrong decode: the references drawn unrefined.
    assert!(reported(&r, &references));
}

/// A dithered gradient: what a halftone region holds.
fn dithered(w: u32, h: u32) -> BitImage {
    let mut image = BitImage::new(w, h).unwrap();
    for y in 0..h {
        for x in 0..w {
            // An ordered (Bayer 4x4) dither of a left-to-right ramp.
            const BAYER: [u32; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
            let level = x * 16 / w;
            image.set(x, y, BAYER[((y % 4) * 4 + x % 4) as usize] < level);
        }
    }
    image
}

/// `image` as [`Bits`].
fn bits_of(image: &BitImage, w: u32, h: u32) -> Bits {
    let mut bits = Bits::new(w, h);
    for y in 0..h {
        for x in 0..w {
            bits.black[(y * w + x) as usize] = image.get(x, y);
        }
    }
    bits
}

/// A halftone region's truth is the encoder's reconstruction of its stream (`halftone_bilevel_preview`): the
/// patterns tiled by the gray values it quantized. Its pattern dictionary and gray-value planes are MMR-coded here:
/// arithmetic-coded (jbig2enc-rust 0.6.2), the stream holds other gray values than its reconstruction says -- `jbig2dec`
/// (`PyMuPDF`) and `hayro-jbig2` decode it alike, 3520 pixels off it (measured 2026-11-27) -- so it is no truth.
#[test]
fn a_halftone_region_draws_its_patterns_and_the_source_is_reported() {
    let (w, h) = (160, 80);
    let source = dithered(w, h);
    let mut config = Jbig2Config::default();
    config.halftone.gray_mmr = true;
    config.halftone.dict_mmr = true;
    let mut region = GenericRegionParams::new(w, h, 300);
    region.tpgdon = false;
    let page = encode_halftone_pdf_fragment_auto(&source, &config, &region, 0, Some(1)).unwrap();
    // A pattern dictionary (16) and an immediate halftone region (22 or 23).
    let found = types(&page);
    assert!(
        found.contains(&16) && found.iter().any(|t| matches!(t, 22 | 23)),
        "{found:?}"
    );
    let truth = bits_of(&halftone_bilevel_preview(&source, &config).unwrap(), w, h);
    let r = drawn(GRAY, &page, "null", &[], &truth);
    assert_eq!(wrong(&r, &truth), Some(0));
    // Planted wrong decode: the dithered source itself, which the halftone only approximates.
    assert!(reported(&r, &bits_of(&source, w, h)));
}

#[test]
fn a_jbig2_stencil_mask_paints_its_black_pixels_and_inverted_is_reported() {
    let bits = text(120, 40);
    let config = Jbig2Context::with_config(Jbig2Config::lossless(), true);
    let encoded = encode_single_image_with_config(&bits.bytes(), bits.w, bits.h, config).unwrap();
    let r = drawn("/ImageMask true", &encoded.page_data, "null", &[], &bits);
    assert_eq!(wrong(&r, &bits), Some(0));
    // The black pixels are painted in the fill color, red.
    let top = r.height - bits.h;
    let first = u32::try_from(bits.black.iter().position(|&b| b).unwrap()).unwrap();
    let i = (((top + first / bits.w) * r.width + first % bits.w) * u32::from(r.channels)) as usize;
    assert_eq!(&r.data[i..i + 3], &[255, 0, 0]);
    // Planted wrong decode: the mask read inverted.
    let inverted = drawn(
        "/ImageMask true /Decode [1 0]",
        &encoded.page_data,
        "null",
        &[],
        &bits,
    );
    assert!(reported(&inverted, &bits));
}

#[test]
fn a_jbig2_image_missing_its_globals_is_drawn_leniently_and_counted_by_why() {
    let bits = text(100, 40);
    let encoded = symbol_coded(&bits);
    assert!(encoded.global_data.is_some());
    let r = drawn(GRAY, &encoded.page_data, "null", &[], &bits);
    assert_eq!(r.undrawn.len(), 1, "{:?}", r.undrawn);
    let why = r.undrawn.keys().next().unwrap();
    assert!(
        why.starts_with(
            "JBIG2Decode image drawn past what does not decode strictly: Segment(InvalidReference)"
        ),
        "{why}"
    );
}

/// A black gray image (object 5) masked by `/Mask 6 0 R`: a JBIG2 stream stating no `/ImageMask` (read as a stencil
/// all the same, as `MuPDF` reads a `/Mask` image), with `extra` objects 7, ...
fn masked_black(mask: &[u8], params: &str, extra: &[String], bits: &Bits) -> Raster {
    let black = vec![0u8; (bits.w * bits.h) as usize];
    let image = common::hex_stream(
        &format!(
            "/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceGray /BitsPerComponent 8 \
             /Mask 6 0 R",
            bits.w, bits.h
        ),
        &black,
    );
    let mut objects = vec![image, jbig2_image("", mask, params, bits)];
    objects.extend(extra.iter().cloned());
    placed(&objects, bits)
}

#[test]
fn a_jbig2_mask_shows_its_image_where_black_and_one_drawn_leniently_is_counted_by_why() {
    let bits = text(100, 40);
    let encoded = symbol_coded(&bits);
    let globals = encoded.global_data.clone().unwrap();
    // A /Mask sample of 1 masks its area out: JBIG2 black decodes to 0, so the image shows where the mask is black.
    let r = masked_black(
        &encoded.page_data,
        "<< /JBIG2Globals 7 0 R >>",
        &[common::hex_stream("", &globals)],
        &bits,
    );
    assert_eq!(wrong(&r, &bits), Some(0));
    // Planted wrong decode: the mask ignored, the image black throughout.
    let mut ignored = bits.clone();
    ignored.black.fill(true);
    assert!(reported(&r, &ignored));
    // Its globals missing: a strict decoder refuses it, a lenient one draws it blank -- counted, never silent.
    let r = masked_black(&encoded.page_data, "null", &[], &bits);
    let why: Vec<_> = r.undrawn.keys().collect();
    assert_eq!(why.len(), 1, "{why:?}");
    assert!(
        why[0].starts_with(
            "a mask's JBIG2Decode image drawn past what does not decode strictly: Segment(InvalidReference)"
        ),
        "{why:?}"
    );
}
