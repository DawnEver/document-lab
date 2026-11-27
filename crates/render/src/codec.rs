//! Image codecs to samples: the data the reader hands over still encoded (`pdf::filter::Codec`) decoded to the
//! samples an image's dictionary describes.
//!
//! - `DCTDecode` (JPEG) by `zune-jpeg`: gray, RGB or CMYK samples, 8 bits each, by the JPEG's own component count
//!   (YCbCr and YCCK converted by the JPEG's transform). CMYK samples are taken as stored: a JPEG an Adobe
//!   application wrote inverted states `/Decode [1 0 1 0 1 0 1 0]` in its PDF, which the image's mapping applies
//!   (as Acrobat and `MuPDF` read it; inverting by the `APP14` marker as well drew such images dark).
//! - `CCITTFaxDecode` (Group 3 one- and two-dimensional, Group 4) by `hayro-ccitt`: one bit a sample, 0 black
//!   unless `/BlackIs1`.
//! - `JPXDecode` (JPEG 2000: a JP2 file or a bare codestream, 7.4.9) by `hayro-jpeg2000`: 8 bits a sample, each
//!   component of `b` bits scaled by `255 / (2^b - 1)` and rounded -- but the samples of an image its dictionary
//!   places in an `Indexed` space are the indices they are (its palette, if any, not applied; its own color space is
//!   `pdf::image::jpx_space`'s to read). Its opacity channel (a JP2 `cdef` box) is handed over beside the color
//!   samples, for `/SMaskInData`.
//! - `JBIG2Decode` (ITU-T T.88, the embedded organization of its Annex D.3, 7.4.7) by `hayro-jbig2`, after the
//!   segments of its `/JBIG2Globals`: generic, refinement, text and halftone regions; one bit a sample, 0 black (a
//!   JBIG2 bitmap's 1 is black: the filter's data is its inverse, as the gray space of a bilevel image reads it).

use pdf::File;
use pdf::color::Space;
use pdf::filter::Codec;
use pdf::object::{Dict, Object};

/// Samples a codec decoded.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    /// The samples, rows top-down, each row ending on a whole byte.
    pub data: Vec<u8>,
    /// Components a sample, when the codec states them (a JPEG's, a JPEG 2000's); None: as the dictionary says.
    pub components: Option<usize>,
    /// Bits a component, as the codec decoded them.
    pub bits: u32,
    /// The opacity the data carries beside its colors (a JPEG 2000's opacity channel), one byte a sample.
    pub opacity: Option<Vec<u8>>,
    /// What of the data was not decoded as it stands, when something was: its rows past the data's end (white), or
    /// what a strict decoding refuses and a lenient one decodes past.
    pub partial: Option<String>,
}

/// The note of an image whose data ends before its rows do.
const ENDS_EARLY: &str = "image whose data ends early (drawn as far as it decodes)";

impl Decoded {
    /// Samples of `bits` that are as the dictionary describes them.
    fn plain(data: Vec<u8>, bits: u32, partial: Option<String>) -> Self {
        Self {
            data,
            components: None,
            bits,
            opacity: None,
            partial,
        }
    }
}

/// `data` decoded from `codec` for an image `width` x `height` samples, its dictionary stating `stated` (None for
/// none, or a mask); `file` holds what the codec's parameters refer to (`/JBIG2Globals`).
///
/// # Errors
///
/// Refuses data that does not decode, and a JPEG 2000 whose size is not the one its dictionary states, saying which.
pub fn decode(
    file: &File,
    codec: &Codec,
    data: &[u8],
    (width, height): (u32, u32),
    stated: Option<&Space>,
) -> Result<Decoded, String> {
    match codec.name.as_slice() {
        b"DCTDecode" => jpeg(data),
        b"CCITTFaxDecode" => fax(data, codec.params.as_ref(), width, height),
        b"JPXDecode" => jpx(data, (width, height), stated),
        b"JBIG2Decode" => jbig2(file, codec.params.as_ref(), data, (width, height)),
        other => Err(format!(
            "{} image (a codec not decoded here)",
            String::from_utf8_lossy(other)
        )),
    }
}

/// A JPEG's samples (see the module).
fn jpeg(data: &[u8]) -> Result<Decoded, String> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let failed = |e: &dyn std::fmt::Debug| format!("DCTDecode image that does not decode: {e:?}");
    let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(data));
    decoder.decode_headers().map_err(|e| failed(&e))?;
    let components = decoder
        .info()
        .map_or(0, |info| usize::from(info.components));
    let out = match components {
        1 => ColorSpace::Luma,
        3 => ColorSpace::RGB,
        4 => ColorSpace::CMYK,
        n => return Err(format!("DCTDecode image of {n} components")),
    };
    decoder.set_options(
        DecoderOptions::default()
            .jpeg_set_out_colorspace(out)
            .set_max_width(1 << 16)
            .set_max_height(1 << 16),
    );
    let samples = decoder.decode().map_err(|e| failed(&e))?;
    Ok(Decoded {
        components: Some(components),
        ..Decoded::plain(samples, 8, None)
    })
}

/// Why a codec's data gave no samples.
enum Refusal {
    /// Its decoder refuses it, saying why.
    Decoder(String),
    /// What it decodes to is not what its dictionary states, in words that say so.
    Stated(String),
}

/// What `decode` gives run strictly, else -- when its decoder refuses that -- run leniently, the strict refusal then
/// what was not decoded as it stands: a lenient decoder skips or repairs what it cannot read without saying so.
fn strictly(
    name: &str,
    decode: impl Fn(bool) -> Result<Decoded, Refusal>,
) -> Result<Decoded, String> {
    let lenient = |why: String| match decode(false) {
        Ok(found) => Ok(Decoded {
            partial: Some(format!(
                "{name} image drawn past what does not decode strictly: {why}"
            )),
            ..found
        }),
        Err(Refusal::Decoder(e)) => Err(format!("{name} image that does not decode: {e}")),
        Err(Refusal::Stated(e)) => Err(e),
    };
    match decode(true) {
        Ok(found) => Ok(found),
        Err(Refusal::Decoder(why)) => lenient(why),
        Err(Refusal::Stated(e)) => Err(e),
    }
}

/// A JPEG 2000's samples and its opacity (see the module).
fn jpx(data: &[u8], size: (u32, u32), stated: Option<&Space>) -> Result<Decoded, String> {
    let indices = matches!(stated, Some(Space::Indexed(..)));
    strictly("JPXDecode", |strict| jpx_read(data, size, indices, strict))
}

/// A JPEG 2000's samples (see the module), decoded `strict` or not; its samples are `indices` or not.
fn jpx_read(
    data: &[u8],
    (width, height): (u32, u32),
    indices: bool,
    strict: bool,
) -> Result<Decoded, Refusal> {
    use hayro_jpeg2000::{DecodeSettings, DecoderContext, Image};
    let failed = |e: &dyn std::fmt::Debug| Refusal::Decoder(format!("{e:?}"));
    let settings = DecodeSettings {
        resolve_palette_indices: !indices,
        strict,
        ..DecodeSettings::default()
    };
    let image = Image::new(data, &settings).map_err(|e| failed(&e))?;
    if (image.width(), image.height()) != (width, height) {
        return Err(Refusal::Stated(format!(
            "JPXDecode image of {}x{} samples where its dictionary states {width}x{height}",
            image.width(),
            image.height()
        )));
    }
    let has_alpha = image.has_alpha();
    let mut context = DecoderContext::default();
    let decoded = image.decode(&mut context).map_err(|e| failed(&e))?;
    let planes = decoded.components();
    let count = width as usize * height as usize;
    if let Some(plane) = planes.iter().find(|p| p.samples().len() < count) {
        return Err(Refusal::Decoder(format!(
            "a component of {} of its {count} samples",
            plane.samples().len()
        )));
    }
    let n = planes.len() - usize::from(has_alpha);
    if n == 0 {
        return Err(Refusal::Decoder("no color component".to_string()));
    }
    let level = |plane: &hayro_jpeg2000::ComponentData, index: bool| {
        let scale = if index {
            1.0
        } else {
            255.0 / (2f32.powi(i32::from(plane.bit_depth())) - 1.0)
        };
        move |v: f32| -> u8 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to a byte"
            )]
            {
                (v * scale + 0.5).floor().clamp(0.0, 255.0) as u8
            }
        }
    };
    if indices && planes[0].bit_depth() > 8 {
        return Err(Refusal::Stated(format!(
            "JPXDecode image of {}-bit indices",
            planes[0].bit_depth()
        )));
    }
    let mut samples = vec![0u8; count * n];
    for (k, plane) in planes[..n].iter().enumerate() {
        let to = level(plane, indices);
        for (out, &v) in samples[k..].iter_mut().step_by(n).zip(plane.samples()) {
            *out = to(v);
        }
    }
    let opacity = has_alpha.then(|| {
        let plane = &planes[n];
        let to = level(plane, false);
        plane.samples()[..count].iter().map(|&v| to(v)).collect()
    });
    Ok(Decoded {
        data: samples,
        components: Some(n),
        bits: 8,
        opacity,
        partial: None,
    })
}

/// A JBIG2 image's samples (see the module): its rows as the dictionary sizes them, a row or column the bitmap lacks
/// white.
fn jbig2(
    file: &File,
    params: Option<&Dict>,
    data: &[u8],
    size: (u32, u32),
) -> Result<Decoded, String> {
    let globals = match params.and_then(|p| p.get(b"JBIG2Globals")) {
        None | Some(Object::Null) => None,
        Some(value) => {
            let Ok(Object::Stream(stream)) = file.resolve(value) else {
                return Err("JBIG2Decode image whose /JBIG2Globals is no stream".to_string());
            };
            let bytes = file
                .decode(&stream)
                .and_then(pdf::filter::Decoded::into_bytes)
                .map_err(|e| format!("JBIG2Decode image whose /JBIG2Globals do not read: {e}"))?;
            Some(bytes)
        }
    };
    strictly("JBIG2Decode", |strict| {
        jbig2_read(data, globals.as_deref(), size, strict)
    })
}

/// A JBIG2 image's samples (see the module) after its `globals` segments, decoded `strict` or not.
fn jbig2_read(
    data: &[u8],
    globals: Option<&[u8]>,
    (width, height): (u32, u32),
    strict: bool,
) -> Result<Decoded, Refusal> {
    use hayro_jbig2::{DecodeSettings, Globals, Page};
    let failed = |e: &dyn std::fmt::Debug| Refusal::Decoder(format!("{e:?}"));
    let settings = DecodeSettings { strict };
    let globals = globals
        .map(|g| Globals::new(g, &settings))
        .transpose()
        .map_err(|e| Refusal::Decoder(format!("its /JBIG2Globals: {e:?}")))?;
    let page = Page::new_embedded(data, globals.as_ref(), &settings).map_err(|e| failed(&e))?;
    let stride = width.div_ceil(8) as usize;
    let mut sink = Bitmap {
        width,
        stride,
        x: 0,
        row: vec![0xFF; stride],
        data: Vec::with_capacity(stride * height as usize),
        wanted: height as usize,
    };
    page.decode(&mut sink).map_err(|e| failed(&e))?;
    if sink.x > 0 {
        hayro_jbig2::Decoder::next_line(&mut sink);
    }
    let decoded = sink.data.len() / stride;
    let mut data = sink.data;
    data.resize(stride * height as usize, 0xFF);
    Ok(Decoded::plain(
        data,
        1,
        (decoded < height as usize).then(|| ENDS_EARLY.to_string()),
    ))
}

/// A JBIG2 bitmap packed one bit a sample, 0 black, to the dictionary's width and height.
struct Bitmap {
    width: u32,
    stride: usize,
    /// The next pixel's column in the bitmap's row.
    x: u32,
    row: Vec<u8>,
    data: Vec<u8>,
    wanted: usize,
}

impl hayro_jbig2::Decoder for Bitmap {
    fn push_pixel(&mut self, black: bool) {
        if black && self.x < self.width {
            self.row[self.x as usize / 8] &= !(0x80 >> (self.x % 8));
        }
        self.x += 1;
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        // Called on a whole byte: 8 pixels a chunk, white already in the row.
        if black {
            let start = (self.x / 8) as usize;
            let end = (start + chunk_count as usize).min(self.stride);
            if start < end {
                self.row[start..end].fill(0);
            }
        }
        self.x += 8 * chunk_count;
    }

    fn next_line(&mut self) {
        if self.data.len() / self.stride < self.wanted {
            self.data.extend_from_slice(&self.row);
        }
        self.row.fill(0xFF);
        self.x = 0;
    }
}

/// A CCITT fax image's samples (ISO 32000-2 Table 11's parameters).
fn fax(data: &[u8], params: Option<&Dict>, width: u32, height: u32) -> Result<Decoded, String> {
    use hayro_ccitt::{DecodeSettings, DecoderContext, EncodingMode};
    let int = |key: &[u8], default: i64| {
        params
            .and_then(|p| p.get(key))
            .and_then(Object::as_integer)
            .unwrap_or(default)
    };
    let flag = |key: &[u8], default: bool| match params.and_then(|p| p.get(key)) {
        Some(Object::Bool(b)) => *b,
        _ => default,
    };
    let k = int(b"K", 0);
    let columns = u32::try_from(int(b"Columns", 1728)).map_err(|_| "a negative /Columns")?;
    let rows = u32::try_from(int(b"Rows", i64::from(height))).unwrap_or(height);
    let settings = DecodeSettings {
        columns,
        rows,
        end_of_block: flag(b"EndOfBlock", true),
        end_of_line: flag(b"EndOfLine", false),
        rows_are_byte_aligned: flag(b"EncodedByteAlign", false),
        encoding: match k {
            ..0 => EncodingMode::Group4,
            0 => EncodingMode::Group3_1D,
            _ => EncodingMode::Group3_2D,
        },
        strict: false,
    };
    let black_is_1 = flag(b"BlackIs1", false);
    let stride = width.div_ceil(8) as usize;
    let mut sink = Rows {
        width,
        stride,
        black_is_1,
        data: Vec::with_capacity(stride * height as usize),
        wanted: height as usize,
    };
    let result = hayro_ccitt::decode(data, &mut sink, &mut DecoderContext::new(settings));
    let decoded = sink.data.len() / stride;
    if decoded == 0 {
        return Err(match result {
            Err(e) => format!("CCITTFaxDecode image that does not decode: {e:?}"),
            Ok(_) => "CCITTFaxDecode image with no rows".to_string(),
        });
    }
    let partial = (decoded < height as usize).then(|| ENDS_EARLY.to_string());
    let white = if black_is_1 { 0x00 } else { 0xFF };
    let mut data = sink.data;
    data.resize(stride * height as usize, white);
    Ok(Decoded::plain(data, 1, partial))
}

/// Fax rows packed one bit a sample.
struct Rows {
    width: u32,
    stride: usize,
    black_is_1: bool,
    data: Vec<u8>,
    wanted: usize,
}

impl hayro_ccitt::Decoder for Rows {
    fn push_row(&mut self, changes: &[u32]) {
        if self.data.len() / self.stride >= self.wanted {
            return;
        }
        // Each row starts white; the runs between changes alternate.
        let (white, black) = if self.black_is_1 { (0, 1) } else { (1, 0) };
        let mut row = vec![if white == 1 { 0xFF } else { 0x00 }; self.stride];
        let mut edges = changes
            .iter()
            .map(|&c| c.min(self.width))
            .collect::<Vec<_>>();
        if edges.len() % 2 == 1 {
            edges.push(self.width);
        }
        for &[start, end] in edges.as_chunks::<2>().0 {
            for x in start..end {
                let (byte, bit) = (x as usize / 8, 7 - x % 8);
                if black == 1 {
                    row[byte] |= 1 << bit;
                } else {
                    row[byte] &= !(1 << bit);
                }
            }
        }
        self.data.extend(row);
    }
}
