//! Image codecs to samples: the data the reader hands over still encoded (`pdf::filter::Codec`) decoded to the
//! samples an image's dictionary describes.
//!
//! - `DCTDecode` (JPEG) by `zune-jpeg`: gray, RGB or CMYK samples, 8 bits each, by the JPEG's own component count
//!   (YCbCr and YCCK converted by the JPEG's transform). CMYK samples are taken as stored: a JPEG an Adobe
//!   application wrote inverted states `/Decode [1 0 1 0 1 0 1 0]` in its PDF, which the image's mapping applies
//!   (as Acrobat and `MuPDF` read it; inverting by the `APP14` marker as well drew such images dark).
//! - `CCITTFaxDecode` (Group 3 one- and two-dimensional, Group 4) by `hayro-ccitt`: one bit a sample, 0 black
//!   unless `/BlackIs1`.
//! - `JPXDecode` (JPEG 2000) and `JBIG2Decode` are not decoded in stage 1: refused by name.

use pdf::filter::Codec;
use pdf::object::{Dict, Object};

/// Samples a codec decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// The samples, rows top-down, each row ending on a whole byte.
    pub data: Vec<u8>,
    /// Components a sample, when the codec states them (a JPEG's); None: as the dictionary says.
    pub components: Option<usize>,
    /// Whether the data ended before every row the image states was decoded (the rows after are white).
    pub partial: bool,
}

/// `data` decoded from `codec` for an image `width` x `height` samples.
///
/// # Errors
///
/// Refuses a codec not decoded here (JPX, JBIG2) and data that does not decode, saying which.
pub fn decode(codec: &Codec, data: &[u8], width: u32, height: u32) -> Result<Decoded, String> {
    match codec.name.as_slice() {
        b"DCTDecode" => jpeg(data),
        b"CCITTFaxDecode" => fax(data, codec.params.as_ref(), width, height),
        other => Err(format!(
            "{} image (not decoded in stage 1)",
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
        data: samples,
        components: Some(components),
        partial: false,
    })
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
    let partial = decoded < height as usize;
    let white = if black_is_1 { 0x00 } else { 0xFF };
    let mut data = sink.data;
    data.resize(stride * height as usize, white);
    Ok(Decoded {
        data,
        components: None,
        partial,
    })
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
