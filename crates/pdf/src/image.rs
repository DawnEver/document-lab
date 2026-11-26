//! An image `XObject`'s samples (ISO 32000-2 section 8.9.5): its size, color space and depth, and its data through
//! every lossless filter -- still in its image codec (`DCTDecode`, `JPXDecode`, ...) when it has one, for whoever
//! turns it into pixels -- and, where they show as they stand, their sRGB ([`Samples::rgb`]).

use crate::color::Space;
use crate::error::{Error, Result};
use crate::file::File;
use crate::filter::Codec;
use crate::icc::Intent;
use crate::object::{Dict, Object};

/// An image's samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    /// Samples across.
    pub width: i64,
    /// Samples down.
    pub height: i64,
    /// Its color space; None for a stencil mask (`/ImageMask true`), one bit a sample.
    pub space: Option<Space>,
    /// Bits per component.
    pub bits: i64,
    /// The data after every lossless filter.
    pub data: Vec<u8>,
    /// The image codec `data` is still encoded in.
    pub codec: Option<Codec>,
    /// Its `/Decode` array, when it states one: how each sample maps into its component's range.
    pub decode: Option<Vec<f64>>,
    /// Whether it states a mask (`/SMask` or `/Mask`): what shows of it is then more than its samples.
    pub masked: bool,
    /// Its `/Intent`; relative colorimetric when it states none (the graphics state's where it is placed is not known
    /// to its samples).
    pub intent: Intent,
}

impl Samples {
    /// Its samples as 8-bit sRGB, three bytes a sample, rows top-down, when they show as they stand: raw (no codec),
    /// unmasked, mapped as stated by default (no `/Decode`, or `[0 1]` for each component), every row present, and
    /// `DeviceGray` of 1, 2, 4 or 8 bits (its levels spread over 0..=255), `DeviceRGB` of 8, or `ICCBased` of those
    /// depths -- gray, RGB or CMYK -- through its profile at [`Samples::intent`]. None otherwise: Indexed, CMYK, 16
    /// bits, a codec, a mask, another `/Decode`, a profile that does not read are a renderer's to show.
    #[must_use]
    pub fn rgb(&self) -> Option<Vec<u8>> {
        let n = self.space.as_ref()?.components();
        let identity = (0..n).flat_map(|_| [0.0, 1.0]).collect::<Vec<_>>();
        if self.codec.is_some()
            || self.masked
            || self.decode.as_ref().is_some_and(|d| *d != identity)
        {
            return None;
        }
        let (width, height) = (
            usize::try_from(self.width).ok()?,
            usize::try_from(self.height).ok()?,
        );
        let bits = match (n, self.bits) {
            (1, 1 | 2 | 4 | 8) | (3 | 4, 8) => usize::try_from(self.bits).ok()?,
            _ => return None,
        };
        let stride = (width * n * bits).div_ceil(8); // each row ends on a whole byte
        let rows = self.data.get(..stride.checked_mul(height)?)?;
        let levels = if bits == 8 {
            rows.to_vec()
        } else {
            let mask = u8::MAX >> (8 - bits);
            let scale = u8::MAX / mask;
            rows.chunks(stride)
                .flat_map(|row| {
                    (0..width).map(move |i| {
                        let (byte, shift) = (i * bits / 8, 8 - bits - i * bits % 8);
                        (row[byte] >> shift & mask) * scale
                    })
                })
                .collect()
        };
        match self.space.as_ref()? {
            Space::Gray => Some(levels.iter().flat_map(|&g| [g; 3]).collect()),
            Space::Rgb => Some(levels),
            Space::Icc(icc) => icc.profile.as_ref().ok()?.rgb8(&levels, self.intent).ok(),
            _ => None,
        }
    }
}

impl File {
    /// The samples of the image `XObject` object `num`.
    ///
    /// # Errors
    ///
    /// Refuses an object that is no image `XObject`, and data that does not decode.
    pub fn image(&self, num: u32) -> Result<Samples> {
        let Object::Stream(stream) = self.object(num)? else {
            return Err(Error::invalid(format!("object {num} is no stream")));
        };
        let dict = &stream.dict;
        if dict.get(b"Subtype").and_then(Object::as_name) != Some(b"Image") {
            return Err(Error::invalid(format!("object {num} is no image")));
        }
        let int = |key: &[u8]| -> Result<Option<i64>> {
            Ok(self.get(dict, key)?.and_then(|v| v.as_integer()))
        };
        let mask = matches!(self.get(dict, b"ImageMask")?, Some(Object::Bool(true)));
        let space = if mask {
            None
        } else {
            let value = self
                .get(dict, b"ColorSpace")?
                .ok_or_else(|| Error::malformed(format!("image {num} states no /ColorSpace")))?;
            Some(self.color_space(&value, &Dict::new())?)
        };
        let decode = self.get(dict, b"Decode")?.and_then(|v| {
            v.as_array()
                .map(|a| a.iter().filter_map(Object::as_number).collect())
        });
        let masked = self.get(dict, b"SMask")?.is_some() || self.get(dict, b"Mask")?.is_some();
        let intent = match self.get(dict, b"Intent")? {
            Some(Object::Name(name)) => Intent::named(&name).unwrap_or_default(),
            _ => Intent::default(),
        };
        let decoded = self.decode(&stream)?;
        Ok(Samples {
            width: int(b"Width")?.unwrap_or(0),
            height: int(b"Height")?.unwrap_or(0),
            space,
            bits: if mask {
                1
            } else {
                int(b"BitsPerComponent")?.unwrap_or(0)
            },
            data: decoded.data,
            codec: decoded.codec,
            decode,
            masked,
            intent,
        })
    }
}
