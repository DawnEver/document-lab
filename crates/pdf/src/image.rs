//! An image `XObject`'s samples (ISO 32000-2 section 8.9.5): its size, color space and depth, and its data through
//! every lossless filter -- still in its image codec (`DCTDecode`, `JPXDecode`, ...) when it has one, for whoever
//! turns it into pixels -- and how a placement shows them ([`Placement`]).

use crate::color::Space;
use crate::error::{Error, Result};
use crate::file::File;
use crate::filter::Codec;
use crate::geometry::Matrix;
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
}

/// A placement matrix entry at most this (points) is zero: the image is square to the page.
const SQUARE: f64 = 1e-6;

/// How an image placed square to the page shows its samples -- turned a quarter or not, mirrored or not: the picture
/// as shown is its rows of samples (`turned`: its columns, each a row), then those rows last to first when
/// `rows_reversed`, and each row last to first when `columns_reversed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// Its rows of samples run down the page, its columns across.
    pub turned: bool,
    /// The rows shown run last to first.
    pub rows_reversed: bool,
    /// Each row shown runs last to first.
    pub columns_reversed: bool,
}

impl Placement {
    /// How `matrix` -- from an image's unit square to page space (y down) -- places its samples; None when it is
    /// skewed: no quarter turn or mirror puts it square to the page.
    #[must_use]
    pub fn of(matrix: &Matrix) -> Option<Self> {
        let [a, b, c, d, _, _] = matrix.0;
        // A sample (row i, column j) sits at u = (j + 1/2) / width, v = 1 - (i + 1/2) / height of the unit square,
        // mapped to x = a u + c v + e, y = b u + d v + f.
        if b.abs() <= SQUARE && c.abs() <= SQUARE {
            Some(Self {
                turned: false,
                rows_reversed: d > 0.0,
                columns_reversed: a < 0.0,
            })
        } else if a.abs() <= SQUARE && d.abs() <= SQUARE {
            Some(Self {
                turned: true,
                rows_reversed: b < 0.0,
                columns_reversed: c > 0.0,
            })
        } else {
            None
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
        })
    }
}
