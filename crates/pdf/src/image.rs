//! An image's samples (ISO 32000-2 section 8.9.5) as its dictionary states them -- the one reading of an image
//! dictionary, an image `XObject`'s, an inline image's or a mask's: its size, color space (a JPEG 2000's own when its
//! dictionary states none, [`jpx_space`]), depth, `/Decode`, masks and intent, and its data through every lossless
//! filter -- still in its image codec (`DCTDecode`, `JPXDecode`, ...) when it has one, for whoever turns it into pixels
//! -- where they show as they stand, their sRGB ([`Samples::rgb`]), and how a placement shows them ([`Placement`]).

use std::rc::Rc;

use crate::color::Space;
use crate::error::{Error, Result};
use crate::file::File;
use crate::filter::{self, Codec, Decoded};
use crate::geometry::Matrix;
use crate::icc::{Icc, Intent};
use crate::interpret::Inline;
use crate::object::{Dict, Object, Stream};

/// An image's samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    /// Samples across; 0 when it states none.
    pub width: i64,
    /// Samples down; 0 when it states none.
    pub height: i64,
    /// Its color space; None for a stencil mask (`/ImageMask true`), one bit a sample.
    pub space: Option<Space>,
    /// Bits per component as its dictionary states them: 1 for a stencil mask; 0 when it states none, as a
    /// `JPXDecode` image may (its codec's data then says).
    pub bits: i64,
    /// The data after every lossless filter.
    pub data: Vec<u8>,
    /// The image codec `data` is still encoded in.
    pub codec: Option<Codec>,
    /// Its `/Decode` array, when it states one: how each sample maps into its component's range.
    pub decode: Option<Vec<f64>>,
    /// Its `/SMask`, as stated (an image `XObject`'s; an inline image has none).
    pub soft_mask: Option<Object>,
    /// Its `/Mask`, as stated: a stencil mask image, or the color key ranges it hides.
    pub mask: Option<Object>,
    /// Its `/SMaskInData` (a `JPXDecode` image's, Table 87): 0 (the default) ignores the opacity its codec's data
    /// carries, 1 takes it as the soft mask, 2 that with the colors preblended with it.
    pub in_data: i64,
    /// Whether it asks to be interpolated when magnified (`/Interpolate true`).
    pub interpolate: bool,
    /// Its `/Intent`; relative colorimetric when it states none (the graphics state's where it is placed is not known
    /// to its samples).
    pub intent: Intent,
}

/// What an image stream is read as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// An image, as its dictionary states it.
    Image,
    /// An `/SMask` (11.6.5.3): one gray component whatever space it states.
    SoftMask,
    /// A `/Mask` image (8.9.6.3): a stencil mask whatever it states, as `MuPDF` reads it.
    Stencil,
}

impl Samples {
    /// Whether it states a mask (`/SMask`, `/Mask`, or an `/SMaskInData` other than 0): what shows of it is then more
    /// than its samples.
    #[must_use]
    pub fn masked(&self) -> bool {
        self.soft_mask.is_some() || self.mask.is_some() || self.in_data != 0
    }

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
            || self.masked()
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
            Space::Icc(icc, _) => icc.profile.as_ref().ok()?.rgb8(&levels, self.intent).ok(),
            _ => None,
        }
    }
}

/// The color space a JPEG 2000's own data states, for an image whose dictionary states none (7.4.9, Table 87): its
/// JP2 `colr` box's -- gray, RGB (sYCC too, its samples being converted to RGB), CMYK, or the ICC profile it embeds --
/// else its components' count (1 gray, 3 RGB, 4 CMYK), an opacity channel apart. Read off the header by
/// `hayro-jpeg2000`, which decodes the samples alike; nothing is decoded here.
///
/// # Errors
///
/// Refuses data whose header does not read, and data that states no space this reader reads.
pub fn jpx_space(data: &[u8]) -> Result<Space> {
    use hayro_jpeg2000::{ColorSpace, DecodeSettings, Image};
    let image = Image::new(data, &DecodeSettings::default()).map_err(|e| {
        Error::malformed(format!("JPXDecode image whose header does not read: {e:?}"))
    })?;
    Ok(match image.color_space() {
        ColorSpace::Gray => Space::Gray,
        ColorSpace::RGB => Space::Rgb,
        ColorSpace::CMYK => Space::Cmyk,
        ColorSpace::Icc {
            profile,
            num_channels,
        } => Space::Icc(
            Rc::new(Icc::read(profile, Some(usize::from(*num_channels)))),
            None,
        ),
        ColorSpace::Unknown { num_channels } => {
            return Err(Error::unsupported(format!(
                "JPXDecode image of {num_channels} components in a color space neither its dictionary nor its \
                 data states"
            )));
        }
    })
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
        if stream.dict.get(b"Subtype").and_then(Object::as_name) != Some(b"Image") {
            return Err(Error::invalid(format!("object {num} is no image")));
        }
        self.image_stream(&stream, Role::Image)
    }

    /// The samples of the image `stream` -- an image `XObject`'s, or a mask's (`/SMask`, `/Mask`) -- read in `role`.
    ///
    /// # Errors
    ///
    /// Refuses data that does not decode, and an image stating no color space where it needs one.
    pub fn image_stream(&self, stream: &Stream, role: Role) -> Result<Samples> {
        let decoded = self.decode(stream)?;
        self.samples(&stream.dict, decoded, &Dict::new(), false, role)
    }

    /// The samples of the inline image `inline` (8.9.7): its keys and names abbreviated or not, a color space it
    /// names one of the resources in force where it is placed.
    ///
    /// # Errors
    ///
    /// Refuses data that does not decode, and an image stating no color space where it needs one.
    pub fn inline_image(&self, inline: &Inline) -> Result<Samples> {
        let decoded = filter::decode(&inline.dict, &inline.data, &|o| self.resolve(o))?;
        self.samples(&inline.dict, decoded, &inline.resources, true, Role::Image)
    }

    /// The samples `dict` states of `decoded` in `role` (see [`Samples`]); an `inline` image's keys may be
    /// abbreviated.
    fn samples(
        &self,
        dict: &Dict,
        decoded: Decoded,
        resources: &Dict,
        inline: bool,
        role: Role,
    ) -> Result<Samples> {
        let get = |long: &[u8], short: &[u8]| -> Result<Option<Object>> {
            let value = dict
                .get(long)
                .or_else(|| if inline { dict.get(short) } else { None });
            value.map(|v| self.resolve(v)).transpose()
        };
        let int = |long: &[u8], short: &[u8]| -> Result<Option<i64>> {
            Ok(get(long, short)?.and_then(|v| v.as_integer()))
        };
        let stencil = role == Role::Stencil
            || role == Role::Image && matches!(get(b"ImageMask", b"IM")?, Some(Object::Bool(true)));
        let space = match get(b"ColorSpace", b"CS")? {
            _ if stencil => None,
            _ if role == Role::SoftMask => Some(Space::Gray),
            Some(value) => Some(self.color_space(&value, resources)?),
            // A JPEG 2000 may leave its space to its data (Table 87).
            None if decoded
                .codec
                .as_ref()
                .is_some_and(|c| c.name == b"JPXDecode") =>
            {
                Some(jpx_space(&decoded.data)?)
            }
            None => return Err(Error::malformed("an image states no /ColorSpace")),
        };
        let numbers = |value: Option<Object>| {
            value.and_then(|v| {
                v.as_array()
                    .map(|a| a.iter().filter_map(Object::as_number).collect())
            })
        };
        let intent = match get(b"Intent", b"Intent")? {
            Some(Object::Name(name)) => Intent::named(&name).unwrap_or_default(),
            _ => Intent::default(),
        };
        Ok(Samples {
            width: int(b"Width", b"W")?.unwrap_or(0),
            height: int(b"Height", b"H")?.unwrap_or(0),
            space,
            bits: if stencil {
                1
            } else {
                int(b"BitsPerComponent", b"BPC")?.unwrap_or(0)
            },
            data: decoded.data,
            codec: decoded.codec,
            decode: numbers(get(b"Decode", b"D")?),
            soft_mask: if inline {
                None
            } else {
                get(b"SMask", b"SMask")?
            },
            mask: get(b"Mask", b"Mask")?,
            in_data: int(b"SMaskInData", b"SMaskInData")?.unwrap_or(0),
            interpolate: matches!(get(b"Interpolate", b"I")?, Some(Object::Bool(true))),
            intent,
        })
    }
}
