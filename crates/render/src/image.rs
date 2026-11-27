//! An image placed on a page to pixels (ISO 32000-2 section 8.9): its samples decoded ([`crate::codec`] for a codec
//! the reader hands over), mapped through `/Decode`, converted to sRGB as the reader converts colors -- `DeviceGray`,
//! `DeviceRGB` and `DeviceCMYK` by the standard's formulas, `ICCBased` through its profile (a profile that does not
//! read: as the device space of its component count), every other space (`Indexed`, `CalGray`/`CalRGB`, `Lab`,
//! `Separation`, `DeviceN`) through `pdf::color::Space::color` -- and masked: a stencil mask (`/ImageMask`) paints
//! the fill color, an `/SMask` (with `/Matte`) or a `/Mask` image gives each sample its alpha, a `/Mask` array of
//! color key ranges hides the samples in them.
//!
//! Placed as viewers place it (measured against `MuPDF`): an image square to the page is grid-fitted -- its box
//! widened to the whole pixels it touches -- and drawn without anti-aliased edges; an image drawn smaller than its
//! samples is first shrunk to its drawn size by averaging the samples each pixel covers; one drawn larger, or turned
//! or skewed, is interpolated (bilinear) -- but magnified more than twice it shows each sample as a block (nearest),
//! unless it asks to be interpolated (`/Interpolate true`).

use std::collections::HashMap;

use pdf::File;
use pdf::color::Space;
use pdf::filter::Codec;
use pdf::geometry::{Matrix, Point};
use pdf::icc::Intent;
use pdf::interpret::ImageEvent;
use pdf::object::{Dict, Object};
use tiny_skia::{IntSize, Pixmap, PixmapRef};

use crate::codec;

/// An image's pixels, decoded and converted, before it is placed: one decoding serves every placement of it.
pub struct Decoded {
    /// Its pixels, premultiplied RGBA, rows top-down.
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    interpolate: bool,
    /// Whether it is a stencil mask, its pixels the fill color it was placed in.
    pub stencil: bool,
    /// What of it is not drawn as the file states it (see `crate::Raster::undrawn`).
    pub notes: Vec<String>,
}

/// An image placed: what to draw where.
pub struct Pixels {
    /// Its pixels shrunk to the size it is drawn at; None when it is drawn at its own size or larger.
    shrunk: Option<Pixmap>,
    /// Whether it is drawn interpolated (bilinear) rather than nearest.
    pub smooth: bool,
    /// The matrix from its pixels to the canvas's.
    pub matrix: Matrix,
    /// Whether it was grid-fitted (square to the page): drawn on whole pixels, edges not anti-aliased.
    pub grid_fitted: bool,
}

impl Pixels {
    /// The pixels to draw, of `decoded` as placed.
    #[must_use]
    pub fn pixmap<'a>(&'a self, decoded: &'a Decoded) -> Option<PixmapRef<'a>> {
        match &self.shrunk {
            Some(pixmap) => Some(pixmap.as_ref()),
            None => PixmapRef::from_bytes(&decoded.rgba, decoded.width, decoded.height),
        }
    }
}

/// The pixels of the image `event` places, in the fill color it is placed in when it is a stencil mask (white when
/// that is a pattern: the renderer paints the pattern through it).
///
/// # Errors
///
/// Refuses what cannot be drawn, in words that name it: a codec not decoded here, data that does not decode, a
/// stencil mask in a color with no RGB, samples whose space gives no RGB.
pub fn decode(file: &File, event: &ImageEvent) -> Result<Decoded, String> {
    let image = Image::of(file, event)?;
    let mut notes = Vec::new();
    let rgba = if image.stencil {
        stencil(&image, event)?
    } else {
        colored(file, &image, &mut notes)?
    };
    if image.partial {
        notes.push("image whose data ends early (drawn as far as it decodes)".to_string());
    }
    Ok(Decoded {
        rgba,
        width: image.width,
        height: image.height,
        interpolate: image.interpolate,
        stencil: image.stencil,
        notes,
    })
}

/// `decoded` placed by `to_pixels`, the matrix from its unit square to the canvas.
///
/// # Errors
///
/// Refuses an image too large to draw.
pub fn place(decoded: &Decoded, to_pixels: Matrix) -> Result<Pixels, String> {
    let (width, height) = (decoded.width, decoded.height);
    let (unit, grid_fitted) = fit(to_pixels);
    let across = unit.0[0].hypot(unit.0[1]);
    let down = unit.0[2].hypot(unit.0[3]);
    let target = |samples: u32, drawn: f64| {
        let drawn = if grid_fitted {
            drawn.round()
        } else {
            drawn.ceil()
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "at least 1 and below the sample count"
        )]
        if drawn < f64::from(samples) {
            drawn.max(1.0) as u32
        } else {
            samples
        }
    };
    let (w, h) = (target(width, across), target(height, down));
    let shrunk = if w < width || h < height {
        let size = IntSize::from_wh(w, h).ok_or("an image of no samples")?;
        let rgba = shrink(&decoded.rgba, (width, height), (w, h));
        Some(Pixmap::from_vec(rgba, size).ok_or("an image too large to draw")?)
    } else {
        None
    };
    // A canvas pixel whose center falls on the line between two samples takes the first (the lower index), as
    // MuPDF's fixed-point sampling does: the samples sit a 1024th of a sample further on.
    let tie = 1.0 / 1024.0;
    let matrix = Matrix([1.0, 0.0, 0.0, 1.0, tie, tie])
        .then(Matrix([
            1.0 / f64::from(w),
            0.0,
            0.0,
            -1.0 / f64::from(h),
            0.0,
            1.0,
        ]))
        .then(unit);
    Ok(Pixels {
        shrunk,
        smooth: smooth(
            across / f64::from(w),
            down / f64::from(h),
            grid_fitted,
            decoded.interpolate,
        ),
        matrix,
        grid_fitted,
    })
}

/// Whether an image whose pixels are drawn `across` x `down` canvas pixels each is interpolated (see the module).
fn smooth(across: f64, down: f64, grid_fitted: bool, interpolate: bool) -> bool {
    let magnified = across > 1.0 || down > 1.0;
    let blocks = across > 2.0 || down > 2.0;
    (magnified || !grid_fitted) && (interpolate || !blocks)
}

/// The matrix from the unit square to the canvas: `to_pixels` itself, or -- for an image square to the page --
/// widened to the whole pixels its box touches; and whether it was.
fn fit(to_pixels: Matrix) -> (Matrix, bool) {
    const SNAP: f64 = 1e-3;
    let [xx, xy, yx, yy, ..] = to_pixels.0;
    let size = xx.abs() + xy.abs() + yx.abs() + yy.abs();
    let tiny = |v: f64| v.abs() <= 1e-6 * size;
    let upright = tiny(xy) && tiny(yx);
    if !(upright || (tiny(xx) && tiny(yy))) || size == 0.0 {
        return (to_pixels, false);
    }
    let corners = [(0.0, 0.0), (1.0, 1.0)].map(|(x, y)| to_pixels.apply(Point { x, y }));
    let span = |p: f64, q: f64| {
        let low = (p.min(q) + SNAP).floor();
        let high = (p.max(q) - SNAP).ceil().max(low + 1.0);
        (low, high)
    };
    let (x0, x1) = span(corners[0].x, corners[1].x);
    let (y0, y1) = span(corners[0].y, corners[1].y);
    // Each axis keeps its direction; its length becomes the whole pixels it spans.
    let axis =
        |v: f64, low: f64, high: f64| (v.signum() * (high - low), if v > 0.0 { low } else { high });
    let fitted = if upright {
        let ((across, left), (down, top)) = (axis(xx, x0, x1), axis(yy, y0, y1));
        [across, 0.0, 0.0, down, left, top]
    } else {
        let ((across, top), (down, left)) = (axis(xy, y0, y1), axis(yx, x0, x1));
        [0.0, across, down, 0.0, left, top]
    };
    (Matrix(fitted), true)
}

/// An image's dictionary read: what its samples are.
struct Image {
    width: u32,
    height: u32,
    stencil: bool,
    bits: u32,
    space: Option<Space>,
    decode: Option<Vec<f64>>,
    interpolate: bool,
    intent: Intent,
    /// Its samples, rows top-down, each row on a whole byte.
    data: Vec<u8>,
    partial: bool,
    /// `/SMask`, `/Mask` (an `XObject`'s), as stated.
    soft_mask: Option<Object>,
    mask: Option<Object>,
}

impl Image {
    fn of(file: &File, event: &ImageEvent) -> Result<Self, String> {
        let failed = |e: pdf::Error| e.to_string();
        let (dict, decoded, resources) = match (&event.inline, event.object) {
            (Some(inline), _) => (
                inline.dict.clone(),
                pdf::filter::decode(&inline.dict, &inline.data, &|o| file.resolve(o))
                    .map_err(failed)?,
                inline.resources.clone(),
            ),
            (None, Some(num)) => {
                let Object::Stream(stream) = file.object(num).map_err(failed)? else {
                    return Err(format!("image object {num} is no stream"));
                };
                let decoded = file.decode(&stream).map_err(failed)?;
                (stream.dict, decoded, Dict::new())
            }
            (None, None) => return Err("an image with neither data nor object".to_string()),
        };
        let inline = event.inline.is_some();
        let get = |long: &[u8], short: &[u8]| -> Result<Option<Object>, String> {
            let value = dict
                .get(long)
                .or_else(|| if inline { dict.get(short) } else { None });
            value.map(|v| file.resolve(v)).transpose().map_err(failed)
        };
        let count = |long: &[u8], short: &[u8]| -> Result<u32, String> {
            get(long, short)?
                .and_then(|v| v.as_integer())
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| format!("an image states no /{}", String::from_utf8_lossy(long)))
        };
        let (width, height) = (count(b"Width", b"W")?, count(b"Height", b"H")?);
        let stencil = matches!(get(b"ImageMask", b"IM")?, Some(Object::Bool(true)));
        let space = if stencil {
            None
        } else {
            let value = get(b"ColorSpace", b"CS")?.ok_or("an image states no /ColorSpace")?;
            Some(file.color_space(&value, &resources).map_err(failed)?)
        };
        let decode = get(b"Decode", b"D")?.and_then(|v| {
            v.as_array()
                .map(|a| a.iter().filter_map(Object::as_number).collect())
        });
        let interpolate = matches!(get(b"Interpolate", b"I")?, Some(Object::Bool(true)));
        let intent = match get(b"Intent", b"Intent")? {
            Some(Object::Name(n)) => Intent::named(&n).unwrap_or_default(),
            _ => Intent::default(),
        };
        let (mut bits, mut data, mut partial) = (1, decoded.data, false);
        if !stencil {
            bits = count(b"BitsPerComponent", b"BPC").or_else(|e| {
                if decoded.codec.is_some() {
                    Ok(8)
                } else {
                    Err(e)
                }
            })?;
        }
        let mut space = space;
        if let Some(stated) = &decoded.codec {
            let found;
            (found, space) = from_codec(stated, &data, (width, height), space)?;
            if found.components.is_some() {
                bits = 8;
            }
            (data, partial) = (found.data, found.partial);
        }
        if !matches!(bits, 1 | 2 | 4 | 8 | 16) {
            return Err(format!("an image of {bits} bits a component"));
        }
        let (soft_mask, mask) = if inline {
            (None, get(b"Mask", b"Mask")?)
        } else {
            (get(b"SMask", b"SMask")?, get(b"Mask", b"Mask")?)
        };
        Ok(Self {
            width,
            height,
            stencil,
            bits,
            space,
            decode,
            interpolate,
            intent,
            data,
            partial,
            soft_mask,
            mask,
        })
    }

    /// Components a sample.
    fn components(&self) -> usize {
        self.space.as_ref().map_or(1, Space::components)
    }

    /// The samples' bytes, rows top-down, when they are 8 bits each (no row ends mid-byte); refused when short.
    fn bytes(&self) -> Result<&[u8], String> {
        let length = self.width as usize * self.height as usize * self.components();
        self.data.get(..length).ok_or_else(|| {
            format!(
                "an image's data holds {} of its {length} bytes",
                self.data.len()
            )
        })
    }

    /// Each sample's raw value, `components()` a sample, rows top-down; refused when the data is short.
    fn samples(&self) -> Result<Vec<u16>, String> {
        let n = self.components();
        let (w, h) = (self.width as usize, self.height as usize);
        let bits = self.bits as usize;
        let stride = (w * n * bits).div_ceil(8);
        let rows = self.data.get(..stride * h).ok_or_else(|| {
            format!(
                "an image's data holds {} of its {} bytes",
                self.data.len(),
                stride * h
            )
        })?;
        let mut out = Vec::with_capacity(w * h * n);
        for row in rows.chunks_exact(stride) {
            match bits {
                8 => out.extend(row[..w * n].iter().map(|&b| u16::from(b))),
                16 => out.extend(
                    row[..w * n * 2]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&pair| u16::from_be_bytes(pair)),
                ),
                _ => {
                    let mask = (1u16 << bits) - 1;
                    out.extend((0..w * n).map(|i| {
                        let (byte, shift) = (i * bits / 8, 8 - bits - i * bits % 8);
                        u16::from(row[byte]) >> shift & mask
                    }));
                }
            }
        }
        Ok(out)
    }
}

/// A stencil mask's pixels: the fill color where its samples paint (0, unless `/Decode [1 0]`); white when the
/// fill is a pattern, which the renderer paints through the stencil.
fn stencil(image: &Image, event: &ImageEvent) -> Result<Vec<u8>, String> {
    let rgb = match (&event.fill.pattern, event.fill.rgb) {
        (Some(_), _) => [1.0; 3], // what a pattern paints through it is the renderer's
        (None, Some(rgb)) => rgb,
        (None, None) => {
            return Err(format!(
                "stencil mask in a {} color with no RGB",
                String::from_utf8_lossy(&event.fill.space)
            ));
        }
    };
    let paints = u16::from(
        image
            .decode
            .as_deref()
            .is_some_and(|d| d.first() > d.get(1)),
    );
    let [r, g, b] = rgb.map(level);
    Ok(image
        .samples()?
        .iter()
        .flat_map(|&s| if s == paints { [r, g, b, 255] } else { [0; 4] })
        .collect())
}

/// A color image's pixels, premultiplied RGBA: its samples converted, then masked.
fn colored(file: &File, image: &Image, notes: &mut Vec<String>) -> Result<Vec<u8>, String> {
    let space = image.space.as_ref().ok_or("an image with no color space")?;
    let n = space.components();
    if n == 0 {
        return Err(format!(
            "an image in a {} space",
            String::from_utf8_lossy(&space.family())
        ));
    }
    let keyed = matches!(image.mask, Some(Object::Array(_)));
    let plain = image.bits == 8 && maps(space, 8, image.decode.as_deref()).is_none();
    let (levels, samples) = if plain && !keyed {
        (image.bytes()?.to_vec(), Vec::new()) // each sample's byte is its level
    } else {
        let samples = image.samples()?;
        (
            levels(&samples, space, image.bits, image.decode.as_deref()),
            samples,
        )
    };
    let rgb = rgb(space, &levels, image.intent, notes)?;
    let count = image.width as usize * image.height as usize;
    if image.mask.is_none() && image.soft_mask.is_none() {
        // Opaque: each pixel as it is, its alpha whole.
        return Ok(rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect());
    }
    let mut alpha = vec![255u8; count];
    if let Some(Object::Array(ranges)) = &image.mask {
        let ranges: Vec<f64> = ranges.iter().filter_map(Object::as_number).collect();
        if ranges.len() == 2 * n {
            for (a, sample) in alpha.iter_mut().zip(samples.chunks_exact(n)) {
                let keyed = sample.iter().enumerate().all(|(k, &s)| {
                    f64::from(s) >= ranges[2 * k] && f64::from(s) <= ranges[2 * k + 1]
                });
                if keyed {
                    *a = 0;
                }
            }
        }
    }
    let mut matte = None;
    for (value, soft) in [(&image.soft_mask, true), (&image.mask, false)] {
        let Some(Object::Stream(stream)) = value else {
            continue;
        };
        let (mask, stated_matte) = mask_alpha(file, stream, soft, (image.width, image.height))?;
        for (a, m) in alpha.iter_mut().zip(mask) {
            *a = mul(*a, m);
        }
        if soft {
            matte = stated_matte;
        }
    }
    let matte = match matte {
        Some(components) if components.len() == n => space
            .color(components, image.intent)
            .ok()
            .and_then(|c| c.rgb)
            .map(|rgb| rgb.map(|v| v * 255.0)),
        _ => None,
    };
    let mut out = Vec::with_capacity(count * 4);
    for (&pixel, &a) in rgb.as_chunks::<3>().0.iter().zip(&alpha) {
        let mut p = pixel;
        if let Some(m) = matte
            && a > 0
        {
            // A preblended sample: c = m + (c' - m) / alpha (11.6.5.3).
            for (c, m) in p.iter_mut().zip(m) {
                let unmatted = m + (f64::from(*c) - m) * 255.0 / f64::from(a);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to a byte"
                )]
                {
                    *c = unmatted.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out.extend([mul(p[0], a), mul(p[1], a), mul(p[2], a), a]);
    }
    Ok(out)
}

/// `a * b / 255`, rounded.
fn mul(a: u8, b: u8) -> u8 {
    let p = u16::from(a) * u16::from(b) + 128;
    u8::try_from((p + (p >> 8)) >> 8).unwrap_or(u8::MAX)
}

/// A component's value in 0..=1 as a byte.
fn level(v: f64) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to a byte"
    )]
    {
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

/// A soft mask's (`soft`) or a stencil `/Mask`'s alpha, sample by sample, resampled (nearest) to `size`; and a soft
/// mask's `/Matte`.
fn mask_alpha(
    file: &File,
    stream: &pdf::Stream,
    soft: bool,
    (width, height): (u32, u32),
) -> Result<(Vec<u8>, Option<Vec<f64>>), String> {
    let failed = |e: pdf::Error| e.to_string();
    let dict = &stream.dict;
    let int = |key: &[u8]| -> Result<u32, String> {
        file.get(dict, key)
            .map_err(failed)?
            .and_then(|v| v.as_integer())
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| format!("a mask states no /{}", String::from_utf8_lossy(key)))
    };
    let (w, h) = (int(b"Width")?, int(b"Height")?);
    let bits = if soft { int(b"BitsPerComponent")? } else { 1 };
    let decoded = file.decode(stream).map_err(failed)?;
    let data = match &decoded.codec {
        Some(c) => codec::decode(c, &decoded.data, w, h)?.data,
        None => decoded.data,
    };
    let decode: Option<Vec<f64>> = file.get(dict, b"Decode").map_err(failed)?.and_then(|v| {
        v.as_array()
            .map(|a| a.iter().filter_map(Object::as_number).collect())
    });
    let mask = Image {
        width: w,
        height: h,
        stencil: !soft,
        bits,
        space: Some(Space::Gray),
        decode: decode.clone(),
        interpolate: false,
        intent: Intent::default(),
        data,
        partial: false,
        soft_mask: None,
        mask: None,
    };
    let samples = mask.samples()?;
    let alpha: Vec<u8> = if soft {
        levels(&samples, &Space::Gray, bits, decode.as_deref())
    } else {
        // A stencil: a sample of 1 (0 under `/Decode [1 0]`) masks its area out.
        let masks = u16::from(!decode.as_deref().is_some_and(|d| d.first() > d.get(1)));
        samples
            .iter()
            .map(|&s| if s == masks { 0 } else { 255 })
            .collect()
    };
    let matte = file.get(dict, b"Matte").map_err(failed)?.and_then(|v| {
        v.as_array()
            .map(|a| a.iter().filter_map(Object::as_number).collect())
    });
    if (w, h) == (width, height) {
        return Ok((alpha, matte));
    }
    let resampled = (0..height)
        .flat_map(|y| {
            let sy = u64::from(y) * u64::from(h) / u64::from(height);
            let alpha = &alpha;
            (0..width).map(move |x| {
                let sx = u64::from(x) * u64::from(w) / u64::from(width);
                alpha[usize::try_from(sy * u64::from(w) + sx).expect("within the mask")]
            })
        })
        .collect();
    Ok((resampled, matte))
}

/// Each component's range in `space` (8.6.5-8.6.6): `L*` 0..=100 and `a*`, `b*` the space's; an index 0..=hival;
/// every other 0..=1.
fn range(space: &Space, k: usize) -> [f64; 2] {
    match (space, k) {
        (Space::Lab(_), 0) => [0.0, 100.0],
        (Space::Lab(lab), _) => [lab.range[2 * k - 2], lab.range[2 * k - 1]],
        (Space::Indexed(base, table), _) => {
            #[expect(clippy::cast_precision_loss, reason = "a table of at most 256 entries")]
            let hival = (table.len() / base.components().max(1)).saturating_sub(1) as f64;
            [0.0, hival]
        }
        _ => [0.0, 1.0],
    }
}

/// Each sample's components as bytes: an index as itself, any other component its place in its range scaled to
/// 0..=255 -- through `/Decode` (8.9.5.2).
fn levels(samples: &[u16], space: &Space, bits: u32, decode: Option<&[f64]>) -> Vec<u8> {
    let n = space.components().max(1);
    let Some(maps) = maps(space, bits, decode) else {
        #[expect(clippy::cast_possible_truncation, reason = "8-bit samples")]
        return samples.iter().map(|&s| s as u8).collect();
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to a byte"
    )]
    samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let [offset, slope] = maps[i % n];
            (offset + f64::from(s) * slope).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// How each component's sample maps to its level ([`levels`]): an offset and a slope; None when every sample is
/// its own level (8 bits mapped as stated by default, not an index).
fn maps(space: &Space, bits: u32, decode: Option<&[f64]>) -> Option<Vec<[f64; 2]>> {
    let n = space.components().max(1);
    let top = f64::from((1u32 << bits) - 1);
    let indexed = matches!(space, Space::Indexed(..));
    let maps: Vec<[f64; 2]> = (0..n)
        .map(|k| {
            let [low, high] = range(space, k);
            let [d0, d1] = match decode {
                Some(d) if d.len() >= 2 * n => [d[2 * k], d[2 * k + 1]],
                _ if indexed => [0.0, top],
                _ => [low, high],
            };
            // A sample s is the value d0 + s (d1 - d0) / top; as a byte, (value - low) / (high - low) * 255.
            if indexed {
                [d0, (d1 - d0) / top]
            } else {
                let span = (high - low).max(f64::EPSILON);
                [(d0 - low) / span * 255.0, (d1 - d0) / top / span * 255.0]
            }
        })
        .collect();
    let identity = !indexed
        && bits == 8
        && maps
            .iter()
            .all(|[offset, slope]| offset.abs() < 1e-9 && (slope - 1.0).abs() < 1e-9);
    (!identity).then_some(maps)
}

/// The samples `data` holds in `stated` (a codec) decoded, and the space they are in: a JPEG's own component count
/// governs its samples, and a device space follows it.
fn from_codec(
    stated: &Codec,
    data: &[u8],
    (width, height): (u32, u32),
    space: Option<Space>,
) -> Result<(codec::Decoded, Option<Space>), String> {
    let found = codec::decode(stated, data, width, height)?;
    let space = match (found.components, space) {
        (Some(n), Some(s)) if n != s.components() => Some(match (n, &s) {
            (1, Space::Gray | Space::Rgb | Space::Cmyk) => Space::Gray,
            (3, Space::Gray | Space::Rgb | Space::Cmyk) => Space::Rgb,
            (4, Space::Gray | Space::Rgb | Space::Cmyk) => Space::Cmyk,
            _ => {
                return Err(format!(
                    "DCTDecode image of {n} components in a space of {}",
                    s.components()
                ));
            }
        }),
        (_, space) => space,
    };
    Ok((found, space))
}

/// Samples' components as bytes ([`levels`]) to 8-bit sRGB, three bytes a sample.
fn rgb(
    space: &Space,
    levels: &[u8],
    intent: Intent,
    notes: &mut Vec<String>,
) -> Result<Vec<u8>, String> {
    let n = space.components();
    Ok(match space {
        Space::Gray => levels.iter().flat_map(|&g| [g; 3]).collect(),
        Space::Rgb => levels.to_vec(),
        Space::Cmyk => levels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[c, m, y, k]| [c, m, y].map(|v| 255 - v.saturating_add(k)))
            .collect(),
        Space::Icc(icc) => match &icc.profile {
            Ok(profile) => profile.rgb8(levels, intent)?,
            Err(why) => {
                notes.push(format!(
                    "image in an ICCBased space drawn as its device space ({why})"
                ));
                let device = match n {
                    1 => Space::Gray,
                    3 => Space::Rgb,
                    4 => Space::Cmyk,
                    _ => {
                        return Err(format!(
                            "image in an ICCBased space of {n} components whose profile does not read"
                        ));
                    }
                };
                return rgb(&device, levels, intent, notes);
            }
        },
        _ => through(space, levels, intent)?,
    })
}

/// Samples converted one distinct color at a time by `space.color` (a table of 256 for one component).
fn through(space: &Space, levels: &[u8], intent: Intent) -> Result<Vec<u8>, String> {
    let n = space.components();
    if n > 8 {
        return Err(format!(
            "image in a {} space of {n} components",
            String::from_utf8_lossy(&space.family())
        ));
    }
    let value = |k: usize, l: u8| {
        let [low, high] = range(space, k);
        if matches!(space, Space::Indexed(..)) {
            f64::from(l).min(high) // an index past the table is its last entry's
        } else {
            low + f64::from(l) / 255.0 * (high - low)
        }
    };
    let convert = |key: &[u8]| -> Result<[u8; 3], String> {
        let components = key.iter().enumerate().map(|(k, &l)| value(k, l)).collect();
        let color = space.color(components, intent).map_err(|e| e.to_string())?;
        color.rgb.map(|rgb| rgb.map(level)).ok_or_else(|| {
            format!(
                "image in a {} space whose colors have no RGB",
                String::from_utf8_lossy(&space.family())
            )
        })
    };
    let mut out = Vec::with_capacity(levels.len() / n * 3);
    if n == 1 {
        let table = (0..=255u8)
            .map(|l| convert(&[l]))
            .collect::<Result<Vec<_>, _>>()?;
        out.extend(levels.iter().flat_map(|&l| table[usize::from(l)]));
    } else {
        let mut seen: HashMap<u64, [u8; 3]> = HashMap::new();
        for key in levels.chunks_exact(n) {
            let packed = key.iter().fold(0u64, |acc, &l| acc << 8 | u64::from(l));
            let rgb = if let Some(rgb) = seen.get(&packed) {
                *rgb
            } else {
                let rgb = convert(key)?;
                seen.insert(packed, rgb);
                rgb
            };
            out.extend(rgb);
        }
    }
    Ok(out)
}

/// Premultiplied RGBA pixels `from` (width, height) shrunk to `to` by averaging what each pixel covers.
fn shrink(rgba: &[u8], from: (u32, u32), to: (u32, u32)) -> Vec<u8> {
    let across = weights(from.0, to.0);
    let down = weights(from.1, to.1);
    let (w, tw) = (from.0 as usize, to.0 as usize);
    let row = |y: usize| -> Vec<f32> {
        let source = &rgba[y * w * 4..(y + 1) * w * 4];
        let mut out = vec![0.0f32; tw * 4];
        for (j, (start, ws)) in across.iter().enumerate() {
            for (k, &wt) in ws.iter().enumerate() {
                let p = &source[(start + k) * 4..(start + k) * 4 + 4];
                for c in 0..4 {
                    out[j * 4 + c] += wt * f32::from(p[c]);
                }
            }
        }
        out
    };
    let mut out = Vec::with_capacity(tw * to.1 as usize * 4);
    let mut cached: Option<(usize, Vec<f32>)> = None;
    for (start, ws) in &down {
        let mut acc = vec![0.0f32; tw * 4];
        for (k, &wt) in ws.iter().enumerate() {
            let y = start + k;
            if cached.as_ref().is_none_or(|(at, _)| *at != y) {
                cached = Some((y, row(y)));
            }
            let source = &cached.as_ref().expect("cached").1;
            for (a, s) in acc.iter_mut().zip(source) {
                *a += wt * s;
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to a byte"
        )]
        out.extend(acc.iter().map(|v| v.round().clamp(0.0, 255.0) as u8));
    }
    out
}

/// For each of `to` pixels along an axis of `from` samples: its first sample and each covered sample's share.
fn weights(from: u32, to: u32) -> Vec<(usize, Vec<f32>)> {
    let ratio = f64::from(from) / f64::from(to);
    (0..to)
        .map(|j| {
            let (x0, x1) = (f64::from(j) * ratio, f64::from(j + 1) * ratio);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a sample index"
            )]
            let (first, last) = (x0.floor() as usize, (x1.ceil() as usize).min(from as usize));
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                reason = "a share"
            )]
            let shares = (first..last)
                .map(|s| (((s + 1) as f64).min(x1) - (s as f64).max(x0)) as f32 / ratio as f32)
                .collect();
            (first, shares)
        })
        .collect()
}
