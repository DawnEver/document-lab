//! ICC color management (ISO 32000-2 section 8.6.5.5, ICC.1): an `ICCBased` space's embedded profile, read by
//! `moxcms`, and its colors converted through it to sRGB at a rendering intent (8.6.5.8).
//!
//! A profile reads when `moxcms` parses it, its color space has the `/N` components the PDF states, and it gives a
//! transform to sRGB at the relative colorimetric intent; otherwise the reason is kept, and its colors have no RGB.
//! Transforms are made when first asked for, one per rendering intent and sample type, and kept with the profile,
//! itself read once per profile object ([`crate::File`] keeps it). An intent whose table the profile lacks converts by
//! its perceptual one, as ICC.1 prescribes (A2B0 stands in for a missing A2B1 or A2B2).
//!
//! No profile stands in for a device space: `DeviceGray`, `DeviceRGB` and `DeviceCMYK` keep the standard's own
//! conversions ([`crate::color`]), as a renderer with color management off converts them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use moxcms::{
    ColorProfile, DataColorSpace, Layout, RenderingIntent, Transform8BitExecutor,
    TransformF64Executor, TransformOptions,
};

/// A rendering intent (8.6.5.8): how colors outside the destination's gamut, and its white, are mapped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Intent {
    /// `AbsoluteColorimetric`.
    AbsoluteColorimetric,
    /// `RelativeColorimetric`: a PDF's default.
    #[default]
    RelativeColorimetric,
    /// `Saturation`.
    Saturation,
    /// `Perceptual`.
    Perceptual,
}

impl Intent {
    /// The intent `name` names; None for a name the standard does not define (the caller then keeps the default).
    #[must_use]
    pub fn named(name: &[u8]) -> Option<Self> {
        match name {
            b"AbsoluteColorimetric" => Some(Self::AbsoluteColorimetric),
            b"RelativeColorimetric" => Some(Self::RelativeColorimetric),
            b"Saturation" => Some(Self::Saturation),
            b"Perceptual" => Some(Self::Perceptual),
            _ => None,
        }
    }

    fn moxcms(self) -> RenderingIntent {
        match self {
            Self::AbsoluteColorimetric => RenderingIntent::AbsoluteColorimetric,
            Self::RelativeColorimetric => RenderingIntent::RelativeColorimetric,
            Self::Saturation => RenderingIntent::Saturation,
            Self::Perceptual => RenderingIntent::Perceptual,
        }
    }
}

/// An `ICCBased` space's profile: its components, and the profile -- or why it does not read.
pub struct Icc {
    /// The components a color of it has: its `/N`.
    pub components: usize,
    /// The profile, or why it does not read.
    pub profile: std::result::Result<Profile, String>,
}

impl fmt::Debug for Icc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Icc")
            .field("components", &self.components)
            .field("profile", &self.profile.as_ref().map(|_| "read"))
            .finish()
    }
}

/// A profile read, and its transforms to sRGB made so far.
pub struct Profile {
    source: ColorProfile,
    layout: Layout,
    srgb: ColorProfile,
    colors: RefCell<HashMap<Intent, Arc<TransformF64Executor>>>,
    samples: RefCell<HashMap<Intent, Arc<Transform8BitExecutor>>>,
}

impl Icc {
    /// The profile `data` of an `ICCBased` space stating `stated` components (`/N`; None: as many as the profile's).
    #[must_use]
    pub fn read(data: &[u8], stated: Option<usize>) -> Self {
        match Profile::read(data, stated) {
            Ok((profile, components)) => Self {
                components,
                profile: Ok(profile),
            },
            Err(why) => Self {
                components: stated.unwrap_or(0),
                profile: Err(why),
            },
        }
    }

    /// The sRGB (0..=1) of `components` at `intent`; None when the profile does not read or there are not as many
    /// components as the space has.
    ///
    /// # Errors
    ///
    /// Refuses a color the profile gives no transform for at `intent` (nor perceptually), saying why.
    pub fn rgb(
        &self,
        components: &[f64],
        intent: Intent,
    ) -> std::result::Result<Option<[f64; 3]>, String> {
        match &self.profile {
            Ok(profile) if components.len() == self.components => {
                profile.rgb(components, intent).map(Some)
            }
            _ => Ok(None),
        }
    }
}

impl Profile {
    /// The profile `data` and its component count, which must be `stated` when stated.
    fn read(data: &[u8], stated: Option<usize>) -> std::result::Result<(Self, usize), String> {
        let profile = ColorProfile::new_from_slice(data)
            .map_err(|e| format!("its ICC profile does not read: {e}"))?;
        let (layout, n) = layout(profile.color_space).ok_or_else(|| {
            format!(
                "its ICC profile is of {:?} data, which this reader does not convert",
                profile.color_space
            )
        })?;
        if let Some(stated) = stated.filter(|s| *s != n) {
            return Err(format!(
                "its ICC profile is of {:?} data ({n} components), not the {stated} its /N states",
                profile.color_space
            ));
        }
        let read = Self {
            source: profile,
            layout,
            srgb: ColorProfile::new_srgb(),
            colors: RefCell::new(HashMap::new()),
            samples: RefCell::new(HashMap::new()),
        };
        read.colors(Intent::RelativeColorimetric)
            .map_err(|e| format!("its ICC profile gives no transform to sRGB: {e}"))?;
        Ok((read, n))
    }

    /// The sRGB of `components` (each 0..=1) at `intent`.
    fn rgb(&self, components: &[f64], intent: Intent) -> std::result::Result<[f64; 3], String> {
        let transform = self.colors(intent)?;
        let source: Vec<f64> = components.iter().map(|c| c.clamp(0.0, 1.0)).collect();
        let mut rgb = [0.0; 3];
        transform
            .transform(&source, &mut rgb[..])
            .map_err(|e| e.to_string())?;
        Ok(rgb.map(|v| v.clamp(0.0, 1.0)))
    }

    /// 8-bit samples, `/N` a sample, as 8-bit sRGB, three bytes a sample, at `intent`.
    ///
    /// # Errors
    ///
    /// Refuses samples the profile gives no transform for at `intent` (nor perceptually), saying why.
    pub fn rgb8(&self, samples: &[u8], intent: Intent) -> std::result::Result<Vec<u8>, String> {
        let transform = made(&self.samples, intent, |options| {
            self.source
                .create_transform_8bit(self.layout, &self.srgb, Layout::Rgb, options)
        })?;
        let n = samples.len() / self.layout.channels();
        let mut rgb = vec![0; n * 3];
        transform
            .transform(&samples[..n * self.layout.channels()], &mut rgb)
            .map_err(|e| e.to_string())?;
        Ok(rgb)
    }

    fn colors(&self, intent: Intent) -> std::result::Result<Arc<TransformF64Executor>, String> {
        made(&self.colors, intent, |options| {
            self.source
                .create_transform_f64(self.layout, &self.srgb, Layout::Rgb, options)
        })
    }
}

/// The transform at `intent` kept in `kept`, made by `make` the first time; by the perceptual intent when the
/// profile has no table for `intent`.
fn made<T: ?Sized>(
    kept: &RefCell<HashMap<Intent, Arc<T>>>,
    intent: Intent,
    make: impl Fn(TransformOptions) -> std::result::Result<Arc<T>, moxcms::CmsError>,
) -> std::result::Result<Arc<T>, String> {
    if let Some(t) = kept.borrow().get(&intent) {
        return Ok(Arc::clone(t));
    }
    let at = |intent: Intent| {
        make(TransformOptions {
            rendering_intent: intent.moxcms(),
            ..TransformOptions::default()
        })
    };
    let transform = at(intent)
        .or_else(|e| {
            if intent == Intent::Perceptual {
                Err(e)
            } else {
                at(Intent::Perceptual)
            }
        })
        .map_err(|e| e.to_string())?;
    kept.borrow_mut().insert(intent, Arc::clone(&transform));
    Ok(transform)
}

/// The sample layout of a profile's device color space, and its components; None for a space of no device colors
/// (CIE, video, hue spaces: their components are not 0..=1 device values).
fn layout(space: DataColorSpace) -> Option<(Layout, usize)> {
    Some(match space {
        DataColorSpace::Gray => (Layout::Gray, 1),
        DataColorSpace::Color2 => (Layout::GrayAlpha, 2),
        DataColorSpace::Rgb | DataColorSpace::Cmy | DataColorSpace::Color3 => (Layout::Rgb, 3),
        DataColorSpace::Cmyk | DataColorSpace::Color4 => (Layout::Rgba, 4),
        DataColorSpace::Color5 => (Layout::Inks5, 5),
        DataColorSpace::Color6 => (Layout::Inks6, 6),
        DataColorSpace::Color7 => (Layout::Inks7, 7),
        DataColorSpace::Color8 => (Layout::Inks8, 8),
        DataColorSpace::Color9 => (Layout::Inks9, 9),
        DataColorSpace::Color10 => (Layout::Inks10, 10),
        DataColorSpace::Color11 => (Layout::Inks11, 11),
        DataColorSpace::Color12 => (Layout::Inks12, 12),
        DataColorSpace::Color13 => (Layout::Inks13, 13),
        DataColorSpace::Color14 => (Layout::Inks14, 14),
        DataColorSpace::Color15 => (Layout::Inks15, 15),
        DataColorSpace::Xyz
        | DataColorSpace::Lab
        | DataColorSpace::Luv
        | DataColorSpace::YCbr
        | DataColorSpace::Yxy
        | DataColorSpace::Hsv
        | DataColorSpace::Hls => return None,
    })
}
