//! Color spaces and colors (ISO 32000-2 section 8.6): the components as the file states them and their RGB, for a
//! `Pattern` color which kind of pattern it paints (8.7).
//!
//! RGB is sRGB in 0..=1:
//!
//! - `DeviceGray`, `DeviceRGB` and `DeviceCMYK` by the standard's own conversions (10.4.2), no profile standing in
//!   for them (a renderer with color management off converts them so): gray is each of red, green and blue; CMYK is
//!   `red = 1 - min(1, cyan + black)` and alike.
//! - `ICCBased` (8.6.5.5) through its embedded profile, at the rendering intent in force ([`crate::icc`]). A
//!   profile that does not read converts through the space's `/Alternate` when it states one (as 8.6.5.5 bids a
//!   reader that cannot use the profile); else its colors keep their components with no RGB, and say why
//!   ([`Space::unread`]).
//! - `CalGray`, `CalRGB` (8.6.5.2, 8.6.5.3) and `Lab` (8.6.5.4) by the CIE formulas, to CIE XYZ under the space's
//!   `/WhitePoint` -- `CalGray` its gray raised to its `/Gamma`, `CalRGB` its components each raised to its `/Gamma`
//!   then through its `/Matrix`, `Lab` L\*a\*b\* by CIE 15 -- adapted to sRGB's D65 white by the Bradford
//!   transform, then to sRGB (IEC 61966-2-1: its matrix, then its transfer curve). That is what an ICC matrix/TRC
//!   profile made of the space's parameters gives at a colorimetric intent; these spaces have no gamut mapping to
//!   choose, so every intent converts alike (absolute colorimetric too: read as relative, their white mapped to
//!   sRGB's).
//! - `Separation` and `DeviceN` (8.6.6.4, 8.6.6.5) through their tint transform ([`crate::function`]) into the
//!   alternate space, then as that space converts. The colorant `None` paints nothing (a color of it has no RGB,
//!   and [`Space::paints`] says so); `All` -- registration, every separation -- is its tint transform's color like
//!   any other.
//! - `Indexed` through its lookup table into its base space.
//!
//! A space this reader cannot read keeps its components with no RGB.

use std::rc::Rc;

use crate::error::{Error, Result};
use crate::file::File;
use crate::filter::Decoded;
use crate::function::Function;
use crate::geometry::Matrix;
use crate::icc::{Icc, Intent};
use crate::object::{Dict, Object};

/// A color space, resolved far enough to count its components and convert them to RGB.
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// `DeviceGray`.
    Gray,
    /// `DeviceRGB`.
    Rgb,
    /// `DeviceCMYK`.
    Cmyk,
    /// `ICCBased`: its profile, read once per profile object, and its `/Alternate` when the profile does not read
    /// and the space states one: its colors then convert through that space.
    Icc(Rc<Icc>, Option<Box<Space>>),
    /// `CalGray` or `CalRGB`.
    Cal(Cal),
    /// `Lab`.
    Lab(Lab),
    /// `Indexed`: a base space and its lookup table.
    Indexed(Box<Space>, Vec<u8>),
    /// `Separation` or `DeviceN`: colorants, converted through a tint transform.
    Tint(Rc<Tint>),
    /// A space this reader cannot read (its family name, or the name it was set by, and its component count): its
    /// components are kept, with no RGB.
    Other(Vec<u8>, usize),
    /// `Pattern`: painted with a pattern, not a color; the space an uncolored tiling pattern's tint is in, when the
    /// space states one (`[/Pattern base]`).
    Pattern(Option<Box<Space>>),
}

impl PartialEq for Icc {
    /// The same profile object: one read is shared by every space naming it.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

/// A `CalGray` or `CalRGB` space's parameters (8.6.5.2, 8.6.5.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cal {
    /// The diffuse white point, CIE XYZ.
    pub white: [f64; 3],
    /// Each component's gamma (`CalGray`'s one first).
    pub gamma: [f64; 3],
    /// The CIE XYZ of each component at full value: `[XA YA ZA]` for the first (`CalGray`'s: its white point, its
    /// gray being neutral).
    pub matrix: [[f64; 3]; 3],
    /// Whether it is `CalGray` (one component).
    pub gray: bool,
}

/// A `Lab` space's parameters (8.6.5.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lab {
    /// The diffuse white point, CIE XYZ.
    pub white: [f64; 3],
    /// The ranges of a\* and b\*: `[a_min, a_max, b_min, b_max]`.
    pub range: [f64; 4],
}

/// A `Separation` (one colorant) or `DeviceN` (several) space: its colorants, the space it stands in for them on a
/// device that has not got them, and the tint transform into that space.
#[derive(Debug, Clone, PartialEq)]
pub struct Tint {
    /// Whether it is `DeviceN` (else `Separation`).
    pub device_n: bool,
    /// The colorant names, one per component (`None` paints nothing; `All` is every separation).
    pub colorants: Vec<Vec<u8>>,
    /// The alternate space.
    pub alternate: Space,
    /// The tint transform: the components in, the alternate space's out.
    pub transform: Function,
}

/// A color: its space's family name, its components, and its RGB when the space gives one.
#[derive(Debug, Clone, PartialEq)]
pub struct Color {
    /// The color space family (`DeviceRGB`, `ICCBased`, `Separation`, ...).
    pub space: Vec<u8>,
    /// The components, as set (an uncolored tiling pattern's tint; none for a colored pattern).
    pub components: Vec<f64>,
    /// Red, green and blue in 0..=1, when the space gives them (see the module).
    pub rgb: Option<[f64; 3]>,
    /// The pattern painted with, when the space is `Pattern`: no one color then.
    pub pattern: Option<Pattern>,
}

/// What a `Pattern` color paints an area with (8.7): the pattern, where its space lies on the page, and what it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    /// Which kind of pattern it is.
    pub kind: PatternKind,
    /// From pattern space to page space: the pattern's `/Matrix`, then the matrix of the default space of the
    /// content stream it is painted in (8.7.2: the page's, a Form's, a glyph procedure's), not the CTM.
    pub matrix: Matrix,
    /// The pattern object: a tiling pattern's stream, a shading pattern's dictionary.
    pub object: Rc<Object>,
    /// The rendering intent a shading pattern's colors convert at: the one in force where the color was set.
    pub intent: Intent,
}

/// The kind of a pattern (8.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// A tiling pattern (`PatternType` 1): a cell repeated across the area.
    Tiling {
        /// Uncolored (`PaintType` 2): its cell is painted in the tint the color states, in the pattern space's base.
        uncolored: bool,
    },
    /// A shading pattern (`PatternType` 2): a gradient across the area.
    Shading,
}

impl Color {
    /// Black in `DeviceGray`: a graphics state's initial color (8.4.1).
    #[must_use]
    pub fn black() -> Self {
        Self {
            space: b"DeviceGray".to_vec(),
            components: vec![0.0],
            rgb: Some([0.0; 3]),
            pattern: None,
        }
    }

    /// Its RGB when its space gives one, else its own components: the most a reader can say of a space it cannot
    /// read, or of an uncolored pattern's tint.
    #[must_use]
    pub fn rgb_or_components(&self) -> Vec<f64> {
        self.rgb
            .map_or_else(|| self.components.clone(), |rgb| rgb.to_vec())
    }
}

/// D65, sRGB's white, in CIE XYZ.
const D65: [f64; 3] = [0.950_47, 1.0, 1.088_83];
/// The Bradford cone response matrix and its inverse.
const BRADFORD: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];
const BRADFORD_INVERSE: [[f64; 3]; 3] = [
    [0.986_992_9, -0.147_054_3, 0.159_962_7],
    [0.432_305_3, 0.518_360_3, 0.049_291_2],
    [-0.008_528_7, 0.040_042_8, 0.968_486_7],
];
/// CIE XYZ (D65) to linear sRGB (IEC 61966-2-1).
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

fn times(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

impl Lab {
    /// The sRGB of `L*`, `a*`, `b*` (see the module): `L*` clipped to 0..=100, `a*` and `b*` to the range.
    #[must_use]
    pub fn rgb(&self, [lightness, a_star, b_star]: [f64; 3]) -> [f64; 3] {
        let lightness = lightness.clamp(0.0, 100.0);
        let a_star = a_star.clamp(self.range[0], self.range[1]);
        let b_star = b_star.clamp(self.range[2], self.range[3]);
        // L*a*b* to XYZ under the space's white (CIE 15).
        let inverse = |t: f64| {
            if t > 6.0 / 29.0 {
                t * t * t
            } else {
                108.0 / 841.0 * (t - 4.0 / 29.0)
            }
        };
        let fy = (lightness + 16.0) / 116.0;
        let xyz = [
            self.white[0] * inverse(fy + a_star / 500.0),
            self.white[1] * inverse(fy),
            self.white[2] * inverse(fy - b_star / 200.0),
        ];
        srgb(self.white, xyz)
    }
}

impl Cal {
    /// The sRGB of the components, each clipped to 0..=1 (see the module); None unless there are as many as the
    /// space has.
    #[must_use]
    pub fn rgb(&self, components: &[f64]) -> Option<[f64; 3]> {
        if components.len() != if self.gray { 1 } else { 3 } {
            return None;
        }
        let mut xyz = [0.0; 3];
        for (k, c) in components.iter().enumerate() {
            let linear = c.clamp(0.0, 1.0).powf(self.gamma[k]);
            for (axis, sum) in xyz.iter_mut().enumerate() {
                *sum += self.matrix[k][axis] * linear;
            }
        }
        Some(srgb(self.white, xyz))
    }
}

/// The sRGB of CIE XYZ seen under `white`: Bradford-adapted to D65, then IEC 61966-2-1's matrix and transfer curve.
fn srgb(white: [f64; 3], xyz: [f64; 3]) -> [f64; 3] {
    // Bradford: from the space's white to D65, by the ratio of the two whites' cone responses.
    let (from, to) = (times(&BRADFORD, white), times(&BRADFORD, D65));
    let cone = times(&BRADFORD, xyz);
    let adapted = times(
        &BRADFORD_INVERSE,
        [0, 1, 2].map(|i| cone[i] * to[i] / from[i]),
    );
    times(&XYZ_TO_SRGB, adapted).map(|v| {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    })
}

impl Space {
    /// How many components a color of this space has.
    #[must_use]
    pub fn components(&self) -> usize {
        match self {
            Self::Gray | Self::Indexed(..) | Self::Cal(Cal { gray: true, .. }) => 1,
            Self::Rgb | Self::Lab(_) | Self::Cal(_) => 3,
            Self::Cmyk => 4,
            Self::Icc(icc, _) => icc.components,
            Self::Other(_, n) => *n,
            Self::Tint(t) => t.colorants.len(),
            Self::Pattern(base) => base.as_ref().map_or(0, |b| b.components()),
        }
    }

    /// The family name.
    #[must_use]
    pub fn family(&self) -> Vec<u8> {
        match self {
            Self::Gray => b"DeviceGray".to_vec(),
            Self::Rgb => b"DeviceRGB".to_vec(),
            Self::Cmyk => b"DeviceCMYK".to_vec(),
            Self::Icc(..) => b"ICCBased".to_vec(),
            Self::Cal(Cal { gray: true, .. }) => b"CalGray".to_vec(),
            Self::Cal(_) => b"CalRGB".to_vec(),
            Self::Lab(_) => b"Lab".to_vec(),
            Self::Indexed(..) => b"Indexed".to_vec(),
            Self::Tint(t) if t.device_n => b"DeviceN".to_vec(),
            Self::Tint(_) => b"Separation".to_vec(),
            Self::Other(name, _) => name.clone(),
            Self::Pattern(_) => b"Pattern".to_vec(),
        }
    }

    /// Why a profile this space converts through does not read -- its own, its base's or its alternate's -- when
    /// one does not: its colors then have no RGB.
    #[must_use]
    pub fn unread(&self) -> Option<&str> {
        match self {
            Self::Icc(_, Some(alternate)) => alternate.unread(),
            Self::Icc(icc, None) => icc.profile.as_ref().err().map(String::as_str),
            Self::Indexed(base, _) | Self::Pattern(Some(base)) => base.unread(),
            Self::Tint(t) => t.alternate.unread(),
            _ => None,
        }
    }

    /// Whether painting in it marks the page: not when its every colorant is `None` (8.6.6.4).
    #[must_use]
    pub fn paints(&self) -> bool {
        match self {
            Self::Tint(t) => t.colorants.iter().any(|c| c != b"None"),
            _ => true,
        }
    }

    /// The components of this space's initial color (8.6.5-8.6.6): black, a tint of 1, index 0; `L*a*b*` 0 brought
    /// into range.
    #[must_use]
    pub fn initial_components(&self) -> Vec<f64> {
        match self {
            Self::Cmyk => vec![0.0, 0.0, 0.0, 1.0],
            Self::Icc(icc, _) if icc.components == 4 => vec![0.0, 0.0, 0.0, 1.0],
            Self::Pattern(_) => Vec::new(), // no pattern yet: it paints nothing
            Self::Tint(t) => vec![1.0; t.colorants.len()],
            Self::Lab(lab) => vec![
                0.0,
                0.0_f64.clamp(lab.range[0], lab.range[1]),
                0.0_f64.clamp(lab.range[2], lab.range[3]),
            ],
            other => vec![0.0; other.components()],
        }
    }

    /// A color of this space from its components, converted at `intent` where the space has a profile.
    ///
    /// # Errors
    ///
    /// Refuses a tint transform that yields no color for them, and a profile that gives no transform at `intent`.
    pub fn color(&self, components: Vec<f64>, intent: Intent) -> Result<Color> {
        Ok(Color {
            space: self.family(),
            rgb: self.rgb(&components, intent)?,
            components,
            pattern: None,
        })
    }

    /// Each component's range (8.6.5-8.6.6): what an `Indexed` table's bytes span, 0 to 255.
    fn range(&self, i: usize) -> [f64; 2] {
        match (self, i) {
            (Self::Lab(_), 0) => [0.0, 100.0],
            (Self::Lab(lab), _) => [lab.range[2 * i - 2], lab.range[2 * i - 1]],
            _ => [0.0, 1.0],
        }
    }

    fn rgb(&self, c: &[f64], intent: Intent) -> Result<Option<[f64; 3]>> {
        let unit = |v: f64| v.clamp(0.0, 1.0);
        Ok(match (self, c) {
            (Self::Gray, [g]) => Some([unit(*g); 3]),
            (Self::Rgb, [r, g, b]) => Some([unit(*r), unit(*g), unit(*b)]),
            (Self::Cmyk, [cy, m, y, k]) => {
                Some([1.0 - unit(cy + k), 1.0 - unit(m + k), 1.0 - unit(y + k)])
            }
            (Self::Icc(icc, Some(alternate)), _) if icc.profile.is_err() => {
                alternate.rgb(c, intent)?
            }
            (Self::Icc(icc, _), _) => icc.rgb(c, intent).map_err(Error::malformed)?,
            (Self::Cal(cal), _) => cal.rgb(c),
            (Self::Lab(lab), [l, a, b]) => Some(lab.rgb([*l, *a, *b])),
            (Self::Pattern(Some(base)), _) if !c.is_empty() => base.rgb(c, intent)?,
            (Self::Tint(_), _) if !self.paints() => None,
            (Self::Tint(t), _) if c.len() == t.colorants.len() => {
                t.alternate.rgb(&t.transform.evaluate(c)?, intent)?
            }
            (Self::Indexed(base, table), [i]) => {
                let n = base.components();
                let index = i.round().max(0.0);
                // An index is a whole number in 0..=hival; the float-to-usize cast saturates.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "rounded, non-negative"
                )]
                let start = index as usize * n;
                let Some(entry) = table.get(start..start + n) else {
                    return Ok(None);
                };
                let entry: Vec<f64> = entry
                    .iter()
                    .enumerate()
                    .map(|(k, &b)| {
                        let [low, high] = base.range(k);
                        low + f64::from(b) / 255.0 * (high - low)
                    })
                    .collect();
                base.rgb(&entry, intent)?
            }
            _ => None,
        })
    }
}

impl File {
    /// The color space `value` names or is, in the context of `resources`.
    ///
    /// # Errors
    ///
    /// Refuses a reference that does not resolve, a `Lab` space with no white point, and a `Separation` or
    /// `DeviceN` space whose tint transform does not read or does not fit its colorants and alternate space.
    pub fn color_space(&self, value: &Object, resources: &Dict) -> Result<Space> {
        self.space_at_depth(value, resources, 0)
    }

    /// Whether the color space `value` names or is, in the context of `resources`, is a `Separation` or `DeviceN`
    /// whose every colorant is `None` (8.6.6.4): painting in it marks nothing. Read from its colorants' names alone
    /// (no tint transform, no lookup table); a space that does not read so is not one.
    #[must_use]
    pub fn colorant_none(&self, value: &Object, resources: &Dict) -> bool {
        let mut value = self.resolve(value).unwrap_or(Object::Null);
        if let Object::Name(name) = &value
            && let Ok(Some(Object::Dict(spaces))) = self.get(resources, b"ColorSpace")
        {
            value = spaces
                .get(name)
                .and_then(|s| self.resolve(s).ok())
                .unwrap_or(Object::Null);
        }
        let Object::Array(items) = value else {
            return false;
        };
        let names = match (
            items.first().and_then(Object::as_name),
            items.get(1).map(|n| self.resolve(n)),
        ) {
            (Some(b"Separation"), Some(Ok(name @ Object::Name(_)))) => vec![name],
            (Some(b"DeviceN"), Some(Ok(Object::Array(names)))) => names,
            _ => return false,
        };
        !names.is_empty()
            && names
                .iter()
                .all(|n| self.resolve(n).ok().as_ref().and_then(Object::as_name) == Some(b"None"))
    }

    /// The pattern `/name` in `resources`, painted in a content stream whose default space maps to page space by
    /// `base`, its colors converted at `intent`.
    ///
    /// # Errors
    ///
    /// Refuses a name `resources` has no pattern for, and a pattern of no known `/PatternType`.
    pub fn pattern(
        &self,
        name: &[u8],
        resources: &Dict,
        base: Matrix,
        intent: Intent,
    ) -> Result<Pattern> {
        let named = || String::from_utf8_lossy(name).into_owned();
        let Some(Object::Dict(patterns)) = self.get(resources, b"Pattern")? else {
            return Err(Error::malformed(format!(
                "no /Pattern resources for /{}",
                named()
            )));
        };
        let pattern = match patterns.get(name) {
            Some(p) => self.resolve(p)?,
            None => return Err(Error::malformed(format!("no pattern /{}", named()))),
        };
        let Some(dict) = pattern.as_dict() else {
            return Err(Error::malformed(format!(
                "pattern /{} is a {}",
                named(),
                pattern.kind()
            )));
        };
        let int = |key: &[u8]| -> Result<Option<i64>> {
            Ok(self.get(dict, key)?.and_then(|t| t.as_integer()))
        };
        let kind = match int(b"PatternType")? {
            Some(1) => PatternKind::Tiling {
                uncolored: int(b"PaintType")? == Some(2),
            },
            Some(2) => PatternKind::Shading,
            _ => {
                return Err(Error::malformed(format!(
                    "pattern /{} has no known /PatternType",
                    named()
                )));
            }
        };
        let own = match self.get(dict, b"Matrix")? {
            Some(Object::Array(m)) => {
                Matrix::from_numbers(&m.iter().filter_map(Object::as_number).collect::<Vec<_>>())
                    .unwrap_or(Matrix::IDENTITY)
            }
            _ => Matrix::IDENTITY,
        };
        Ok(Pattern {
            kind,
            matrix: own.then(base),
            object: Rc::new(pattern),
            intent,
        })
    }

    fn space_at_depth(&self, value: &Object, resources: &Dict, depth: usize) -> Result<Space> {
        let value = self.resolve(value)?;
        if depth > 8 {
            return Ok(Space::Other(b"Unknown".to_vec(), 1));
        }
        match &value {
            Object::Name(name) => Ok(match name.as_slice() {
                b"DeviceGray" | b"G" => Space::Gray,
                b"DeviceRGB" | b"RGB" => Space::Rgb,
                b"DeviceCMYK" | b"CMYK" => Space::Cmyk,
                b"Pattern" => Space::Pattern(None),
                _ => match self.get(resources, b"ColorSpace")? {
                    Some(Object::Dict(spaces)) => match spaces.get(name) {
                        Some(named) => self.space_at_depth(named, resources, depth + 1)?,
                        None => Space::Other(name.clone(), 1),
                    },
                    _ => Space::Other(name.clone(), 1),
                },
            }),
            Object::Array(items) => {
                let family = items
                    .first()
                    .and_then(Object::as_name)
                    .unwrap_or(b"")
                    .to_vec();
                Ok(match family.as_slice() {
                    b"DeviceGray" => Space::Gray,
                    b"DeviceRGB" => Space::Rgb,
                    b"DeviceCMYK" => Space::Cmyk,
                    b"Pattern" => Space::Pattern(match items.get(1) {
                        Some(base) => {
                            Some(Box::new(self.space_at_depth(base, resources, depth + 1)?))
                        }
                        None => None,
                    }),
                    b"Lab" => Space::Lab(self.lab(items.get(1))?),
                    b"CalGray" | b"CalRGB" => {
                        Space::Cal(self.cal(items.get(1), family == b"CalGray")?)
                    }
                    b"ICCBased" => {
                        let icc = self.icc(items.get(1))?;
                        let alternate = match icc.profile {
                            Ok(_) => None,
                            Err(_) => self.icc_alternate(items.get(1), &icc, resources, depth)?,
                        };
                        Space::Icc(icc, alternate.map(Box::new))
                    }
                    b"Indexed" | b"I" => {
                        let base = match items.get(1) {
                            Some(b) => self.space_at_depth(b, resources, depth + 1)?,
                            None => Space::Rgb,
                        };
                        let table = match items.get(3).map(|t| self.resolve(t)).transpose()? {
                            Some(Object::String(s)) => s,
                            Some(Object::Stream(s)) => self.decode(&s)?.into_bytes()?,
                            _ => Vec::new(),
                        };
                        Space::Indexed(Box::new(base), table)
                    }
                    b"Separation" | b"DeviceN" => {
                        Space::Tint(Rc::new(self.tint(items, resources, depth)?))
                    }
                    _ => Space::Other(family, 1),
                })
            }
            _ => Ok(Space::Other(b"Unknown".to_vec(), 1)),
        }
    }

    /// An `ICCBased` space's profile stream read, once per profile object: the profile, or why it does not read.
    /// Its components are its `/N`, else its profile's.
    fn icc(&self, stream: Option<&Object>) -> Result<Rc<Icc>> {
        match stream.and_then(Object::as_reference) {
            Some(r) => self.read_once(r.num, || self.icc_read(stream)),
            None => self.icc_read(stream).map(Rc::new),
        }
    }

    /// An `ICCBased` space's `/Alternate` (8.6.5.5), read when its profile `icc` does not: None when it states
    /// none.
    ///
    /// # Errors
    ///
    /// Refuses an alternate that does not read, or whose components are not the space's.
    fn icc_alternate(
        &self,
        stream: Option<&Object>,
        icc: &Icc,
        resources: &Dict,
        depth: usize,
    ) -> Result<Option<Space>> {
        let Some(Object::Stream(s)) = stream.map(|s| self.resolve(s)).transpose()? else {
            return Ok(None);
        };
        let Some(value) = s.dict.get(b"Alternate") else {
            return Ok(None);
        };
        let alternate = self.space_at_depth(value, resources, depth + 1)?;
        if alternate.components() != icc.components {
            return Err(Error::malformed(format!(
                "an ICCBased space of {} components states an /Alternate of {}",
                icc.components,
                alternate.components()
            )));
        }
        Ok(Some(alternate))
    }

    fn icc_read(&self, stream: Option<&Object>) -> Result<Icc> {
        let Some(Object::Stream(s)) = stream.map(|s| self.resolve(s)).transpose()? else {
            return Err(Error::malformed("an ICCBased space has no profile stream"));
        };
        let stated = match self.get(&s.dict, b"N")? {
            Some(n) => Some(
                n.as_integer()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| (1..=15).contains(n))
                    .ok_or_else(|| {
                        Error::malformed("an ICCBased space's /N is no count of 1 to 15")
                    })?,
            ),
            None => None,
        };
        Ok(match self.decode(&s).and_then(Decoded::into_bytes) {
            Ok(data) => Icc::read(&data, stated),
            Err(e) => Icc {
                components: stated.unwrap_or(0),
                profile: Err(format!("its ICC profile stream does not decode: {e}")),
            },
        })
    }

    /// A `CalGray` or `CalRGB` space's dictionary read: its white point (required), its gamma (1 when not stated)
    /// and, for `CalRGB`, its matrix (the identity when not stated).
    fn cal(&self, dict: Option<&Object>, gray: bool) -> Result<Cal> {
        let what = if gray {
            "a CalGray space"
        } else {
            "a CalRGB space"
        };
        let Some(Object::Dict(dict)) = dict.map(|d| self.resolve(d)).transpose()? else {
            return Err(Error::malformed(format!("{what} has no dictionary")));
        };
        let numbers = |key: &[u8]| -> Result<Option<Vec<f64>>> {
            Ok(match self.get(&dict, key)? {
                Some(Object::Array(items)) => items.iter().map(Object::as_number).collect(),
                Some(one) => one.as_number().map(|n| vec![n]),
                None => None,
            })
        };
        let white = numbers(b"WhitePoint")?
            .and_then(|w| <[f64; 3]>::try_from(w).ok())
            .filter(|w| w[0] > 0.0 && w[2] > 0.0 && (w[1] - 1.0).abs() < 1e-9)
            .ok_or_else(|| Error::malformed(format!("{what} has no valid /WhitePoint")))?;
        let gamma = match (numbers(b"Gamma")?, gray) {
            (None, _) => [1.0; 3],
            (Some(g), true) if g.len() == 1 => [g[0]; 3],
            (Some(g), false) if g.len() == 3 => [g[0], g[1], g[2]],
            _ => {
                return Err(Error::malformed(format!(
                    "{what}'s /Gamma is not one per component"
                )));
            }
        };
        if gamma.iter().any(|g| *g <= 0.0) {
            return Err(Error::malformed(format!("{what}'s /Gamma is not positive")));
        }
        let matrix = if gray {
            [white; 3]
        } else {
            match numbers(b"Matrix")? {
                None => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                Some(m) if m.len() == 9 => {
                    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
                }
                Some(_) => {
                    return Err(Error::malformed(format!(
                        "{what}'s /Matrix is not nine numbers"
                    )));
                }
            }
        };
        Ok(Cal {
            white,
            gamma,
            matrix,
            gray,
        })
    }

    /// A `Lab` space's dictionary read: its white point (required) and the a\* and b\* range (-100..100 each when
    /// not stated).
    fn lab(&self, dict: Option<&Object>) -> Result<Lab> {
        let Some(Object::Dict(dict)) = dict.map(|d| self.resolve(d)).transpose()? else {
            return Err(Error::malformed("a Lab space has no dictionary"));
        };
        let numbers = |key: &[u8]| -> Result<Option<Vec<f64>>> {
            Ok(match self.get(&dict, key)? {
                Some(Object::Array(items)) => items.iter().map(Object::as_number).collect(),
                _ => None,
            })
        };
        let white = numbers(b"WhitePoint")?
            .and_then(|w| <[f64; 3]>::try_from(w).ok())
            .filter(|w| w[0] > 0.0 && w[2] > 0.0 && (w[1] - 1.0).abs() < 1e-9)
            .ok_or_else(|| Error::malformed("a Lab space has no valid /WhitePoint"))?;
        let range = match numbers(b"Range")? {
            None => [-100.0, 100.0, -100.0, 100.0],
            Some(r) => <[f64; 4]>::try_from(r)
                .map_err(|_| Error::malformed("a Lab space's /Range is not four numbers"))?,
        };
        Ok(Lab { white, range })
    }

    /// A `Separation` (`[/Separation name alternate transform]`) or `DeviceN` (`[/DeviceN names alternate
    /// transform attributes?]`) space.
    fn tint(&self, items: &[Object], resources: &Dict, depth: usize) -> Result<Tint> {
        let device_n = items.first().and_then(Object::as_name) == Some(b"DeviceN");
        let what = if device_n {
            "a DeviceN space"
        } else {
            "a Separation space"
        };
        let (Some(names), Some(alternate), Some(transform)) =
            (items.get(1), items.get(2), items.get(3))
        else {
            return Err(Error::malformed(format!(
                "{what} misses its colorants, alternate space or tint transform"
            )));
        };
        let names = self.resolve(names)?;
        let colorants: Vec<Vec<u8>> = match (&names, device_n) {
            (Object::Name(name), false) => vec![name.clone()],
            (Object::Array(names), true) => names
                .iter()
                .map(|n| {
                    self.resolve(n)
                        .ok()
                        .and_then(|n| n.as_name().map(<[u8]>::to_vec))
                })
                .collect::<Option<_>>()
                .ok_or_else(|| Error::malformed(format!("{what}'s colorants are not names")))?,
            _ => {
                return Err(Error::malformed(format!(
                    "{what}'s colorants are not names"
                )));
            }
        };
        let alternate = self.space_at_depth(alternate, resources, depth + 1)?;
        let transform = self.function(transform)?;
        if transform.inputs() != colorants.len() || transform.outputs() != alternate.components() {
            return Err(Error::malformed(format!(
                "{what}'s tint transform takes {} and gives {}, for {} colorants into {} components",
                transform.inputs(),
                transform.outputs(),
                colorants.len(),
                alternate.components()
            )));
        }
        Ok(Tint {
            device_n,
            colorants,
            alternate,
            transform,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    #[test]
    fn device_colors_convert_to_rgb() {
        assert_eq!(
            Space::Gray.color(vec![0.5], Intent::default()).unwrap().rgb,
            Some([0.5; 3])
        );
        assert_eq!(
            Space::Cmyk
                .color(vec![1.0, 0.0, 0.0, 0.0], Intent::default())
                .unwrap()
                .rgb,
            Some([0.0, 1.0, 1.0])
        );
        let rgb = Space::Cmyk
            .color(vec![0.2, 0.5, 0.1, 0.1], Intent::default())
            .unwrap()
            .rgb
            .unwrap();
        assert!(near(rgb, [0.7, 0.4, 0.8], 1e-12), "{rgb:?}");
        assert_eq!(
            Space::Cmyk
                .color(Space::Cmyk.initial_components(), Intent::default())
                .unwrap()
                .rgb,
            Some([0.0; 3])
        );
    }

    #[test]
    fn an_indexed_color_looks_up_its_base() {
        let space = Space::Indexed(Box::new(Space::Rgb), vec![0, 0, 0, 255, 0, 0]);
        assert_eq!(
            space.color(vec![1.0], Intent::default()).unwrap().rgb,
            Some([1.0, 0.0, 0.0])
        );
        assert_eq!(space.color(vec![9.0], Intent::default()).unwrap().rgb, None);
    }

    const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

    #[test]
    fn a_lab_color_converts_by_the_cie_formulas() {
        let lab = Lab {
            white: D50,
            range: [-100.0, 100.0, -100.0, 100.0],
        };
        assert!(near(lab.rgb([100.0, 0.0, 0.0]), [1.0; 3], 1e-4));
        assert_eq!(lab.rgb([0.0, 0.0, 0.0]), [0.0; 3]);
        // Lab D50 (50, 20, -30) is sRGB (0.5212, 0.4237, 0.6685) by the CIE and IEC formulas (computed apart, with
        // numpy); MuPDF's color-managed conversion (lcms2) states (0.5211, 0.4237, 0.6684).
        let rgb = lab.rgb([50.0, 20.0, -30.0]);
        assert!(near(rgb, [0.5212, 0.4237, 0.6685], 1e-4), "{rgb:?}");
        // The white point is the space's white: L* 100 is white under any.
        let d65 = Lab { white: D65, ..lab };
        assert!(near(d65.rgb([100.0, 0.0, 0.0]), [1.0; 3], 1e-4));
        // a* and b* are clipped to the range.
        let narrow = Lab {
            range: [-10.0, 10.0, -10.0, 10.0],
            ..lab
        };
        assert_eq!(narrow.rgb([50.0, 90.0, 0.0]), lab.rgb([50.0, 10.0, 0.0]));
    }

    #[test]
    fn a_color_reads_as_its_rgb_else_its_components() {
        let rgb = Space::Cmyk
            .color(vec![0.0, 0.0, 0.0, 1.0], Intent::default())
            .unwrap();
        assert_eq!(rgb.rgb_or_components(), [0.0; 3]);
        let unread = Space::Other(b"Foo".to_vec(), 1)
            .color(vec![0.3], Intent::default())
            .unwrap();
        assert_eq!(unread.rgb_or_components(), [0.3]);
    }
}
