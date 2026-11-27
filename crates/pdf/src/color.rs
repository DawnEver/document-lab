//! Color spaces and colors (ISO 32000-2 section 8.6): the components as the file states them and their RGB, for a
//! `Pattern` color which kind of pattern it paints (8.7).
//!
//! RGB is sRGB in 0..=1, reached without a color profile:
//!
//! - `DeviceGray`, `DeviceRGB` and `DeviceCMYK` by the standard's own conversions (10.4.2): gray is each of red,
//!   green and blue; CMYK is `red = 1 - min(1, cyan + black)` and alike. `CalGray`, `CalRGB` and `ICCBased` are
//!   read as the device space of their component count -- their calibration or profile is not applied.
//! - `Lab` (8.6.5.4) by the CIE formulas: L\*a\*b\* to XYZ under the space's `/WhitePoint`, adapted to sRGB's D65
//!   white by the Bradford transform, then to sRGB (IEC 61966-2-1: its matrix, then its transfer curve).
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
use crate::function::Function;
use crate::object::{Dict, Object};

/// A color space, resolved far enough to count its components and convert them to RGB.
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// `DeviceGray` / `CalGray`.
    Gray,
    /// `DeviceRGB` / `CalRGB`.
    Rgb,
    /// `DeviceCMYK`.
    Cmyk,
    /// `ICCBased` with `n` components (converted as gray, RGB or CMYK by `n`).
    Icc(usize),
    /// `Lab`.
    Lab(Lab),
    /// `Indexed`: a base space and its lookup table.
    Indexed(Box<Space>, Vec<u8>),
    /// `Separation` or `DeviceN`: colorants, converted through a tint transform.
    Tint(Rc<Tint>),
    /// A space this reader cannot read (its family name, or the name it was set by, and its component count): its
    /// components are kept, with no RGB.
    Other(Vec<u8>, usize),
    /// `Pattern`: painted with a pattern, not a color.
    Pattern,
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

/// What a `Pattern` color paints an area with (8.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// A tiling pattern (`PatternType` 1): a cell repeated across the area; the cell is not interpreted.
    Tiling,
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
        // Bradford: from the space's white to D65, by the ratio of the two whites' cone responses.
        let (from, to) = (times(&BRADFORD, self.white), times(&BRADFORD, D65));
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
}

impl Space {
    /// How many components a color of this space has.
    #[must_use]
    pub fn components(&self) -> usize {
        match self {
            Self::Gray | Self::Indexed(..) => 1,
            Self::Rgb | Self::Lab(_) => 3,
            Self::Cmyk => 4,
            Self::Icc(n) | Self::Other(_, n) => *n,
            Self::Tint(t) => t.colorants.len(),
            Self::Pattern => 0,
        }
    }

    /// The family name.
    #[must_use]
    pub fn family(&self) -> Vec<u8> {
        match self {
            Self::Gray => b"DeviceGray".to_vec(),
            Self::Rgb => b"DeviceRGB".to_vec(),
            Self::Cmyk => b"DeviceCMYK".to_vec(),
            Self::Icc(_) => b"ICCBased".to_vec(),
            Self::Lab(_) => b"Lab".to_vec(),
            Self::Indexed(..) => b"Indexed".to_vec(),
            Self::Tint(t) if t.device_n => b"DeviceN".to_vec(),
            Self::Tint(_) => b"Separation".to_vec(),
            Self::Other(name, _) => name.clone(),
            Self::Pattern => b"Pattern".to_vec(),
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
            Self::Cmyk | Self::Icc(4) => vec![0.0, 0.0, 0.0, 1.0],
            Self::Tint(t) => vec![1.0; t.colorants.len()],
            Self::Lab(lab) => vec![
                0.0,
                0.0_f64.clamp(lab.range[0], lab.range[1]),
                0.0_f64.clamp(lab.range[2], lab.range[3]),
            ],
            other => vec![0.0; other.components()],
        }
    }

    /// A color of this space from its components.
    ///
    /// # Errors
    ///
    /// Refuses a tint transform that yields no color for them.
    pub fn color(&self, components: Vec<f64>) -> Result<Color> {
        Ok(Color {
            space: self.family(),
            rgb: self.rgb(&components)?,
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

    fn rgb(&self, c: &[f64]) -> Result<Option<[f64; 3]>> {
        let unit = |v: f64| v.clamp(0.0, 1.0);
        Ok(match (self, c) {
            (Self::Gray | Self::Icc(1), [g]) => Some([unit(*g); 3]),
            (Self::Rgb | Self::Icc(3), [r, g, b]) => Some([unit(*r), unit(*g), unit(*b)]),
            (Self::Cmyk | Self::Icc(4), [cy, m, y, k]) => {
                Some([1.0 - unit(cy + k), 1.0 - unit(m + k), 1.0 - unit(y + k)])
            }
            (Self::Lab(lab), [l, a, b]) => Some(lab.rgb([*l, *a, *b])),
            (Self::Tint(_), _) if !self.paints() => None,
            (Self::Tint(t), _) if c.len() == t.colorants.len() => {
                t.alternate.rgb(&t.transform.evaluate(c)?)?
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
                base.rgb(&entry)?
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

    /// The pattern `/name` in `resources`.
    ///
    /// # Errors
    ///
    /// Refuses a name `resources` has no pattern for, and a pattern of no known `/PatternType`.
    pub fn pattern(&self, name: &[u8], resources: &Dict) -> Result<Pattern> {
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
        let kind = match pattern.as_dict() {
            Some(d) => self.get(d, b"PatternType")?.and_then(|t| t.as_integer()),
            None => None,
        };
        match kind {
            Some(1) => Ok(Pattern::Tiling),
            Some(2) => Ok(Pattern::Shading),
            _ => Err(Error::malformed(format!(
                "pattern /{} has no known /PatternType",
                named()
            ))),
        }
    }

    fn space_at_depth(&self, value: &Object, resources: &Dict, depth: usize) -> Result<Space> {
        let value = self.resolve(value)?;
        if depth > 8 {
            return Ok(Space::Other(b"Unknown".to_vec(), 1));
        }
        match &value {
            Object::Name(name) => Ok(match name.as_slice() {
                b"DeviceGray" | b"G" | b"CalGray" => Space::Gray,
                b"DeviceRGB" | b"RGB" | b"CalRGB" => Space::Rgb,
                b"DeviceCMYK" | b"CMYK" => Space::Cmyk,
                b"Pattern" => Space::Pattern,
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
                    b"DeviceGray" | b"CalGray" => Space::Gray,
                    b"DeviceRGB" | b"CalRGB" => Space::Rgb,
                    b"DeviceCMYK" => Space::Cmyk,
                    b"Pattern" => Space::Pattern,
                    b"Lab" => Space::Lab(self.lab(items.get(1))?),
                    b"ICCBased" => {
                        let n = match items.get(1).map(|s| self.resolve(s)).transpose()? {
                            Some(Object::Stream(s)) => {
                                s.dict.get(b"N").and_then(Object::as_integer).unwrap_or(3)
                            }
                            _ => 3,
                        };
                        Space::Icc(usize::try_from(n).unwrap_or(3))
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
        assert_eq!(Space::Gray.color(vec![0.5]).unwrap().rgb, Some([0.5; 3]));
        assert_eq!(
            Space::Cmyk.color(vec![1.0, 0.0, 0.0, 0.0]).unwrap().rgb,
            Some([0.0, 1.0, 1.0])
        );
        let rgb = Space::Cmyk
            .color(vec![0.2, 0.5, 0.1, 0.1])
            .unwrap()
            .rgb
            .unwrap();
        assert!(near(rgb, [0.7, 0.4, 0.8], 1e-12), "{rgb:?}");
        assert_eq!(
            Space::Cmyk
                .color(Space::Cmyk.initial_components())
                .unwrap()
                .rgb,
            Some([0.0; 3])
        );
    }

    #[test]
    fn an_indexed_color_looks_up_its_base() {
        let space = Space::Indexed(Box::new(Space::Rgb), vec![0, 0, 0, 255, 0, 0]);
        assert_eq!(space.color(vec![1.0]).unwrap().rgb, Some([1.0, 0.0, 0.0]));
        assert_eq!(space.color(vec![9.0]).unwrap().rgb, None);
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
        let rgb = Space::Cmyk.color(vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(rgb.rgb_or_components(), [0.0; 3]);
        let unread = Space::Other(b"Foo".to_vec(), 1).color(vec![0.3]).unwrap();
        assert_eq!(unread.rgb_or_components(), [0.3]);
    }
}
